/**
 * ============================================================================
 * Cloud-Edge Collaborative Digital Twin Framework
 * ESP32 Physical Twin Sensor Node Firmware (Breadboard / CNC Machine Twin)
 * ============================================================================
 * Target: ESP32-WROOM-32 / DevKitC
 * Framework: Arduino / ESP-IDF
 *
 * Hardware Peripherals:
 * - I2C Bus: SDA = GPIO 21, SCL = GPIO 22 (400 kHz Fast-Mode)
 * - Power Sensor: INA226 (I2C Address: 0x40, Shunt: 10 mOhm)
 * - Accelerometer: ADXL345 (I2C Address: 0x53, Full-Res +/- 16g or +/- 2g)
 * - Actuators:
 *     - SSR Standby Power: GPIO 25
 *     - SSR Main CNC Power: GPIO 26
 *
 * Signal Processing:
 * - Sampling Frequency: 200 Hz (5 ms / sample)
 * - Power Filtering: SimpleKalmanFilter1D (3-tap Median + 1D Kalman Filter)
 * - Vibration Calculation: 20-sample sliding window RMS (100 ms window)
 * - Telemetry Output: 10 Hz (100 ms interval) via UART (115200 bps)
 * - Serial Commands: CMD:IDLE, CMD:STANDBY, CMD:OFF, CMD:CUTTING
 * ============================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <math.h>

// ----------------------------------------------------------------------------
// Pin Configurations & Constants
// ----------------------------------------------------------------------------
#define PIN_I2C_SDA         21
#define PIN_I2C_SCL         22
#define PIN_SSR_STANDBY     25
#define PIN_SSR_MAIN        26

#define INA226_I2C_ADDR     0x40
#define ADXL345_I2C_ADDR    0x53
#define I2C_FREQ_HZ         400000

#define SAMPLE_INTERVAL_US  5000       // 200 Hz sampling (5 ms)
#define TELEMETRY_INTERVAL_MS 100      // 10 Hz telemetry (100 ms)
#define VIB_WINDOW_SIZE     20         // 20 samples = 100 ms at 200 Hz

// ----------------------------------------------------------------------------
// Machine States
// ----------------------------------------------------------------------------
enum MachineState {
    STATE_OFF = 0,
    STATE_STANDBY,
    STATE_IDLE,
    STATE_CUTTING
};

volatile MachineState current_state = STATE_IDLE;

const char* stateToString(MachineState s) {
    switch (s) {
        case STATE_CUTTING: return "CUTTING";
        case STATE_IDLE:    return "IDLE";
        case STATE_STANDBY: return "STANDBY";
        case STATE_OFF:     return "OFF";
        default:            return "UNKNOWN";
    }
}

// ----------------------------------------------------------------------------
// Digital Signal Processing: 1D Kalman Filter with 3-Tap Median Pre-Filter
// ----------------------------------------------------------------------------
class SimpleKalmanFilter1D {
private:
    float _err_measure;
    float _err_estimate;
    float _q;
    float _current_estimate;
    float _last_estimate;
    float _kalman_gain;
    bool  _initialized;

    // 3-tap circular buffer for impulse spike rejection
    float _med_buf[3];
    uint8_t _med_count;
    uint8_t _med_head;

    float compute_median(float val) {
        _med_buf[_med_head] = val;
        _med_head = (_med_head + 1) % 3;
        if (_med_count < 3) {
            _med_count++;
        }

        if (_med_count == 1) return _med_buf[0];
        if (_med_count == 2) return 0.5f * (_med_buf[0] + _med_buf[1]);

        // Exact 3-element median without dynamic allocation
        float a = _med_buf[0];
        float b = _med_buf[1];
        float c = _med_buf[2];
        return fmaxf(fminf(a, b), fminf(fmaxf(a, b), c));
    }

public:
    SimpleKalmanFilter1D(float mea_e = 4.0f, float est_e = 4.0f, float q = 0.20f)
        : _err_measure(mea_e), _err_estimate(est_e), _q(q),
          _current_estimate(0.0f), _last_estimate(0.0f), _kalman_gain(0.0f),
          _initialized(false), _med_count(0), _med_head(0) {
        _med_buf[0] = 0.0f;
        _med_buf[1] = 0.0f;
        _med_buf[2] = 0.0f;
    }

    void reset(float initial_val = 0.0f) {
        _current_estimate = initial_val;
        _last_estimate = initial_val;
        _err_estimate = _err_measure;
        _initialized = false;
        _med_count = 0;
        _med_head = 0;
        _med_buf[0] = initial_val;
        _med_buf[1] = initial_val;
        _med_buf[2] = initial_val;
    }

    float update_estimate(float mea) {
        float med = compute_median(mea);

        if (!_initialized) {
            _current_estimate = med;
            _last_estimate = med;
            _initialized = true;
            return med;
        }

        _kalman_gain = _err_estimate / (_err_estimate + _err_measure);
        _current_estimate = _last_estimate + _kalman_gain * (med - _last_estimate);
        _err_estimate = (1.0f - _kalman_gain) * _err_estimate + fabsf(_last_estimate - _current_estimate) * _q;
        _last_estimate = _current_estimate;

        return _current_estimate;
    }

    float get_estimate() const {
        return _current_estimate;
    }
};

// ----------------------------------------------------------------------------
// Vibration RMS Calculator (20-sample sliding window = 100ms at 200Hz)
// ----------------------------------------------------------------------------
class VibrationRMSCalculator {
private:
    float _buf[VIB_WINDOW_SIZE];
    uint8_t _head;
    uint8_t _count;
    float _sq_sum;

public:
    VibrationRMSCalculator() : _head(0), _count(0), _sq_sum(0.0f) {
        for (int i = 0; i < VIB_WINDOW_SIZE; i++) {
            _buf[i] = 0.0f;
        }
    }

    void reset() {
        _head = 0;
        _count = 0;
        _sq_sum = 0.0f;
        for (int i = 0; i < VIB_WINDOW_SIZE; i++) {
            _buf[i] = 0.0f;
        }
    }

    float update(float ax, float ay, float az) {
        float mag_sq = ax * ax + ay * ay + az * az;

        if (_count >= VIB_WINDOW_SIZE) {
            _sq_sum -= _buf[_head];
            if (_sq_sum < 0.0f) _sq_sum = 0.0f; // Prevent floating point underflow
        } else {
            _count++;
        }

        _buf[_head] = mag_sq;
        _sq_sum += mag_sq;
        _head = (_head + 1) % VIB_WINDOW_SIZE;

        return sqrtf(_sq_sum / (float)_count);
    }
};

// ----------------------------------------------------------------------------
// Global Instances & Sensor Status
// ----------------------------------------------------------------------------
SimpleKalmanFilter1D powerFilter(4.0f, 4.0f, 0.20f);
VibrationRMSCalculator vibCalculator;

bool ina226_connected = false;
bool adxl345_connected = false;
uint16_t i2c_consecutive_errors = 0;

// Telemetry State Variables
float latest_v_bus = 0.0f;
float latest_i_amp = 0.0f;
float latest_p_raw = 0.0f;
float latest_p_filt = 0.0f;
float latest_vib_rms = 0.0f;

// ----------------------------------------------------------------------------
// I2C Bus Recovery: Generates 9 SCL Clock Pulses and STOP Condition
// ----------------------------------------------------------------------------
void recoverI2CBus() {
    Wire.end();
    pinMode(PIN_I2C_SDA, INPUT_PULLUP);
    pinMode(PIN_I2C_SCL, OUTPUT);

    // Toggle SCL 9 times to allow stuck slave to release SDA
    for (int i = 0; i < 9; i++) {
        digitalWrite(PIN_I2C_SCL, HIGH);
        delayMicroseconds(10);
        digitalWrite(PIN_I2C_SCL, LOW);
        delayMicroseconds(10);
    }

    // Generate standard I2C STOP condition: SDA transitions from LOW to HIGH while SCL is HIGH
    pinMode(PIN_I2C_SDA, OUTPUT);
    digitalWrite(PIN_I2C_SDA, LOW);
    delayMicroseconds(10);
    digitalWrite(PIN_I2C_SCL, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_I2C_SDA, HIGH);
    delayMicroseconds(10);

    // Re-initialize Wire peripheral
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, I2C_FREQ_HZ);
    Wire.setTimeOut(50);
}

// ----------------------------------------------------------------------------
// SSR Actuator Control
// ----------------------------------------------------------------------------
void applySSRState(MachineState state) {
    current_state = state;
    switch (state) {
        case STATE_CUTTING:
            digitalWrite(PIN_SSR_MAIN, HIGH);
            digitalWrite(PIN_SSR_STANDBY, LOW);
            break;
        case STATE_IDLE:
            digitalWrite(PIN_SSR_MAIN, HIGH);
            digitalWrite(PIN_SSR_STANDBY, LOW);
            break;
        case STATE_STANDBY:
            digitalWrite(PIN_SSR_MAIN, LOW);
            digitalWrite(PIN_SSR_STANDBY, HIGH);
            break;
        case STATE_OFF:
            digitalWrite(PIN_SSR_MAIN, LOW);
            digitalWrite(PIN_SSR_STANDBY, LOW);
            break;
    }
}

// ----------------------------------------------------------------------------
// Sensor Initialization & Communication
// ----------------------------------------------------------------------------
bool initINA226() {
    Wire.beginTransmission(INA226_I2C_ADDR);
    if (Wire.endTransmission() != 0) {
        return false;
    }
    // Configuration Register (0x00): Reset & configure continuous shunt + bus, 1.1ms conversion
    Wire.beginTransmission(INA226_I2C_ADDR);
    Wire.write(0x00);
    Wire.write(0x41); // Config MSB
    Wire.write(0x27); // Config LSB (1.1ms, avg 1, shunt + bus continuous)
    return (Wire.endTransmission() == 0);
}

bool initADXL345() {
    Wire.beginTransmission(ADXL345_I2C_ADDR);
    if (Wire.endTransmission() != 0) {
        return false;
    }
    // Data Format (0x31): Full resolution mode, +/- 16g (0x0B) or +/- 2g (0x08)
    Wire.beginTransmission(ADXL345_I2C_ADDR);
    Wire.write(0x31);
    Wire.write(0x08); // Full resolution, +/- 2g (3.9 mg/LSB)
    if (Wire.endTransmission() != 0) return false;

    // Power Control (0x2D): Measurement Mode
    Wire.beginTransmission(ADXL345_I2C_ADDR);
    Wire.write(0x2D);
    Wire.write(0x08); // Measure bit ON
    return (Wire.endTransmission() == 0);
}

bool readINA226Data(float &v_bus, float &i_amp, float &p_raw) {
    if (!ina226_connected) return false;

    // Read Bus Voltage (Register 0x02, LSB = 1.25 mV)
    Wire.beginTransmission(INA226_I2C_ADDR);
    Wire.write(0x02);
    if (Wire.endTransmission() != 0) return false;
    if (Wire.requestFrom((uint8_t)INA226_I2C_ADDR, (uint8_t)2) != 2) return false;
    uint16_t raw_v = ((uint16_t)Wire.read() << 8) | Wire.read();
    v_bus = (float)raw_v * 0.00125f;

    // Read Shunt Voltage (Register 0x01, LSB = 2.5 uV, Signed 16-bit)
    Wire.beginTransmission(INA226_I2C_ADDR);
    Wire.write(0x01);
    if (Wire.endTransmission() != 0) return false;
    if (Wire.requestFrom((uint8_t)INA226_I2C_ADDR, (uint8_t)2) != 2) return false;
    int16_t raw_shunt = ((int16_t)Wire.read() << 8) | Wire.read();
    
    // Shunt resistor = 10 mOhm (0.010 Ohm) -> I = V_shunt / R_shunt
    i_amp = ((float)raw_shunt * 0.0000025f) / 0.010f;
    if (i_amp < 0.0f) i_amp = 0.0f;

    p_raw = v_bus * i_amp;
    return true;
}

bool readADXL345Data(float &ax, float &ay, float &az) {
    if (!adxl345_connected) return false;

    // Read 6 bytes starting from DATAX0 (0x32)
    Wire.beginTransmission(ADXL345_I2C_ADDR);
    Wire.write(0x32);
    if (Wire.endTransmission() != 0) return false;
    if (Wire.requestFrom((uint8_t)ADXL345_I2C_ADDR, (uint8_t)6) != 6) return false;

    int16_t rx = Wire.read() | ((int16_t)Wire.read() << 8);
    int16_t ry = Wire.read() | ((int16_t)Wire.read() << 8);
    int16_t rz = Wire.read() | ((int16_t)Wire.read() << 8);

    // 3.9 mg/LSB in full resolution mode
    ax = (float)rx * 0.0039f;
    ay = (float)ry * 0.0039f;
    az = (float)rz * 0.0039f;
    return true;
}

// ----------------------------------------------------------------------------
// Breadboard / Hardware-in-the-Loop Simulation Fallback
// Generates realistic signals if physical sensors are disconnected or on bench
// ----------------------------------------------------------------------------
void generateSimulatedSensors(float &v_bus, float &i_amp, float &p_raw, float &ax, float &ay, float &az) {
    static uint32_t sim_tick = 0;
    sim_tick++;

    // Random Gaussian-like noise via central limit approximation
    float noise1 = ((float)(random(-100, 100)) / 100.0f) * 1.5f;
    float noise2 = ((float)(random(-100, 100)) / 100.0f) * 0.05f;

    // Intermittent PWM motor switching spike (every ~35 samples, +25W)
    float pwm_spike = (random(0, 100) < 5) ? 25.0f : 0.0f;

    switch (current_state) {
        case STATE_CUTTING: {
            v_bus = 24.05f + (float)random(-5, 5) * 0.01f;
            float p_base = 57.6f + noise1 + pwm_spike;
            if (p_base < 0.0f) p_base = 0.0f;
            p_raw = p_base;
            i_amp = p_raw / v_bus;

            // Spindle vibration at ~12000 RPM (fundamental + harmonics)
            float t = (float)sim_tick * 0.005f;
            ax = 0.55f * sinf(2.0f * M_PI * 200.0f * t) + noise2;
            ay = 0.72f * cosf(2.0f * M_PI * 200.0f * t) + noise2;
            az = 0.85f * sinf(2.0f * M_PI * 400.0f * t) + noise2;
            break;
        }
        case STATE_IDLE: {
            v_bus = 24.02f + (float)random(-3, 3) * 0.01f;
            float p_base = 15.6f + (noise1 * 0.3f);
            if (p_base < 0.0f) p_base = 0.0f;
            p_raw = p_base;
            i_amp = p_raw / v_bus;

            ax = 0.04f + noise2 * 0.2f;
            ay = 0.05f + noise2 * 0.2f;
            az = 0.06f + noise2 * 0.2f;
            break;
        }
        case STATE_STANDBY: {
            v_bus = 24.00f;
            p_raw = 3.6f + (noise1 * 0.1f);
            if (p_raw < 0.0f) p_raw = 0.0f;
            i_amp = p_raw / v_bus;

            ax = 0.01f;
            ay = 0.01f;
            az = 0.01f;
            break;
        }
        case STATE_OFF:
        default: {
            v_bus = 0.0f;
            i_amp = 0.0f;
            p_raw = 0.0f;
            ax = 0.0f;
            ay = 0.0f;
            az = 0.0f;
            break;
        }
    }
}

// ----------------------------------------------------------------------------
// Serial Command Processing
// ----------------------------------------------------------------------------
char serial_rx_buf[64];
uint8_t serial_rx_idx = 0;

void processCommand(const char* cmd) {
    // Expected format: CMD:IDLE, CMD:STANDBY, CMD:OFF, CMD:CUTTING
    if (strncmp(cmd, "CMD:", 4) == 0) {
        const char* sub = cmd + 4;
        if (strcmp(sub, "CUTTING") == 0) {
            applySSRState(STATE_CUTTING);
            Serial.println("{\"ack\": \"CMD:CUTTING\", \"status\": \"OK\"}");
        } else if (strcmp(sub, "IDLE") == 0) {
            applySSRState(STATE_IDLE);
            Serial.println("{\"ack\": \"CMD:IDLE\", \"status\": \"OK\"}");
        } else if (strcmp(sub, "STANDBY") == 0) {
            applySSRState(STATE_STANDBY);
            Serial.println("{\"ack\": \"CMD:STANDBY\", \"status\": \"OK\"}");
        } else if (strcmp(sub, "OFF") == 0) {
            applySSRState(STATE_OFF);
            Serial.println("{\"ack\": \"CMD:OFF\", \"status\": \"OK\"}");
        } else {
            Serial.printf("{\"error\": \"UNKNOWN_CMD\", \"cmd\": \"%s\"}\n", cmd);
        }
    }
}

void handleSerialInput() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serial_rx_idx > 0) {
                serial_rx_buf[serial_rx_idx] = '\0';
                processCommand(serial_rx_buf);
                serial_rx_idx = 0;
            }
        } else {
            if (serial_rx_idx < sizeof(serial_rx_buf) - 1) {
                serial_rx_buf[serial_rx_idx++] = c;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Arduino Setup & Main Loop
// ----------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(200);

    // Initialize Actuator GPIOs
    pinMode(PIN_SSR_STANDBY, OUTPUT);
    pinMode(PIN_SSR_MAIN, OUTPUT);
    applySSRState(STATE_IDLE); // Default initial state: IDLE

    // Initialize I2C Bus at 400kHz
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, I2C_FREQ_HZ);
    Wire.setTimeOut(50);

    // Detect and initialize I2C sensors
    ina226_connected = initINA226();
    adxl345_connected = initADXL345();

    if (!ina226_connected || !adxl345_connected) {
        Serial.printf("[WARN] I2C Sensor Status - INA226: %s, ADXL345: %s (Fallback to Simulation)\n",
                      ina226_connected ? "FOUND" : "NOT_FOUND",
                      adxl345_connected ? "FOUND" : "NOT_FOUND");
    }

    powerFilter.reset(15.6f);
    vibCalculator.reset();
}

void loop() {
    static uint32_t last_sample_us = 0;
    static uint32_t last_telemetry_ms = 0;

    handleSerialInput();

    uint32_t current_us = micros();
    // ------------------------------------------------------------------------
    // 200 Hz Sampling & DSP Task (5 ms interval)
    // ------------------------------------------------------------------------
    if (current_us - last_sample_us >= SAMPLE_INTERVAL_US) {
        last_sample_us = current_us;

        float v_bus = 0.0f;
        float i_amp = 0.0f;
        float p_raw = 0.0f;
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        bool read_ok = true;
        if (ina226_connected && adxl345_connected) {
            read_ok &= readINA226Data(v_bus, i_amp, p_raw);
            read_ok &= readADXL345Data(ax, ay, az);
        } else {
            read_ok = false;
        }

        if (!read_ok) {
            i2c_consecutive_errors++;
            if (i2c_consecutive_errors >= 5) {
                recoverI2CBus();
                i2c_consecutive_errors = 0;
            }
            // Generate breadboard simulated data
            generateSimulatedSensors(v_bus, i_amp, p_raw, ax, ay, az);
        } else {
            i2c_consecutive_errors = 0;
        }

        // Apply DSP Filtering
        float p_filt = powerFilter.update_estimate(p_raw);
        float vib_rms = vibCalculator.update(ax, ay, az);

        latest_v_bus = v_bus;
        latest_i_amp = i_amp;
        latest_p_raw = p_raw;
        latest_p_filt = p_filt;
        latest_vib_rms = vib_rms;
    }

    // ------------------------------------------------------------------------
    // 10 Hz Telemetry Broadcast Task (100 ms interval)
    // ------------------------------------------------------------------------
    uint32_t current_ms = millis();
    if (current_ms - last_telemetry_ms >= TELEMETRY_INTERVAL_MS) {
        last_telemetry_ms = current_ms;

        JsonDocument doc;
        doc["ts_ms"] = current_ms;
        doc["v_bus"] = roundf(latest_v_bus * 100.0f) / 100.0f;
        doc["i_amp"] = roundf(latest_i_amp * 100.0f) / 100.0f;
        doc["p_raw_w"] = roundf(latest_p_raw * 10.0f) / 10.0f;
        doc["p_filt_w"] = roundf(latest_p_filt * 10.0f) / 10.0f;
        doc["vib_rms_g"] = roundf(latest_vib_rms * 100.0f) / 100.0f;
        doc["ssr_state"] = stateToString(current_state);

        serializeJson(doc, Serial);
        Serial.println();
    }
}
