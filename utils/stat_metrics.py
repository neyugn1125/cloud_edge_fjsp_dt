"""
Statistical and Multi-Objective Evaluation Metrics for Green-FJSP.

Provides:
1. Multi-Objective Normalization: Min-Max scaling of (C_max, TOU_Cost, TWT) onto [0, 1].
2. Hypervolume (HV): Standard multi-objective metric using pymoo with ref_point=[1.1, 1.1, 1.1].
3. Inverted Generational Distance (IGD): Convergence and diversity indicator using pymoo.
4. Wilcoxon Signed-Rank Test: Non-parametric statistical hypothesis testing returning
   (p_value, symbol) with '+' (better), '-' (worse), '≈' (statistically equivalent).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from scipy import stats


def normalize_multiobjective_fronts(
    fronts: Union[List[np.ndarray], Dict[str, np.ndarray], np.ndarray],
    ideal_point: Optional[np.ndarray] = None,
    nadir_point: Optional[np.ndarray] = None,
) -> Tuple[Union[List[np.ndarray], Dict[str, np.ndarray], np.ndarray], np.ndarray, np.ndarray]:
    """
    Min-Max normalization of 3 objectives (C_max, TOU_Cost, TWT) to [0, 1]
    based on the pooled set of all non-dominated solutions.

    Args:
        fronts: Single front (N, 3), list of fronts [F_1, F_2, ...], or dict {algo_name: F}.
        ideal_point: Optional vector of global minima [min_f1, min_f2, min_f3].
        nadir_point: Optional vector of global maxima [max_f1, max_f2, max_f3].

    Returns:
        Tuple of (normalized_fronts, ideal_point, nadir_point).
    """
    is_dict = isinstance(fronts, dict)
    is_single = isinstance(fronts, np.ndarray) and fronts.ndim == 2

    if is_dict:
        front_list = [np.asarray(f, dtype=np.float64) for f in fronts.values()]
    elif is_single:
        front_list = [np.asarray(fronts, dtype=np.float64)]
    else:
        front_list = [np.asarray(f, dtype=np.float64) for f in fronts]

    # Pool valid non-empty solution vectors
    valid_pts = [f for f in front_list if f.ndim == 2 and f.shape[0] > 0]
    if not valid_pts:
        raise ValueError("Cannot normalize: all provided fronts are empty.")

    all_pooled = np.vstack(valid_pts)

    if ideal_point is None:
        ideal = np.min(all_pooled, axis=0)
    else:
        ideal = np.asarray(ideal_point, dtype=np.float64)

    if nadir_point is None:
        nadir = np.max(all_pooled, axis=0)
    else:
        nadir = np.asarray(nadir_point, dtype=np.float64)

    denom = nadir - ideal
    # Prevent division by zero if all solutions have identical objective value
    denom = np.where(denom > 1e-9, denom, 1.0)

    def _norm(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == 0:
            return arr.copy()
        scaled = (arr - ideal) / denom
        return np.clip(scaled, 0.0, 1.0)

    if is_dict:
        norm_result = {k: _norm(np.asarray(v, dtype=np.float64)) for k, v in fronts.items()}
        return norm_result, ideal, nadir
    elif is_single:
        return _norm(np.asarray(fronts, dtype=np.float64)), ideal, nadir
    else:
        norm_result = [_norm(f) for f in front_list]
        return norm_result, ideal, nadir


def calculate_hypervolume(
    front: np.ndarray,
    ref_point: np.ndarray = np.array([1.1, 1.1, 1.1]),
) -> float:
    """
    Calculate Hypervolume (HV) indicator for a 3-objective Pareto front.

    Args:
        front: Array of normalized objective vectors in [0, 1] with shape (N, 3).
        ref_point: Reference point vector, default [1.1, 1.1, 1.1].

    Returns:
        float: Hypervolume metric (higher is better, in [0.0, 1.1^3 = 1.331]).
    """
    pts = np.asarray(front, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return 0.0

    ref = np.asarray(ref_point, dtype=np.float64)

    # Filter out points that do not strictly dominate the reference point
    valid_mask = np.all(pts < ref, axis=1)
    valid_pts = pts[valid_mask]
    if len(valid_pts) == 0:
        return 0.0

    hv_indicator = HV(ref_point=ref)
    return float(hv_indicator(valid_pts))


def calculate_igd(
    front: np.ndarray,
    true_pareto_front: np.ndarray,
) -> float:
    """
    Calculate Inverted Generational Distance (IGD) indicator.
    Measures both convergence to the reference Pareto front and diversity along it.

    Args:
        front: Algorithm's obtained Pareto front of shape (N, 3).
        true_pareto_front: Reference / true Pareto front of shape (M, 3).

    Returns:
        float: IGD metric (lower is better, 0.0 = perfect coverage).
    """
    pts = np.asarray(front, dtype=np.float64)
    ref = np.asarray(true_pareto_front, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[0] == 0:
        return float("inf")
    if ref.ndim != 2 or ref.shape[0] == 0:
        raise ValueError("Reference Pareto front must be non-empty.")

    igd_indicator = IGD(ref)
    return float(igd_indicator(pts))


def run_wilcoxon_test(
    proposed_scores: Union[List[float], np.ndarray],
    baseline_scores: Union[List[float], np.ndarray],
    higher_is_better: bool = False,
    alpha: float = 0.05,
) -> Tuple[float, str]:
    """
    Execute Wilcoxon Signed-Rank statistical test comparing Proposed against Baseline.

    Hypothesis Testing:
        - If higher_is_better is True (e.g. Hypervolume):
            '+' : Proposed > Baseline with p < alpha
            '-' : Proposed < Baseline with p < alpha
            '≈' : Statistically equivalent with p >= alpha
        - If higher_is_better is False (e.g. IGD, Makespan, TOU Cost, Tardiness):
            '+' : Proposed < Baseline with p < alpha (lower is superior)
            '-' : Proposed > Baseline with p < alpha (higher is inferior)
            '≈' : Statistically equivalent with p >= alpha

    Args:
        proposed_scores: Scores across independent runs for Proposed algorithm.
        baseline_scores: Scores across independent runs for Baseline algorithm.
        higher_is_better: True if higher score is better, False for minimization objectives.
        alpha: Significance level (default: 0.05).

    Returns:
        Tuple of (p_value, significance_symbol).
    """
    prop = np.asarray(proposed_scores, dtype=np.float64)
    base = np.asarray(baseline_scores, dtype=np.float64)

    if len(prop) != len(base):
        raise ValueError(f"Sample length mismatch: proposed={len(prop)}, baseline={len(base)}")
    if len(prop) < 2:
        raise ValueError(f"At least 2 samples are required for Wilcoxon test, got {len(prop)}")

    diff = prop - base
    # If all paired samples are identical, p-value is 1.0 (equivalent)
    if np.allclose(diff, 0.0):
        return 1.0, "≈"

    # Execute Wilcoxon signed-rank test
    res = stats.wilcoxon(prop, base)
    p_val = float(res.pvalue)

    if p_val < alpha:
        if higher_is_better:
            is_better = float(np.mean(prop)) > float(np.mean(base))
        else:
            is_better = float(np.mean(prop)) < float(np.mean(base))
        symbol = "+" if is_better else "-"
    else:
        symbol = "≈"

    return p_val, symbol

