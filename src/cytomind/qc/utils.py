"""
Utility functions for QC evaluation.

Provides shared utilities for:
- Threshold validation
- Event metric computation
- Outlier detection
- Gate diagnostics extraction

Used across all QC evaluators to avoid code duplication.
"""
from __future__ import annotations
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
from scipy.stats import median_abs_deviation as mad, chi2

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from cytomind.domain.gates import GateNode
else:
    NDArray = object
    GateNode = object


# ============================================================================
# Validator helpers for threshold checks
# ============================================================================

def validate_percentage(name: str, value: float) -> None:
    """Raise ValueError if value is not in [0,1]."""
    if value is None:
        return
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1] range.")


def validate_warn_severe(
    warn_name: str,
    warn: float,
    severe_name: str,
    severe: float,
    are_percentage: bool = True
) -> None:
    """Validate two percentage thresholds: each in [0,1] and severe >= warn."""
    if are_percentage:
        validate_percentage(warn_name, warn)
        validate_percentage(severe_name, severe)
    if severe < warn:
        raise ValueError(f"{severe_name} must be greater than or equal to {warn_name}.")


def validate_percentage_range(name: str, value: tuple[float, float]) -> None:
    """Raise ValueError if values are not valid percentages."""
    if value is None or len(value) != 2:
        return
    if not (0.0 <= value[0] <= 1.0 and 0.0 <= value[1] <= 1.0):
        raise ValueError(f"{name} must be a tuple of values in [0, 1] range.")
    if value[0] > value[1]:
        raise ValueError(f"{name}: min threshold must be <= max threshold.")


# ============================================================================
# Gate event metric computation
# ============================================================================

def compute_gate_event_metrics(
    gate_mask: NDArray[np.bool_],
    parent_mask: NDArray[np.bool_] | None = None,
    total_events: int | None = None,
) -> dict[str, Any]:
    """
    Compute event count and ratio metrics for a gate mask.

    Parameters
    ----------
    gate_mask : NDArray[np.bool_]
        Boolean mask for events passing the gate
    parent_mask : NDArray[np.bool_] | None
        Optional parent gate mask for computing ratio_parent
    total_events : int | None
        Total number of events. If None, uses len(gate_mask)

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - n_events_passing: int - Number of events passing the gate
        - n_events_total: int - Total number of events
        - ratio_total: float - Ratio of passing events to total events
        - ratio_parent: float - Ratio of passing events to parent events (nan if no parent)
        - n_parent_events: int - Number of events in parent gate (only if parent_mask provided)

    Examples
    --------
    >>> gate_mask = np.array([True, False, True, True])
    >>> metrics = compute_gate_event_metrics(gate_mask)
    >>> metrics['n_events_passing']
    3
    >>> metrics['ratio_total']
    0.75
    """
    gate_mask = np.asarray(gate_mask, dtype=bool)
    n_total = total_events if total_events is not None else len(gate_mask)
    n_passing = int(np.sum(gate_mask))

    metrics = {
        "n_events_passing": n_passing,
        "n_events_total": n_total,
        "ratio_total": float(n_passing / n_total) if n_total > 0 else 0.0,
        "ratio_parent": float('nan'),
    }

    # Compute ratio relative to parent if provided
    if parent_mask is not None:
        parent_mask = np.asarray(parent_mask, dtype=bool)
        n_parent = int(np.sum(parent_mask))
        metrics["n_parent_events"] = n_parent

        if n_parent > 0:
            # Count events that pass both parent and gate masks
            n_passing_in_parent = int(np.sum(gate_mask & parent_mask))
            metrics["ratio_parent"] = float(n_passing_in_parent / n_parent)
        else:
            metrics["ratio_parent"] = float('nan')
    else:
        # If no parent, ratio_parent is same as ratio_total
        metrics["ratio_parent"] = metrics["ratio_total"]
        metrics["n_parent_events"] = n_total

    return metrics


# ============================================================================
# Outlier detection
# ============================================================================

def dict_iqr_score(
    values: dict[str, float],
    use_mad: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Compute an IQR-based outlier score.

    Score is 0 for values inside [Q1, Q3].
    Outside that interval, score is the distance from the nearest quartile
    expressed in IQR units.

    Parameters
    ----------
    values : dict[str, float]
        Mapping of sample_id → metric value
    use_mad : bool
        Ignored for IQR method (included for consistent signature with Z-score method)

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Mapping of sample_id → IQR score (0 inside [Q1, Q3], positive outside)
        and additional statistics (Q1, Q3, IQR)

    """
    if len(values) < 4:
        raise ValueError("At least 4 samples are required for IQR outlier detection.")

    sample_ids = list(values.keys())
    value_array = np.array(list(values.values()), dtype=float)

    # Compute quartiles on valid (non-NaN) values only.
    valid_values = value_array[~np.isnan(value_array)]
    if len(valid_values) < 4:
        raise ValueError("At least 4 non-NaN samples are required for IQR outlier detection.")

    q1, q3 = np.percentile(valid_values, [25, 75])
    iqr = q3 - q1
    meta = {"q1": q1, "q3": q3, "iqr": iqr}

    if iqr == 0:
        return {sample_id: 0.0 for sample_id in sample_ids}, meta

    score = np.where(
        value_array < q1,
        (value_array - q1) / iqr,
        np.where(value_array > q3, (value_array - q3) / iqr, 0.0),
    )

    return dict(zip(sample_ids, score.tolist())), meta


def dict_zscore(
    values: dict[str, float],
    use_mad: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Detect outliers using Z-score method.

    Identifies values with |z-score| > threshold, where z-score measures
    how many standard deviations a value is from the mean.

    Parameters
    ----------
    values : dict[str, float]
        Mapping of sample_id → metric value
    use_mad : bool
        If True, use Median Absolute Deviation (MAD) for robust statistics.
        If False, use mean and standard deviation (sensitive to outliers).

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Mapping of sample_id → Z-score
        and additional statistics (center, scale)

    """
    if len(values) < 3:
        raise ValueError("At least 3 samples are required for Z-score outlier detection.")

    value_array = np.array(list(values.values()))

    # Filter out NaN values for statistics computation
    valid_values = value_array[~np.isnan(value_array)]
    if len(valid_values) < 3:
        raise ValueError("At least 3 non-NaN samples are required for Z-score outlier detection.")

    if use_mad:
        # Robust statistics using median and MAD
        center = np.median(valid_values)
        scale = mad(valid_values) * 1.4826  # Scale MAD to match std dev for normal distribution
    else:
        # Classical statistics using mean and std
        center = np.mean(valid_values)
        scale = np.std(valid_values, ddof=1)

    meta = {"center": center, "scale": scale}
    if scale == 0:
        return {sample_id: 0. for sample_id in values}, meta

    z_score = (value_array - center) / scale
    return dict(zip(values.keys(), z_score.tolist())) , meta


# TODO: simplify options just use scikit-learn robust EllipticEnvelope and OneClassSVM
def dict_mahalanobis_score(
    values: Mapping[str, Mapping[str, float]],
    variant: str = "empirical",
    regularization: float = 1e-6,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Compute multivariate outlier scores using Mahalanobis distance.

    Parameters
    ----------
    values : dict[str, Mapping[str, float]]
        Mapping of sample_id → {feature_name: value}
    variant : str
        Covariance estimation variant:
        - "empirical": mean + full empirical covariance
        - "robust_elliptic": robust covariance from sklearn.covariance.EllipticEnvelope
        - "robust_diag_mad": compatibility alias for "robust_elliptic"
        - "robust_diag_iqr": compatibility alias for "robust_elliptic"
    regularization : float
        Ridge factor used to stabilize covariance inversion.

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Mapping of sample_id → Mahalanobis distance (non-negative)
        and metadata including feature list and suggested chi-square threshold.
    """
    if len(values) < 3:
        raise ValueError("At least 3 samples are required for Mahalanobis outlier detection.")
    if regularization < 0:
        raise ValueError("regularization must be >= 0.")

    sample_ids = list(values.keys())
    feature_names = sorted(set().union(*(set(v.keys()) for v in values.values())))
    if not feature_names:
        raise ValueError("No numeric features available for Mahalanobis outlier detection.")

    n_samples = len(sample_ids)
    n_features = len(feature_names)

    x = np.full((n_samples, n_features), np.nan, dtype=float)
    for i, sample_id in enumerate(sample_ids):
        row = values[sample_id]
        for j, feature_name in enumerate(feature_names):
            try:
                x[i, j] = float(row.get(feature_name, np.nan))
            except (TypeError, ValueError):
                x[i, j] = np.nan

    valid_mask = ~np.isnan(x).any(axis=1)
    x_valid = x[valid_mask]
    min_valid = max(3, n_features + 1)
    if x_valid.shape[0] < min_valid:
        raise ValueError(
            f"At least {min_valid} complete samples are required for Mahalanobis outlier detection with {n_features} dimensions."
        )

    normalized_variant = variant
    if variant in {"robust_diag_mad", "robust_diag_iqr"}:
        normalized_variant = "robust_elliptic"

    if variant == "empirical":
        center = np.mean(x_valid, axis=0)
        cov = np.cov(x_valid, rowvar=False, ddof=1)
    elif normalized_variant == "robust_elliptic":
        # Use sklearn robust covariance estimator instead of custom robust covariance code.
        from sklearn.covariance import EllipticEnvelope

        estimator = EllipticEnvelope(contamination=0.1, support_fraction=None, random_state=0)
        estimator.fit(x_valid)
        center = np.asarray(estimator.location_, dtype=float)
        cov = np.asarray(estimator.covariance_, dtype=float)
    else:
        raise ValueError(
            "Invalid Mahalanobis variant: "
            f"{variant}. Must be one of 'empirical', 'robust_elliptic', 'robust_diag_mad', 'robust_diag_iqr'."
        )

    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape != (n_features, n_features):
        raise ValueError(f"Invalid covariance shape {cov.shape}, expected {(n_features, n_features)}.")

    if regularization > 0:
        trace = float(np.trace(cov))
        ridge = regularization * (trace / n_features if trace > 0 else 1.0)
        cov = cov + ridge * np.eye(n_features)

    inv_cov = np.linalg.pinv(cov)

    scores = np.full(n_samples, np.nan, dtype=float)
    centered = x_valid - center
    md2 = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    md2 = np.maximum(md2, 0.0)
    scores[valid_mask] = np.sqrt(md2)

    chi2_alpha = 0.9973
    suggested_threshold = float(np.sqrt(chi2.ppf(chi2_alpha, n_features)))
    meta = {
        "variant": variant,
        "normalized_variant": normalized_variant,
        "feature_names": feature_names,
        "n_dimensions": n_features,
        "n_complete_samples": int(x_valid.shape[0]),
        "chi2_alpha": chi2_alpha,
        "suggested_threshold": suggested_threshold,
    }
    return dict(zip(sample_ids, scores.tolist())), meta
