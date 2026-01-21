"""
Compensation QC Evaluator.

Performs detailed QC analysis on compensated data using statistical tests.
"""
from __future__ import annotations
from typing import Any, Mapping, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import median_abs_deviation as mad, spearmanr, gaussian_kde
from scipy.signal import find_peaks
from cytomind.domain.transforms import transform_registry

from cytomind.qc.base import StepQCEvaluator
from cytomind.qc import QCEvaluatorRegistry
from cytomind.domain.pipeline import StepRun, QCRunStatus, QCFlag, QCTestRecord
from cytomind.visualization import build_histogram1d
from cytomind.visualization.transforms import apply_transform, get_default_transformations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from cytomind.infra.repo import ProjectRepository
else:
    ProjectRepository = object

# ============================================================================
# Validator helpers for threshold checks
# ============================================================================

def _validate_percentage(name: str, value: float) -> None:
	"""Raise ValueError if value is not in [0,1]."""
	if value is None:
		return
	if not (0.0 <= value <= 1.0):
		raise ValueError(f"{name} must be in [0, 1] range.")

def _validate_warn_severe(warn_name: str, warn: float, severe_name: str, severe: float, are_percentage: bool=True) -> None:
	"""Validate two percentage thresholds: each in [0,1] and severe >= warn."""
	if are_percentage:
		_validate_percentage(warn_name, warn)
		_validate_percentage(severe_name, severe)
	if severe < warn:
		raise ValueError(f"{severe_name} must be greater than or equal to {warn_name}.")


# ============================================================================
# QC Test Classes
# ============================================================================

class NegativeFluorescenceTest:
    """Test for negative fluorescence in a single channel."""

    def __init__(
        self,
        neg_warn: float = 0.15,
        neg_severe: float = 0.30,
    ):
        _validate_warn_severe("neg_warn", neg_warn, "neg_severe", neg_severe)
        self.neg_warn = neg_warn
        self.neg_severe = neg_severe

    def fit(self, values: np.ndarray) -> tuple[QCTestRecord, dict[str, Any]]:
        """Compute test metrics."""
        test = QCTestRecord(
            test_type="compensation_channel",
            test_name="negative_fluorescence",
            metadata={},
            metrics={
                "ratio_neg": 0.0,
            },
            status="PENDING",
        )

        n_events = len(values)
        if n_events == 0:
            test.status = "SKIP"
            test.message = "No events to test."
            return test, {}

        # Basic negative statistics
        neg_mask = values < 0
        test.metrics["ratio_neg"] = float(neg_mask.mean())

        return test, {'neg_mask': neg_mask}

    def classify(
        self,
        test: QCTestRecord,
        neg_warn: float | None = None,
        neg_severe: float | None = None,
    ) -> QCTestRecord:
        """Classify test results based on thresholds."""
        _neg_warn = neg_warn if neg_warn is not None else self.neg_warn
        _neg_severe = neg_severe if neg_severe is not None else self.neg_severe
        _validate_warn_severe("neg_warn", _neg_warn, "neg_severe", _neg_severe)

        test.thresholds["ratio_neg"] = (_neg_warn, _neg_severe)

        if test.status == "SKIP":
            return test

        if test.metrics["ratio_neg"] > _neg_severe:
            test.status = "SEVERE"
            test.message = "High proportion of negative events."
        elif test.metrics["ratio_neg"] > _neg_warn:
            test.status = "WARN"
            test.message = "Moderate proportion of negative events."
        else:
            test.status = "PASS"

        return test

    def fit_classify(self, values: np.ndarray, **classify_kwargs) -> QCTestRecord:
        """Compute metrics and classify in one step."""
        test, _ = self.fit(values)
        return self.classify(test, **classify_kwargs)

    def plot(
        self,
        values: np.ndarray,
        test: QCTestRecord | None = None,
        channel: str | None = None,
        nbins: int = 128,
        **kwargs
    ) -> go.Figure:
        """Generate data for plotting negative fluorescence histogram."""

        channel = channel if channel is not None else "Channel"
        fig = build_histogram1d(
            values=values,
            nbins=nbins,
            title=f'Histogram of {channel} Fluorescence Values',
            xaxis_title='Fluorescence Intensity',
            yaxis_title='Event Count',
            **kwargs
        )

        x_min = np.min(values)
        if x_min < 0.0:
            bar_color = kwargs.get('color', 'blue')
            fig.add_vline(
                x=0.0,
                line=dict(color="red" if bar_color != "red" else "blue", dash="dash"),
                annotation_text="0",
                annotation_position="top right",
            )

        return fig



class StrongNegativeFluorescenceTest:
    """Test for strong negative fluorescence in a single channel."""

    def __init__(
        self,
        min_neg_events_for_sigma: int = 50,
        strong_k: float = 4.0,
        strong_warn: float = 0.01,
        strong_severe: float = 0.05,
    ):
        if min_neg_events_for_sigma <= 0:
            raise ValueError("min_neg_events_for_sigma must be positive.")
        self.min_neg_events_for_sigma = int(min_neg_events_for_sigma)
        self.strong_k = abs(strong_k)

        _validate_warn_severe("strong_warn", strong_warn, "strong_severe", strong_severe)
        self.strong_warn = strong_warn
        self.strong_severe = strong_severe

    def fit(self, values: np.ndarray) -> tuple[QCTestRecord, dict[str, Any]]:
        """Compute test metrics."""
        test = QCTestRecord(
            test_type="compensation_channel",
            test_name="strong_negative_fluorescence",
            metadata={
                "min_events_for_sigma": self.min_neg_events_for_sigma,
                "strong_k": self.strong_k,
            },
            metrics={
                "ratio_strong_neg": 0.0,
            },
            status="PENDING",
        )

        n_events = len(values)
        if n_events == 0:
            test.status = "SKIP"
            test.message = "No events to test."
            return test, {}

        # Basic negative statistics
        neg_mask = values < 0

        # Robust sigma estimation
        try:
            global_sigma = float(mad(values, scale=1.4826))
        except Exception:
            global_sigma = 1.0

        if not np.isfinite(global_sigma) or global_sigma <= 0:
            global_sigma = 1.0

        # Per-channel robust sigma from negative values if sufficient data
        sigma_neg = global_sigma
        if neg_mask.sum() >= self.min_neg_events_for_sigma:
            try:
                sigma_neg = float(mad(values[neg_mask], scale=1.4826))
                if not np.isfinite(sigma_neg) or sigma_neg <= 0:
                    sigma_neg = global_sigma
            except Exception:
                sigma_neg = global_sigma

        # Strong negatives
        strong_cutoff = -self.strong_k * sigma_neg
        test.metrics["ratio_strong_neg"] = float(np.mean(values < strong_cutoff))
        test.metadata["strong_neg_cutoff"] = strong_cutoff

        return test, {'strong_cutoff': strong_cutoff}

    def classify(
        self,
        test: QCTestRecord,
        strong_warn: float | None = None,
        strong_severe: float | None = None,
    ) -> QCTestRecord:
        """Classify test results based on thresholds."""
        _strong_warn = strong_warn if strong_warn is not None else self.strong_warn
        _strong_severe = strong_severe if strong_severe is not None else self.strong_severe
        _validate_warn_severe("strong_warn", _strong_warn, "strong_severe", _strong_severe)

        test.thresholds["ratio_strong_neg"] = (_strong_warn, _strong_severe)

        if test.status == "SKIP":
            return test

        if test.metrics["ratio_strong_neg"] > _strong_severe:
            test.status = "SEVERE"
            test.message = "High proportion of strong negative events."
        elif test.metrics["ratio_strong_neg"] > _strong_warn:
            test.status = "WARN"
            test.message = "Moderate proportion of strong negative events."
        else:
            test.status = "PASS"

        return test

    def fit_classify(self, values: np.ndarray, **classify_kwargs) -> QCTestRecord:
        """Compute metrics and classify in one step."""
        test, _ = self.fit(values)
        return self.classify(test, **classify_kwargs)

    def plot(
        self,
        values: np.ndarray,
        test: QCTestRecord | None = None,
        channel: str | None = None,
        **kwargs
    ) -> go.Figure:
        """Generate data for plotting negative fluorescence histogram."""
        channel = channel if channel is not None else "Channel"
        fig = build_histogram1d(
            values=values,
            title=f'Histogram of {channel} Fluorescence Values',
            xaxis_title='Fluorescence Intensity',
            yaxis_title='Event Count',
            **kwargs
        )

        bar_color = kwargs.get('color', 'blue')
        redish_bars = bar_color in set(['red', 'orange', 'pink', 'crimson', 'darkred'])
        x_min = np.min(values)
        if x_min < 0.0:
            fig.add_vline(
                x=0.0,
                line=dict(color="red" if redish_bars else "blue", dash="dash"),
                annotation_text="0",
                annotation_position="top right",
            )

            if not test:
                test, _ = self.fit(values)
            try:
                strong_cutoff = test.metadata["strong_neg_cutoff"]
            except KeyError:
                raise ValueError("Wrong test provided: metadata are missing 'strong_neg_cutoff' value required for plotting.")

            if strong_cutoff > x_min:
                fig.add_vline(
                    x=strong_cutoff,
                    line=dict(color="orange" if not redish_bars else "cyan", dash="dot"),
                    annotation_text="Strong Negative Cutoff",
                    annotation_position="top left",
                )

        return fig


class NegativeEnrichmentTest:
    """Test for negative receiver fluorescence when donor is high."""

    def __init__(
        self,
        high_quantile: float = 0.99,
        min_high_events: int = 1000,
        neg_warn: float = 0.05,
        neg_severe: float = 0.15,
    ):
        if high_quantile <= 0.0 or high_quantile >= 1.0:
            raise ValueError("high_quantile must be in (0, 1) range.")
        self.high_quantile = high_quantile
        if min_high_events <= 0:
            raise ValueError("min_high_events must be positive.")
        self.min_high_events = min_high_events

        _validate_warn_severe("neg_warn", neg_warn, "neg_severe", neg_severe)
        self.neg_warn = neg_warn
        self.neg_severe = neg_severe

    def fit(self, donor: np.ndarray, receiver: np.ndarray) -> tuple[QCTestRecord, dict[str, Any]]:
        """Compute test metrics."""
        high_quantile = self.high_quantile
        donor_thr = np.quantile(donor, high_quantile)
        donor_high = donor >= donor_thr
        n_high = float(donor_high.sum())
        if n_high < self.min_high_events:
            high_quantile = 1.0 - (self.min_high_events / len(donor))
            donor_thr = np.quantile(donor, high_quantile)
            donor_high = donor >= donor_thr
            n_high = float(donor_high.sum())

        test = QCTestRecord(
            test_type="compensation_pair",
            test_name="negative_enrichment",
            metadata={
                "high_quantile": high_quantile,
                "n_donor_high": 0,
            },
            metrics={
                "p_neg_given_high_donor": 0.0,
            },
            status="PENDING"
        )

        if donor.shape != receiver.shape:
            raise ValueError("Donor and receiver arrays must have the same shape.")

        test.metadata["n_donor_high"] = int(donor_high.sum())
        if test.metadata["n_donor_high"] > 0:
            test.metrics["p_neg_given_high_donor"] = float(np.mean(receiver[donor_high] < 0))

        return test, {'donor_high': donor_high, 'donor_thr': donor_thr}

    def classify(
        self,
        test: QCTestRecord,
        min_high_events: int | None = None,
        neg_warn: float | None = None,
        neg_severe: float | None = None,
    ) -> QCTestRecord:
        """Classify test results based on thresholds."""
        _min_high_events = min_high_events if min_high_events is not None else self.min_high_events
        _neg_warn = neg_warn if neg_warn is not None else self.neg_warn
        _neg_severe = neg_severe if neg_severe is not None else self.neg_severe
        _validate_warn_severe("neg_warn", _neg_warn, "neg_severe", _neg_severe)

        test.thresholds["p_neg_given_high_donor"] = (_neg_warn, _neg_severe)
        test.thresholds["n_donor_high"] = (_min_high_events,)

        if test.status == "SKIP":
            return test

        n_high = test.metadata["n_donor_high"]
        p_neg = test.metrics["p_neg_given_high_donor"]

        if n_high < _min_high_events:
            test.status = "SKIP"
            test.message = f"Insufficient high-donor events ({n_high} < {_min_high_events})"
        elif p_neg > _neg_severe:
            test.status = "SEVERE"
            test.message = "High proportion of negative receiver events given high values of donor."
        elif p_neg > _neg_warn:
            test.status = "WARN"
            test.message = "Moderate proportion of negative receiver events given high values of donor."
        else:
            test.status = "PASS"

        return test

    def fit_classify(self, donor: np.ndarray, receiver: np.ndarray, **classify_kwargs) -> QCTestRecord:
        """Compute metrics and classify in one step."""
        test, _ = self.fit(donor, receiver)
        return self.classify(test, **classify_kwargs)

    def plot(
        self,
        donor: np.ndarray,
        receiver: np.ndarray,
        test: QCTestRecord | None = None,
        donor_channel: str | None = None,
        receiver_channel: str | None = None,
        n_bins: int = 100,
        **kwargs
    ) -> go.Figure | None:
        """Plot negative receiver rate across donor quantile bins."""
        donor_channel = donor_channel if donor_channel is not None else "Donor Channel"
        receiver_channel = receiver_channel if receiver_channel is not None else "Receiver Channel"
        if donor.shape != receiver.shape:
            raise ValueError("Donor and receiver arrays must have the same shape.")

        # Sort both arrays by donor values for efficient bin-based computation
        sort_idx = np.argsort(donor)
        donor_sorted = donor[sort_idx]
        receiver_sorted = receiver[sort_idx]

        n = len(donor_sorted)
        bin_size = n // n_bins

        x_vals: list[float] = []
        y_vals: list[float] = []
        for i in range(n_bins):
            idx_lo = i * bin_size
            idx_hi = (i + 1) * bin_size if i < n_bins - 1 else n

            if idx_lo >= idx_hi:
                continue

            x_vals.append(float(np.mean(donor_sorted[idx_lo:idx_hi])))
            y_vals.append(float(np.mean(receiver_sorted[idx_lo:idx_hi] < 0)))

        line_color = kwargs.get("color", "blue")
        fig = go.Figure()
        fig.add_scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Negative rate",
            line={"color": line_color},
            marker={"color": line_color},
        )

        if not test:
            test, _ = self.fit(donor, receiver)
        try:
            high_quantile = test.metadata["high_quantile"]
        except KeyError:
            raise ValueError("Wrong test provided: metadata are missing 'high_quantile' value required for plotting.")
        donor_thr = np.quantile(donor, high_quantile)
        fig.add_vline(
            x=donor_thr,
            line={"color": "orange", "dash": "dot"},
            annotation_text=f"{donor_channel} {high_quantile:.2%} Quantile",
            annotation_position="top left",
        )

        fig.update_layout(
            title=f"Negative rate of {receiver_channel} vs {donor_channel} quantiles",
            xaxis_title=f"{donor_channel} intensity",
            yaxis_title=f"Negative rate in {receiver_channel}",
        )
        return fig

class HighDonorCorrelationTest:
    """Test for correlation between donor and receiver when donor is high."""

    def __init__(
        self,
        high_quantile: float = 0.99,
        min_high_events: int = 1000,
        cor_warn: float = 0.5,
        cor_severe: float = 0.8,
    ):
        if high_quantile <= 0.0 or high_quantile >= 1.0:
            raise ValueError("high_quantile must be in (0, 1) range.")
        self.high_quantile = high_quantile
        if min_high_events <= 0:
            raise ValueError("min_high_events must be positive.")
        self.min_high_events = min_high_events

        _validate_warn_severe("cor_warn", cor_warn, "cor_severe", cor_severe)
        self.cor_warn = cor_warn
        self.cor_severe = cor_severe

    def fit(self, donor: np.ndarray, receiver: np.ndarray) -> tuple[QCTestRecord, dict[str, Any]]:
        """Compute test metrics."""
        high_quantile = self.high_quantile
        donor_thr = np.quantile(donor, high_quantile)
        donor_high = donor >= donor_thr
        n_high = float(donor_high.sum())
        if n_high < self.min_high_events:
            high_quantile = 1.0 - (self.min_high_events / len(donor))
            donor_thr = np.quantile(donor, high_quantile)
            donor_high = donor >= donor_thr
            n_high = float(donor_high.sum())

        test = QCTestRecord(
            test_type="compensation_pair",
            test_name="high_donor_correlation",
            metadata={
                "high_quantile": high_quantile,
                "n_donor_high": 0,
            },
            metrics={
                "spearman_given_high_donor": 0.0,
            },
            status="PENDING"
        )

        if donor.shape != receiver.shape:
            raise ValueError("Donor and receiver arrays must have the same shape.")

        test.metadata["n_donor_high"] = int(donor_high.sum())
        if test.metadata["n_donor_high"] > 0:
            corr, _ = spearmanr(donor[donor_high], receiver[donor_high])  # pyright: ignore
            test.metrics["spearman_given_high_donor"] = float(corr)  # pyright: ignore

        return test, {'donor_high': donor_high, 'donor_thr': donor_thr}

    def classify(
        self,
        test: QCTestRecord,
        spill_coeff: float | None = None,
        min_high_events: int | None = None,
        cor_warn: float | None = None,
        cor_severe: float | None = None,
    ) -> QCTestRecord:
        """Classify test results based on thresholds."""
        _min_high_events = min_high_events if min_high_events is not None else self.min_high_events
        _cor_warn = cor_warn if cor_warn is not None else self.cor_warn
        _cor_severe = cor_severe if cor_severe is not None else self.cor_severe
        _validate_warn_severe("cor_warn", _cor_warn, "cor_severe", _cor_severe)

        test.thresholds["spearman_given_high_donor"] = (_cor_warn, _cor_severe)
        test.thresholds["n_donor_high"] = (_min_high_events,)

        if test.status == "SKIP":
            return test

        n_high = test.metadata["n_donor_high"]
        if n_high < _min_high_events:
            test.status = "SKIP"
            test.message = f"Insufficient high-donor events ({n_high} < {_min_high_events})"
            return test

        if spill_coeff is None:
            cor = np.abs(test.metrics["spearman_given_high_donor"])
        else:
            cor = np.sign(spill_coeff) * test.metrics["spearman_given_high_donor"]

        if cor > _cor_severe:
            test.status = "SEVERE"
            test.message = "High Spearman correlation in high-donor subset."
        elif cor > _cor_warn:
            test.status = "WARN"
            test.message = "Moderate Spearman correlation in high-donor subset."
        else:
            test.status = "PASS"

        return test

    def fit_classify(self, donor: np.ndarray, receiver: np.ndarray, **classify_kwargs) -> QCTestRecord:
        """Compute metrics and classify in one step."""
        test, _ = self.fit(donor, receiver)
        return self.classify(test, **classify_kwargs)

    def _plot_simple(
        self,
        x_vals: list[float],
        y_vals: list[float],
        donor_channel: str,
        receiver_channel: str,
        donor_thr: float,
        high_quantile: float,
        line_color: str,
    ) -> go.Figure:
        """Create a simple correlation plot without 2D histogram.

        Parameters
        ----------
        x_vals : list[float]
            Mean donor values per bin.
        y_vals : list[float]
            Spearman correlation per bin.
        donor_channel : str
            Donor channel name.
        receiver_channel : str
            Receiver channel name.
        donor_thr : float
            Donor threshold value for high quantile.
        high_quantile : float
            High quantile value for annotation.
        line_color : str
            Color for the line plot.

        Returns
        -------
        go.Figure
            Simple correlation plot.
        """
        fig = go.Figure()
        fig.add_scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Spearman ρ",
            line={"color": line_color},
            marker={"color": line_color},
        )

        fig.update_layout(
            title=f"Spearman correlation of {receiver_channel} vs {donor_channel} quantiles",
            xaxis_title=f"{donor_channel} intensity",
            yaxis_title=f"Spearman ρ ({donor_channel}, {receiver_channel})",
        )

        # Clamp scatter y-axis to reach 1 while extending slightly below the minimum
        ymin = min(y_vals) if y_vals else 0.0
        y_lower = ymin - 0.1 * abs(ymin)
        fig.update_yaxes(range=[y_lower, 1.0])

        fig.add_vline(
            x=donor_thr,
            line={"color": "orange", "dash": "dot"},
            annotation_text=f"{donor_channel} {high_quantile:.2%} Quantile",
            annotation_position="top left",
        )

        return fig

    def _plot_with_histogram2d(
        self,
        donor: np.ndarray,
        receiver: np.ndarray,
        x_vals: list[float],
        y_vals: list[float],
        donor_channel: str,
        receiver_channel: str,
        donor_thr: float,
        high_quantile: float,
        line_color: str,
        transformation: str = "logicle",
    ) -> go.Figure:
        """Create correlation plot with 2D histogram using subplots.

        Parameters
        ----------
        donor : np.ndarray
            Donor channel values.
        receiver : np.ndarray
            Receiver channel values.
        x_vals : list[float]
            Mean donor values per bin.
        y_vals : list[float]
            Spearman correlation per bin.
        donor_channel : str
            Donor channel name.
        receiver_channel : str
            Receiver channel name.
        donor_thr : float
            Donor threshold value for high quantile.
        high_quantile : float
            High quantile value for annotation.
        line_color : str
            Color for the line plot.
        transformation : str
            Transformation to apply to donor and receiver data for heatmap.

        Returns
        -------
        go.Figure
            Figure with 2D histogram on bottom and correlation scatter on top, sharing x-axis.
        """
        # Apply transformation to donor and receiver data for heatmap
        donor_transformed = apply_transform(donor, transformation=transformation)
        receiver_transformed = apply_transform(receiver, transformation=transformation)
        x_vals_transformed = apply_transform(np.array(x_vals), transformation=transformation)

        # Create subplots: scatter plot on top (row 1), density heatmap on bottom (row 2)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.3, 0.7],
            vertical_spacing=0.1,
        )

        # Top subplot: correlation scatter plot
        fig.add_trace(
            go.Scatter(
                x=x_vals_transformed,
                y=y_vals,
                mode="lines+markers",
                name="Spearman ρ",
                line={"color": line_color},
                marker={"color": line_color},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Bottom subplot: 2D density heatmap with transformed data
        n_bins = 2 * len(x_vals)
        fig.add_trace(
            go.Histogram2d(
                x=donor_transformed,
                y=receiver_transformed,
                nbinsx=n_bins,
                nbinsy=n_bins,
                colorscale="Viridis",
                name="Density",
            ),
            row=2,
            col=1,
        )

        # Update axes titles and y-range so the upper limit is 1
        ymin = min(y_vals) if y_vals else 0.0
        y_lower = ymin - 0.1 * abs(ymin)
        fig.update_xaxes(title_text=f"{donor_channel} intensity", row=2, col=1)
        fig.update_yaxes(title_text=f"Spearman ρ", row=1, col=1, range=[y_lower, 1.0])
        fig.update_yaxes(title_text=receiver_channel, row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f"Spearman correlation of {receiver_channel} vs {donor_channel}",
            height=700,
        )

        # Add shared vertical line for high quantile threshold
        donor_thr = float(np.quantile(donor_transformed, high_quantile))
        fig.add_vline(
            x=donor_thr,
            line={"color": "orange", "dash": "dot"},
            annotation_text=f"{donor_channel} {high_quantile:.2%} Quantile",
            annotation_position="top left",
        )

        return fig

    def plot(
        self,
        donor: np.ndarray,
        receiver: np.ndarray,
        test: QCTestRecord | None = None,
        donor_channel: str | None = None,
        receiver_channel: str | None = None,
        n_bins: int = 100,
        add_histogram2d: bool = True,
        transformation: str = "logicle",
        **kwargs
    ) -> go.Figure:
        """Plot Spearman correlation across donor quantile bins.

        Parameters
        ----------
        donor : np.ndarray
            Donor channel values.
        receiver : np.ndarray
            Receiver channel values.
        test : QCTestRecord | None
            Test record with metadata.
        donor_channel : str | None
            Donor channel name.
        receiver_channel : str | None
            Receiver channel name.
        n_bins : int
            Number of quantile bins for correlation plot.
        add_histogram2d : bool
            If True, add a 2D histogram with correlation in top subplot.
        transformation : str
            Transformation to apply to donor and receiver data for heatmap. Default is "logicle".
        **kwargs
            Additional keyword arguments (e.g., color).

        Returns
        -------
        go.Figure
            Plotly figure with correlation plot and optional 2D histogram.
        """
        donor_channel = donor_channel if donor_channel is not None else "Donor Channel"
        receiver_channel = receiver_channel if receiver_channel is not None else "Receiver Channel"
        if donor.shape != receiver.shape:
            raise ValueError("Donor and receiver arrays must have the same shape.")

        # Sort both arrays by donor values for efficient bin-based computation
        sort_idx = np.argsort(donor)
        donor_sorted = donor[sort_idx]
        receiver_sorted = receiver[sort_idx]

        # Use direct index arithmetic for quantile bins
        n = len(donor_sorted)
        bin_size = n // n_bins

        x_vals: list[float] = []
        y_vals: list[float] = []
        for i in range(n_bins):
            idx_lo = i * bin_size
            idx_hi = (i + 1) * bin_size if i < n_bins - 1 else n

            if idx_lo >= idx_hi or (idx_hi - idx_lo) < 10:  # Need at least 10 points for correlation
                continue

            corr, _ = spearmanr(donor_sorted[idx_lo:idx_hi], receiver_sorted[idx_lo:idx_hi])  # pyright: ignore
            x_vals.append(float(np.mean(donor_sorted[idx_lo:idx_hi])))
            y_vals.append(float(corr))  # pyright: ignore

        line_color = kwargs.get("color", "blue")

        # Get threshold for vertical line
        if not test:
            test, _ = self.fit(donor, receiver)
        try:
            high_quantile = test.metadata["high_quantile"]
        except KeyError:
            raise ValueError("Wrong test provided: metadata are missing 'high_quantile' value required for plotting.")
        donor_thr = np.quantile(donor, high_quantile)

        # Call appropriate internal plot function based on flag
        if add_histogram2d:
            return self._plot_with_histogram2d(
                donor=donor,
                receiver=receiver,
                x_vals=x_vals,
                y_vals=y_vals,
                donor_channel=donor_channel,
                receiver_channel=receiver_channel,
                donor_thr=donor_thr,
                high_quantile=high_quantile,
                line_color=line_color,
                transformation=transformation,
            )
        else:
            return self._plot_simple(
                x_vals=x_vals,
                y_vals=y_vals,
                donor_channel=donor_channel,
                receiver_channel=receiver_channel,
                donor_thr=donor_thr,
                high_quantile=high_quantile,
                line_color=line_color,
            )


class ResidualBandingTest:
    """Test for banding artifacts using PLS residual analysis."""

    def __init__(
        self,
        nbins: int = 256,
        sigma: str | float = 'silverman',
        prominence: float = 0.05,
        residual_clip_quantile: float = 0.005,
        warn_threshold: int = 3,
        severe_threshold: int = 5,
    ):
        self.nbins = int(nbins)
        self.sigma = sigma
        self.prominence = prominence
        self.r_clip = residual_clip_quantile

        if warn_threshold <= 1:
            raise ValueError("warn_threshold must be greater than 1.")
        _validate_warn_severe("warn_threshold", warn_threshold,
                              "severe_threshold", severe_threshold,
                              are_percentage=False)
        self.warn_threshold = warn_threshold
        self.severe_threshold = severe_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[QCTestRecord, dict[str, Any]]:
        """Compute banding metrics using PLS and KDE."""
        test = QCTestRecord(
            test_type="compensation_pair",
            test_name="residual_banding",
            metadata={
                "nbins": self.nbins,
                "min_peak_prominence": self.prominence,
            },
            metrics={
                "n_peaks": 0,
                "pls_angle_deg": 0.0,
            },
            status="PENDING"
        )

        eps = 1e-12
        n = x.size
        if n < 1000:
            test.status = "SKIP"
            test.message = "Insufficient data points (< 1000) for banding score estimation."
            return test, {}

        x_c = x - float(np.mean(x))
        y_c = y - float(np.mean(y))

        A = x_c.reshape(-1, 1)
        try:
            a, *_ = np.linalg.lstsq(A, y_c, rcond=None)[0]
            slope = float(a)
        except Exception:
            test.status = "SKIP"
            test.message = "OLS fit failed; cannot compute banding score."
            return test, {}

        if not np.isfinite(slope):
            test.status = "SKIP"
            test.message = "Degenerate slope; cannot compute banding score."
            return test, {}

        pls_vec = np.array([1.0, slope], dtype=float)
        pls_vec /= np.linalg.norm(pls_vec)
        pls_angle_rad = np.arctan2(pls_vec[1], pls_vec[0])
        test.metrics["pls_angle_deg"] = int(np.round(np.degrees(pls_angle_rad)))

        perp_vec = np.array([-pls_vec[1], pls_vec[0]])
        stacked = np.column_stack([x_c, y_c])
        residuals = stacked @ perp_vec
        r = residuals.ravel()

        r_lo, r_hi = np.quantile(r, [self.r_clip, 1 - self.r_clip])
        if r_lo == r_hi:
            test.status = "SKIP"
            test.message = "Degenerate residual range; cannot compute banding score."
            return test, {'slope': slope, 'residuals': r}

        r_clipped = np.clip(r, r_lo, r_hi)
        x_eval = np.linspace(r_lo, r_hi, self.nbins)

        try:
            kde = gaussian_kde(r_clipped, bw_method=self.sigma)
            density = kde(x_eval)
            density /= density.max() + eps
        except (np.linalg.LinAlgError, ValueError) as e:
            test.status = "SKIP"
            test.message = f"KDE failed ({str(e)}); cannot compute banding score."
            return test, {'slope': slope, 'residuals': r}

        peaks_idx, properties = find_peaks(
            density,
            prominence=self.prominence,
            distance=int(self.nbins * 0.02)
        )

        test.metadata["prominence_values"] = properties.get("prominences", []).tolist()
        test.metrics["n_peaks"] = len(peaks_idx)

        return test, {'slope': slope, 'residuals': r, 'x_eval': x_eval, 'density': density, 'peaks_idx': peaks_idx}

    def classify(
        self,
        test: QCTestRecord,
        warn_threshold: int | None = None,
        severe_threshold: int | None = None,
    ) -> QCTestRecord:
        """Classify banding results based on number of peaks."""
        _warn = warn_threshold if warn_threshold is not None else self.warn_threshold
        _severe = severe_threshold if severe_threshold is not None else self.severe_threshold

        test.thresholds["n_peaks"] = (_warn, _severe)

        if test.status == "SKIP":
            return test

        if test.metrics["n_peaks"] > _severe:
            test.status = "SEVERE"
        elif test.metrics["n_peaks"] > _warn:
            test.status = "WARN"
        else:
            test.status = "PASS"

        return test

    def fit_classify(self, x: np.ndarray, y: np.ndarray, **classify_kwargs) -> QCTestRecord:
        """Compute metrics and classify in one step."""
        test, _ = self.fit(x, y)
        return self.classify(test, **classify_kwargs)

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        test: QCTestRecord | None = None,
        n_bins: int = 256,
    ) -> go.Figure:

        raise NotImplementedError("Plotting not yet implemented for ResidualBandingTest.")


class QuantizationLatticeTest:
    """Test for quantization/lattice artifacts."""

    def __init__(
        self,
        x_bins: int = 64,
        min_points_per_bin: int = 300,
        jump_k: float = 4.0,
        warn_threshold: float | None = None,
        severe_threshold: float | None = None,
    ):
        self.x_bins = x_bins
        self.min_points_per_bin = min_points_per_bin
        self.jump_k = jump_k
        self.warn_threshold = warn_threshold if warn_threshold is not None else 5.0
        self.severe_threshold = severe_threshold if severe_threshold is not None else 7.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[QCTestRecord, dict[str, Any]]:
        """Compute lattice metrics by analyzing y-levels within x-bins."""
        test = QCTestRecord(
            test_type="compensation_pair",
            test_name="quantization_lattice",
            metadata={
                "min_points_per_bin": self.min_points_per_bin,
                "jump_k": self.jump_k,
                "x_bins_used": 0,
            },
            metrics={
                "median_cluster_count": 0.0,
            },
            status="PENDING"
        )

        n = x.size
        if n < 6 * self.min_points_per_bin:
            test.status = "SKIP"
            test.message = "Insufficient data points for lattice score estimation."
            return test, {}

        x_lo, x_hi = float(np.min(x)), float(np.max(x))
        if x_lo == x_hi:
            test.status = "SKIP"
            test.message = "Degenerate x range."
            return test, {}

        edges = np.linspace(x_lo, x_hi, self.x_bins + 1)
        bin_ids = np.searchsorted(edges, x, side="right") - 1
        valid = (bin_ids >= 0) & (bin_ids < self.x_bins)
        bin_ids = bin_ids[valid]
        yv = y[valid]

        levels_per_bin = []
        used_bins = 0

        for b in range(self.x_bins):
            idx = np.where(bin_ids == b)[0]
            if idx.size < self.min_points_per_bin:
                continue

            used_bins += 1
            ys = np.sort(yv[idx])

            diffs = np.diff(ys)
            if diffs.size == 0:
                continue

            sigma = float(mad(diffs, scale=1.4826))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = np.median(diffs) + 1e-6

            jump_thr = self.jump_k * sigma
            n_clusters = 1 + int(np.sum(diffs > jump_thr))
            levels_per_bin.append(n_clusters)

        if used_bins == 0:
            test.status = "SKIP"
            test.message = "No x bins with sufficient points for lattice score estimation."
            return test, {}

        test.metadata["x_bins_used"] = used_bins
        levels = np.array(levels_per_bin, dtype=float)
        test.metrics["median_cluster_count"] = float(np.median(levels))

        return test, {'levels_per_bin': levels_per_bin}

    def classify(
        self,
        test: QCTestRecord,
        warn_threshold: float | None = None,
        severe_threshold: float | None = None,
    ) -> QCTestRecord:
        """Classify lattice results based on median cluster count."""
        _warn = warn_threshold if warn_threshold is not None else self.warn_threshold
        _severe = severe_threshold if severe_threshold is not None else self.severe_threshold

        test.thresholds["median_cluster_count"] = (_warn, _severe)

        if test.status == "SKIP":
            return test

        if test.metrics["median_cluster_count"] > _severe:
            test.status = "SEVERE"
        elif test.metrics["median_cluster_count"] > _warn:
            test.status = "WARN"
        else:
            test.status = "PASS"

        return test

    def fit_classify(self, x: np.ndarray, y: np.ndarray, **classify_kwargs) -> QCTestRecord:
        """Compute metrics and classify in one step."""
        test, _ = self.fit(x, y)
        return self.classify(test, **classify_kwargs)

    def plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        test: QCTestRecord | None = None,
        n_bins: int = 64,
    ) -> go.Figure:

        raise NotImplementedError("Plotting not yet implemented for QuantizationLatticeTest.")


# ============================================================================
# Stateless QC Test Orchestrators (module-level functions)
# ============================================================================

def _run_single_channel_tests(x: np.ndarray, cfg: Mapping[str, Any]) -> list[QCTestRecord]:
    """
    Run negative-fluorescence checks for a single channel (stateless).

    Parameters
    ----------
    x : np.ndarray
        Channel values
    cfg : Mapping[str, Any]
        Configuration dict with thresholds:
        - neg_warn, neg_severe
        - strong_warn, strong_severe
        - min_neg_events_for_sigma

    Returns
    -------
    list[QCTestRecord]
        Computed test records
    """
    neg_test_obj = NegativeFluorescenceTest(
        neg_warn=cfg["neg_warn"],
        neg_severe=cfg["neg_severe"],
    )
    strong_neg_test_obj = StrongNegativeFluorescenceTest(
        min_neg_events_for_sigma=cfg["min_neg_events_for_sigma"],
        strong_k=cfg.get("strong_k", 4.0),
        strong_warn=cfg["strong_warn"],
        strong_severe=cfg["strong_severe"],
    )
    return [
        neg_test_obj.fit_classify(x),
        strong_neg_test_obj.fit_classify(x),
    ]


def _run_channel_pair_tests(
    x_donor: np.ndarray,
    x_recv: np.ndarray,
    cfg: Mapping[str, Any],
) -> list[QCTestRecord]:
    """
    Run pairwise donor->receiver checks (stateless).

    Parameters
    ----------
    x_donor : np.ndarray
        Raw donor channel values
    x_donor_trans : np.ndarray
        Logicle-transformed donor values
    x_recv : np.ndarray
        Raw receiver channel values
    x_recv_trans : np.ndarray
        Logicle-transformed receiver values
    cfg : Mapping[str, Any]
        Configuration dict with thresholds:
        - high_quantile, min_high_events
        - tail_neg_warn, tail_neg_severe
        - tail_cor_warn, tail_cor_severe
        - (optional) banding_*, lattice_* for disabled tests

    Returns
    -------
    list[QCTestRecord]
        Computed test records
    """
    neg_enrich_test = NegativeEnrichmentTest(
        high_quantile=cfg.get("high_quantile", 0.90),
        min_high_events=cfg.get("min_high_events", 200),
        neg_warn=cfg.get("tail_neg_warn", 0.20),
        neg_severe=cfg.get("tail_neg_severe", 0.40),
    )
    neg_enrich = neg_enrich_test.fit_classify(x_donor, x_recv)

    corr_test = HighDonorCorrelationTest(
        high_quantile=cfg.get("high_quantile", 0.90),
        min_high_events=cfg.get("min_high_events", 200),
        cor_warn=cfg.get("tail_cor_warn", 0.50),
        cor_severe=cfg.get("tail_cor_severe", 0.80),
    )
    corr = corr_test.fit_classify(x_donor, x_recv)

    return [neg_enrich, corr]


def run_channel_tests(
    adata: ad.AnnData,
    config: Mapping[str, Any],
) -> list[QCTestRecord]:
    """
    Run channel-level QC tests on compensated AnnData (stateless).

    Parameters
    ----------
    adata : ad.AnnData
        Compensated data to analyze
    config : Mapping[str, Any]
        Configuration dict with QC thresholds

    Returns
    -------
    list[QCTestRecord]
        Channel-level test records with metadata["channel"] populated
    """
    if adata.n_obs == 0 or adata.X is None:
        return []

    all_tests: list[QCTestRecord] = []
    cfg = config

    # Get fluorescence channels
    fluoro_idx = np.where(adata.var["type"] == "fluorescence")[0]
    if len(fluoro_idx) == 0:
        raise ValueError("No fluorescence channels found in adata.var['type']")

    fluoro_labels = adata.var.index[fluoro_idx].tolist()

    # Single-channel tests
    for j, name in zip(fluoro_idx, fluoro_labels):
        x_col = np.asarray(adata.X[:, j]).ravel()
        tests = _run_single_channel_tests(x_col, cfg)
        for test in tests:
            test.metadata["channel"] = name
            all_tests.append(test)

    return all_tests


def run_pairwise_tests(
    adata: ad.AnnData,
    config: Mapping[str, Any],
) -> list[QCTestRecord]:
    """
    Run pairwise QC tests on compensated AnnData (stateless).

    Parameters
    ----------
    adata : ad.AnnData
        Compensated data to analyze
    config : Mapping[str, Any]
        Configuration dict with QC thresholds

    Returns
    -------
    list[QCTestRecord]
        Pairwise test records with metadata["donor_channel"] and
        metadata["receiver_channel"] populated
    """
    if adata.n_obs == 0 or adata.X is None:
        return []

    all_tests: list[QCTestRecord] = []
    cfg = config

    # Get fluorescence channels
    fluoro_idx = np.where(adata.var["type"] == "fluorescence")[0]
    if len(fluoro_idx) == 0:
        raise ValueError("No fluorescence channels found in adata.var['type']")

    fluoro_labels = adata.var.index[fluoro_idx].tolist()

    # Pairwise tests
    X_dense = np.asarray(adata.X)
    # transform_name = cfg.get("transform_func", "logicle")
    # transform_ref = get_default_transformations().get(transform_name)
    # if transform_ref is None:
    #     raise ValueError(f"Unknown transform function: {transform_name}")
    # transform = transform_registry[transform_name](**transform_ref.params)
    # Xtrans = transform.apply(X_dense)
    # clip_quantile = cfg.get("clip_quantile", 0.01)
    # if clip_quantile > 0:
    #     x_lo, x_hi = np.quantile(Xtrans, [clip_quantile, 1.0 - clip_quantile], axis=0)
    #     np.clip(Xtrans, x_lo, x_hi, out=Xtrans)

    for i, donor_name in zip(fluoro_idx, fluoro_labels):
        for j, recv_name in zip(fluoro_idx, fluoro_labels):
            if i == j:
                continue
            x_donor = np.asarray(X_dense[:, i]).ravel()
            x_recv = np.asarray(X_dense[:, j]).ravel()

            tests = _run_channel_pair_tests(
                x_donor=x_donor,
                x_recv=x_recv,
                cfg=cfg,
            )

            for test in tests:
                test.metadata["donor_channel"] = donor_name
                test.metadata["receiver_channel"] = recv_name
                all_tests.append(test)

    return all_tests


def run_compensation_tests(
    adata: ad.AnnData,
    config: Mapping[str, Any],
) -> list[QCTestRecord]:
    """
    Run all QC tests on compensated AnnData without StepQCEvaluator dependency.

    This stateless function computes all channel and pairwise QC tests on the
    provided AnnData object. It's designed for reuse by both the main QC evaluator
    and revision handlers that need QC analysis on visualization subsets.

    Parameters
    ----------
    adata : ad.AnnData
        Compensated data to analyze (can be full sample or subset)
    config : Mapping[str, Any]
        Configuration dict with QC thresholds. See CompensationQCEvaluator.default_config
        for all available parameters.

    Returns
    -------
    list[QCTestRecord]
        All test records (channel-level and pairwise). Records have:
        - test_type: "compensation_channel" or "compensation_pair"
        - test_name: identifier for the test
        - status: "PASS", "WARN", "FAIL", or "SKIP"
        - metadata: empty dict (caller should populate with sample_id, comp_id, etc.)
        - metrics: computed values
        - thresholds: applied threshold tuples

    Notes
    -----
    - Caller is responsible for adding sample_id, comp_id, and channel names to
      test.metadata after this function returns.
    - Does not modify adata or config in-place.
    - Raises ValueError if adata has no fluorescence channels.
    """
    all_tests = run_channel_tests(adata, config)

    # Only run pairwise tests if enabled
    if config.get("compute_pairwise", True):
        all_tests.extend(run_pairwise_tests(adata, config))

    return all_tests


@QCEvaluatorRegistry.register("compensate")
class CompensationQCEvaluator(StepQCEvaluator):
    """QC evaluator for compensation step runs."""

    default_config = {
        "compute_pairwise": True,
        "clip_quantile": 0.01,
        "subsample": 1.0,
        "smooth_sigma": "silverman",
        "high_quantile": 0.95,
        "neg_warn": 0.15,
        "neg_severe": 0.30,
        "strong_warn": 0.01,
        "strong_severe": 0.05,
        "min_neg_events_for_sigma": 50,
        "tail_neg_warn": 0.20,
        "tail_neg_severe": 0.40,
        "tail_cor_warn": 0.7,
        "tail_cor_severe": 0.9,
        "banding_warn": 3,
        "banding_severe": 5,
        "banding_bins": 256,
        "banding_prominence": 0.05,
        "lattice_warn": 5,
        "lattice_severe": 7,
        "lattice_x_bins": 64,
        "lattice_min_points_per_bin": 300,
        "lattice_jump_k": 4.0,
        "transform_func": "logicle",
    }

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(self.default_config)
        if config:
            cfg.update(config)
        self.config = cfg

    def run_step_qc(self, repo: ProjectRepository, step_run: StepRun) -> StepRun:
        """
        Evaluate compensation QC for each sample in the step run.

        Stores all test results as CSV in qc_dir/tables and keeps only warn/fail
        tests in per_sample_qc for space efficiency.
        """
        qc_dir = repo.step_dir(step_run.id) / "QC"
        tables_dir = qc_dir / "tables"
        figures_dir = qc_dir / "figures"
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        sample_ids: list[str] = step_run.inputs.get("sample_ids", [])
        comp_map: dict[str, str] | str = step_run.config["comp_id"]
        if isinstance(comp_map, str):
            comp_map = {sid: comp_map for sid in sample_ids}

        per_sample_qc: dict[str, QCRunStatus] = {}
        all_tests: dict[str, list[QCTestRecord]] = {"channel": [], "pairwise": []}

        for sid in sample_ids:
            qc = self.get_sample_qc(sid, step_run)
            if qc.overall_flag == QCFlag.FAIL:
                per_sample_qc[sid] = qc
                continue
            adata = self._load_comp_adata(repo, sid, comp_map[sid], qc)
            if adata:
                tests = self.check_compensated_adata(
                    adata=adata,
                    qc=qc,
                    comp_id=comp_map[sid],
                    sample_id=sid,
                )
                all_tests["channel"].extend(tests["channel"])
                all_tests["pairwise"].extend(tests["pairwise"])

                # Generate plots for WARN/SEVERE tests from this sample
                for test in tests["channel"]:
                    if test.status in {"WARN", "SEVERE"}:
                        self._save_channel_plot(
                            test=test,
                            adata=adata,
                            sample_id=sid,
                            cfg=self.config,
                            figures_dir=figures_dir,
                        )
                for test in tests["pairwise"]:
                    if test.status in {"WARN", "SEVERE"}:
                        self._save_pair_plot(
                            test=test,
                            adata=adata,
                            sample_id=sid,
                            cfg=self.config,
                            figures_dir=figures_dir,
                        )
            per_sample_qc[sid] = qc

        # Filter per_sample_qc to only keep WARN/SEVERE tests (save space)
        per_sample_qc = self._filter_qc_to_issues(per_sample_qc)

        # Save all test results to CSV files
        summary = self._save_qc_tables(all_tests, comp_map, repo, tables_dir)

        # Update step_run with filtered QC and basic summary only
        step_run.per_sample_qc = per_sample_qc
        step_run.qc_summary = {"basic_summary": self._get_basic_summary(step_run),
                               "detailed_summary": summary}

        return step_run

    def _filter_qc_to_issues(self, per_sample_qc: dict[str, QCRunStatus]) -> dict[str, QCRunStatus]:
        """
        Filter QCRunStatus to only keep tests with WARN or SEVERE status.

        Removes PASS tests to save space in the stored per_sample_qc.
        """
        for sample_id, qc in per_sample_qc.items():
            if qc.overall_flag == QCFlag.FAIL:
                continue  # Keep all tests if overall is FAIL

            for step_name, qc_step in list(qc.steps.items()):
                for reason_code, reason_detail in list(qc_step.reasons.items()):
                    tests = reason_detail.get("tests", [])
                    # Keep only tests with WARN or SEVERE status
                    filtered_tests = [t for t in tests if t.status in ("WARN", "SEVERE")]

                    if filtered_tests:
                        reason_detail["tests"] = filtered_tests
                    else:
                        # Remove reason if no issue-level tests remain
                        del qc_step.reasons[reason_code]

                # Remove step if no reasons remain
                if not qc_step.reasons:
                    del qc.steps[step_name]

        return per_sample_qc

    def _save_qc_tables(
        self,
        all_tests: dict[str, list[QCTestRecord]],
        comp_map: dict[str, str],
        repo: ProjectRepository,
        tables_dir,
    ) -> dict[str, Any]:
        """
        Save all test results to CSV files.

        Creates channel_tests.csv and pairwise_tests.csv in tables_dir.
        """
        # Prepare channel-level tests
        channel_records = []
        for test in all_tests["channel"]:
            channel_records.append({
                "sample_id": test.metadata.get("sample_id"),
                "compensation": test.metadata.get("compensation"),
                "channel": test.metadata.get("channel"),
                "test_name": test.test_name,
                "status": test.status,
                **test.metrics,
            })

        # Prepare pairwise tests
        pairwise_records = []
        matrices = {}
        for test in all_tests["pairwise"]:
            sample_id = test.metadata.get("sample_id")
            comp_id = test.metadata.get("compensation")

            # Cache compensation matrix to get spill coefficients
            if comp_id and comp_id not in matrices:
                matrices[comp_id] = repo.get_spill_df(comp_id)
            spill = matrices.get(comp_id)

            donor = test.metadata.get("donor_channel")
            receiver = test.metadata.get("receiver_channel")
            coef = None
            if spill is not None:
                try:
                    coef = spill.at[receiver, donor]
                except (KeyError, AttributeError):
                    coef = None

            pairwise_records.append({
                "sample_id": sample_id,
                "compensation": comp_id,
                "coefficient": coef,
                "donor": donor,
                "receiver": receiver,
                "test_name": test.test_name,
                "status": test.status,
                **test.metrics,
            })

        # Save to CSV
        qc = {}
        if channel_records:
            df_channel = pd.DataFrame.from_records(channel_records)
            # Melt to long format and drop NA values
            id_vars = ["sample_id", "compensation", "channel", "test_name", "status"]
            df_channel = df_channel.melt(id_vars=id_vars, var_name="metric_name", value_name="metric_value")
            df_channel = df_channel.dropna(subset=["metric_value"])
            table_path = tables_dir / "channel_tests.csv"
            df_channel.to_csv(table_path, index=False)
            qc["channel"] = table_path.as_posix()

        if pairwise_records:
            df_pairwise = pd.DataFrame.from_records(pairwise_records)
            # Melt to long format and drop NA values
            id_vars = ["sample_id", "compensation", "coefficient", "donor", "receiver", "test_name", "status"]
            df_pairwise = df_pairwise.melt(id_vars=id_vars, var_name="metric_name", value_name="metric_value")
            df_pairwise = df_pairwise.dropna(subset=["metric_value"])
            table_path = tables_dir / "pairwise_tests.csv"
            df_pairwise.to_csv(table_path, index=False)
            qc["channel_pair"] = table_path.as_posix()

        return {"tables": qc}

    def _get_basic_summary(self, step_run: StepRun) -> dict[str, Any]:
        """Get basic QC summary from parent class."""
        return super()._summarize_qc(step_run)

    def _load_comp_adata(
        self,
        repo: ProjectRepository,
        sample_id: str,
        comp_id: str,
        qc: QCRunStatus
    ) -> ad.AnnData | None:

        try:
            adata = repo.load_sample_adata(sample_id, layer="comp")
        except Exception:
            step = qc.get_step("load_anndata_comp")
            step.flag = QCFlag.FAIL
            step.add_reason(
                code="LOAD_ANNDATA_ERROR",
                message="Compensated layer missing; cannot run QC.",
            )
            return None


        if adata.uns["compensation_id"] != comp_id:
            step = qc.get_step("load_anndata_comp")
            step.flag = QCFlag.FAIL
            step.add_reason(
                code="COMP_MISMATCH",
                message=f"Compensated layer compensation ID "
                        f"({adata.uns['compensation_id']}) does not match expected ({comp_id}).",
            )
            return None

        return adata


    def _summarize_qc(self, step_run: StepRun, repo: ProjectRepository, **kwargs) -> dict[str, Any]:
        """Generate basic summary only (detailed tables are saved as CSV)."""
        return {"basic_summary": super()._summarize_qc(step_run)}

    def check_compensated_adata(
        self,
        adata: ad.AnnData,
        qc: QCRunStatus,
        comp_id: str,
        sample_id: str,
    ) -> dict[str, list[QCTestRecord]]:
        """Run channel and pairwise QC on compensated data.

        Uses the stateless run_compensation_tests() function internally,
        then aggregates results into QCRunStatus for step-level tracking.

        Returns dict with all tests (channel and pairwise).
        """
        all_tests = {"channel": [], "pairwise": []}

        if adata.n_obs == 0 or adata.X is None:
            step = qc.get_step("comp_qc_no_events")
            step.flag = QCFlag.FAIL
            step.add_reason(code="NO_EVENTS", message="No events selected for compensation QC.")
            return all_tests

        # Call stateless function to get all test results
        tests = run_compensation_tests(adata, self.config)

        step = qc.get_step("COMP_QC_OVERVIEW")

        # Separate into channel and pairwise, add context metadata
        for test in tests:
            test.metadata["sample_id"] = sample_id
            test.metadata["compensation"] = comp_id

            if test.test_type == "compensation_channel":
                all_tests["channel"].append(test)
            else:  # compensation_pair
                all_tests["pairwise"].append(test)

            # Only add to QC report if WARN or SEVERE
            if test.status in {"SEVERE", "WARN"}:
                step.flag = QCFlag.WARN
                channel_info = test.metadata.get("channel", "?")
                pair_info = (
                    f"{test.metadata.get('donor_channel', '?')}"
                    f" -> {test.metadata.get('receiver_channel', '?')}"
                    if test.test_type == "compensation_pair"
                    else channel_info
                )
                step.add_reason(
                    code=f"COMP_{test.test_type.upper()}_{pair_info}_{test.test_name}_{test.status}",
                    message=f"{'Channel' if test.test_type == 'compensation_channel' else 'Pair'} {pair_info}: {test.test_name}",
                    test=test,
                )

        return all_tests


    def _save_channel_plot(
        self,
        test: QCTestRecord,
        adata: ad.AnnData,
        sample_id: str,
        cfg: Mapping[str, Any],
        figures_dir: Path | None,
    ) -> None:
        """Persist plot for WARN/SEVERE single-channel tests if available."""
        if figures_dir is None:
            return

        channel = test.metadata.get("channel")
        if not channel:
            return

        # Extract values from adata using channel name
        try:
            col_idx = adata.var.index.get_loc(channel)
            values = np.asarray(adata.X[:, col_idx]).ravel() # pyright: ignore[reportOptionalSubscript]
        except (KeyError, IndexError):
            return

        fig = None
        if test.test_name == "negative_fluorescence":
            obj = NegativeFluorescenceTest(
                neg_warn=cfg["neg_warn"],
                neg_severe=cfg["neg_severe"],
            )
            fig = obj.plot(values, test=test, channel=channel)
        elif test.test_name == "strong_negative_fluorescence":
            obj = StrongNegativeFluorescenceTest(
                min_neg_events_for_sigma=cfg["min_neg_events_for_sigma"],
                strong_k=cfg.get("strong_k", 4.0),
                strong_warn=cfg["strong_warn"],
                strong_severe=cfg["strong_severe"],
            )
            fig = obj.plot(values, test=test, channel=channel)

        if fig is None:
            return

        fig_dir = figures_dir / test.test_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        safe_channel = channel.replace("/", "_")
        filename = f"{sample_id}_{safe_channel}.html"
        fig.write_html(fig_dir / filename, include_plotlyjs="cdn")

    def _save_pair_plot(
        self,
        test: QCTestRecord,
        adata: ad.AnnData,
        sample_id: str,
        cfg: Mapping[str, Any],
        figures_dir: Path | None,
    ) -> None:
        """Persist plot for WARN/SEVERE pairwise tests if available."""
        if figures_dir is None:
            return

        donor_channel = test.metadata.get("donor_channel")
        receiver_channel = test.metadata.get("receiver_channel")

        if not donor_channel or not receiver_channel:
            return

        # Extract donor and receiver values from adata
        try:
            donor_idx = adata.var.index.get_loc(donor_channel)
            receiver_idx = adata.var.index.get_loc(receiver_channel)
            donor = np.asarray(adata.X[:, donor_idx]).ravel() # pyright: ignore[reportOptionalSubscript]
            receiver = np.asarray(adata.X[:, receiver_idx]).ravel() # pyright: ignore[reportOptionalSubscript]
        except (KeyError, IndexError):
            return

        fig = None
        if test.test_name == "negative_enrichment":
            obj = NegativeEnrichmentTest(
                high_quantile=cfg.get("high_quantile", 0.90),
                min_high_events=cfg.get("min_high_events", 200),
                neg_warn=cfg.get("tail_neg_warn", 0.20),
                neg_severe=cfg.get("tail_neg_severe", 0.40),
            )
            fig = obj.plot(
                donor=donor,
                receiver=receiver,
                test=test,
                donor_channel=donor_channel,
                receiver_channel=receiver_channel,
            )
        elif test.test_name == "high_donor_correlation":
            obj = HighDonorCorrelationTest(
                high_quantile=cfg.get("high_quantile", 0.90),
                min_high_events=cfg.get("min_high_events", 200),
                cor_warn=cfg.get("tail_cor_warn", 0.50),
                cor_severe=cfg.get("tail_cor_severe", 0.80),
            )
            fig = obj.plot(
                donor=donor,
                receiver=receiver,
                test=test,
                donor_channel=donor_channel,
                receiver_channel=receiver_channel,
            )

        if fig is None:
            return

        fig_dir = figures_dir / test.test_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        safe_donor = donor_channel.replace("/", "_")
        safe_recv = receiver_channel.replace("/", "_")
        filename = f"{sample_id}_{safe_donor}_{safe_recv}.html"
        fig.write_html(fig_dir / filename, include_plotlyjs="cdn")

    def generate_review_summary(self, repo: ProjectRepository, step_run: StepRun) -> dict[str, Any]:
        return {}