"""
Compensation QC Evaluator.

Performs detailed QC analysis on compensated data using statistical tests.
"""
from __future__ import annotations
from typing import Any, Hashable, Iterable, Mapping, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import median_abs_deviation as mad, spearmanr

from cytomind.domain.flow import CompensationRef
from cytomind.domain.pipeline import SampleRef
from cytomind.domain.qc import EntityQCStatus, QCRunStatus, QCFlag, QCTestRecord
from cytomind.steps.compensation import apply_compensation
from cytomind.utils import now_iso
from cytomind.visualization import build_histogram1d
from cytomind.visualization.transforms import apply_transform

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    PathLike = Path | str

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

class NegativeFluorescenceTest(QCTester):
    """Test for negative fluorescence in a single channel."""

    test_type = "compensation_channel"
    test_name = "negative_fluorescence"
    default_config = {}
    default_thresholds = {"ratio_neg": (0.15, 0.30)} # warn, severe thresholds for ratio of negative events

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        super().__init__(config=config, thresholds=thresholds)
        neg_warn, neg_severe = self.thresholds["ratio_neg"]
        _validate_warn_severe("neg_warn", neg_warn, "neg_severe", neg_severe, are_percentage=True)

    @property
    def neg_warn(self) -> float:
        return self.thresholds["ratio_neg"][0]

    @property
    def neg_severe(self) -> float:
        return self.thresholds["ratio_neg"][1]

    def fit(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
        channel: str | None = None,
        **kwargs
    ) -> tuple[Hashable, QCTestRecord]:
        """Compute test metrics."""
        if channel is None:
            raise ValueError("Channel name must be provided for NegativeFluorescenceTest.")

        test = QCTestRecord(
            test_type=self.test_type,
            test_name=self.test_name,
            metadata={"compensation_id": entity.id, "channel": channel},
            thresholds=self.thresholds,
            metrics={"ratio_neg": 0.0},
            status="PENDING",
        )

        if adata.X is None or adata.n_obs == 0:
            test.status = "SKIP"
            test.message = "No events to test."
            return self.make_key(test), test

        values = adata[:, channel].X.ravel() # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        test.metrics["ratio_neg"] = np.mean(values < 0)

        return self.make_key(test), test

    def make_key(self, test: QCTestRecord) -> Hashable:
        """Generate unique key from channel name."""
        self._check_test_record(test)
        return (test.metadata["compensation_id"], test.metadata["channel"], test.test_type, test.test_name)

    def _check_test_record(self, test: QCTestRecord):
        base = super()._check_test_record(test)
        return base and "compensation_id" in test.metadata and "channel" in test.metadata

    def classify(self, test: QCTestRecord, neg_warn: float | None = None, neg_severe: float | None = None, **kwargs) -> QCTestRecord:
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

    def plot(
        self,
        adata: ad.AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        nbins: int = 128,
        **kwargs
    ) -> go.Figure:
        """Generate data for plotting negative fluorescence histogram."""

        self._check_test_record(test)
        channel = test.metadata["channel"]
        values = adata[:, channel].X.ravel() # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
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

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)
        return fig


class VeryNegativeFluorescenceTest(QCTester):
    """Test for very negative fluorescence in a single channel."""

    test_type = "compensation_channel"
    test_name = "very_negative_fluorescence"

    default_config = {
        "min_neg_events_for_sigma": 50,
        "k_sigma_threshold": 4.0,
    }
    default_thresholds = {"ratio_very_neg": (0.01, 0.05)} # warn, severe thresholds for ratio of very negative events

    def __init__(
        self,
        config: Mapping[str, Any] = {},
        thresholds: Mapping[str, Any] = {},
    ):
        super().__init__(config, thresholds)

        # Validate config
        self.metadata["min_neg_events_for_sigma"] = int(self.metadata["min_neg_events_for_sigma"])
        if self.metadata["min_neg_events_for_sigma"] < 0:
            raise ValueError("min_neg_events_for_sigma must be positive.")
        if self.metadata["k_sigma_threshold"] <= 0:
            raise ValueError("k_sigma_threshold must be positive.")

        # Validate thresholds
        _validate_warn_severe("ratio_very_neg", self.very_warn, "ratio_very_neg", self.very_severe, are_percentage=True)

    @property
    def very_warn(self) -> float:
        return self.thresholds["ratio_very_neg"][0]

    @property
    def very_severe(self) -> float:
        return self.thresholds["ratio_very_neg"][1]

    def fit(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
        channel: str | None = None,
        **kwargs
    ) -> tuple[Hashable, QCTestRecord]:
        """Compute test metrics."""
        if channel is None:
            raise ValueError("Channel name must be provided for VeryNegativeFluorescenceTest.")

        # Extract channel values
        min_neg_events = int(self.metadata["min_neg_events_for_sigma"])
        k_sigma_threshold = float(self.metadata["k_sigma_threshold"])

        test = QCTestRecord(
            test_type=self.test_type,
            test_name=self.test_name,
            metadata={
                "compensation_id": entity.id,
                "channel": channel,
                "min_events_for_sigma": min_neg_events,
                "k_sigma_threshold": k_sigma_threshold,
            },
            metrics={
                "ratio_very_neg": 0.0,
            },
            status="PENDING",
        )

        key = self.make_key(test)
        if adata.X is None or adata.n_obs == 0:
            test.status = "SKIP"
            test.message = "No events to test."
            return key, test

        values = adata[:, channel].X.ravel() # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        neg_mask = values < 0  # Basic negative statistics
        # Robust sigma estimation
        try:
            sigma_global: float = mad(values, scale=1.4826)
        except Exception:
            sigma_global = 1.

        if not np.isfinite(sigma_global) or sigma_global <= 0:
            sigma_global = 1.

        # Per-channel robust sigma from negative values if sufficient data
        sigma_neg = sigma_global  # standard deviation of negative values
        if neg_mask.sum() >= min_neg_events:
            try:
                sigma_neg: float = mad(values[neg_mask], scale=1.4826)
            except Exception:
                pass
            if not np.isfinite(sigma_neg) or sigma_neg <= 0:
                sigma_neg = sigma_global

        # very negatives
        very_cutoff = -k_sigma_threshold * sigma_neg
        test.metadata["very_neg_cutoff"] = very_cutoff
        test.metadata["sigma_negative"] = sigma_neg
        test.metrics["ratio_very_neg"] = np.mean(values < very_cutoff)

        return key, test

    def classify(self, test: QCTestRecord, **kwargs) -> QCTestRecord:
        """Classify test results based on thresholds."""
        _very_warn = kwargs.get("very_warn", self.very_warn)
        _very_severe = kwargs.get("very_severe", self.very_severe)
        _validate_warn_severe("very_warn", _very_warn, "very_severe", _very_severe)

        test.thresholds["ratio_very_neg"] = (_very_warn, _very_severe)

        if test.status == "SKIP":
            return test

        if test.metrics["ratio_very_neg"] > _very_severe:
            test.status = "SEVERE"
            test.message = "High proportion of very negative events."
        elif test.metrics["ratio_very_neg"] > _very_warn:
            test.status = "WARN"
            test.message = "Moderate proportion of very negative events."
        else:
            test.status = "PASS"

        return test

    def make_key(self, test: QCTestRecord) -> Hashable:
        """Generate unique key from channel name."""
        self._check_test_record(test)
        return (test.metadata["compensation_id"], test.metadata["channel"], test.test_type, test.test_name)

    def _check_test_record(self, test: QCTestRecord):
        base = super()._check_test_record(test)
        return base and "channel" in test.metadata and "compensation_id" in test.metadata

    def plot(
        self,
        adata: ad.AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        nbins: int = 128,
        **kwargs
    ) -> go.Figure:
        """Generate data for plotting negative fluorescence histogram."""

        self._check_test_record(test)
        channel = test.metadata["channel"]
        values = adata[:, channel].X.flatten() # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        fig = build_histogram1d(
            values=values,
            nbins=nbins,
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

            try:
                very_cutoff = test.metadata["very_neg_cutoff"]
                if very_cutoff > x_min:
                    fig.add_vline(
                        x=very_cutoff,
                        line=dict(color="orange" if not redish_bars else "cyan", dash="dot"),
                        annotation_text="very Negative Cutoff",
                        annotation_position="top left",
                    )
            except (KeyError, TypeError):
                pass  # No test provided, skip cutoff line

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig


class NegativeEnrichmentTest(QCTester):
    """Test for negative receiver fluorescence when donor is high."""

    test_type = "compensation_pair"
    test_name = "negative_enrichment"

    default_config = {
        "high_quantile": 0.99,
        "min_high_events": 1000,
    }
    default_thresholds = {"p_neg_given_high_donor": (0.05, 0.15)} # warn, severe thresholds for proportion of negative receiver events given high donor values

    def __init__(
        self,
        config: Mapping[str, Any] = {},
        thresholds: Mapping[str, Any] = {},
    ):
        super().__init__(config, thresholds)

        # Validate config
        high_q = self.metadata.get("high_quantile", 0.99)
        if high_q <= 0.0 or high_q >= 1.0:
            raise ValueError("high_quantile must be in (0, 1) range.")
        self.metadata["high_quantile"] = high_q

        min_events = self.metadata.get("min_high_events", 1000)
        if min_events <= 0:
            raise ValueError("min_high_events must be positive.")
        self.metadata["min_high_events"] = min_events

        # Validate thresholds
        warn_val = self.thresholds.get("p_neg_given_high_donor", (0.05, 0.15))[0]
        severe_val = self.thresholds.get("p_neg_given_high_donor", (0.05, 0.15))[1]
        _validate_warn_severe("p_neg_given_high_donor", warn_val, "p_neg_given_high_donor", severe_val)

    def fit(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
        donor: str | None = None,
        receiver: str | None = None,
        **kwargs
    ) -> tuple[Hashable, QCTestRecord]:
        if donor is None or receiver is None:
            raise ValueError("Donor and receiver channel names must be provided for NegativeEnrichmentTest.")

        high_quantile = self.metadata["high_quantile"]
        min_high_events = self.metadata["min_high_events"]
        coef = entity.spill.at[donor, receiver]

        # Initialize test
        test = QCTestRecord(
            test_type=self.test_type,
            test_name=self.test_name,
            metadata={
                "compensation_id": entity.id,
                "donor_channel": donor,
                "receiver_channel": receiver,
                "compensation_coefficient": coef,
                "high_quantile": high_quantile,
            },
            metrics={
                "p_neg_given_high_donor": 0.0,
                "p_neg_given_low_donor": 0.0,
            },
            status="PENDING"
        )

        key = self.make_key(test)

        # Extract channel values
        donor_values = np.asarray(adata[:, donor].X).ravel()
        receiver_values = np.asarray(adata[:, receiver].X).ravel()

        # Identify events with high donor values based on quantile threshold,
        # ensuring at least min_high_events are included
        donor_thr = np.quantile(donor_values, high_quantile)
        donor_high = donor_values >= donor_thr
        n_high = donor_high.sum(dtype=int)
        if n_high < min_high_events:
            high_quantile = 1.0 - (min_high_events / len(donor_values))
            donor_thr = np.quantile(donor_values, high_quantile)
            donor_high = donor_values >= donor_thr
            n_high = donor_high.sum(dtype=int)

        test.metadata["n_donor_high"] = n_high

        if n_high > 0:
            test.metrics["p_neg_given_high_donor"] = np.mean(receiver_values[donor_high] < 0)
            test.metrics["p_neg_given_low_donor"] = np.mean(receiver_values[~donor_high] < 0)

        return key, test

    def classify(
        self,
        test: QCTestRecord,
        **kwargs,
    ) -> QCTestRecord:
        """Classify test results based on thresholds."""
        _min_high_events = kwargs.get("min_high_events", self.metadata["min_high_events"])
        _neg_warn = kwargs.get("neg_warn", self.thresholds["p_neg_given_high_donor"][0])
        _neg_severe = kwargs.get("neg_severe", self.thresholds["p_neg_given_high_donor"][1])
        _validate_warn_severe("p_neg_given_high_donor", _neg_warn, "p_neg_given_high_donor", _neg_severe)

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

    def _check_test_record(self, test: QCTestRecord):
        base = super()._check_test_record(test)
        has_channels = "donor_channel" in test.metadata and "receiver_channel" in test.metadata
        return base and has_channels and "compensation_id" in test.metadata

    def make_key(self, test: QCTestRecord) -> Hashable:
        """Generate unique key from donor and receiver channel names."""
        self._check_test_record(test)
        donor = test.metadata["donor_channel"]
        receiver = test.metadata["receiver_channel"]
        return (test.metadata["compensation_id"], donor, receiver, self.test_type, self.test_name)

    def plot(
        self,
        adata: ad.AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        **kwargs
    ) -> go.Figure:
        """Plot negative receiver rate across donor quantile bins."""
        donor_channel = test.metadata.get("donor_channel")
        receiver_channel = test.metadata.get("receiver_channel")

        # Extract channel values
        donor = np.asarray(adata[:, donor_channel].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        receiver = np.asarray(adata[:, receiver_channel].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

        n_bins = kwargs.get("n_bins", 100)

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

        high_quantile = test.metadata["high_quantile"]
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

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig


class HighDonorCorrelationTest(QCTester):
    """Test for correlation between donor and receiver when donor is high."""

    test_type = "compensation_pair"
    test_name = "high_donor_correlation"
    default_config = {"high_quantile": 0.99, "min_high_events": 1000}
    default_thresholds = {"spearman_given_high_donor": (0.3, 0.5)} # warn, severe thresholds for Spearman correlation

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        super().__init__(config, thresholds)
        _validate_percentage("high_quantile", self.metadata["high_quantile"])
        if self.metadata["min_high_events"] <= 0:
            raise ValueError("min_high_events must be positive.")

        cor_warn = self.thresholds["spearman_given_high_donor"][0]
        cor_severe = self.thresholds["spearman_given_high_donor"][1]
        _validate_warn_severe("spearman_given_high_donor_warn", cor_warn, "spearman_given_high_donor_severe", cor_severe)

    def fit(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
        donor: str | None = None,
        receiver: str | None = None,
        **kwargs
    ) -> tuple[Hashable, QCTestRecord]:
        if donor is None or receiver is None:
            raise ValueError("Donor and receiver channel names must be provided for HighDonorCorrelationTest.")

        donor_values = np.asarray(adata[:, donor].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        receiver_values = np.asarray(adata[:, receiver].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        coef = entity.spill.at[donor, receiver]

        high_quantile = self.metadata["high_quantile"]
        min_high_events = self.metadata["min_high_events"]
        test = QCTestRecord(
            test_type=self.test_type,
            test_name=self.test_name,
            metadata={
                "compensation_id": entity.id,
                "spillover_coefficient": coef,
                "donor_channel": donor,
                "receiver_channel": receiver,
                "high_quantile": high_quantile,
            },
            metrics={
                "spearman_given_high_donor": 0.0,
            },
            status="PENDING"
        )

        key = self.make_key(test)

        if adata.X is None or adata.n_obs <= min_high_events * 2:
            test.status = "SKIP"
            test.message = "Not enough events to test."
            return key, test

        # Identify events with high donor values based on quantile threshold
        donor_thr = np.quantile(donor_values, high_quantile)
        donor_high = donor_values >= donor_thr
        n_high = donor_high.sum()

        # Ensure at least min_high_events are included
        if n_high < min_high_events:
            high_quantile = 1.0 - (min_high_events / len(donor_values))
            donor_thr = np.quantile(donor_values, high_quantile)
            donor_high = donor_values >= donor_thr
            n_high = donor_high.sum()

        test.metadata["high_threshold"] = donor_thr
        test.metadata["n_high_donor"] = n_high

        corr, _ = spearmanr(donor_values[donor_high], receiver_values[donor_high])  # pyright: ignore
        test.metrics["spearman_given_high_donor"] = float(corr)  # pyright: ignore

        return key, test

    def classify(
        self,
        test: QCTestRecord,
        **kwargs
    ) -> QCTestRecord:
        """Classify test results based on thresholds."""
        _min_high_events = kwargs.get("min_high_events", self.metadata["min_high_events"])
        _cor_warn = kwargs.get("cor_warn", self.thresholds["spearman_given_high_donor"][0])
        _cor_severe = kwargs.get("cor_severe", self.thresholds["spearman_given_high_donor"][1])
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

        spill_coeff = test.metadata.get("spillover_coefficient", None)
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

    def make_key(self, test: QCTestRecord) -> Hashable:
        """Generate unique key from donor and receiver channel names."""
        self._check_test_record(test)
        donor = test.metadata["donor_channel"]
        receiver = test.metadata["receiver_channel"]
        return (self.test_type, self.test_name, donor, receiver)

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
        adata: ad.AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        **kwargs
    ) -> go.Figure:
        """Plot Spearman correlation across donor quantile bins.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data matrix with channel data.
        test : QCTestRecord
            Test record with metadata containing donor_channel and receiver_channel.
        output_path : PathLike | None
            Path to save the figure (not used in this implementation).
        **kwargs
            Additional keyword arguments (n_bins, add_histogram2d, transformation, color).

        Returns
        -------
        go.Figure
            Plotly figure with correlation plot and optional 2D histogram.
        """
        donor_channel = test.metadata.get("donor_channel", "Donor Channel")
        receiver_channel = test.metadata.get("receiver_channel", "Receiver Channel")

        # Extract channel values
        donor = np.asarray(adata[:, donor_channel].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        receiver = np.asarray(adata[:, receiver_channel].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

        n_bins = kwargs.get("n_bins", 100)
        add_histogram2d = kwargs.get("add_histogram2d", True)
        transformation = kwargs.get("transformation", "logicle")

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
        high_quantile = test.metadata["high_quantile"]
        donor_thr = test.metadata["high_threshold"]

        # Call appropriate internal plot function based on flag
        if add_histogram2d:
            fig = self._plot_with_histogram2d(
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
            fig = self._plot_simple(
                x_vals=x_vals,
                y_vals=y_vals,
                donor_channel=donor_channel,
                receiver_channel=receiver_channel,
                donor_thr=donor_thr,
                high_quantile=high_quantile,
                line_color=line_color,
            )

        if output_path:
            output_path = Path(output_path)
            if output_path.suffix != ".html":
                raise ValueError("Output path must have .html extension for Plotly figure.")
            fig.write_html(output_path)

        return fig


@EntityQCEvaluatorRegistry.register("compensation")
class CompensationQCEvaluator(EntityQCEvaluator):
    """QC evaluator for compensation entities."""

    default_config = {
        "compute_pairwise": True,
        "subsample": 1.0,
        "high_quantile": 0.95,
        "neg_warn": 0.15,
        "neg_severe": 0.30,
        "very_warn": 0.01,
        "very_severe": 0.05,
        "min_neg_events_for_sigma": 50,
        "k_sigma_threshold": 4.0,
        "tail_neg_warn": 0.20,
        "tail_neg_severe": 0.40,
        "tail_cor_warn": 0.7,
        "tail_cor_severe": 0.9,
        "transform_func": "logicle",
    }

    tester_registry: dict[str, type[QCTester]] = {
            "negative_fluorescence": NegativeFluorescenceTest,
            "very_negative_fluorescence": VeryNegativeFluorescenceTest,
            "negative_enrichment": NegativeEnrichmentTest,
            "high_donor_correlation": HighDonorCorrelationTest,
    }

    def run_entity_qc(
        self,
        entity_id: str,
        *,
        sample_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> EntityQCStatus:
        """Evaluate compensation QC against a sample context."""
        comp = self.project.compensations[entity_id]
        if sample_ids is None:
            sample_ids = comp.batch

        try:
            qc_status = self.repo.load_qc_entity_status(entity_type="compensation", entity_id=entity_id)
        except FileNotFoundError:
            qc_status = EntityQCStatus(
                entity_type="compensation",
                entity_id=entity_id,
                context=context or {},
                generated_at=now_iso(),
            )

        all_tests: dict[str, list[tuple[str, QCTestRecord]]] = {"channel": [], "pairwise": []}

        for sid in sample_ids:
            sref = self.project.samples[sid]
            qc = qc_status.get_sample_steps(sid)
            adata = self._load_comp_adata(sref, comp, qc)
            if adata is None:
                continue
            tests = self.check_compensated_adata(
                adata=adata,
                qc=qc,
                comp_id=entity_id,
                sample_id=sid,
            )
            all_tests["channel"].extend(tests["channel"])
            all_tests["pairwise"].extend(tests["pairwise"])

        qc_status.summary = self.summarize(qc_status)
        qc_status.summary["artifacts"] = {
            "tables": self._collect_table_artifacts(qc_status),
        }
        return qc_status

    def check_compensated_adata(
        self,
        adata: ad.AnnData,
        qc: QCRunStatus,
        comp_id: str,
        sample_id: str,
    ) -> dict[str, list[tuple[str, QCTestRecord]]]:
        """Run channel and pairwise QC on compensated data.

        Uses the instance compensation test runner internally,
        then aggregates results into QCRunStatus for step-level tracking.

        Returns dict with all tests (channel and pairwise).
        """
        all_tests: dict[str, list[tuple[str, QCTestRecord]]] = {"channel": [], "pairwise": []}

        if adata.n_obs == 0 or adata.X is None:
            step = qc.get_step("comp_qc_no_events")
            step.flag = QCFlag.FAIL
            step.add_reason(code="NO_EVENTS", message="No events selected for compensation QC.")
            return all_tests

        # Call stateless function to get all test results
        comp_ref = self.project.compensations[comp_id]
        tests = self._run_compensation_tests(comp_ref, adata)
        step = qc.get_step("COMP_QC_OVERVIEW")

        # Separate into channel and pairwise
        for test_key, test in tests:
            if test.test_type == "compensation_channel":
                all_tests["channel"].append((sample_id, test))
            else:  # compensation_pair
                all_tests["pairwise"].append((sample_id, test))

            # Only add to QC report if WARN or SEVERE
            if test.status in {"SEVERE", "WARN"}:
                step.add_reason(
                    code=f"COMP_QC_{test.status}",
                    message=test.message,
                    test={test_key: test},
                )
            else:
                step.add_test(test_key, test)

        return all_tests

    def generate_figure(
        self,
        entity: str | EntityQCStatus,
        sample_id: str,
        test_key: Any,
        step_id: str | None = None,
        adata: ad.AnnData | None = None,
    ) -> go.Figure:
        """Generate a figure for a specific test on demand.

        Loads the QC status, retrieves the test by key, loads the sample data,
        and generates the plot using the appropriate tester class.

        Parameters
        ----------
        entity : str | EntityQCStatus
            Compensation entity ID or QC status object.
        sample_id : str
            Sample ID.
        test_key : Hashable
            Test key to look up in the QC status (from QCStepStatus.tests).
        step_id : str | None
            Optional step ID to narrow down the search for the test.
            If None, searches all steps for the sample and returns the first match.
        adata : ad.AnnData | None
            Optional pre-loaded AnnData to use for plotting.
            If None, the method will attempt to load the compensated data for the sample.
        artifact_format : str | None
            Format to generate ("plotly_json", "png", "svg"). Format handling is
            delegated to the caller; this parameter is provided for reference.

        Returns
        -------
        go.Figure
            Plotly figure object ready to be serialized or displayed.
        """
        # Load the QC status from disk
        if not isinstance(entity, EntityQCStatus):
            entity = self.repo.load_qc_entity_status(entity_type="compensation", entity_id=entity)

        # Retrieve the test from the QC status
        if sample_id not in entity.sample_qc:
            raise KeyError(f"Sample {sample_id} not found in QC status for entity {entity.entity_id}")
        sample_run = entity.sample_qc[sample_id]

        # Look for the test in all steps for this sample
        test = None
        if step_id is not None:
            step = sample_run.steps.get(step_id)
            if step and test_key in step.tests:
                test = step.tests[test_key]
        else:
            for step in sample_run.steps.values():
                if test_key in step.tests:
                    test = step.tests[test_key]
                    break

        if test is None:
            raise KeyError(f"Test {test_key} not found for sample {sample_id} in QC status")

        test_name = test.test_name

        # Get tester class from registry
        try:
            tester_class = self.tester_registry[test_name]
        except KeyError:
            raise ValueError(f"Unknown test name '{test_name}'. Available: {list(self.tester_registry.keys())}")

        # Load compensated data for the sample
        if adata is None:
            sref = self.project.samples[sample_id]
            comp = self.project.compensations[entity.entity_id]
            qc_temp = QCRunStatus()  # Temporary QC tracking for data loading
            adata = self._load_comp_adata(sref, comp, qc_temp)
            if adata is None:
                raise RuntimeError(f"Failed to load compensated data for sample {sample_id}")

        # Generate figure using tester
        tester = tester_class.from_dict(test)
        fig = tester.plot(adata=adata, test=test)
        if fig is None:
            raise RuntimeError(f"Plot generation failed for test '{test_name}' on sample {sample_id}")

        return fig

    def generate_table(
        self,
        entity: str | EntityQCStatus,
        table_type: str = "channel_tests",
    ) -> pd.DataFrame:
        """Generate a table for a compensation entity on demand.

        Reconstructs the specified table (channel_tests or pairwise_tests) from
        the stored test records in the QC status. The caller is responsible for
        persisting the result if needed.

        Parameters
        ----------
        entity : str | EntityQCStatus
            Compensation entity ID or QC status object.
            If str, loads the QC status from disk.
            If EntityQCStatus, uses the in-memory object directly.
        table_type : str, default "channel_tests"
            Type of table to generate: "channel_tests" or "pairwise_tests".

        Returns
        -------
        pd.DataFrame
            Table with columns matching the QC output format.
            For channel_tests: sample_id, compensation, channel, test_name, status,
                              metric_name, metric_value.
            For pairwise_tests: sample_id, compensation, coefficient, donor, receiver,
                                test_name, status, metric_name, metric_value.
        """
        # Load or use provided QC status
        if not isinstance(entity, EntityQCStatus):
            entity = self.repo.load_qc_entity_status(entity_type="compensation", entity_id=entity)

        if table_type == "channel_tests":
            return self._generate_channel_tests_table(entity)
        elif table_type == "pairwise_tests":
            return self._generate_pairwise_tests_table(entity)
        else:
            raise ValueError(f"Unknown table_type '{table_type}'. Must be 'channel_tests' or 'pairwise_tests'.")

    def _generate_channel_tests_table(self, entity: EntityQCStatus) -> pd.DataFrame:
        """Generate channel-level tests table."""
        records = []
        for sample_id, sample_run in entity.sample_qc.items():
            for step in sample_run.steps.values():
                for test in step.tests.values():
                    # Check if this is a channel-level test
                    if hasattr(test, "metadata") and "channel" in test.metadata:
                        records.append({
                            "sample_id": sample_id,
                            "compensation": test.metadata["compensation_id"],
                            "channel": test.metadata["channel"],
                            "test_name": test.test_name,
                            "status": test.status,
                            **test.metrics,
                        })
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)
        id_vars = ["sample_id", "compensation", "channel", "test_name", "status"]
        df = df.melt(id_vars=id_vars, var_name="metric_name", value_name="metric_value")
        df = df.dropna(subset=["metric_value"])

        return df

    def _generate_pairwise_tests_table(self, entity: EntityQCStatus) -> pd.DataFrame:
        """Generate pairwise tests table."""
        records = []
        for sample_id, sample_run in entity.sample_qc.items():
            for step in sample_run.steps.values():
                for test in step.tests.values():
                    # Check if this is a pairwise test
                    if hasattr(test, "metadata") and "donor_channel" in test.metadata:
                        records.append({
                            "sample_id": sample_id,
                            "compensation": test.metadata["compensation_id"],
                            "coefficient": test.metadata.get("spillover_coefficient", None),
                            "donor": test.metadata["donor_channel"],
                            "receiver": test.metadata["receiver_channel"],
                            "test_name": test.test_name,
                            "status": test.status,
                            **test.metrics,
                        })
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)
        id_vars = ["sample_id", "compensation", "coefficient", "donor", "receiver", "test_name", "status"]
        df = df.melt(id_vars=id_vars, var_name="metric_name", value_name="metric_value")
        df = df.dropna(subset=["metric_value"])

        return df

    def _collect_table_artifacts(self, entity: EntityQCStatus) -> list[dict[str, Any]]:
        """Collect table artifacts from EntityQCStatus by generating and persisting tables."""
        artifacts = []

        # Generate channel tests table and persist
        df_channel = self.generate_table(entity=entity, table_type="channel_tests")
        if not df_channel.empty:
            table_dir = self.repo.qc_entity_tables_dir(entity_type="compensation", entity_id=entity.entity_id)
            path = table_dir / "channel_tests.csv"
            df_channel.to_csv(path, index=False)
            artifacts.append({"scope": "entity", "name": "channel_tests", "format": "csv", "path": path.as_posix()})

        # Generate pairwise tests table and persist
        df_pairwise = self.generate_table(entity=entity, table_type="pairwise_tests")
        if not df_pairwise.empty:
            table_dir = self.repo.qc_entity_tables_dir(entity_type="compensation", entity_id=entity.entity_id)
            path = table_dir / "pairwise_tests.csv"
            df_pairwise.to_csv(path, index=False)
            artifacts.append({"scope": "entity", "name": "pairwise_tests", "format": "csv", "path": path.as_posix()})

        return artifacts

    def _run_single_channel_tests(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
        channel: str,
    ) -> Iterable[tuple[Hashable, QCTestRecord]]:
        """Run negative-fluorescence checks for a single channel."""
        cfg = self.config

        neg_test_obj = NegativeFluorescenceTest(
            thresholds={"ratio_neg": (cfg["neg_warn"], cfg["neg_severe"])},
        )
        yield neg_test_obj.fit_classify(entity, adata, channel=channel)

        very_neg_test_obj = VeryNegativeFluorescenceTest(
            config={
                "min_neg_events_for_sigma": cfg["min_neg_events_for_sigma"],
                "k_sigma_threshold": cfg.get("k_sigma_threshold", 4.0),
            },
            thresholds={"ratio_very_neg": (cfg["very_warn"], cfg["very_severe"])},
        )
        yield very_neg_test_obj.fit_classify(entity, adata, channel=channel)

    def _run_channel_pair_tests(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
        donor: str,
        receiver: str,
    ) -> Iterable[tuple[Hashable, QCTestRecord]]:
        """Run pairwise donor->receiver checks."""
        cfg = self.config

        neg_enrich_test = NegativeEnrichmentTest(
            config={
                "high_quantile": cfg.get("high_quantile", 0.90),
                "min_high_events": cfg.get("min_high_events", 200),
            },
            thresholds={
                "p_neg_given_high_donor": (
                    cfg.get("tail_neg_warn", 0.20),
                    cfg.get("tail_neg_severe", 0.40),
                ),
            },
        )
        yield neg_enrich_test.fit_classify(
            entity,
            adata,
            donor=donor,
            receiver=receiver,
        )

        corr_test = HighDonorCorrelationTest(
            config={
                "high_quantile": cfg.get("high_quantile", 0.90),
                "min_high_events": cfg.get("min_high_events", 200),
            },
            thresholds={
                "spearman_given_high_donor": (
                    cfg.get("tail_cor_warn", 0.50),
                    cfg.get("tail_cor_severe", 0.80),
                ),
            },
        )

        yield corr_test.fit_classify(
            entity,
            adata,
            donor=donor,
            receiver=receiver,
        )

    def _run_channel_tests(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
    ) -> Iterable[tuple[Hashable, QCTestRecord]]:
        """Run channel-level QC tests on compensated AnnData."""
        if adata.n_obs == 0 or adata.X is None:
            return

        fluoro_idx = np.where(adata.var["type"] == "fluorescence")[0]
        if len(fluoro_idx) == 0:
            raise ValueError("No fluorescence channels found in adata.var['type']")

        fluoro_labels = adata.var.index[fluoro_idx].tolist()

        for name in fluoro_labels:
            yield from self._run_single_channel_tests(entity=entity, adata=adata, channel=name)

    def _run_pairwise_tests(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
    ) -> Iterable[tuple[Hashable, QCTestRecord]]:
        """Run pairwise QC tests on compensated AnnData."""
        if adata.n_obs == 0 or adata.X is None:
            return

        fluoro_idx = np.where(adata.var["type"] == "fluorescence")[0]
        if len(fluoro_idx) == 0:
            raise ValueError("No fluorescence channels found in adata.var['type']")

        fluoro_labels = adata.var.index[fluoro_idx].tolist()

        for donor_name in fluoro_labels:
            for recv_name in fluoro_labels:
                if donor_name == recv_name:
                    continue

                tests = self._run_channel_pair_tests(
                    entity=entity,
                    adata=adata,
                    donor=donor_name,
                    receiver=recv_name,
                )
                yield from tests

    def _run_compensation_tests(
        self,
        entity: CompensationRef,
        adata: ad.AnnData,
    ) -> Iterable[tuple[Hashable, QCTestRecord]]:
        """Run all QC tests on compensated AnnData without step-level dependencies."""
        yield from self._run_channel_tests(entity, adata)

        if self.config.get("compute_pairwise", True):
            yield from self._run_pairwise_tests(entity, adata)

    def _load_comp_adata(
        self,
        sample_ref: SampleRef,
        comp_ref: CompensationRef,
        qc: QCRunStatus
    ) -> ad.AnnData | None:

        cur_comp_id = sample_ref.compensation
        if cur_comp_id == comp_ref.id:
            try:
                return self._load_adata(sample_ref, layer="comp", qc=qc)
            except Exception:
                return None

        if cur_comp_id is None:
            try:
                raw = self._load_adata(sample_ref, layer="raw", qc=qc)
            except Exception:
                return None
        else:
            cur_comp = self.project.compensations[cur_comp_id]
            try:
                adata = self._load_adata(sample_ref, layer="comp", qc=qc)
            except Exception:
                return None
            try:
                raw = apply_compensation(adata, cur_comp, invert=True)
            except Exception as e:
                step = qc.get_step("COMP_QC_INVERT_COMP_FAIL")
                step.flag = QCFlag.FAIL
                step.add_reason(
                    code="APPLY_COMP_FAIL",
                    message=f"Failed to invert current compensation {cur_comp_id} to sample {sample_ref.id}: {str(e)}"
                )
                return None

        # Save the raw data back to the repo for potential reuse
        self.repo.save_sample_adata(sample_ref.id, layer="raw", adata=raw)
        try:
            return apply_compensation(raw, comp_ref)
        except Exception as e:
            step = qc.get_step("COMP_QC_APPLY_COMP_FAIL")
            step.flag = QCFlag.FAIL
            step.add_reason(
                code="APPLY_COMP_FAIL",
                message=f"Failed to apply compensation {comp_ref.id} to sample {sample_ref.id}: {str(e)}"
            )
            return None

    def _load_adata(self, sample_ref: SampleRef, layer: str, qc: QCRunStatus) -> ad.AnnData:
        """Helper to load adata with error handling."""
        try:
            return self.repo.load_sample_adata(sample_ref.id, layer=layer)
        except Exception as e:
            step = qc.get_step(f"LOAD_{layer.upper()}_FAIL")
            step.flag = QCFlag.FAIL
            step.add_reason(code="LOAD_FAIL", message=f"Failed to load {layer} data for sample {sample_ref.id}: {str(e)}")
            raise e
