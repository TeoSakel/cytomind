"""
Compensation QC Evaluator.

Performs detailed QC analysis on compensated data using statistical tests.
"""
from __future__ import annotations
from typing import Any, Hashable, Iterable, Mapping, TYPE_CHECKING
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import median_abs_deviation as mad, spearmanr

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cytomind.domain.qc import EntityQCStatus, QCTestRecord
from cytomind.visualization import build_histogram1d
from cytomind.visualization.transforms import apply_transform

from . import EntityQCEvaluatorRegistry
from .base import EntityQCEvaluator, QCTester

if TYPE_CHECKING:
    from anndata import AnnData
    from cytomind.infra.dataloader import UnifiedDataLoader
    from cytomind.domain.flow import CompensationRef
    from cytomind.domain.constants import PathLike
else:
    AnnData = object
    UnifiedDataLoader = object
    CompensationRef = object
    PathLike = object

# ============================================================================
# Validator helpers for threshold checks
# ============================================================================

def _validate_percentage(name: str, value: float) -> None:
	"""Raise ValueError if value is not in [0,1]."""
	if value is None:
		return
	if not (0.0 <= value <= 1.0):
		raise ValueError(f"{name} must be in [0, 1] range.")

def _validate_warn_severe(
    warn_name: str,
    warn: float,
    severe_name: str,
    severe: float,
    are_percentage: bool=True
) -> None:
	"""Validate two percentage thresholds: each in [0,1] and severe >= warn."""
	if are_percentage:
		_validate_percentage(warn_name, warn)
		_validate_percentage(severe_name, severe)
	if severe < warn:
		raise ValueError(f"{severe_name} must be greater than or equal to {warn_name}.")


# ============================================================================
# QC Test Classes
# ============================================================================

# TODO: Refactor tests based on common test_type to reduce code duplication

class NegativeFluorescenceTest(QCTester):
    """Test for negative fluorescence in a single channel."""

    test_type = "compensation_channel"
    test_name = "negative_fluorescence"
    target_keys = ("compensation_id", "sample_id", "mask")
    meta_keys = ("channel",)
    default_config = {}
    default_thresholds = {"ratio_neg": (0.15, 0.30)} # warn, severe thresholds for ratio of negative events
    plot_type = "histogram"
    plot_description = "Distribution of compensated fluorescence values in channel"

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

    def _yield_skip_channel_records(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        message: str = "No events to test.",
    ) -> Iterable[QCTestRecord]:
        """Yield skip records for all channels."""
        for channel in entity.detectors:
            metadata = {"channel": channel}
            yield QCTestRecord(
                id=self.make_key(targets, metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                thresholds=self.thresholds,
                metrics={"ratio_neg": 0.0},
                status="SKIP",
                message=message,
            )

    def fit(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        adata: AnnData,
        *,
        mask: dict[str, np.ndarray] | None = None,
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for all channels in entity.detectors.

        Yields one QCTestRecord per channel.
        """
        mask_key = targets.get("mask", "root")
        if adata.X is None or adata.n_obs == 0:
            yield from self._yield_skip_channel_records(
                entity,
                targets,
                message="No events to test."
            )
            return

        # Handle mask parameter for subsetting (should be single parent mask only)
        if mask is not None:
            mask_array = mask[mask_key]
            adata = adata[mask_array]
        elif mask_key != "root":
            raise ValueError(
                f"Mask key '{mask_key}' specified in targets but no mask provided."
                "Mask parameter should contain only parent mask, not multiple gates."
            )

        if adata.n_obs == 0:
            yield from self._yield_skip_channel_records(
                entity,
                targets,
                message="No events to test (empty after mask)."
            )
            return

        # Iterate through all channels
        for channel in entity.detectors:
            if channel not in adata.var_names:
                # Skip channels not in adata
                metadata = {"channel": channel}
                test = QCTestRecord(
                    id=self.make_key(targets, metadata),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=metadata,
                    thresholds=self.thresholds,
                    metrics={"ratio_neg": 0.0},
                    status="SKIP",
                    message=f"Channel '{channel}' not found in adata.",
                )
                yield test
                continue

            # Extract values for this channel
            values = adata[:, channel].X.ravel() # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

            # Compute ratio of negative events
            ratio_neg = float(np.mean(values < 0))

            metadata = {"channel": channel}
            test = QCTestRecord(
                id=self.make_key(targets, metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                thresholds=self.thresholds,
                metrics={"ratio_neg": ratio_neg},
                status="PENDING",
            )
            yield test

    def classify(
        self,
        test: QCTestRecord,
        neg_warn: float | None = None,
        neg_severe: float | None = None,
        **kwargs
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

    def plot(
        self,
        adata: AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        nbins: int = 128,
        **kwargs
    ) -> go.Figure:
        """Generate data for plotting negative fluorescence histogram."""

        self._check_test_record(test)
        channel = test.metadata["channel"]
        values = adata[:, channel].X.ravel() # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        cap_percentile = kwargs.pop("cap_percentile", 95)
        if cap_percentile is None:
            capped_values = values
        else:
            cap_percentile = float(cap_percentile)
            if not 0 < cap_percentile <= 100:
                raise ValueError("cap_percentile must be in (0, 100].")
            cap_value = float(np.nanpercentile(values, cap_percentile))
            capped_values = np.minimum(values, cap_value)
        fig = build_histogram1d(
            values=capped_values,
            nbins=nbins,
            title=f'Histogram of {channel} Fluorescence Values',
            xaxis_title='Fluorescence Intensity',
            yaxis_title='Frequency',
            **kwargs
        )

        x_min = np.min(values)
        if x_min < 0.0:
            bar_color = kwargs.get('color', 'blue')
            fig.add_vline(
                x=0.0,
                line=dict(color="red" if bar_color != "red" else "blue", dash="dash"),
                annotation_text="",
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
    target_keys = ("compensation_id", "sample_id", "mask")
    meta_keys = ("channel",)

    default_config = {
        "min_neg_events_for_sigma": 50,
        "k_sigma_threshold": 4.0,
    }
    default_thresholds = {"ratio_very_neg": (0.01, 0.05)} # warn, severe thresholds for ratio of very negative events
    plot_type = "histogram"
    plot_description = "Distribution showing very negative events and sigma cutoff thresholds"

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
        _validate_warn_severe(
            "ratio_very_neg",
            self.very_warn,
            "ratio_very_neg",
            self.very_severe,
            are_percentage=True
        )

    @property
    def very_warn(self) -> float:
        return self.thresholds["ratio_very_neg"][0]

    @property
    def very_severe(self) -> float:
        return self.thresholds["ratio_very_neg"][1]

    def _yield_skip_channel_records(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        min_neg_events: int,
        k_sigma_threshold: float,
        message: str = "No events to test.",
    ) -> Iterable[QCTestRecord]:
        """Yield skip records for all channels."""
        for channel in entity.detectors:
            metadata = {
                "channel": channel,
                "min_events_for_sigma": min_neg_events,
                "k_sigma_threshold": k_sigma_threshold,
            }
            yield QCTestRecord(
                id=self.make_key(targets, metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                metrics={"ratio_very_neg": 0.0},
                status="SKIP",
                message=message,
            )

    def fit(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        adata: AnnData,
        *,
        mask: dict[str, np.ndarray] | None = None,
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for all channels in entity.detectors.

        Yields one QCTestRecord per channel.
        """
        # Extract config once
        min_neg_events = int(self.metadata["min_neg_events_for_sigma"])
        k_sigma_threshold = float(self.metadata["k_sigma_threshold"])
        mask_key = targets.get("mask", "root")

        if adata.X is None or adata.n_obs == 0:
            yield from self._yield_skip_channel_records(
                entity,
                targets,
                min_neg_events,
                k_sigma_threshold,
                message="No events to test.",
            )
            return

        # Handle mask parameter for subsetting (should be single parent mask only)
        if mask is not None:
            mask_array = mask[mask_key]
            adata = adata[mask_array]
        elif mask_key != "root":
            raise ValueError(
                f"Mask key '{mask_key}' specified in targets but no mask provided."
                "Mask parameter should contain only parent mask, not multiple gates."
            )

        if adata.n_obs == 0:
            yield from self._yield_skip_channel_records(
                entity,
                targets,
                min_neg_events,
                k_sigma_threshold,
                message="No events to test (empty after mask).",
            )
            return

        # Iterate through all channels
        missing = [channel for channel in entity.detectors if channel not in adata.var_names]
        if missing:
            raise ValueError(f"Channels not found in adata: {', '.join(missing)}")

        for channel in entity.detectors:
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
            ratio_very_neg = float(np.mean(values < very_cutoff))

            metadata = {
                "channel": channel,
                "min_events_for_sigma": min_neg_events,
                "k_sigma_threshold": k_sigma_threshold,
                "very_neg_cutoff": very_cutoff,
                "sigma_negative": sigma_neg,
            }
            test = QCTestRecord(
                id=self.make_key(targets, metadata),
                test_type=self.test_type,
                test_name=self.test_name,
                targets=targets,
                metadata=metadata,
                metrics={"ratio_very_neg": ratio_very_neg},
                status="PENDING",
            )
            yield test

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

    def plot(
        self,
        adata: AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        nbins: int = 128,
        **kwargs
    ) -> go.Figure:
        """Generate data for plotting negative fluorescence histogram."""

        self._check_test_record(test)
        channel = test.metadata["channel"]
        values = adata[:, channel].X.flatten() # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        cap_percentile = kwargs.pop("cap_percentile", 95)
        if cap_percentile is None:
            capped_values = values
        else:
            cap_percentile = float(cap_percentile)
            if not 0 < cap_percentile <= 100:
                raise ValueError("cap_percentile must be in (0, 100].")
            cap_value = float(np.nanpercentile(values, cap_percentile))
            capped_values = np.minimum(values, cap_value)
        fig = build_histogram1d(
            values=capped_values,
            nbins=nbins,
            title=f'Histogram of {channel} Fluorescence Values',
            xaxis_title='Fluorescence Intensity',
            yaxis_title='Frequency',
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
    target_keys = ("compensation_id", "sample_id", "mask")
    meta_keys = ("donor_channel", "receiver_channel")

    default_config = {
        "high_quantile": 0.99,
        "min_high_events": 1000,
    }
    default_thresholds = {"p_neg_given_high_donor": (0.05, 0.15)}
    plot_type = "scatter"
    plot_description = "Donor-receiver fluorescence correlation in high-donor population"

    def __init__(
        self,
        config: Mapping[str, Any] = {},
        thresholds: Mapping[str, Any] = {},
    ):
        super().__init__(config, thresholds)

        # Validate config
        high_q = self.metadata["high_quantile"]
        if high_q <= 0.0 or high_q >= 1.0:
            raise ValueError("high_quantile must be in (0, 1) range.")
        self.metadata["high_quantile"] = high_q

        min_events = self.metadata["min_high_events"]
        if min_events <= 0:
            raise ValueError("min_high_events must be positive.")
        self.metadata["min_high_events"] = min_events

        # Validate thresholds
        warn_val = self.thresholds.get("p_neg_given_high_donor", (0.05, 0.15))[0]
        severe_val = self.thresholds.get("p_neg_given_high_donor", (0.05, 0.15))[1]
        _validate_warn_severe("p_neg_given_high_donor", warn_val, "p_neg_given_high_donor", severe_val)

    def _yield_skip_pair_records(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        message: str = "No events to test.",
    ) -> Iterable[QCTestRecord]:
        """Yield skip records for all donor-receiver pairs."""
        high_quantile = self.metadata["high_quantile"]
        for donor in entity.detectors:
            for receiver in entity.detectors:
                if donor == receiver:
                    continue
                metadata = {
                    "donor_channel": donor,
                    "receiver_channel": receiver,
                    "compensation_coefficient": entity.spill.at[receiver, donor],
                    "high_quantile": high_quantile,
                }
                yield QCTestRecord(
                    id=self.make_key(targets, metadata),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=metadata,
                    metrics={
                        "p_neg_given_high_donor": 0.0,
                        "p_neg_given_low_donor": 0.0,
                    },
                    status="SKIP",
                    message=message,
                )

    def fit(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        adata: AnnData,
        *,
        mask: dict[str, np.ndarray] | None = None,
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for all donor-receiver pairs.

        Yields one QCTestRecord per pair.
        """
        high_quantile = self.metadata["high_quantile"]
        min_high_events = self.metadata["min_high_events"]
        mask_key = targets.get("mask", "root")

        if adata.X is None or adata.n_obs == 0:
            yield from self._yield_skip_pair_records(entity, targets, message="No events to test.")
            return

        # Handle mask parameter for subsetting (should be single parent mask only)
        if mask is not None:
            mask_array = mask[mask_key]
            adata = adata[mask_array]
        elif mask_key != "root":
            raise ValueError(
                f"Mask key '{mask_key}' specified in targets but no mask provided."
                "Mask parameter should contain only parent mask, not multiple gates."
            )

        if adata.n_obs == 0:
            yield from self._yield_skip_pair_records(
                entity,
                targets,
                message="No events to test (empty after mask)."
            )
            return

        # Iterate through all donor-receiver pairs
        for donor in entity.detectors:
            donor_values = np.asarray(adata[:, donor].X).ravel()

            # Compute donor threshold once (donor-specific, not receiver-specific)
            donor_thr = np.quantile(donor_values, high_quantile)
            donor_high = donor_values >= donor_thr
            n_high = donor_high.sum(dtype=int)
            if n_high < min_high_events:
                high_quantile_adj = 1.0 - (min_high_events / len(donor_values))
                donor_thr = np.quantile(donor_values, high_quantile_adj)
                donor_high = donor_values >= donor_thr
                n_high = donor_high.sum(dtype=int)

            for receiver in entity.detectors:
                if donor == receiver:
                    continue

                coef = entity.spill.at[receiver, donor]
                metadata = {
                    "donor_channel": donor,
                    "receiver_channel": receiver,
                    "compensation_coefficient": coef,
                    "high_quantile": high_quantile,
                    "n_donor_high": n_high,
                }
                test = QCTestRecord(
                    id=self.make_key(targets, metadata),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=metadata,
                    metrics={
                        "p_neg_given_high_donor": 0.0,
                        "p_neg_given_low_donor": 0.0,
                    },
                    status="PENDING"
                )

                # Extract receiver values
                receiver_values = np.asarray(adata[:, receiver].X).ravel()

                if n_high > 0:
                    test.metrics["p_neg_given_high_donor"] = \
                        float(np.mean(receiver_values[donor_high] < 0))
                    test.metrics["p_neg_given_low_donor"] = float(np.mean(receiver_values[~donor_high] < 0))

                yield test

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

    def plot(
        self,
        adata: AnnData,
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
    target_keys = ("compensation_id", "sample_id", "mask")
    meta_keys = ("donor_channel", "receiver_channel")
    default_config = {"high_quantile": 0.99, "min_high_events": 1000}
    default_thresholds = {"spearman_given_high_donor": (0.3, 0.5)} # warn, severe thresholds for Spearman correlation
    plot_type = "scatter"
    plot_description = "Donor-receiver correlation scatter in high-donor population"

    def __init__(self, config: Mapping[str, Any] = {}, thresholds: Mapping[str, Any] = {}):
        super().__init__(config, thresholds)
        _validate_percentage("high_quantile", self.metadata["high_quantile"])
        if self.metadata["min_high_events"] <= 0:
            raise ValueError("min_high_events must be positive.")

        cor_warn = self.thresholds["spearman_given_high_donor"][0]
        cor_severe = self.thresholds["spearman_given_high_donor"][1]
        _validate_warn_severe("spearman_given_high_donor_warn", cor_warn, "spearman_given_high_donor_severe", cor_severe)

    def _yield_skip_pair_records(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        message: str = "Not enough events to test.",
    ) -> Iterable[QCTestRecord]:
        """Yield skip records for all donor-receiver pairs."""
        high_quantile = self.metadata["high_quantile"]
        for donor in entity.detectors:
            for receiver in entity.detectors:
                if donor == receiver:
                    continue
                metadata = {
                    "spillover_coefficient": entity.spill.at[receiver, donor],
                    "donor_channel": donor,
                    "receiver_channel": receiver,
                    "high_quantile": high_quantile,
                }
                yield QCTestRecord(
                    id=self.make_key(targets, metadata),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=metadata,
                    metrics={"spearman_given_high_donor": 0.0},
                    status="SKIP",
                    message=message,
                )

    def fit(
        self,
        entity: CompensationRef,
        targets: dict[str, Any],
        adata: AnnData,
        *,
        mask: dict[str, np.ndarray] | None = None,
        **kwargs
    ) -> Iterable[QCTestRecord]:
        """Compute test metrics for all donor-receiver pairs.

        Yields one QCTestRecord per pair.
        """
        high_quantile = self.metadata["high_quantile"]
        min_high_events = self.metadata["min_high_events"]
        mask_key = targets.get("mask", "root")

        if adata.X is None or adata.n_obs <= min_high_events * 2:
            yield from self._yield_skip_pair_records(entity, targets, message="Not enough events to test.")
            return

        # Handle mask parameter for subsetting (should be single parent mask only)
        if mask is not None:
            mask_array = mask[mask_key]
            adata = adata[mask_array]

        if adata.X is None or adata.n_obs <= min_high_events * 2:
            yield from self._yield_skip_pair_records(entity, targets, message="Not enough events to test (after mask).")
            return

        # Iterate through all donor-receiver pairs
        for donor in entity.detectors:
            donor_values = np.asarray(adata[:, donor].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

            # Compute donor threshold once (donor-specific, not receiver-specific)
            donor_thr = np.quantile(donor_values, high_quantile)
            donor_high = donor_values >= donor_thr
            n_high = donor_high.sum()

            # Ensure at least min_high_events are included
            if n_high < min_high_events:
                high_quantile_adj = 1.0 - (min_high_events / len(donor_values))
                donor_thr = np.quantile(donor_values, high_quantile_adj)
                donor_high = donor_values >= donor_thr
                n_high = donor_high.sum()

            for receiver in entity.detectors:
                if donor == receiver:
                    continue

                receiver_values = np.asarray(adata[:, receiver].X).ravel()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                coef = entity.spill.at[receiver, donor]

                metadata = {
                    "spillover_coefficient": coef,
                    "donor_channel": donor,
                    "receiver_channel": receiver,
                    "high_quantile": high_quantile,
                    "high_threshold": donor_thr,
                    "n_donor_high": n_high,
                }
                test = QCTestRecord(
                    id=self.make_key(targets, metadata),
                    test_type=self.test_type,
                    test_name=self.test_name,
                    targets=targets,
                    metadata=metadata,
                    metrics={"spearman_given_high_donor": 0.0},
                    status="PENDING"
                )

                corr, _ = spearmanr(donor_values[donor_high], receiver_values[donor_high])  # pyright: ignore
                test.metrics["spearman_given_high_donor"] = corr  # pyright: ignore

                yield test

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
        adata: AnnData,
        test: QCTestRecord,
        output_path: PathLike | None = None,
        **kwargs
    ) -> go.Figure:
        """Plot Spearman correlation across donor quantile bins.

        Parameters
        ----------
        adata : AnnData
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

    entity_type = "compensation"
    targets = ("compensation_id", "sample_id", "mask")
    _supported_tables = {
        "compensation_channel": {
            "description": "Per-channel negative fluorescence metrics across samples",
            "input_params": {}
        },
        "compensation_pair": {
            "description": "Pairwise spillover metrics (correlation, enrichment) across samples",
            "input_params": {}
        },
    }
    _supported_figures = {}  # Test plots are auto-discovered from registered tests

    default_config = {
        "subsample": 1.0,
        "high_quantile": 0.95,
        "neg_warn": 0.15,
        "neg_severe": 0.30,
        "very_warn": 0.01,
        "very_severe": 0.05,
        "min_neg_events_for_sigma": 50,
        "k_sigma_threshold": 4.0,
        "min_high_events": 1000,
        "tail_neg_warn": 0.20,
        "tail_neg_severe": 0.40,
        "tail_cor_warn": 0.7,
        "tail_cor_severe": 0.9,
        "transform_func": "logicle",
    }

    @classmethod
    def get_tests(cls, entity: CompensationRef | None = None) -> dict[str, type[QCTester]]:
        """Return dictionary of test classes for compensation QC.

        Parameters
        ----------
        entity : Any, optional
            Entity parameter (ignored for compensation evaluator).

        Returns
        -------
        dict[str, type[QCTester]]
            Mapping of test_name → QCTester subclass
        """
        return {
            "negative_fluorescence": NegativeFluorescenceTest,
            "very_negative_fluorescence": VeryNegativeFluorescenceTest,
            "negative_enrichment": NegativeEnrichmentTest,
            "high_donor_correlation": HighDonorCorrelationTest,
        }

    def required_layer(self, entity: CompensationRef | None = None) -> str:
        return "comp"

    def load_entity(self, dataloader: UnifiedDataLoader, entity_id: Hashable) -> CompensationRef:
        project = dataloader.load_data("project")
        if entity_id not in project.compensations:
            raise KeyError(f"Compensation '{entity_id}' not found in project.")
        return project.compensations[str(entity_id)]

    def update_sample_qc(
        self,
        entity: CompensationRef,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        """Evaluate compensation QC against a sample context.

        Parameters
        ----------
        entity : CompensationRef
            The compensation entity to evaluate.
        entity_qc : EntityQCStatus
            QC status to update.
        dataloader : UnifiedDataLoader | None
            UnifiedDataLoader for loading sample data on the fly
        dataloader_context : dict[str, Any] | None
            Context with sample_ids, layer, etc. or specific pre-loaded adata
        context : dict[str, Any]
            Optional metadata to attach to the QC status.

        Returns
        -------
        EntityQCStatus
            Updated QC status with test results and summary.
        """
        # Default context to empty dict to avoid None checks
        dataloader_context = dataloader_context or {}

        config = self.config.copy()
        config.update(context)
        entity_qc.context = config

        # Create adata iterator - either from pre-loaded data or via dataloader
        if "adata" in dataloader_context:
            # Use pre-loaded tuple iterator of (sample_id, adata)
            adata_iter = dataloader_context["adata"]
        elif dataloader is not None:
            # Create iterator by loading data on the fly
            sample_ids = dataloader_context.get("sample_ids") or list(entity_qc.sample_qc.keys())
            layer = self.required_layer(entity)
            adata_iter = ((sid, dataloader.load_adata(sid, layer=layer)) for sid in sample_ids)
        else:
            # No data source available - raise error
            raise ValueError(
                "Cannot evaluate compensation QC: dataloader_context must contain 'adata' "
                "or a dataloader must be provided to load sample data."
            )

        # Process all samples from the iterator
        for sample_id, adata in adata_iter:
            self.check_compensated_adata(
                entity=entity,
                adata=adata,
                entity_qc=entity_qc,
                sample_id=sample_id,
            )

        return entity_qc

    def update_batch_qc(
        self,
        entity: CompensationRef,
        entity_qc: EntityQCStatus,
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        *,
        context: dict[str, Any] = {},
    ) -> EntityQCStatus:
        """Update batch-level QC tests for compensation.

        Compensation has no batch-level tests, so this is a no-op.

        Parameters
        ----------
        entity : CompensationRef
            The compensation entity being evaluated
        entity_qc : EntityQCStatus
            QC status to update
        all_samples : Iterable[tuple[str, AnnData]] | None
            Iterable of (sample_id, adata) tuples (unused)
        context : dict[str, Any]
            Optional evaluation context (unused)

        Returns
        -------
        EntityQCStatus
            Unchanged entity_qc (no batch tests for compensation)
        """
        return entity_qc

    def summarize_entity_qc(self, entity_qc: EntityQCStatus) -> dict[str, Any]:
        # TODO: check for outlier samples in metrics
        return {}

    def check_compensated_adata(
        self,
        entity: CompensationRef,
        adata: AnnData,
        entity_qc: EntityQCStatus,
        sample_id: str,
    ) -> dict[str, list[tuple[str, QCTestRecord]]]:
        """Run channel and pairwise QC on compensated data.

        Uses the instance compensation test runner internally,
        then aggregates results into QCRunStatus for step-level tracking.

        Parameters
        ----------
        entity : CompensationRef
            The compensation entity being evaluated.
        adata : AnnData
            Compensated sample data.
        qc : QCRunStatus
            QC run status for this sample.
        sample_id : str
            Sample identifier.

        Returns
        -------
        dict[str, list[tuple[str, QCTestRecord]]]
            Dict with all tests (channel and pairwise).
        """
        comp_id = adata.uns.get("compensation_id", None)
        if not comp_id or comp_id != entity.id:
            warnings.warn(f"Sample {sample_id} compensation_id '{comp_id}' does not match entity ID '{entity.id}'")

        all_tests: dict[str, list[tuple[str, QCTestRecord]]] = {"channel": [], "pairwise": []}

        # Call stateless function to get all test results
        tests = self._run_compensation_tests(entity, adata, sample_id=sample_id, config=entity_qc.context)
        step = entity_qc.get_sample_steps(sample_id).get_step("COMP_QC_OVERVIEW")

        # Separate into channel and pairwise
        for test in tests:
            if test.test_type == "compensation_channel":
                all_tests["channel"].append((sample_id, test))
            else:  # compensation_pair
                all_tests["pairwise"].append((sample_id, test))

            # Only add to QC report if WARN or SEVERE
            if test.status in {"SEVERE", "WARN"}:
                step.add_reason(
                    code=f"COMP_QC_{test.status}",
                    message=test.message,
                    tests=[test],
                )
            else:
                step.add_test(test)

        return all_tests

    def generate_figure(
        self,
        entity_qc: EntityQCStatus,
        test_key: Mapping[str, str],
        dataloader: UnifiedDataLoader | None = None,
        dataloader_context: dict[str, Any] | None = None,
        step_id: str | None = None,
        figure_dir: PathLike | None = None,
        **kwargs
    ) -> go.Figure:
        """Generate a figure for a specific test on demand.

        Parameters
        ----------
        entity_qc : EntityQCStatus
            The QC status object containing test records.
        test_key : Hashable
            Test key to look up in the QC status (from QCStepStatus.tests).
        dataloader : UnifiedDataLoader | None
            Optional UnifiedDataLoader for loading sample data (AnnData) for plotting.
        dataloader_context : dict[str, Any] | None
            Optional context parameters for the dataloader (e.g., sample_id or sample_ids, layer).
            Can optionally contain 'adata' with an iterable of (sample_id, adata) tuples.
            If 'adata' not provided, it will be created on-the-fly from dataloader.
            Note: Only single-sample figures are currently supported; multiple samples will raise an error.
        step_id : str | None
            Optional step ID to narrow down the search for the test.
            If None, searches all steps for the sample and returns the first match.
        figure_dir : PathLike | None
            Optional directory to save the figure.

        Returns
        -------
        go.Figure
            Plotly figure object ready to be serialized or displayed.

        Raises
        ------
        ValueError
            If multiple samples are provided (faceted plots planned for future).
        """
        # Default context to empty dict to avoid None checks
        dataloader_context = dataloader_context or {}

        # Get sample_id: support both singular 'sample_id' and plural 'sample_ids'

        # Build adata iterator: either from context or create on-the-fly from dataloader
        if "adata" in dataloader_context:
            # Use provided iterable of (sample_id, adata) tuples
            adata_itr: Iterable[tuple[str, AnnData]] = dataloader_context["adata"]
            sample_id, adata = next(iter(adata_itr), (None, None))
        else:
            # Create iterator on-the-fly from dataloader
            if dataloader is None:
                raise ValueError("dataloader must be provided if 'adata' is not in dataloader_context")

            sample_ids: list[str] = list(dataloader_context.get("sample_ids") or entity_qc.sample_qc.keys())

            # Validate and resolve sample_id
            if len(sample_ids) > 1:
                raise NotImplementedError(
                    f"Multiple samples ({len(sample_ids)}) are not yet supported for figure generation. "
                    "Faceted plots will be supported in a future release."
                )
            if len(sample_ids) == 0:
                raise ValueError("sample_ids must not be empty")
            # adata_itr = (
            #     (sample_id, dataloader.load_adata(sample_id, layer="comp"))
            #     for sample_id in sample_ids
            # )
            sample_id = sample_ids[0]
            layer = self.required_layer()
            adata = dataloader.load_adata(sample_id, layer="comp")


        if adata is None:
            raise ValueError(f"No AnnData found for sample {sample_id}")

        # Retrieve QC status for the sample
        if sample_id not in entity_qc.sample_qc:
            raise KeyError(f"Sample {sample_id} not found in QC status for entity {entity_qc.entity_id}")
        sample_run = entity_qc.sample_qc[sample_id]

        # Parse and validate test_key using helper
        tester_class, test_key_dict = self._parse_test_key(test_key)

        # Generate test key tuple using tester's make_key method
        tester = tester_class()
        test_key_tuple = tester.make_key(test_key_dict)

        # Recover test record from QC status using the step_id and test_key
        test = None
        if step_id is not None:
            step = sample_run.steps.get(step_id)
            if step and test_key_tuple in step.tests:
                test = step.tests[test_key_tuple]
        else:
            for step in sample_run.steps.values():
                if test_key_tuple in step.tests:
                    test = step.tests[test_key_tuple]
                    break
        if test is None:
            raise KeyError(f"Test {test_key_tuple} not found for sample {sample_id} in QC status")

        # Generate figure using tester
        tester = tester_class.from_dict(test)
        return tester.plot(adata=adata, test=test)

    def _run_compensation_tests(
        self,
        entity: CompensationRef,
        adata: AnnData,
        sample_id: str,
        config: Mapping[str, Any],
    ) -> Iterable[QCTestRecord]:
        """Run all compensation QC testers using a unified fit_classify interface."""

        missing = [channel for channel in entity.detectors if channel not in adata.var_names]
        if missing:
            raise ValueError(f"Channels {missing} from compensation entity not found in adata.var_names")
        targets = {"compensation_id": entity.id, "sample_id": sample_id, "mask": "root"}  # TODO: handle multiple masks/gates in the future

        testers: list[QCTester] = [
            NegativeFluorescenceTest(
                thresholds={
                    "ratio_neg": (
                        config.get("neg_warn", self.default_config["neg_warn"]),
                        config.get("neg_severe", self.default_config["neg_severe"]),
                    )
                },
            ),
            VeryNegativeFluorescenceTest(
                config={
                    "min_neg_events_for_sigma": config.get(
                        "min_neg_events_for_sigma", self.default_config["min_neg_events_for_sigma"]
                    ),
                    "k_sigma_threshold": config.get("k_sigma_threshold", self.default_config["k_sigma_threshold"]),
                },
                thresholds={
                    "ratio_very_neg": (
                        config.get("very_warn", self.default_config["very_warn"]),
                        config.get("very_severe", self.default_config["very_severe"]),
                    )
                },
            ),
                NegativeEnrichmentTest(
                    config={
                        "high_quantile": config.get("high_quantile", self.default_config["high_quantile"]),
                        "min_high_events": config.get("min_high_events", self.default_config["min_high_events"]),
                    },
                    thresholds={
                        "p_neg_given_high_donor": (
                            config.get("tail_neg_warn", self.default_config["tail_neg_warn"]),
                            config.get("tail_neg_severe", self.default_config["tail_neg_severe"]),
                        )
                    },
                ),
                HighDonorCorrelationTest(
                    config={
                        "high_quantile": config.get("high_quantile", self.default_config["high_quantile"]),
                        "min_high_events": config.get("min_high_events", self.default_config["min_high_events"]),
                    },
                    thresholds={
                        "spearman_given_high_donor": (
                            config.get("tail_cor_warn", self.default_config["tail_cor_warn"]),
                            config.get("tail_cor_severe", self.default_config["tail_cor_severe"]),
                        )
                    },
                ),
            ]

        for tester in testers:
            yield from tester.fit_classify(entity=entity, targets=targets, adata=adata)
