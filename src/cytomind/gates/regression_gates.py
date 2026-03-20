"""Regression-based gates that fit models to 2D data and create polygon boundaries."""

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from cytomind.gates.glm_gates import PolygonGate
from . import GateRegistry

if TYPE_CHECKING:
    from cytomind.domain.constants import FloatArray
    from pandas import DataFrame
else:
    DataFrame = object
    FloatArray = object


class RegressionGate(PolygonGate):
    """
    Abstract base class for regression-based gates.

    Regression gates fit a model to (x, y) data and create a polygon boundary
    based on confidence intervals around the fitted curve. The polygon has vertices
    along both the upper and lower confidence bounds.

    Subclasses must implement:
    - _fit_regression(x, y): Fit the regression model to data
    - _predict_with_confidence(x): Return predictions with confidence intervals
    """

    gate_type = "regression"
    tunable = True

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        alpha: float = 0.99,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        num_vertices: int = 5,
        subsample: float = 1.0,
        min_points: int = 50,
        use_as_complement: bool = False,
        vertices = [[0, 0], [0, 1], [1, 0]]
    ) -> None:
        """
        Initialize regression gate.

        Parameters
        ----------
        gate_name : str
            Name of the gate
        dimensions : Sequence[str]
            Exactly 2 dimensions [x_dim, y_dim] where x is independent, y is dependent
        alpha : float, optional
            Confidence level for prediction interval (default: 0.95)
        x_range : Sequence[float] | None, optional
            Optional [min, max] bounds for x predictions
        y_range : Sequence[float] | None, optional
            Optional [min, max] bounds to clip y predictions
        num_vertices : int, optional
            Number of vertices to sample along the curve (default: 50)
        subsample : float, optional
            Fraction of data to use for fitting (default: 1.0)
        min_points : int, optional
            Minimum number of points required for fitting (default: 50)
        use_as_complement : bool, optional
            Whether to use gate as complement (default: False)
        """
        if len(dimensions) != 2:
            raise ValueError("RegressionGate requires exactly two dimensions [x, y].")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1 (exclusive).")
        if num_vertices < 2:
            raise ValueError("num_vertices must be at least 2.")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1].")
        if min_points < 1:
            raise ValueError("min_points must be >= 1.")
        if x_range is not None:
            if len(x_range) != 2:
                raise ValueError("x_range must be a sequence of two values [min, max].")
            x_low, x_high = float(x_range[0]), float(x_range[1])
            if x_high < x_low:
                raise ValueError("x_range max must be greater than or equal to min.")
            if x_low < 0.0:
                raise ValueError("x_range min must be non-negative.")
        if y_range is not None:
            if len(y_range) != 2:
                raise ValueError("y_range must be a sequence of two values [min, max].")
            y_low, y_high = float(y_range[0]), float(y_range[1])
            if y_high < y_low:
                raise ValueError("y_range max must be greater than or equal to min.")

        # Initialize with placeholder vertices (will be set during fitting)
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            vertices=vertices,
            use_as_complement=use_as_complement
        )

        self._hyperparams.update({
            "alpha": alpha,
            "x_range": x_range,
            "y_range": y_range,
            "num_vertices": num_vertices,
            "subsample": float(subsample),
            "min_points": int(min_points),
        })

    @abstractmethod
    def _fit_regression(self, x: FloatArray, y: FloatArray) -> None:
        """
        Fit regression model to data.

        Parameters
        ----------
        x : FloatArray
            Independent variable values
        y : FloatArray
            Dependent variable values

        Notes
        -----
        Implementations should store the fitted model as instance attributes
        for use by _predict_with_confidence(). Note that `x` and `y` may be
        a random subsample of the original events (controlled by the
        `subsample` and `min_points` hyperparameters).
        """
        pass

    @abstractmethod
    def _predict_with_confidence(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """
        Predict y values with confidence intervals for given x values.

        Parameters
        ----------
        x : NDArray
            x values at which to make predictions

        Returns
        -------
        y_pred : NDArray
            Predicted y values
        y_lower : NDArray
            Lower confidence bound
        y_upper : NDArray
            Upper confidence bound
        """
        pass

    def _get_x_bouns(self, x: FloatArray) -> tuple[float, float]:
        """Get x bounds for prediction based on hyperparams and data."""
        x_min, x_max = x.min(), x.max()
        x_range_data = x_max - x_min
        x_range_param = self.hyperparams["x_range"]
        if x_range_param is None:
            padding = 0.1 * x_range_data
            x_low = max(0.0, x_min - padding)
            x_high = x_max + padding
        else:
            x_low = float(x_range_param[0])
            x_high = float(x_range_param[1])
        return x_low, x_high

    def _fit_gate(self, events_slice: DataFrame) -> None:
        """
        Fit regression model and generate polygon vertices.

        Parameters
        ----------
        events_slice : DataFrame
            Event data with columns matching self.dimensions
        """
        dim_x, dim_y = self.dimensions

        # Extract data and convert to numpy arrays
        x = events_slice[dim_x].to_numpy(np.float32)
        y = events_slice[dim_y].to_numpy(np.float32)

        # Decide whether to subsample for fitting
        n_total = int(len(x))
        subsample = float(self.hyperparams.get("subsample", 1.0))
        min_points = int(self.hyperparams.get("min_points", 1))

        # Compute number of points to use for fitting
        k = int(np.ceil(subsample * n_total)) if n_total > 0 else 0
        k = int(max(min_points, k))
        k = min(k, n_total)

        if k < n_total and k > 0:
            rng = np.random.default_rng()
            idx = rng.choice(n_total, size=k, replace=False)
            x_fit = x[idx]
            y_fit = y[idx]
        else:
            x_fit = x
            y_fit = y

        # Fit the regression model (implemented by subclass) on the chosen subset
        self._fit_regression(x_fit, y_fit)

        # Store sample sizes in params (required for confidence interval calculations)
        self.params["n_obs"] = int(len(x_fit))

        # Generate x values for predictions
        x_low, x_high = self._get_x_bouns(x)
        x_pred = np.linspace(x_low, x_high, self.hyperparams["num_vertices"])

        # Get predictions with confidence intervals
        y_pred, y_lower, y_upper = self._predict_with_confidence(x_pred)

        # Apply y_range clipping if specified
        y_range = self.hyperparams["y_range"]
        if y_range is not None:
            y_lower = np.clip(y_lower, y_range[0], y_range[1])
            y_upper = np.clip(y_upper, y_range[0], y_range[1])

        # Build polygon vertices: left to right along upper bound,
        # then right to left along lower bound (creates closed polygon)
        vertices = []

        # Upper confidence bound (left to right)
        for x, yu in zip(x_pred, y_upper):
            vertices.append([x, yu])

        # Lower confidence bound (right to left, reversed)
        for x, yl in zip(reversed(x_pred), reversed(y_lower)):
            vertices.append([x, yl])

        # Store vertices (required by apply)
        self.params["vertices"] = np.array(vertices)

        # Store x/y ranges used and observed (required for reproducibility)
        self.params["x_range_used"] = (float(x_low), float(x_high))
        self.params["y_range_fitted"] = (float(y.min()), float(y.max()))


@GateRegistry.register("linear_regression")
class LinearRegressionGate(RegressionGate):
    """
    Regression gate using ordinary least squares (OLS) linear regression.

    Fits a simple linear regression line and creates a polygon boundary based
    on prediction intervals. Suitable for data with normally distributed errors
    and no significant outliers.
    """

    gate_type = "linear_regression"

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        alpha: float = 0.99,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        num_vertices: int = 5,
        subsample: float = 1.0,
        min_points: int = 50,
        use_as_complement: bool = False,
    ) -> None:
        """
        Initialize linear regression gate.

        Parameters
        ----------
        gate_name : str
            Name of the gate
        dimensions : Sequence[str]
            Exactly 2 dimensions [x_dim, y_dim] where x is independent, y is dependent
        alpha : float, optional
            Confidence level for prediction interval (default: 0.95)
        x_range : Sequence[float] | None, optional
            Optional [min, max] bounds for x predictions
        y_range : Sequence[float] | None, optional
            Optional [min, max] bounds to clip y predictions
        num_vertices : int, optional
            Number of vertices to sample along the curve (default: 50)
        use_as_complement : bool, optional
            Whether to use gate as complement (default: False)
        """
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            alpha=alpha,
            x_range=x_range,
            y_range=y_range,
            num_vertices=num_vertices,
            subsample=subsample,
            min_points=min_points,
            use_as_complement=use_as_complement,
        )

        # Will store fitted model
        self._ols_fit = None

    def _fit_regression(self, x: FloatArray, y: FloatArray) -> None:
        """Fit ordinary least squares linear regression."""
        # Prepare design matrix with intercept
        X = sm.add_constant(x)

        # Fit OLS model
        ols_model = sm.OLS(y, X)
        self._ols_fit = ols_model.fit()
        ols_inf = OLSInfluence(self._ols_fit)

        # Store model parameters required for vertex re-calculation (JSON serializable)
        self.params["coefficients"] = {
            "intercept": float(self._ols_fit.params[0]),
            "slope": float(self._ols_fit.params[1])
        }
        self.params["scale"] = float(self._ols_fit.mse_resid)
        self.params["mean_x"] = float(np.mean(x))
        self.params["var_x"] = float(np.var(x, ddof=1))  # Sample variance of x

        # Store fit quality diagnostics
        self.diagnostics["r_squared"] = float(self._ols_fit.rsquared)
        self.diagnostics["residual_std"] = float(np.sqrt(self._ols_fit.mse_resid))
        cook_pval = ols_inf.cooks_distance[1] * len(x)  # Adjust p-values for number of observations
        self.diagnostics["n_outliers"] = int(np.sum(cook_pval < 0.05))

    def _predict_with_confidence(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Predict using OLS model with confidence intervals."""
        if self._ols_fit is None:
            raise RuntimeError("Model not fitted. Call _fit_regression first.")

        alpha = self.hyperparams["alpha"]

        # Prepare design matrix
        X = sm.add_constant(x)

        # Get predictions with confidence intervals
        prediction = self._ols_fit.get_prediction(X)
        pred_summary = prediction.summary_frame(alpha=1 - alpha)

        y_pred = pred_summary["mean"].values
        y_lower = pred_summary["obs_ci_lower"].values
        y_upper = pred_summary["obs_ci_upper"].values

        return y_pred, y_lower, y_upper


@GateRegistry.register("robust_regression")
class RobustRegressionGate(RegressionGate):
    """
    Regression gate using Huber's robust linear regression.

    Fits a robust regression line using Huber's T method, which is less
    sensitive to outliers than ordinary least squares. Confidence intervals
    are computed using weighted least squares with the robust weights.
    """

    gate_type = "robust_regression"

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        alpha: float = 0.99,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        num_vertices: int = 5,
        subsample: float = 1.0,
        min_points: int = 50,
        huber_t: float = 2.01,
        use_as_complement: bool = False,
        vertices = [[0, 0], [0, 1], [1, 0]]
    ) -> None:
        """
        Initialize robust regression gate.

        Parameters
        ----------
        gate_name : str
            Name of the gate
        dimensions : list[str]
            Exactly 2 dimensions [x_dim, y_dim] where x is independent, y is dependent
        alpha : float, optional
            Confidence level for prediction interval (default: 0.95)
        x_range : Sequence[float] | None, optional
            Optional [min, max] bounds for x predictions
        y_range : Sequence[float] | None, optional
            Optional [min, max] bounds to clip y predictions
        num_vertices : int, optional
            Number of vertices to sample along the curve (default: 50)
        subsample : float, optional
            Fraction of data to use for fitting (default: 1.0)
        min_points : int, optional
            Minimum number of points required for fitting (default: 50)
        huber_t : float, optional
            Tuning constant for Huber's T function (default: 1.345 for 95% efficiency)
        use_as_complement : bool, optional
            Whether to use gate as complement (default: False)
        """
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            alpha=alpha,
            x_range=x_range,
            y_range=y_range,
            num_vertices=num_vertices,
            subsample=subsample,
            min_points=min_points,
            use_as_complement=use_as_complement,
        )
        self._hyperparams["huber_t"] = huber_t

        # Will store fitted models
        self._rlm_fit = None
        self._wls_fit = None

    def _fit_regression(self, x: FloatArray, y: FloatArray) -> None:
        """Fit Huber's robust regression and weighted least squares."""
        # Prepare design matrix with intercept
        X = sm.add_constant(x)

        # Fit robust regression using Huber's T
        huber_t = self.hyperparams["huber_t"]
        rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=huber_t))
        self._rlm_fit = rlm_model.fit()

        # Fit WLS using robust weights for confidence intervals
        self._wls_fit = sm.WLS(y, X, weights=self._rlm_fit.weights).fit()

        # Store model parameters required for vertex re-calculation (JSON serializable)
        self.params["coefficients"] = {
            "intercept": float(self._wls_fit.params[0]),
            "slope": float(self._wls_fit.params[1])
        }
        self.params["scale"] = float(self._wls_fit.scale)
        self.params["mean_x"] = float(np.mean(x))
        self.params["var_x"] = float(np.var(x, ddof=1))  # Sample variance of x

        # Store fit quality diagnostics
        self.diagnostics["r_squared"] = float(self._wls_fit.rsquared)
        self.diagnostics["resid_std"] = float(np.sqrt(self._wls_fit.mse_resid))
        self.diagnostics["n_outliers"] = int(np.sum(self._rlm_fit.weights < 0.9))

    def _predict_with_confidence(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Predict using WLS model with confidence intervals."""
        if self._wls_fit is None:
            raise RuntimeError("Model not fitted. Call _fit_regression first.")

        alpha = self.hyperparams["alpha"]

        # Prepare design matrix
        X = sm.add_constant(x)

        # Get predictions with confidence intervals
        prediction = self._wls_fit.get_prediction(X)
        pred_summary = prediction.summary_frame(alpha=1 - alpha)

        y_pred = pred_summary["mean"].values
        y_lower = pred_summary["obs_ci_lower"].values
        y_upper = pred_summary["obs_ci_upper"].values

        return y_pred, y_lower, y_upper


@GateRegistry.register("linear_gam")
class LinearGAMGate(RegressionGate):
    """
    Regression gate using a linear GAM with B-splines.

    Fits a generalized additive model with a B-spline basis on x to produce
    a smooth curve. Confidence intervals are derived from the GAM prediction.
    """
    gate_type = "linear_gam"

    def __init__(
        self,
        gate_name: str,
        dimensions: Sequence[str],
        alpha: float = 0.99,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        num_vertices: int = 50,
        subsample: float = 1.0,
        min_points: int = 50,
        df: int = 6,
        degree: int = 3,
        use_as_complement: bool = False,
    ) -> None:
        """
        Initialize linear GAM gate.

        Parameters
        ----------
        gate_name : str
            Name of the gate
        dimensions : Sequence[str]
            Exactly 2 dimensions [x_dim, y_dim] where x is independent, y is dependent
        alpha : float, optional
            Confidence level for prediction interval (default: 0.99)
        x_range : Sequence[float] | None, optional
            Optional [min, max] bounds for x predictions
        y_range : Sequence[float] | None, optional
            Optional [min, max] bounds to clip y predictions
        num_vertices : int, optional
            Number of vertices to sample along the curve (default: 50)
        df : int, optional
            Degrees of freedom for the spline basis (default: 6)
        degree : int, optional
            Degree of the spline (default: 3)
        use_as_complement : bool, optional
            Whether to use gate as complement (default: False)
        """
        if df < 3:
            raise ValueError("df must be at least 3.")
        if degree < 1:
            raise ValueError("degree must be at least 1.")
        super().__init__(
            gate_name=gate_name,
            dimensions=dimensions,
            alpha=alpha,
            x_range=x_range,
            y_range=y_range,
            num_vertices=num_vertices,
            subsample=subsample,
            min_points=min_points,
            use_as_complement=use_as_complement,
        )
        self._hyperparams.update({
            "df": df,
            "degree": degree,
        })

        self._bs: Any = None
        self._gam_fit: Any = None

    def _fit_regression(self, x: FloatArray, y: FloatArray) -> None:
        """Fit a linear GAM using B-spline basis on x."""
        from statsmodels.gam.api import BSplines, GLMGam
        df = self.hyperparams["df"]
        degree = self.hyperparams["degree"]

        # Create BSplines with custom knot range that extends beyond data
        # This allows some extrapolation beyond the training data
        x_low, x_high = self._get_x_bouns(x)
        knot_kwd = {"spacing": "quantile", "lower_bound": x_low, "upper_bound": x_high}
        self._bs = BSplines(x[:, np.newaxis], df=[df], degree=[degree], knot_kwds=knot_kwd)
        gam_model = GLMGam(y, smoother=self._bs)
        self._gam_fit = gam_model.fit()

        # Store model parameters required by apply
        self.params["scale"] = float(self._gam_fit.scale)
        self.params["gam_edf"] = float(self._gam_fit.edf.mean())  # Effective degrees of freedom

        # Store fit quality diagnostics
        self.diagnostics.update({
            "mean_edf": float(self._gam_fit.edf.mean()),
            "cv": float(self._gam_fit.cv),
            "aic": float(self._gam_fit.aic),
        })

    def _predict_with_confidence(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Predict using GAM with confidence intervals."""
        if self._gam_fit is None:
            raise RuntimeError("Model not fitted. Call _fit_regression first.")

        alpha = 1. - self.hyperparams["alpha"]
        prediction = self._gam_fit.get_prediction(exog=None, exog_smooth=x[:, np.newaxis])
        pred_summary = prediction.summary_frame(alpha=alpha)

        y_pred = pred_summary["mean"].values

        # Try to get prediction intervals from summary
        if "obs_ci_lower" in pred_summary.columns and "obs_ci_upper" in pred_summary.columns:
            # Use available prediction intervals
            y_lower = pred_summary["obs_ci_lower"].values
            y_upper = pred_summary["obs_ci_upper"].values
            return y_pred, y_lower, y_upper

        # Construct prediction intervals manually for Gaussian family
        # Prediction interval = mean prediction ± t * sqrt(SE(mean)^2 + residual_var)
        # This accounts for both parameter uncertainty and inherent data variability
        from scipy import stats

        # Get standard errors of the mean if available
        if "mean_se" in pred_summary.columns:
            se_mean = pred_summary["mean_se"].values
        else:
            se_mean = np.zeros_like(y_pred)

        # For Gaussian family, residual variance is the scale parameter
        residual_var = self._gam_fit.scale
        se_obs = np.sqrt(se_mean**2 + residual_var)

        # Critical value for prediction interval
        df_resid = self._gam_fit.df_resid if hasattr(self._gam_fit, 'df_resid') else len(y_pred) - 1
        t_crit = stats.t.ppf(1. - .5 * alpha, df_resid)

        # Calculate prediction intervals
        y_disp = t_crit * se_obs
        y_lower = y_pred - y_disp
        y_upper = y_pred + y_disp

        return y_pred, y_lower, y_upper


def huber_efficiency(t: float) -> float:
    """
    Asymptotic efficiency of Huber's T M-estimator for Normal(0,1) errors
    as a function of tuning constant t (used in statsmodels.robust.norms.HuberT).

    Efficiency formula:
        Eff(t) = (E[psi'(Z)])^2 / E[psi(Z)^2],  Z ~ N(0,1)
    where psi(z) = z for |z|<=t and t*sign(z) otherwise.
    Returns a float in (0, 1).
    """
    if t <= 0:
        raise ValueError("t must be > 0")

    from scipy.stats import norm

    Phi = norm.cdf
    phi = norm.pdf

    A = 2.0 * Phi(t) - 1.0
    # E[Z^2 * I(|Z|<=t)] has closed form:
    # 2 * ∫_0^t z^2 phi(z) dz = 2 * (Phi(t) - 0.5 - t*phi(t))
    Ez2_trunc = 2.0 * (Phi(t) - 0.5 - t * phi(t))

    tail = 2.0 * (1.0 - Phi(t))
    B = Ez2_trunc + (t**2) * tail               # E[psi(Z)^2]

    return ((A**2) / B)[0]

def huber_c_for_efficiency(
        target_eff: float = 0.95,
        bracket=(1e-6, 50.0),
        xtol: float = 1e-12,
        rtol: float = 1e-12,
        maxiter: int = 20
    ) -> float:
    """
    Find the Huber tuning constant c such that huber_efficiency(c) ~= target_eff.

    Parameters
    ----------
    target_eff : float
        Desired asymptotic efficiency in (0, 1). Common values: 0.95, 0.99.
    bracket : tuple(float, float)
        Search interval for c. Should be wide enough; (1e-6, 50) is safe.
    xtol, rtol, maxiter : passed to scipy.optimize.brentq.

    Returns
    -------
    t : float
        The tuning constant (same as statsmodels' HuberT(t)).
    """
    if not (0.0 < target_eff < 1.0):
        raise ValueError("target_eff must be in (0, 1)")

    a, b = bracket
    if a <= 0 or b <= a:
        raise ValueError("Invalid bracket; need 0 < a < b")

    from scipy.optimize import brentq

    f = lambda t: huber_efficiency(t) - target_eff

    # Ensure the bracket actually brackets a root.
    fa, fb = f(a), f(b)
    if fa > 0:
        # Already above target at tiny t: essentially impossible for valid target,
        # but keep a helpful error message.
        raise RuntimeError("Bracket lower end already above target efficiency; adjust bracket/target.")
    if fb < 0:
        raise RuntimeError("Bracket upper end still below target efficiency; increase bracket upper end.")

    return brentq(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter, full_output=False) # pyright: ignore[reportArgumentType, reportReturnType]
