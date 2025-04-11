"""PyFinQuant - A Python library for financial quantitative analysis."""

# Core functionality
from .core import (
    correlation,
    covariance,
    cumulative_returns,
    information_ratio,
    kurtosis,
    log_returns,
    maximize_sharpe,
    maximize_sortino,
    mean,
    minimize_risk,
    minimize_tracking_error,
    returns,
    rolling_correlation,
    rolling_covariance,
    rolling_mean,
    rolling_std,
    sharpe_ratio,
    skew,
    sortino_ratio,
    std,
)

from .greeks.analytical import AnalyticalGreeks
from .models.black_scholes import BlackScholes
from .models.hull_white import HullWhite

from .risk import (
    conditional_var,
    drawdown,
    drawdown_duration,
    expected_shortfall,
    historical_var,
    max_drawdown,
    monte_carlo_var,
    parametric_var,
    recovery_time,
)

from .utils import Numeric

__all__ = [
    "mean",
    "std",
    "skew",
    "kurtosis",
    "correlation",
    "covariance",
    "rolling_mean",
    "rolling_std",
    "rolling_correlation",
    "rolling_covariance",
    "returns",
    "log_returns",
    "cumulative_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "information_ratio",
    "minimize_risk",
    "maximize_sharpe",
    "maximize_sortino",
    "minimize_tracking_error",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "conditional_var",
    "expected_shortfall",
    "drawdown",
    "max_drawdown",
    "drawdown_duration",
    "recovery_time",
    "AnalyticalGreeks", 
    "BlackScholes",
    "HullWhite",
    "Numeric",
]

__version__ = "0.1.0"
