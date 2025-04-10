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

# Option pricing and interest rate models
from .greeks.analytical import AnalyticalGreeks
# from .instruments.option import Option, OptionType # Removed as Option class is deprecated
from .models.black_scholes import BlackScholes
from .models.hull_white import HullWhite

# Risk management
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

# Utilities
from .utils import Numeric

# Define __all__ to specify the public API explicitly
__all__ = [
    # Core functions
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
    # Risk management
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "conditional_var",
    "expected_shortfall",
    "drawdown",
    "max_drawdown",
    "drawdown_duration",
    "recovery_time",
    # Models
    "AnalyticalGreeks", # Technically Greeks, but closely tied to models
    "BlackScholes",
    "HullWhite",
    # "Option", # Removed
    # "OptionType", # Removed
    # Utilities
    "Numeric",
]

# Package version
__version__ = "0.1.0"
