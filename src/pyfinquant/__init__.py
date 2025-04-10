"""PyFinQuant - A Python library for financial quantitative analysis."""

# Core functionality
from .core import (
    mean, std, skew, kurtosis,
    correlation, covariance,
    rolling_mean, rolling_std,
    rolling_correlation, rolling_covariance,
    returns, log_returns, cumulative_returns,
    sharpe_ratio, sortino_ratio, information_ratio,
    minimize_risk, maximize_sharpe,
    maximize_sortino, minimize_tracking_error
)

# Risk management
from .risk import (
    historical_var, parametric_var, monte_carlo_var,
    conditional_var, expected_shortfall,
    drawdown, max_drawdown, drawdown_duration, recovery_time
)

# Option pricing and Greeks
from .greeks.analytical import AnalyticalGreeks
from .instruments.option import Option, OptionType
from .models.black_scholes import BlackScholes

# Utilities
from .utils import Numeric

# Define __all__ to specify the public API explicitly
__all__ = [
    # Core functions
    "mean", "std", "skew", "kurtosis",
    "correlation", "covariance",
    "rolling_mean", "rolling_std",
    "rolling_correlation", "rolling_covariance",
    "returns", "log_returns", "cumulative_returns",
    "sharpe_ratio", "sortino_ratio", "information_ratio",
    "minimize_risk", "maximize_sharpe",
    "maximize_sortino", "minimize_tracking_error",
    
    # Risk management
    "historical_var", "parametric_var", "monte_carlo_var",
    "conditional_var", "expected_shortfall",
    "drawdown", "max_drawdown", "drawdown_duration", "recovery_time",
    
    # Option pricing
    "AnalyticalGreeks",
    "BlackScholes",
    "Option",
    "OptionType",
    
    # Utilities
    "Numeric"
]

# Package version
__version__ = "0.1.0"
