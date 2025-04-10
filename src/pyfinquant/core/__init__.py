"""
Core mathematical tools for quantitative finance.
"""

from .optimization import (
    maximize_sharpe,
    maximize_sortino,
    minimize_risk,
    minimize_tracking_error,
)
from .statistics import (
    correlation,
    covariance,
    kurtosis,
    mean,
    rolling_correlation,
    rolling_covariance,
    rolling_mean,
    rolling_std,
    skew,
    std,
)
from .time_series import (
    cumulative_returns,
    information_ratio,
    log_returns,
    returns,
    sharpe_ratio,
    sortino_ratio,
)

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
]
