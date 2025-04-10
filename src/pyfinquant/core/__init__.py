"""
Core mathematical tools for quantitative finance.
"""

from .statistics import (
    mean,
    std,
    skew,
    kurtosis,
    correlation,
    covariance,
    rolling_mean,
    rolling_std,
    rolling_correlation,
    rolling_covariance,
)

from .time_series import (
    returns,
    log_returns,
    cumulative_returns,
    sharpe_ratio,
    sortino_ratio,
    information_ratio,
)

from .optimization import (
    minimize_risk,
    maximize_sharpe,
    maximize_sortino,
    minimize_tracking_error,
)

__all__ = [
    'mean',
    'std',
    'skew',
    'kurtosis',
    'correlation',
    'covariance',
    'rolling_mean',
    'rolling_std',
    'rolling_correlation',
    'rolling_covariance',
    'returns',
    'log_returns',
    'cumulative_returns',
    'sharpe_ratio',
    'sortino_ratio',
    'information_ratio',
    'minimize_risk',
    'maximize_sharpe',
    'maximize_sortino',
    'minimize_tracking_error',
] 