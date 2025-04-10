"""
Risk management module for PyFinQuant.
"""

from .value_at_risk import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    conditional_var,
    expected_shortfall
)

from .drawdown import (
    drawdown,
    max_drawdown,
    drawdown_duration,
    recovery_time
)

from .risk_metrics import (
    calculate_risk_metrics,
    calculate_portfolio_risk
)

__all__ = [
    # Value at Risk
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "conditional_var",
    "expected_shortfall",
    
    # Drawdown
    "drawdown",
    "max_drawdown",
    "drawdown_duration",
    "recovery_time",
    
    # Risk Metrics
    "calculate_risk_metrics",
    "calculate_portfolio_risk"
]
