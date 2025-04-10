import numpy as np
import pandas as pd
import pytest
from pyfinquant.risk import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    conditional_var,
    expected_shortfall,
    drawdown,
    max_drawdown,
    drawdown_duration,
    recovery_time
)

def test_historical_var():
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    expected_var = np.percentile(returns, 5) # Use np.percentile for exact expected value
    # Adjust tolerance or expected value based on calculation
    # np.percentile gives -0.018 for this data
    assert abs(historical_var(returns) - expected_var) < 1e-6

def test_parametric_var():
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    var = parametric_var(returns, confidence_level=0.95)
    assert var < 0  # VaR should be negative for losses
    assert abs(var - (-0.02)) < 0.01  # Approximate value

def test_monte_carlo_var():
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    var = monte_carlo_var(returns, confidence_level=0.95, n_simulations=1000)
    assert var < 0  # VaR should be negative for losses
    assert abs(var - (-0.02)) < 0.01  # Approximate value

def test_conditional_var():
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    cvar = conditional_var(returns, confidence_level=0.95)
    assert cvar < 0  # CVaR should be negative for losses
    assert cvar <= historical_var(returns, confidence_level=0.95)  # CVaR should be more extreme than VaR

def test_expected_shortfall():
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    es = expected_shortfall(returns, confidence_level=0.95)
    assert es < 0  # ES should be negative for losses
    assert es <= historical_var(returns, confidence_level=0.95)  # ES should be more extreme than VaR

def test_drawdown():
    prices = np.array([100, 110, 105, 115, 120, 115])
    dd = drawdown(prices)
    expected = np.array([0, 0, -0.04545, 0, 0, -0.04167])
    np.testing.assert_allclose(dd, expected, rtol=1e-4)

def test_max_drawdown():
    prices = np.array([100, 110, 105, 115, 120, 115])
    mdd = max_drawdown(prices)
    assert abs(mdd - (-0.04545)) < 0.001

def test_drawdown_duration():
    prices = np.array([100, 110, 105, 115, 120, 115])
    start, end = drawdown_duration(prices)
    assert start == 1
    assert end == 2

def test_recovery_time():
    prices = np.array([100, 110, 105, 115, 120, 115])
    recovery = recovery_time(prices)
    assert recovery == 2  # Takes 2 periods to recover from the drawdown 