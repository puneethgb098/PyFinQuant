import numpy as np
import pandas as pd
import pytest
from pyfinquant.core import (
    returns, log_returns, cumulative_returns,
    sharpe_ratio, sortino_ratio, information_ratio
)

def test_returns():
    # Test with numpy array
    x = np.array([100, 110, 121, 133.1])
    expected = np.array([np.nan, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(returns(x), expected, rtol=1e-5)
    
    # Test with pandas Series
    s = pd.Series(x)
    pd.testing.assert_series_equal(returns(s), pd.Series(expected))

def test_log_returns():
    x = np.array([100, 110, 121, 133.1])
    expected = np.array([np.nan, np.log(1.1), np.log(1.1), np.log(1.1)])
    np.testing.assert_allclose(log_returns(x), expected, rtol=1e-5)

def test_cumulative_returns():
    x = np.array([100, 110, 121, 133.1])
    expected = np.array([0, 0.1, 0.21, 0.331])
    np.testing.assert_allclose(cumulative_returns(x), expected, rtol=1e-5)

def test_sharpe_ratio():
    x = np.array([0.01, 0.02, -0.01, 0.03])
    expected = 0.5  # Approximate value
    assert abs(sharpe_ratio(x) - expected) < 0.1

def test_sortino_ratio():
    x = np.array([0.01, 0.02, -0.01, 0.03])
    expected = 0.7  # Approximate value
    assert abs(sortino_ratio(x) - expected) < 0.1

def test_information_ratio():
    x = np.array([0.01, 0.02, -0.01, 0.03])
    benchmark = np.array([0.005, 0.015, -0.005, 0.025])
    expected = 0.4  # Approximate value
    assert abs(information_ratio(x, benchmark) - expected) < 0.1 