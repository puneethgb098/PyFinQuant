"""
Tests for utility functions.
"""

import pytest
import numpy as np
import pandas as pd
from pyfinquant.utils.helpers import (
    check_positive,
    check_non_negative,
    calculate_returns
)


def test_check_positive():
    """Test check_positive function."""
    check_positive(1.0)
    check_positive(0.1)
    check_positive(100.0)
    
    with pytest.raises(ValueError):
        check_positive(0.0)
    
    with pytest.raises(ValueError):
        check_positive(-1.0)
    
    with pytest.raises(ValueError):
        check_positive(-0.1)


def test_check_non_negative():
    """Test check_non_negative function."""
    check_non_negative(0.0)
    check_non_negative(1.0)
    check_non_negative(0.1)
    check_non_negative(100.0)
    
    with pytest.raises(ValueError):
        check_non_negative(-1.0)
    
    with pytest.raises(ValueError):
        check_non_negative(-0.1)


def test_calculate_returns():
    """Test calculate_returns function."""
    dates = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=dates)
    
    arithmetic_returns = calculate_returns(prices, method='arithmetic')
    expected_arithmetic = pd.Series([
        np.nan, 0.01, 0.009901, 0.009804, 0.009709,
        0.009615, 0.009524, 0.009434, 0.009346, 0.009259
    ], index=dates)
    pd.testing.assert_series_equal(arithmetic_returns, expected_arithmetic,atol=1e-3)
    
    log_returns = calculate_returns(prices, method='log')
    expected_log = pd.Series([
        np.nan, 0.00995, 0.009852, 0.009756, 0.009662,
        0.009569, 0.009478, 0.009389, 0.009301, 0.009215
    ], index=dates)
    pd.testing.assert_series_equal(log_returns, expected_log, atol=1e-3)
    
    with pytest.raises(ValueError):
        calculate_returns(prices, method='invalid')
    
    df_prices = pd.DataFrame({
        'A': prices,
        'B': prices * 1.1
    })
    df_returns = calculate_returns(df_prices)
    assert isinstance(df_returns, pd.DataFrame)
    assert df_returns.shape == df_prices.shape
    assert all(col in df_returns.columns for col in df_prices.columns) 