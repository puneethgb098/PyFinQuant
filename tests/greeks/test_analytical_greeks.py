"""
Tests for analytical Greeks calculations.
"""

import pytest
import numpy as np
from pyfinquant.greeks.analytical import (
    delta,
    gamma,
    vega,
    theta,
    rho,
    rho_dividend
)
from pyfinquant.instruments.option import Option


@pytest.fixture
def sample_call_option() -> Option:
    """Create a sample call option for testing."""
    return Option(
<<<<<<< HEAD
        underlying_price=100.0,
=======
        spot_price=100.0,
>>>>>>> 77d228180a2f5aa0c116c49388d15823995d88c1
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.02,
        option_type='call'
    )


@pytest.fixture
def sample_put_option() -> Option:
    """Create a sample put option for testing."""
    return Option(
<<<<<<< HEAD
        underlying_price=100.0,
=======
        spot_price=100.0,
>>>>>>> 77d228180a2f5aa0c116c49388d15823995d88c1
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.02,
        option_type='put'
    )


def test_delta(sample_call_option, sample_put_option):
    """Test delta calculation."""
    call_delta = delta(sample_call_option)
    put_delta = delta(sample_put_option)
    
<<<<<<< HEAD
    assert -1 <= call_delta <= 1
    assert -1 <= put_delta <= 1
    
=======
    # Delta should be between -1 and 1
    assert -1 <= call_delta <= 1
    assert -1 <= put_delta <= 1
    
    # Call delta should be positive, put delta should be negative
>>>>>>> 77d228180a2f5aa0c116c49388d15823995d88c1
    assert call_delta > 0
    assert put_delta < 0


def test_gamma(sample_call_option, sample_put_option):
    """Test gamma calculation."""
    call_gamma = gamma(sample_call_option)
    put_gamma = gamma(sample_put_option)
    
    assert call_gamma > 0
    assert put_gamma > 0
    
    assert np.isclose(call_gamma, put_gamma)


def test_vega(sample_call_option, sample_put_option):
    """Test vega calculation."""
    call_vega = vega(sample_call_option)
    put_vega = vega(sample_put_option)
    
    assert call_vega > 0
    assert put_vega > 0
    
    assert np.isclose(call_vega, put_vega)


def test_theta(sample_call_option, sample_put_option):
    """Test theta calculation."""
    call_theta = theta(sample_call_option)
    put_theta = theta(sample_put_option)
    
    assert call_theta < 0
    assert put_theta < 0


def test_rho(sample_call_option, sample_put_option):
    """Test rho calculation."""
    call_rho = rho(sample_call_option)
    put_rho = rho(sample_put_option)
    
    assert call_rho > 0
    assert put_rho < 0


def test_rho_dividend(sample_call_option, sample_put_option):
    """Test rho dividend calculation."""
    call_rho_div = rho_dividend(sample_call_option)
    put_rho_div = rho_dividend(sample_put_option)
    
    assert call_rho_div < 0
    assert put_rho_div > 0


def test_greeks_at_maturity(sample_call_option, sample_put_option):
    """Test Greeks at maturity."""
    sample_call_option.time_to_maturity = 0.0
    sample_put_option.time_to_maturity = 0.0
    
    call_delta = delta(sample_call_option)
    put_delta = delta(sample_put_option)
    
    if sample_call_option.underlying_price > sample_call_option.strike_price:
        assert np.isclose(call_delta, 1.0)
        assert np.isclose(put_delta, 0.0)
    elif sample_call_option.underlying_price < sample_call_option.strike_price:
        assert np.isclose(call_delta, 0.0)
        assert np.isclose(put_delta, -1.0)
    else:
        assert np.isclose(call_delta, 0.5)
        assert np.isclose(put_delta, -0.5)
    
    assert np.isclose(gamma(sample_call_option), 0.0)
    assert np.isclose(gamma(sample_put_option), 0.0)
    
    assert np.isclose(vega(sample_call_option), 0.0)
    assert np.isclose(vega(sample_put_option), 0.0)
