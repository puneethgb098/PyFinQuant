"""Tests for the Black-Scholes model."""

import pytest
import numpy as np
from pyfinquant.models.black_scholes import BlackScholes

TOL = 1e-6


@pytest.fixture
def sample_call_model() -> BlackScholes:
    """Provides a sample European call option model."""
    return BlackScholes(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
        option_type='call',
        q=0.01
    )


@pytest.fixture
def sample_put_model() -> BlackScholes:
    """Provides a sample European put option model."""
    return BlackScholes(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
        option_type='put',
        q=0.01
    )


def test_black_scholes_call_price(sample_call_model):
    """Test BSM call price against a known benchmark value."""
    expected_price = 9.826298
    calculated_price = sample_call_model.price()
    assert abs(calculated_price - expected_price) < TOL


def test_black_scholes_put_price(sample_put_model):
    """Test BSM put price against a known benchmark value."""
    expected_price = 5.944257
    calculated_price = sample_put_model.price()
    assert abs(calculated_price - expected_price) < TOL


def test_put_call_parity(sample_call_model, sample_put_model):
    """Verify Put-Call Parity: C - P = S*exp(-q*T) - K*exp(-r*T)"""
    call_price = sample_call_model.price()
    put_price = sample_put_model.price()
    S = sample_call_model.S
    K = sample_call_model.K
    r = sample_call_model.r
    q = sample_call_model.q
    T = sample_call_model.T

    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < TOL


def test_zero_time_to_maturity():
    """Test option price at expiration (T=0)."""
    call_t0_itm = BlackScholes(S=110, K=100, T=1e-12, r=0.05, sigma=0.2, option_type='call', q=0.01)
    call_t0_otm = BlackScholes(S=90, K=100, T=1e-12, r=0.05, sigma=0.2, option_type='call', q=0.01)
    put_t0_itm = BlackScholes(S=90, K=100, T=1e-12, r=0.05, sigma=0.2, option_type='put', q=0.01)
    put_t0_otm = BlackScholes(S=110, K=100, T=1e-12, r=0.05, sigma=0.2, option_type='put', q=0.01)

    call_t0_itm_exact = BlackScholes(S=110, K=100, T=0.0, r=0.05, sigma=0.2, option_type='call', q=0.01)
    put_t0_itm_exact = BlackScholes(S=90, K=100, T=0.0, r=0.05, sigma=0.2, option_type='put', q=0.01)

    assert abs(call_t0_itm.price() - (110 - 100)) < TOL
    assert abs(call_t0_otm.price() - 0.0) < TOL
    assert abs(put_t0_itm.price() - (100 - 90)) < TOL
    assert abs(put_t0_otm.price() - 0.0) < TOL

    assert abs(call_t0_itm_exact.price() - (110 - 100)) < TOL
    assert abs(put_t0_itm_exact.price() - (100 - 90)) < TOL


def test_zero_volatility():
    """Test option price with zero volatility."""
    call_zero_vol = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=1e-12, option_type='call', q=0.01)
    put_zero_vol = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=1e-12, option_type='put', q=0.01)

    S = 100
    K = 100
    r = 0.05
    q = 0.01
    T = 1.0
    forward_price = S * np.exp((r - q) * T)
    discount = np.exp(-r * T)

    expected_call_price = max(0.0, forward_price - K) * discount
    expected_put_price = max(0.0, K - forward_price) * discount

    assert abs(call_zero_vol.price() - expected_call_price) < TOL
    assert abs(put_zero_vol.price() - expected_put_price) < TOL


def test_invalid_option_type():
    """Test invalid option type handling."""
    with pytest.raises(ValueError):
        BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='invalid')


def test_negative_time_to_maturity():
    """Test negative time to maturity handling."""
    with pytest.raises(ValueError):
        BlackScholes(S=100, K=100, T=-1, r=0.05, sigma=0.2, option_type='call')
