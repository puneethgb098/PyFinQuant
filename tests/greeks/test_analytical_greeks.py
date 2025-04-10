"""Tests for analytical Greeks calculations."""

import pytest
import numpy as np
from pyfinquant.models import BlackScholes
from pyfinquant.greeks import AnalyticalGreeks

TOL = 1e-6

@pytest.fixture
def sample_call_model() -> BlackScholes:
    """Provides a sample Black-Scholes model for a European call option."""
    return BlackScholes(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
        option_type='call',
        q=0.01  # Include dividend yield
    )

@pytest.fixture
def sample_put_model() -> BlackScholes:
    """Provides a sample Black-Scholes model for a European put option."""
    return BlackScholes(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
        option_type='put',
        q=0.01  # Include dividend yield
    )

@pytest.fixture
def greeks_calculator_call(sample_call_model) -> AnalyticalGreeks:
    """Provides an AnalyticalGreeks instance for the call model."""
    return AnalyticalGreeks(sample_call_model)

@pytest.fixture
def greeks_calculator_put(sample_put_model) -> AnalyticalGreeks:
    """Provides an AnalyticalGreeks instance for the put model."""
    return AnalyticalGreeks(sample_put_model)


def test_delta_call(greeks_calculator_call):
    """Test the delta calculation for a call option."""
    delta = greeks_calculator_call.delta()
    expected_delta = 0.58315 # Value for S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.01
    assert isinstance(delta, float)
    assert abs(delta - expected_delta) < TOL

def test_delta_put(greeks_calculator_put):
    """Test the delta calculation for a put option."""
    delta = greeks_calculator_put.delta()
    expected_delta = -0.41685 # Value for S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.01
    assert isinstance(delta, float)
    assert abs(delta - expected_delta) < TOL

def test_gamma(greeks_calculator_call, greeks_calculator_put):
    """Test the gamma calculation."""
    gamma_call = greeks_calculator_call.gamma()
    gamma_put = greeks_calculator_put.gamma()
    expected_gamma = 0.019249 # Value for S=100, K=100, T=1, r=0.05, sigma=0.2, q=0.01
    assert isinstance(gamma_call, float)
    assert gamma_call >= 0
    assert abs(gamma_call - expected_gamma) < TOL
    # Gamma should be the same for calls and puts
    assert abs(gamma_call - gamma_put) < TOL

def test_theta(greeks_calculator_call, greeks_calculator_put):
    """Test the theta calculation."""
    theta_call = greeks_calculator_call.theta() # Daily theta
    theta_put = greeks_calculator_put.theta() # Daily theta
    expected_theta_call = -6.05449 / 365.0 # Value / 365
    expected_theta_put = -1.27893 / 365.0 # Value / 365
    assert isinstance(theta_call, float)
    assert abs(theta_call - expected_theta_call) < TOL
    assert abs(theta_put - expected_theta_put) < TOL

def test_vega(greeks_calculator_call, greeks_calculator_put):
    """Test the vega calculation."""
    vega_call = greeks_calculator_call.vega()
    vega_put = greeks_calculator_put.vega()
    expected_vega = 38.4975 / 100 # Value / 100
    assert isinstance(vega_call, float)
    assert vega_call >= 0
    assert abs(vega_call - expected_vega) < TOL
    # Vega should be the same for calls and puts
    assert abs(vega_call - vega_put) < TOL

def test_rho(greeks_calculator_call, greeks_calculator_put):
    """Test the rho calculation."""
    rho_call = greeks_calculator_call.rho()
    rho_put = greeks_calculator_put.rho()
    expected_rho_call = 47.864 / 100 # Value / 100
    expected_rho_put = -47.136 / 100 # Value / 100
    assert isinstance(rho_call, float)
    assert abs(rho_call - expected_rho_call) < TOL
    assert abs(rho_put - expected_rho_put) < TOL
