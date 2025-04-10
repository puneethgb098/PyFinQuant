import pytest
import numpy as np
from pyfinquant.greeks import AnalyticalGreeks
from pyfinquant.instruments import Option, OptionType

TOL = 1e-6

@pytest.fixture
def sample_call_option() -> Option:
    """Provides a sample European call option."""
    return Option(
        underlying_price=100.0,
        strike_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        option_type=OptionType.CALL,
        dividend_yield=0.0
    )

@pytest.fixture
def sample_put_option() -> Option:
    """Provides a sample European put option."""
    return Option(
        underlying_price=100.0,
        strike_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        option_type=OptionType.PUT,
        dividend_yield=0.0
    )

def test_delta_call(sample_call_option):
    """Test the delta calculation for a call option."""
    delta = AnalyticalGreeks.delta(sample_call_option)
    assert isinstance(delta, float)
    assert 0 <= delta <= 1
    # Known value for ATM call with these parameters
    expected_delta = 0.5  # Approximately 0.5 for ATM option
    assert abs(delta - expected_delta) < 0.1  # Allow some deviation

def test_delta_put(sample_put_option):
    """Test the delta calculation for a put option."""
    delta = AnalyticalGreeks.delta(sample_put_option)
    assert isinstance(delta, float)
    assert -1 <= delta <= 0
    # Known value for ATM put with these parameters
    expected_delta = -0.5  # Approximately -0.5 for ATM option
    assert abs(delta - expected_delta) < 0.1  # Allow some deviation

def test_gamma(sample_call_option):
    """Test the gamma calculation."""
    gamma = AnalyticalGreeks.gamma(sample_call_option)
    assert isinstance(gamma, float)
    assert gamma >= 0
    # Gamma should be the same for calls and puts
    gamma_put = AnalyticalGreeks.gamma(sample_put_option)
    assert abs(gamma - gamma_put) < TOL

def test_theta(sample_call_option):
    """Test the theta calculation."""
    theta = AnalyticalGreeks.theta(sample_call_option)
    assert isinstance(theta, float)
    # Theta should be negative for ATM options
    assert theta < 0

def test_vega(sample_call_option):
    """Test the vega calculation."""
    vega = AnalyticalGreeks.vega(sample_call_option)
    assert isinstance(vega, float)
    assert vega >= 0
    # Vega should be the same for calls and puts
    vega_put = AnalyticalGreeks.vega(sample_put_option)
    assert abs(vega - vega_put) < TOL

def test_rho(sample_call_option):
    """Test the rho calculation."""
    rho = AnalyticalGreeks.rho(sample_call_option)
    assert isinstance(rho, float)
    # Rho should be positive for calls and negative for puts
    assert rho > 0
    rho_put = AnalyticalGreeks.rho(sample_put_option)
    assert rho_put < 0
