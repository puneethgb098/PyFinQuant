import pytest
import numpy as np
from pyfinquant import Option, OptionType, BlackScholes

TOL = 1e-3


@pytest.fixture
def sample_call_option() -> Option:
    """Provides a sample European call option."""
    return Option(
        underlying_price=100.0,
        strike_price=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1.0,
        option_type=OptionType.CALL,
        dividend_yield=0.01,
    )


@pytest.fixture
def sample_put_option() -> Option:
    """Provides a sample European put option."""
    return Option(
        underlying_price=100.0,
        strike_price=100.0,
        risk_free_rate=0.05,
        volatility=0.20,
        time_to_maturity=1.0,
        option_type=OptionType.PUT,
        dividend_yield=0.01,
    )


def test_black_scholes_call_price(sample_call_option):
    """Test BSM call price against a known benchmark value."""
    expected_price = 9.826
    calculated_price = BlackScholes.price(sample_call_option)
    assert abs(calculated_price - expected_price) < TOL


def test_black_scholes_put_price(sample_put_option):
    """Test BSM put price against a known benchmark value."""
    expected_price = 5.944
    calculated_price = BlackScholes.price(sample_put_option)
    assert abs(calculated_price - expected_price) < TOL


def test_put_call_parity(sample_call_option, sample_put_option):
    """Verify Put-Call Parity: C - P = S*exp(-q*T) - K*exp(-r*T)"""
    call_price = BlackScholes.price(sample_call_option)
    put_price = BlackScholes.price(sample_put_option)
    S = sample_call_option.underlying_price
    K = sample_call_option.strike_price
    r = sample_call_option.risk_free_rate
    q = sample_call_option.dividend_yield
    T = sample_call_option.time_to_maturity

    lhs = call_price - put_price
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(lhs - rhs) < TOL


def test_zero_time_to_maturity(sample_call_option, sample_put_option):
    """Test option price at expiration (T=0)."""
    call_t0_itm = Option(underlying_price=110, strike_price=100, time_to_maturity=1e-12, risk_free_rate=0.05, volatility=0.2, option_type=OptionType.CALL, dividend_yield=0.01)
    call_t0_otm = Option(underlying_price=90, strike_price=100, time_to_maturity=1e-12, risk_free_rate=0.05, volatility=0.2, option_type=OptionType.CALL, dividend_yield=0.01)
    put_t0_itm = Option(underlying_price=90, strike_price=100, time_to_maturity=1e-12, risk_free_rate=0.05, volatility=0.2, option_type=OptionType.PUT, dividend_yield=0.01)
    put_t0_otm = Option(underlying_price=110, strike_price=100, time_to_maturity=1e-12, risk_free_rate=0.05, volatility=0.2, option_type=OptionType.PUT, dividend_yield=0.01)

    call_t0_itm_exact = Option(underlying_price=110, strike_price=100, time_to_maturity=0.0, risk_free_rate=0.05, volatility=0.2, option_type=OptionType.CALL, dividend_yield=0.01)
    put_t0_itm_exact = Option(underlying_price=90, strike_price=100, time_to_maturity=0.0, risk_free_rate=0.05, volatility=0.2, option_type=OptionType.PUT, dividend_yield=0.01)

    assert abs(BlackScholes.price(call_t0_itm) - (110 - 100)) < TOL
    assert abs(BlackScholes.price(call_t0_otm) - 0.0) < TOL
    assert abs(BlackScholes.price(put_t0_itm) - (100 - 90)) < TOL
    assert abs(BlackScholes.price(put_t0_otm) - 0.0) < TOL

    assert abs(BlackScholes.price(call_t0_itm_exact) - (110 - 100)) < TOL
    assert abs(BlackScholes.price(put_t0_itm_exact) - (100 - 90)) < TOL


def test_zero_volatility(sample_call_option, sample_put_option):
    """Test option price with zero volatility."""
    call_zero_vol = Option(underlying_price=100, strike_price=100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0, option_type=OptionType.CALL, dividend_yield=0.01)
    put_zero_vol = Option(underlying_price=100, strike_price=100, time_to_maturity=1.0, risk_free_rate=0.05, volatility=0, option_type=OptionType.PUT, dividend_yield=0.01)

    S = 100
    K = 100
    r = 0.05
    q = 0.01
    T = 1.0
    forward_price = S * np.exp((r - q) * T)
    discount = np.exp(-r * T)

    expected_call_price = max(0.0, forward_price - K) * discount
    expected_put_price = max(0.0, K - forward_price) * discount

    assert abs(BlackScholes.price(call_zero_vol) - expected_call_price) < TOL
    assert abs(BlackScholes.price(put_zero_vol) - expected_put_price) < TOL
