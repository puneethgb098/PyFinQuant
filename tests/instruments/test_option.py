"""
Tests for option instruments.
"""

import pytest
from pyfinquant.instruments.option import Option


def test_option_initialization():
    """Test option initialization."""
    option = Option(
        underlying_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.02,
        option_type='call'
    )
    
    assert option.underlying_price == 100.0
    assert option.strike_price == 100.0
    assert option.time_to_maturity == 1.0
    assert option.risk_free_rate == 0.05
    assert option.volatility == 0.2
    assert option.dividend_yield == 0.02
    assert option.option_type == OptionType.CALL


def test_option_invalid_type():
    """Test invalid option type."""
    with pytest.raises(ValueError):
        Option(
            underlying_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.02,
            option_type='invalid'
        )


def test_option_invalid_parameters():
    """Test invalid option parameters."""
    with pytest.raises(ValueError):
        Option(
            underlying_price=-100.0,  # Invalid spot price
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.02,
            option_type='call'
        )
    
    with pytest.raises(ValueError):
        Option(
            underlying_price=100.0,
            strike_price=-100.0,  # Invalid strike price
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.02,
            option_type='call'
        )
    
    with pytest.raises(ValueError):
        Option(
            underlying_price=100.0,
            strike_price=100.0,
            time_to_maturity=-1.0,  # Invalid time to maturity
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.02,
            option_type='call'
        )


def test_option_call_put():
    """Test call and put option properties."""
    call_option = Option(
        underlying_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.02,
        option_type='call'
    )
    
    put_option = Option(
        underlying_price=100.0,
        strike_price=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        dividend_yield=0.02,
        option_type='put'
    )
    
    assert call_option.is_call()
    assert not call_option.is_put()
    assert put_option.is_put()
    assert not put_option.is_call() 