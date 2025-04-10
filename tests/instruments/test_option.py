import pytest
from pyfinquant.instruments import Option, OptionType

def test_option_creation():
    """Test the creation of an Option object."""
    option = Option(
        underlying_price=100.0,
        strike_price=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        option_type=OptionType.CALL,
        dividend_yield=0.0
    )
    assert option.underlying_price == 100.0
    assert option.strike_price == 100.0
    assert option.risk_free_rate == 0.05
    assert option.volatility == 0.2
    assert option.time_to_maturity == 1.0
    assert option.option_type == OptionType.CALL
    assert option.dividend_yield == 0.0

def test_option_validation():
    """Test the validation of an Option object."""
    # Test with invalid underlying price
    with pytest.raises(ValueError):
        Option(
            underlying_price=-1.0,
            strike_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            dividend_yield=0.0
        )

    # Test with invalid strike price
    with pytest.raises(ValueError):
        Option(
            underlying_price=100.0,
            strike_price=-1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            dividend_yield=0.0
        )

    # Test with invalid volatility
    with pytest.raises(ValueError):
        Option(
            underlying_price=100.0,
            strike_price=100.0,
            risk_free_rate=0.05,
            volatility=-0.2,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            dividend_yield=0.0
        )

    # Test with invalid time to maturity
    with pytest.raises(ValueError):
        Option(
            underlying_price=100.0,
            strike_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_maturity=-1.0,
            option_type=OptionType.CALL,
            dividend_yield=0.0
        )

    # Test with invalid dividend yield
    with pytest.raises(ValueError):
        Option(
            underlying_price=100.0,
            strike_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            option_type=OptionType.CALL,
            dividend_yield=-0.1
        ) 