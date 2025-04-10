import numpy as np
import pytest
from pyfinquant.instruments.option import Option, OptionType
from pyfinquant.models.black_scholes import BlackScholes
from pyfinquant.greeks.analytical import AnalyticalGreeks

def test_option_creation():
    option = Option(
        S=100,  # Spot price
        K=100,  # Strike price
        T=1.0,  # Time to maturity
        r=0.05,  # Risk-free rate
        sigma=0.2,  # Volatility
        option_type=OptionType.CALL
    )
    assert option.S == 100
    assert option.K == 100
    assert option.T == 1.0
    assert option.r == 0.05
    assert option.sigma == 0.2
    assert option.option_type == OptionType.CALL

def test_black_scholes_price():
    model = BlackScholes()
    price = model.price(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    # Expected price for ATM call with these parameters is around 10.45
    assert abs(price - 10.45) < 0.1

def test_analytical_greeks():
    greeks = AnalyticalGreeks()
    
    # Test delta
    delta = greeks.delta(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    assert abs(delta - 0.64) < 0.01  # Expected delta for ATM call
    
    # Test gamma
    gamma = greeks.gamma(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    assert abs(gamma - 0.018) < 0.001  # Expected gamma for ATM call
    
    # Test vega
    vega = greeks.vega(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    assert abs(vega - 39.58) < 0.1  # Expected vega for ATM call
    
    # Test theta
    theta = greeks.theta(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    assert abs(theta - (-6.35)) < 0.1  # Expected theta for ATM call
    
    # Test rho
    rho = greeks.rho(
        S=100,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type=OptionType.CALL
    )
    assert abs(rho - 51.77) < 0.1  # Expected rho for ATM call 