"""
Tests for the Hull-White two-factor interest rate model.
"""

import numpy as np
import pytest
from pyfinquant.models.hull_white import HullWhite2F


def constant_rate(t: float) -> float:
    """Constant forward rate function."""
    return 0.05


def constant_bond_price(t: float) -> float:
    """Constant bond price function."""
    return np.exp(-0.05 * t)


@pytest.fixture
def hull_white_model():
    """Create a Hull-White model instance for testing."""
    return HullWhite2F(
        a1=0.1,
        a2=0.2,
        sigma1=0.01,
        sigma2=0.02,
        rho=0.5,
        initial_fwd_rate_func=constant_rate,
        initial_bond_price_func=constant_bond_price
    )


def test_hull_white_initialization():
    """Test Hull-White model initialization."""
    model = HullWhite2F(
        a1=0.1,
        a2=0.2,
        sigma1=0.01,
        sigma2=0.02,
        rho=0.5,
        initial_fwd_rate_func=constant_rate,
        initial_bond_price_func=constant_bond_price
    )
    assert model.a1 == 0.1
    assert model.a2 == 0.2
    assert model.sigma1 == 0.01
    assert model.sigma2 == 0.02
    assert model.rho == 0.5
    
    with pytest.raises(ValueError):
        HullWhite2F(
            a1=-0.1,
            a2=0.2,
            sigma1=0.01,
            sigma2=0.02,
            rho=0.5,
            initial_fwd_rate_func=constant_rate,
            initial_bond_price_func=constant_bond_price
        )
    
    with pytest.raises(ValueError):
        HullWhite2F(
            a1=0.1,
            a2=0.2,
            sigma1=0.01,
            sigma2=0.02,
            rho=1.5,
            initial_fwd_rate_func=constant_rate,
            initial_bond_price_func=constant_bond_price
        )
    
    with pytest.raises(ValueError):
        HullWhite2F(
            a1=0.1,
            a2=0.2,
            sigma1=0.01,
            sigma2=0.02,
            rho=0.5,
            initial_fwd_rate_func="not_callable",
            initial_bond_price_func=constant_bond_price
        )


def test_B_functions(hull_white_model):
    """Test B(t,T) function calculation."""
    B1 = hull_white_model._B1(0, 1)
    expected_B1 = (1 - np.exp(-0.1 * 1)) / 0.1
    assert np.isclose(B1, expected_B1)
    
    B2 = hull_white_model._B2(0, 1)
    expected_B2 = (1 - np.exp(-0.2 * 1)) / 0.2
    assert np.isclose(B2, expected_B2)


def test_A_function(hull_white_model):
    """Test A(t,T) function calculation."""
    A = hull_white_model._A(0, 1)
    B1 = hull_white_model._B1(0, 1)
    B2 = hull_white_model._B2(0, 1)
    
    V1 = (0.01**2 / (2 * 0.1**2)) * (
        1 + (2 / 0.1) * np.exp(-0.1) -
        (1 / (2 * 0.1)) * np.exp(-2 * 0.1) - 3 / (2 * 0.1)
    )
    
    V2 = (0.02**2 / (2 * 0.2**2)) * (
        1 + (2 / 0.2) * np.exp(-0.2) -
        (1 / (2 * 0.2)) * np.exp(-2 * 0.2) - 3 / (2 * 0.2)
    )
    
    V12 = (0.5 * 0.01 * 0.02 / (0.1 * 0.2)) * (
        1 + (1 / (0.1 + 0.2)) * (np.exp(-(0.1 + 0.2)) - 1) -
        (1 / 0.1) * (np.exp(-0.1) - 1) -
        (1 / 0.2) * (np.exp(-0.2) - 1)
    )
    
    expected = np.log(constant_bond_price(1) / constant_bond_price(0)) + \
               0.05 * (B1 + B2) - 0.5 * (V1 + V2 + 2 * V12)
    
    assert np.isclose(A, expected)


def test_zero_coupon_bond_price(hull_white_model):
    """Test zero-coupon bond price calculation."""
    price = hull_white_model.zero_coupon_bond_price(0, 1, 0.03, 0.02)
    A = hull_white_model._A(0, 1)
    B1 = hull_white_model._B1(0, 1)
    B2 = hull_white_model._B2(0, 1)
    expected = np.exp(A - B1 * 0.03 - B2 * 0.02)
    assert np.isclose(price, expected)
    
    # Test invalid maturity time
    with pytest.raises(ValueError):
        hull_white_model.zero_coupon_bond_price(1, 0.5, 0.03, 0.02)


def test_forward_rate(hull_white_model):
    """Test forward rate calculation."""
    rate = hull_white_model.forward_rate(0, 1, 2, 0.03, 0.02)
    P1 = hull_white_model.zero_coupon_bond_price(0, 1, 0.03, 0.02)
    P2 = hull_white_model.zero_coupon_bond_price(0, 2, 0.03, 0.02)
    expected = np.log(P1 / P2)
    assert np.isclose(rate, expected)
    
    with pytest.raises(ValueError):
        hull_white_model.forward_rate(1, 0.5, 2, 0.03, 0.02)
    with pytest.raises(ValueError):
        hull_white_model.forward_rate(0, 2, 1, 0.03, 0.02)


def test_short_rate_simulation(hull_white_model):
    """Test short rate simulation."""
    np.random.seed(42)
    rates1, rates2 = hull_white_model.short_rate_simulation(0, 1, 0.03, 0.02, n_steps=100)
    assert len(rates1) == 101  
    assert len(rates2) == 101
    assert rates1[0] == 0.03
    assert rates2[0] == 0.02
    
    with pytest.raises(ValueError):
        hull_white_model.short_rate_simulation(1, 0.5, 0.03, 0.02)
    
    with pytest.raises(ValueError):
        hull_white_model.short_rate_simulation(0, 1, 0.03, 0.02, n_steps=0)


def test_get_short_rate(hull_white_model):
    short_rate = hull_white_model.get_short_rate(0.03, 0.02)
    assert short_rate == 0.05 
