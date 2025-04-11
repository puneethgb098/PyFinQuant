"""Tests for the Hull-White interest rate model."""

import pytest
import numpy as np
from pyfinquant.models import HullWhite

TOL = 1e-6

# --- Fixtures for a Flat Initial Term Structure ---
FLAT_RATE = 0.03

@pytest.fixture
def flat_fwd_rate_func():
    """Initial instantaneous forward rate function for a flat curve."""
    return lambda t: FLAT_RATE

@pytest.fixture
def flat_bond_price_func():
    """Initial zero-coupon bond price function for a flat curve."""
    return lambda t: np.exp(-FLAT_RATE * t)

@pytest.fixture
def hw_model_flat(flat_fwd_rate_func, flat_bond_price_func) -> HullWhite:
    """Provides a HullWhite model instance calibrated to a flat curve."""
    return HullWhite(
        a=0.1, 
        sigma=0.01, 
        initial_fwd_rate_func=flat_fwd_rate_func, 
        initial_bond_price_func=flat_bond_price_func
    )
# -------------------------------------------------

def test_hull_white_init():
    """Test HullWhite model initialization."""
    with pytest.raises(ValueError): # Test negative mean reversion
        HullWhite(a=-0.1, sigma=0.01, initial_fwd_rate_func=lambda t: 0.03, initial_bond_price_func=lambda t: np.exp(-0.03*t))
    with pytest.raises(ValueError): # Test negative volatility
        HullWhite(a=0.1, sigma=-0.01, initial_fwd_rate_func=lambda t: 0.03, initial_bond_price_func=lambda t: np.exp(-0.03*t))

def test_B_calculation(hw_model_flat):
    """Test the B(t, T) calculation."""
    t, T = 0.5, 1.0
    expected_B = (1.0 / hw_model_flat.a) * (1.0 - np.exp(-hw_model_flat.a * (T - t)))
    calculated_B = hw_model_flat.B(t, T)
    assert abs(calculated_B - expected_B) < TOL
    
    # Test T=t case
    assert abs(hw_model_flat.B(T, T) - 0.0) < TOL
    
    with pytest.raises(ValueError): # Test t > T
        hw_model_flat.B(T + 0.1, T)

def test_A_calculation_flat_curve(hw_model_flat):
    """Test the A(t, T) calculation specifically for a flat initial curve.
       Verify the consistency relationship P(0,T) = A(0,T) * exp(-B(0,T) * r(0)).
    """
    t, T = 0.0, 1.0 # Test at t=0
    a = hw_model_flat.a
    sigma = hw_model_flat.sigma
    FLAT_RATE = hw_model_flat.f0(0) # Get r(0)
    
    # Calculate A(0, T) and B(0, T) using the model functions
    A_0_T_model = hw_model_flat.A(t, T)
    B_0_T_model = hw_model_flat.B(t, T)
    
    # Verify the fundamental relationship: P(0, T) = A(0, T) * exp(-B(0, T) * r(0))
    price_recalc = A_0_T_model * np.exp(-B_0_T_model * FLAT_RATE)
    expected_price = hw_model_flat.P0(T)
    
    # Use original tight tolerance
    assert abs(price_recalc - expected_price) < TOL
    
    # Optional: Also test A(t,T) for t > 0 against the calculated P(t,T) if needed
    # This requires knowing r(t) which is stochastic, so maybe less direct.
    # A direct check of the A(t,T) formula against a known result might be better
    # if testing for t>0.

def test_zero_coupon_bond_price(hw_model_flat):
    """Test the zero-coupon bond price calculation P(t, T)."""
    t, T, r_t = 0.5, 1.0, 0.04 # Assume some short rate r(t)
    
    # Calculate A and B
    A_t_T = hw_model_flat.A(t, T)
    B_t_T = hw_model_flat.B(t, T)
    
    expected_price = A_t_T * np.exp(-B_t_T * r_t)
    calculated_price = hw_model_flat.zero_coupon_bond_price(t, T, r_t)
    
    assert abs(calculated_price - expected_price) < TOL
    assert 0 < calculated_price <= 1.0 # Price should be positive and <= 1
    
    # Test T=t case
    assert abs(hw_model_flat.zero_coupon_bond_price(T, T, r_t) - 1.0) < TOL
    
    with pytest.raises(ValueError): # Test t > T
        hw_model_flat.zero_coupon_bond_price(T + 0.1, T, r_t) 