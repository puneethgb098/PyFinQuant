"""Hull-White interest rate model implementation."""

import numpy as np
from typing import Callable

class HullWhite:
    """Hull-White (Extended Vasicek) single-factor interest rate model.

    The model describes the short rate r(t) dynamics as:
    dr(t) = (theta(t) - a * r(t)) dt + sigma * dW(t)
    
    where 'a' is the mean-reversion speed and 'sigma' is the volatility.
    theta(t) is chosen to fit the model to the initial term structure.
    """

    def __init__(self, a: float, sigma: float, initial_fwd_rate_func: Callable[[float], float], initial_bond_price_func: Callable[[float], float]):
        """
        Initialize the Hull-White model.

        Args:
            a (float): Mean-reversion speed.
            sigma (float): Volatility parameter.
            initial_fwd_rate_func (Callable[[float], float]): Function f(0, t) giving the initial instantaneous forward rate for maturity t.
            initial_bond_price_func (Callable[[float], float]): Function P(0, t) giving the initial zero-coupon bond price for maturity t.
        """
        if a <= 0:
            raise ValueError("Mean reversion 'a' must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility 'sigma' must be positive.")
            
        self.a = a
        self.sigma = sigma
        self.f0 = initial_fwd_rate_func
        self.P0 = initial_bond_price_func
        self._sigma2 = sigma**2

    def B(self, t: float, T: float) -> float:
        """Calculates the B(t, T) term for bond pricing."""
        if t > T:
            raise ValueError("Evaluation time t cannot be greater than maturity T.")
        return (1.0 / self.a) * (1.0 - np.exp(-self.a * (T - t)))

    def A(self, t: float, T: float) -> float:
        """Calculates the A(t, T) term for bond pricing."""
        if t > T:
            raise ValueError("Evaluation time t cannot be greater than maturity T.")
        
        # P(0, T) / P(0, t)
        term1 = np.log(self.P0(T) / self.P0(t))
        
        # B(t, T) * f(0, t)
        term2 = self.B(t, T) * self.f0(t)
        
        # (sigma^2 / (4 * a)) * (1 - exp(-2 * a * t)) * B(t, T)^2
        b_t_T = self.B(t, T)
        term3 = (self._sigma2 / (4.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t)) * (b_t_T ** 2)
        
        log_A = term1 - term2 - term3
        return np.exp(log_A)

    def zero_coupon_bond_price(self, t: float, T: float, r_t: float) -> float:
        """
        Calculates the price P(t, T) of a zero-coupon bond at time t for maturity T,
        given the short rate r(t) = r_t.

        Args:
            t (float): Current time.
            T (float): Maturity time.
            r_t (float): Short rate at time t.

        Returns:
            float: Price of the zero-coupon bond.
        """
        if t > T:
            raise ValueError("Evaluation time t cannot be greater than maturity T.")
        if t == T:
            return 1.0
            
        a_t_T = self.A(t, T)
        b_t_T = self.B(t, T)
        
        price = a_t_T * np.exp(-b_t_T * r_t)
        return price

    # Potential future additions:
    # - Method to derive theta(t) if not provided implicitly via P0/f0
    # - Method to simulate short rate paths using discretization (Euler, Milstein)
    # - Methods for pricing European bond options (using Jamshidian's trick or integration)
    # - Methods for pricing caps/floors/swaptions

    def __repr__(self):
        return f"HullWhite(a={self.a}, sigma={self.sigma}, initial_fwd_rate_func={self.f0}, initial_bond_price_func={self.P0})" 