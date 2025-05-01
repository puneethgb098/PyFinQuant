"""
Implementation of the Hull-White two-factor interest rate model.
"""

import numpy as np
from typing import Callable, Tuple, Union
from dataclasses import dataclass
from pyfinquant.utils.helpers import check_positive, check_non_negative


@dataclass
class HullWhite2F:
    """
    Hull-White two-factor interest rate model implementation.
    
    The two-factor model allows for better fitting of the term structure
    and more realistic interest rate dynamics by introducing a second
    stochastic factor.
    
    Parameters:
    -----------
    a1 : float
        Mean reversion speed of first factor (must be positive)
    a2 : float
        Mean reversion speed of second factor (must be positive)
    sigma1 : float
        Volatility of first factor (must be positive)
    sigma2 : float
        Volatility of second factor (must be positive)
    rho : float
        Correlation between factors (must be between -1 and 1)
    initial_fwd_rate_func : Callable
        Function that returns the initial forward rate for a given time
    initial_bond_price_func : Callable
        Function that returns the initial bond price for a given time
    """
    
    a1: float
    a2: float
    sigma1: float
    sigma2: float
    rho: float
    initial_fwd_rate_func: Callable[[float], float]
    initial_bond_price_func: Callable[[float], float]
    
    def __post_init__(self):
        """Validate parameters."""
        check_positive(self.a1, "First factor mean reversion speed")
        check_positive(self.a2, "Second factor mean reversion speed")
        check_positive(self.sigma1, "First factor volatility")
        check_positive(self.sigma2, "Second factor volatility")
        
        if not -1 <= self.rho <= 1:
            raise ValueError("Correlation must be between -1 and 1")
        
        if not callable(self.initial_fwd_rate_func):
            raise ValueError("initial_fwd_rate_func must be callable")
        if not callable(self.initial_bond_price_func):
            raise ValueError("initial_bond_price_func must be callable")
    
    def _B1(self, t: float, T: float) -> float:
        """Calculate B1(t,T) function for first factor."""
        return (1 - np.exp(-self.a1 * (T - t))) / self.a1
    
    def _B2(self, t: float, T: float) -> float:
        """Calculate B2(t,T) function for second factor."""
        return (1 - np.exp(-self.a2 * (T - t))) / self.a2
    
    def _A(self, t: float, T: float) -> float:
        """Calculate A(t,T) function."""
        B1 = self._B1(t, T)
        B2 = self._B2(t, T)
        P_t = self.initial_bond_price_func(t)
        P_T = self.initial_bond_price_func(T)
        f_t = self.initial_fwd_rate_func(t)
        
        V1 = (self.sigma1**2 / (2 * self.a1**2)) * (
            (T - t) + (2 / self.a1) * np.exp(-self.a1 * (T - t)) -
            (1 / (2 * self.a1)) * np.exp(-2 * self.a1 * (T - t)) - 3 / (2 * self.a1)
        )
        
        V2 = (self.sigma2**2 / (2 * self.a2**2)) * (
            (T - t) + (2 / self.a2) * np.exp(-self.a2 * (T - t)) -
            (1 / (2 * self.a2)) * np.exp(-2 * self.a2 * (T - t)) - 3 / (2 * self.a2)
        )
        
        V12 = (self.rho * self.sigma1 * self.sigma2 / (self.a1 * self.a2)) * (
            (T - t) + (1 / (self.a1 + self.a2)) * (
                np.exp(-(self.a1 + self.a2) * (T - t)) - 1
            ) -
            (1 / self.a1) * (np.exp(-self.a1 * (T - t)) - 1) -
            (1 / self.a2) * (np.exp(-self.a2 * (T - t)) - 1)
        )
        
        return np.log(P_T / P_t) + f_t * (B1 + B2) - 0.5 * (V1 + V2 + 2 * V12)
    
    def zero_coupon_bond_price(
        self,
        t: float,
        T: float,
        r1_t: float,
        r2_t: float
    ) -> float:
        """
        Calculate zero-coupon bond price.
        
        Parameters:
        -----------
        t : float
            Current time
        T : float
            Maturity time
        r1_t : float
            Current value of first factor
        r2_t : float
            Current value of second factor
        
        Returns:
        --------
        float
            Zero-coupon bond price
        """
        check_non_negative(t, "Current time")
        check_non_negative(T, "Maturity time")
        if T < t:
            raise ValueError("Maturity time must be greater than current time")
        
        A = self._A(t, T)
        B1 = self._B1(t, T)
        B2 = self._B2(t, T)
        
        return np.exp(A - B1 * r1_t - B2 * r2_t)
    
    def forward_rate(
        self,
        t: float,
        T1: float,
        T2: float,
        r1_t: float,
        r2_t: float
    ) -> float:
        """
        Calculate forward rate.
        
        Parameters:
        -----------
        t : float
            Current time
        T1 : float
            Start time of forward period
        T2 : float
            End time of forward period
        r1_t : float
            Current value of first factor
        r2_t : float
            Current value of second factor
        
        Returns:
        --------
        float
            Forward rate
        """
        check_non_negative(t, "Current time")
        check_non_negative(T1, "Start time")
        check_non_negative(T2, "End time")
        if T1 < t:
            raise ValueError("Start time must be greater than current time")
        if T2 <= T1:
            raise ValueError("End time must be greater than start time")
        
        P1 = self.zero_coupon_bond_price(t, T1, r1_t, r2_t)
        P2 = self.zero_coupon_bond_price(t, T2, r1_t, r2_t)
        
        return (np.log(P1 / P2)) / (T2 - T1)
    
    def short_rate_simulation(
        self,
        t: float,
        T: float,
        r1_t: float,
        r2_t: float,
        n_steps: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate short rate paths for both factors.
        
        Parameters:
        -----------
        t : float
            Current time
        T : float
            End time
        r1_t : float
            Initial value of first factor
        r2_t : float
            Initial value of second factor
        n_steps : int, optional
            Number of time steps (default: 100)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Arrays of simulated short rates for both factors
        """
        check_non_negative(t, "Current time")
        check_non_negative(T, "End time")
        check_positive(n_steps, "Number of steps")
        if T <= t:
            raise ValueError("End time must be greater than current time")
        
        dt = (T - t) / n_steps
        times = np.linspace(t, T, n_steps + 1)
        rates1 = np.zeros(n_steps + 1)
        rates2 = np.zeros(n_steps + 1)
        rates1[0] = r1_t
        rates2[0] = r2_t
        
        for i in range(1, n_steps + 1):
            t_prev = times[i-1]
            r1_prev = rates1[i-1]
            r2_prev = rates2[i-1]
            
            # Calculate drift terms
            f_t = self.initial_fwd_rate_func(t_prev)
            df_dt = (self.initial_fwd_rate_func(t_prev + dt) - f_t) / dt
            
            drift1 = df_dt + self.a1 * (f_t - r1_prev)
            drift2 = df_dt + self.a2 * (f_t - r2_prev)
            
            z1 = np.random.normal(0, 1)
            z2 = np.random.normal(0, 1)
            dW1 = np.sqrt(dt) * z1
            dW2 = np.sqrt(dt) * (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2)
            
            rates1[i] = r1_prev + drift1 * dt + self.sigma1 * dW1
            rates2[i] = r2_prev + drift2 * dt + self.sigma2 * dW2
        
        return rates1, rates2
    
    def get_short_rate(self, r1_t: float, r2_t: float) -> float:
        """
        Get the short rate from the two factors.
        
        Parameters:
        -----------
        r1_t : float
            Value of first factor
        r2_t : float
            Value of second factor
        
        Returns:
        --------
        float
            Short rate
        """
        return r1_t + r2_t 
