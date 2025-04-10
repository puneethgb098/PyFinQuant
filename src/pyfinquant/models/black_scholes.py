from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm
from ..utils.types import Numeric

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from pyfinquant.instruments.option import Option


class BlackScholes:
    """
    Calculates the price of a European option using the Black-Scholes-Merton model.
    Handles continuous dividend yields.
    """

    def __init__(self, S: Numeric, K: Numeric, T: Numeric, r: Numeric, sigma: Numeric, option_type: str, q: Numeric = 0.0):
        """
        Initialize the Black-Scholes model.
        
        Parameters
        ----------
        S : Numeric
            Current price of the underlying asset
        K : Numeric
            Strike price of the option
        T : Numeric
            Time to maturity in years
        r : Numeric
            Risk-free interest rate
        sigma : Numeric
            Volatility of the underlying asset
        option_type : str
            Type of option, either 'call' or 'put'
        q : Numeric, optional
            Continuous dividend yield, by default 0.0
        """
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")
        
        if self.T < 0:
            raise ValueError("Time to maturity must be positive.")
        
        # Calculate d1 and d2
        self._calculate_d1_d2()
    
    def _calculate_d1_d2(self) -> None:
        """Calculate d1 and d2 parameters."""
        if self.T == 0 or self.sigma == 0:
            self.d1 = np.inf if self.S > self.K else -np.inf
            self.d2 = self.d1
        else:
            sigma_sqrt_T = self.sigma * np.sqrt(self.T)
            self.d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / sigma_sqrt_T
            self.d2 = self.d1 - sigma_sqrt_T
    
    def price(self) -> float:
        """
        Calculate the option price using the Black-Scholes formula.
        
        Returns
        -------
        float
            Option price
        """
        if self.T == 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)
        
        if self.sigma == 0:
            if self.option_type == 'call':
                return max(0, self.S * np.exp((self.r - self.q) * self.T) - self.K)
            else:
                return max(0, self.K - self.S * np.exp((self.r - self.q) * self.T))
        
        if self.option_type == 'call':
            return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
