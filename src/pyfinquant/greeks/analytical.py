from typing import TYPE_CHECKING, Union, Optional

import numpy as np
from scipy.stats import norm
from ..utils.types import Numeric
from ..models.black_scholes import BlackScholes

if TYPE_CHECKING:
    from pyfinquant.instruments.option import Option


class AnalyticalGreeks:
    """
    Calculates analytical Greeks for European options based on the Black-Scholes-Merton model.

    This class provides methods to calculate the following Greeks:
    - Delta: First derivative of option price with respect to underlying price
    - Gamma: Second derivative of option price with respect to underlying price
    - Vega: First derivative of option price with respect to volatility
    - Theta: First derivative of option price with respect to time
    - Rho: First derivative of option price with respect to risk-free rate
    - Rho_dividend: First derivative of option price with respect to dividend yield
    """

    def __init__(self, model: BlackScholes):
        """
        Initialize the AnalyticalGreeks calculator.
        
        Parameters
        ----------
        model : BlackScholes
            Black-Scholes model instance
        """
        self.model = model

    def delta(self) -> float:
        """
        Calculate the option's delta.
        
        Returns
        -------
        float
            Option delta
        """
        if self.model.option_type == 'call':
            return np.exp(-self.model.q * self.model.T) * norm.cdf(self.model.d1)
        else:
            return np.exp(-self.model.q * self.model.T) * (norm.cdf(self.model.d1) - 1)

    def gamma(self) -> float:
        """
        Calculate the option's gamma.
        
        Returns
        -------
        float
            Option gamma
        """
        if self.model.T == 0 or self.model.sigma == 0:
            return 0.0
        return np.exp(-self.model.q * self.model.T) * norm.pdf(self.model.d1) / (self.model.S * self.model.sigma * np.sqrt(self.model.T))

    def vega(self) -> float:
        """
        Calculate the option's vega.
        
        Returns
        -------
        float
            Option vega
        """
        if self.model.T == 0:
            return 0.0
        return self.model.S * np.exp(-self.model.q * self.model.T) * np.sqrt(self.model.T) * norm.pdf(self.model.d1) / 100

    def rho(self) -> float:
        """
        Calculate the option's rho.
        
        Returns
        -------
        float
            Option rho
        """
        if self.model.T == 0:
            return 0.0
        if self.model.option_type == 'call':
            return self.model.K * self.model.T * np.exp(-self.model.r * self.model.T) * norm.cdf(self.model.d2) / 100
        else:
            return -self.model.K * self.model.T * np.exp(-self.model.r * self.model.T) * norm.cdf(-self.model.d2) / 100

    def theta(self) -> float:
        """
        Calculate the option's theta.
        
        Returns
        -------
        float
            Option theta (daily theta)
        """
        if self.model.T == 0:
            return 0.0
        
        S = self.model.S
        K = self.model.K
        r = self.model.r
        q = self.model.q
        sigma = self.model.sigma
        T = self.model.T
        
        d1 = self.model.d1
        d2 = self.model.d2
        
        if self.model.option_type == 'call':
            theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T))) - \
                   (r * K * np.exp(-r * T) * norm.cdf(d2)) + \
                   (q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T))) + \
                   (r * K * np.exp(-r * T) * norm.cdf(-d2)) - \
                   (q * S * np.exp(-q * T) * norm.cdf(-d1))
        
        return theta / 365.0  # Convert to daily theta

    @staticmethod
    def rho_dividend(option: "Option") -> float:
        """
        Calculates the sensitivity to dividend yield (Psi).

        Psi measures the rate of change in the option price with respect to changes in the dividend yield.
        Psi is negative for call options and positive for put options.

        Args:
            option: The Option object containing all necessary parameters.

        Returns:
            The calculated psi (dividend sensitivity) of the option.
        """
        try:
            from pyfinquant.models.black_scholes import BlackScholes

            d1, d2 = BlackScholes._calculate_d1_d2(option)
        except ImportError:
            S = option.underlying_price
            K = option.strike_price
            r = option.risk_free_rate
            q = option.dividend_yield
            sigma = option.volatility
            T = option.time_to_maturity
            sigma_sqrt_T = sigma * np.sqrt(T)
            if sigma_sqrt_T == 0:
                d1 = np.inf if S > K else -np.inf
                d2 = d1
            else:
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
                d2 = d1 - sigma_sqrt_T

        S = option.underlying_price
        q = option.dividend_yield
        T = option.time_to_maturity

        if T == 0:
            return 0.0

        cdf_d1 = norm.cdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)

        if option.is_call():
            psi = -S * T * np.exp(-q * T) * cdf_d1
        elif option.is_put():
            psi = S * T * np.exp(-q * T) * cdf_neg_d1
        else:
            raise ValueError("Invalid option type.")

        return psi
