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
    - Vega: First derivative of option price with respect to volatility (per 1% change)
    - Theta: First derivative of option price with respect to time (per calendar day)
    - Rho: First derivative of option price with respect to risk-free rate (per 1% change)
    - Psi (Rho_dividend): First derivative of option price with respect to dividend yield
    """

    def __init__(self, model: BlackScholes):
        """
        Initialize the AnalyticalGreeks calculator.

        Parameters
        ----------
        model : BlackScholes
            An initialized Black-Scholes model instance containing all necessary
            parameters (S, K, r, q, sigma, T, option_type) and calculated
            d1 and d2 values.
        """
        if not isinstance(model, BlackScholes):
            raise TypeError("model must be an instance of BlackScholes")
        self.model = model

    def delta(self) -> float:
        """
        Calculate the option's delta.

        Delta measures the rate of change of the option price with respect
        to a $1 change in the underlying asset's price.

        Returns
        -------
        float
            Option delta
        """
        if self.model.T <= 0:
            if self.model.option_type == 'call':
                return 1.0 if self.model.S > self.model.K else 0.0
            else:
                return -1.0 if self.model.S < self.model.K else 0.0

        exp_neg_qt = np.exp(-self.model.q * self.model.T)
        d1 = self.model.d1

        if self.model.option_type == 'call':
            return exp_neg_qt * norm.cdf(d1)
        elif self.model.option_type == 'put':
            return -exp_neg_qt * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type in model.")

    def gamma(self) -> float:
        """
        Calculate the option's gamma.

        Gamma measures the rate of change in the delta with respect to a $1
        change in the underlying asset's price.

        Returns
        -------
        float
            Option gamma
        """
        if self.model.T <= 0 or self.model.sigma <= 0:
            return 0.0

        S = self.model.S
        T = self.model.T
        sigma = self.model.sigma
        d1 = self.model.d1
        q = self.model.q

        exp_neg_qt = np.exp(-q * T)
        pdf_d1 = norm.pdf(d1)
        denominator = S * sigma * np.sqrt(T)

        return exp_neg_qt * pdf_d1 / denominator

    def vega(self) -> float:
        """
        Calculate the option's vega.

        Vega measures sensitivity to volatility. It is the change in the
        option price per 1% change in implied volatility.

        Returns
        -------
        float
            Option vega (scaled by 1/100)
        """
        if self.model.T <= 0:
            return 0.0

        S = self.model.S
        T = self.model.T
        d1 = self.model.d1
        q = self.model.q

        exp_neg_qt = np.exp(-q * T)
        pdf_d1 = norm.pdf(d1)

        return S * exp_neg_qt * np.sqrt(T) * pdf_d1 / 100.0

    def theta(self) -> float:
        """
        Calculate the option's theta.

        Theta measures sensitivity to the passage of time (time decay).
        It represents the change in option price per calendar day decrease
        in time to expiration. Theta is typically negative for long options.

        Returns
        -------
        float
            Option theta (per calendar day)
        """
        if self.model.T <= 0:
            return 0.0

        S = self.model.S
        K = self.model.K
        r = self.model.r
        q = self.model.q
        sigma = self.model.sigma
        T = self.model.T
        d1 = self.model.d1
        d2 = self.model.d2

        exp_neg_qt = np.exp(-q * T)
        exp_neg_rt = np.exp(-r * T)
        pdf_d1 = norm.pdf(d1)

        common_term = -(S * sigma * exp_neg_qt * pdf_d1) / (2 * np.sqrt(T))

        if self.model.option_type == 'call':
            theta = (common_term 
                    - r * K * exp_neg_rt * norm.cdf(d2) 
                    + q * S * exp_neg_qt * norm.cdf(d1))
        elif self.model.option_type == 'put':
            theta = (common_term 
                    + r * K * exp_neg_rt * norm.cdf(-d2) 
                    - q * S * exp_neg_qt * norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type in model.")

        return theta / 365.0

    def rho(self) -> float:
        """
        Calculate the option's rho.

        Rho measures sensitivity to the risk-free interest rate. It is the
        change in the option price per 1% change in the risk-free rate.

        Returns
        -------
        float
            Option rho (scaled by 1/100)
        """
        if self.model.T <= 0:
            return 0.0

        K = self.model.K
        T = self.model.T
        r = self.model.r
        d2 = self.model.d2

        exp_neg_rt = np.exp(-r * T)

        if self.model.option_type == 'call':
            return K * T * exp_neg_rt * norm.cdf(d2) / 100.0
        elif self.model.option_type == 'put':
            return -K * T * exp_neg_rt * norm.cdf(-d2) / 100.0
        else:
            raise ValueError("Invalid option type in model.")

    def psi(self) -> float:
        """
        Calculate the option's sensitivity to dividend yield (Psi).

        Psi measures the rate of change in the option price with respect
        to changes in the continuous dividend yield.
        Psi is typically negative for call options and positive for put options.

        Returns
        -------
        float
            The calculated psi (dividend sensitivity) of the option.
        """
        if self.model.T <= 0:
            return 0.0

        S = self.model.S
        T = self.model.T
        q = self.model.q
        d1 = self.model.d1

        exp_neg_qt = np.exp(-q * T)

        if self.model.option_type == 'call':
            return -S * T * exp_neg_qt * norm.cdf(d1) / 100.0
        elif self.model.option_type == 'put':
            return S * T * exp_neg_qt * norm.cdf(-d1) / 100.0
        else:
            raise ValueError("Invalid option type in model.")

    def rho_dividend(self) -> float:
        """Alias for psi()."""
        return self.psi()
