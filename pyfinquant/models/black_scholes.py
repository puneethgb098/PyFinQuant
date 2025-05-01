import numpy as np
from scipy.stats import norm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyfinquant.instruments.option import Option, OptionType


class BlackScholes:
    """
    Calculates the price of a European option using the Black-Scholes-Merton model.
    Handles continuous dividend yields.
    """

    @staticmethod
    def _calculate_d1_d2(option: "Option") -> tuple[float, float]:
        """
        Calculates the d1 and d2 parameters used in the Black-Scholes formula.

        Args:
            option: The Option object containing parameters S, K, r, q, sigma, T.

        Returns:
            A tuple containing (d1, d2).
        """
        S = option.underlying_price
        K = option.strike_price
        r = option.risk_free_rate
        q = option.dividend_yield
        sigma = option.volatility
        T = option.time_to_maturity

        sigma_sqrt_T = sigma * np.sqrt(T)
        if sigma_sqrt_T == 0:
            if S * np.exp(-q * T) >= K * np.exp(-r * T):
                return np.inf, np.inf
            else:
                return -np.inf, -np.inf

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        return d1, d2

    @classmethod
    def price(cls, option: "Option") -> float:
        """
        Calculates the Black-Scholes-Merton price for a European option.

        Args:
            option: The Option object containing all necessary parameters.

        Returns:
            The calculated price of the option.
        """
        S = option.underlying_price
        K = option.strike_price
        r = option.risk_free_rate
        q = option.dividend_yield
        T = option.time_to_maturity

        if T == 0:
            if option.is_call():
                return max(0.0, S - K)
            else:  # Put
                return max(0.0, K - S)

        d1, d2 = cls._calculate_d1_d2(option)

        if option.is_call():
            price = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        elif option.is_put():
            price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type specified.")

        return max(0.0, price)
