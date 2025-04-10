from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from pyfinquant.instruments.option import Option

class BlackScholes:
    """
    Calculates the price of a European option using the Black-Scholes-Merton model.
    Handles continuous dividend yields.
    """

    @staticmethod
    def _calculate_d1_d2(option: 'Option') -> tuple[float, float]:
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

        # Handle potential division by zero or log of non-positive if T or sigma is zero,
        # though Option validation should prevent T <= 0 and sigma <= 0.
        # However, sigma * sqrt(T) could still be very small or zero.
        sigma_sqrt_T = sigma * np.sqrt(T)
        if sigma_sqrt_T == 0:
             # If time or volatility is zero, the option price is its intrinsic value.
             # We can return values for d1/d2 that lead to this, or handle it in the price func.
             # For simplicity, let's return large values that push N(d) to 0 or 1.
             # A more robust approach might handle T=0 or sigma=0 directly in price().
             # If S > K for call, d1->inf, d2->inf. If S < K for put, d1->-inf, d2->-inf.
             # Arbitrarily large/small number:
             if S * np.exp(-q * T) >= K * np.exp(-r * T): # Equivalent to intrinsic value > 0
                 return np.inf, np.inf
             else:
                 return -np.inf, -np.inf

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        return d1, d2

    @classmethod
    def price(cls, option: 'Option') -> float:
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

        # Handle edge case T=0 or very small T: Option price is intrinsic value
        if T < 1e-10:  # Use a small threshold instead of exact zero
            if option.is_call():
                return max(0.0, S - K)
            else:  # Put
                return max(0.0, K - S)

        # Handle edge case sigma=0: Future price is deterministic
        if option.volatility < 1e-10:  # Use a small threshold instead of exact zero
            forward_price = S * np.exp((r - q) * T)
            discount = np.exp(-r * T)
            if option.is_call():
                return max(0.0, forward_price - K) * discount
            else:  # Put
                return max(0.0, K - forward_price) * discount

        d1, d2 = cls._calculate_d1_d2(option)

        if option.is_call():
            # Price = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
            price = (S * np.exp(-q * T) * norm.cdf(d1)) - \
                    (K * np.exp(-r * T) * norm.cdf(d2))
        elif option.is_put():
            # Price = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)
            price = (K * np.exp(-r * T) * norm.cdf(-d2)) - \
                    (S * np.exp(-q * T) * norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type specified.")

        # Price cannot be negative (arbitrage)
        return max(0.0, price)
