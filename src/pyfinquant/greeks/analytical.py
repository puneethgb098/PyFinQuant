from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

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

    @staticmethod
    def delta(option: 'Option') -> float:
        """
        Calculates the Delta of the option.
        
        Delta measures the rate of change in the option price with respect to changes in the underlying asset's price.
        For a call option, delta ranges from 0 to 1.
        For a put option, delta ranges from -1 to 0.
        
        Args:
            option: The Option object containing all necessary parameters.
            
        Returns:
            The calculated delta of the option.
        """
        try:
            from pyfinquant.models.black_scholes import BlackScholes
            d1, _ = BlackScholes._calculate_d1_d2(option)
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
            else:
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T

        q = option.dividend_yield
        T = option.time_to_maturity

        # Handle edge cases
        if T < 1e-10:  # Use a small threshold instead of exact zero
            if option.is_call():
                return 1.0 if option.underlying_price > option.strike_price else 0.0
            else:  # Put
                return -1.0 if option.underlying_price < option.strike_price else 0.0
        
        if option.volatility < 1e-10:  # Use a small threshold instead of exact zero
            # For zero volatility, delta is based on the deterministic forward price
            forward_price = option.underlying_price * np.exp((option.risk_free_rate - q) * T)
            if option.is_call():
                return np.exp(-q * T) if forward_price > option.strike_price else 0.0
            else:  # Put
                return -np.exp(-q * T) if forward_price < option.strike_price else 0.0

        # Normal case
        if option.is_call():
            return np.exp(-q * T) * norm.cdf(d1)
        elif option.is_put():
            return -np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type.")

    @staticmethod
    def gamma(option: 'Option') -> float:
        """
        Calculates the Gamma of the option.
        
        Gamma measures the rate of change in delta with respect to changes in the underlying asset's price.
        Gamma is always positive for both call and put options.
        
        Args:
            option: The Option object containing all necessary parameters.
            
        Returns:
            The calculated gamma of the option.
        """
        try:
            from pyfinquant.models.black_scholes import BlackScholes
            d1, _ = BlackScholes._calculate_d1_d2(option)
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
            else:
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T

        S = option.underlying_price
        sigma = option.volatility
        T = option.time_to_maturity
        q = option.dividend_yield

        # Handle edge cases
        if T < 1e-10 or sigma < 1e-10:  # Use a small threshold instead of exact zero
            return 0.0  # Gamma is zero at expiration or with zero volatility

        sigma_sqrt_T = sigma * np.sqrt(T)
        if sigma_sqrt_T < 1e-10:  # Use a small threshold instead of exact zero
            return 0.0  # Gamma is zero when sigma_sqrt_T is zero

        # Normal case
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma_sqrt_T)

    @staticmethod
    def vega(option: 'Option') -> float:
        """
        Calculates the Vega of the option.
        
        Vega measures the rate of change in the option price with respect to changes in the volatility of the underlying asset.
        Vega is always positive for both call and put options.
        
        Args:
            option: The Option object containing all necessary parameters.
            
        Returns:
            The calculated vega of the option.
        """
        try:
            from pyfinquant.models.black_scholes import BlackScholes
            d1, _ = BlackScholes._calculate_d1_d2(option)
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
            else:
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T

        S = option.underlying_price
        q = option.dividend_yield
        T = option.time_to_maturity

        if T < 1e-10 or option.volatility < 1e-10:  # Use a small threshold instead of exact zero
            return 0.0

        pdf_d1 = norm.pdf(d1)
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
        return vega

    @staticmethod
    def theta(option: 'Option') -> float:
        """
        Calculates the Theta of the option.
        
        Theta measures the rate of change in the option value with respect to the passage of time.
        Theta is typically negative for both call and put options, indicating time decay.
        
        Args:
            option: The Option object containing all necessary parameters.
            
        Returns:
            The calculated theta of the option.
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
        K = option.strike_price
        r = option.risk_free_rate
        q = option.dividend_yield
        sigma = option.volatility
        T = option.time_to_maturity

        # Handle edge cases
        if T < 1e-10:  # Use a small threshold instead of exact zero
            return 0.0  # Theta is zero at expiration
        if sigma < 1e-10:  # Use a small threshold instead of exact zero
            if option.is_call():
                return -q * S * np.exp(-q * T) if S > K else 0.0
            else:
                return q * S * np.exp(-q * T) if S < K else 0.0

        sigma_sqrt_T = sigma * np.sqrt(T)
        if sigma_sqrt_T < 1e-10:  # Use a small threshold instead of exact zero
            return 0.0  # Theta is zero when sigma_sqrt_T is zero

        # Normal case
        term1 = -S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T))
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)

        if option.is_call():
            return (term1 + term2 + term3) / 365.0  # Convert to daily theta
        elif option.is_put():
            return (term1 + term2 - term3) / 365.0  # Convert to daily theta
        else:
            raise ValueError("Invalid option type.")

    @staticmethod
    def rho(option: 'Option') -> float:
        """
        Calculates the Rho of the option.
        
        Rho measures the rate of change in the option price with respect to changes in the risk-free interest rate.
        Rho is positive for call options and negative for put options.
        
        Args:
            option: The Option object containing all necessary parameters.
            
        Returns:
            The calculated rho of the option.
        """
        try:
            from pyfinquant.models.black_scholes import BlackScholes
            _, d2 = BlackScholes._calculate_d1_d2(option)
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

        K = option.strike_price
        r = option.risk_free_rate
        T = option.time_to_maturity

        if T < 1e-10:  # Use a small threshold instead of exact zero
            return 0.0

        cdf_d2 = norm.cdf(d2)
        cdf_neg_d2 = norm.cdf(-d2)

        if option.is_call():
            rho = K * T * np.exp(-r * T) * cdf_d2
        elif option.is_put():
            rho = -K * T * np.exp(-r * T) * cdf_neg_d2
        else:
            raise ValueError("Invalid option type.")

        return rho

    @staticmethod
    def rho_dividend(option: 'Option') -> float:
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
             S = option.underlying_price; K = option.strike_price; r = option.risk_free_rate; q = option.dividend_yield; sigma = option.volatility; T = option.time_to_maturity
             sigma_sqrt_T = sigma * np.sqrt(T)
             if sigma_sqrt_T == 0: d1 = np.inf if S > K else -np.inf; d2 = d1
             else: d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T; d2 = d1 - sigma_sqrt_T

        S = option.underlying_price
        q = option.dividend_yield
        T = option.time_to_maturity

        if T == 0: return 0.0

        cdf_d1 = norm.cdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)

        if option.is_call():
            psi = -S * T * np.exp(-q * T) * cdf_d1
        elif option.is_put():
            psi = S * T * np.exp(-q * T) * cdf_neg_d1
        else:
            raise ValueError("Invalid option type.")

        return psi
