import numpy as np
from scipy.stats import norm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyfinquant.instruments.option import Option
    from pyfinquant.models.black_scholes import BlackScholes


class AnalyticalGreeks:
    """
    Calculates analytical Greeks for European options based on the Black-Scholes-Merton model.
    """

    @staticmethod
    def delta(option: "Option") -> float:
        """Calculates the Delta of the option."""

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

        if T == 0:
            return (
                1.0
                if option.is_call() and option.underlying_price > option.strike_price
                else (
                    -1.0
                    if option.is_put() and option.underlying_price < option.strike_price
                    else 0.0
                )
            )
        if option.volatility == 0:
            return (
                np.exp(-q * T) if option.is_call() else -np.exp(-q * T)
            )  # Based on deterministic forward price sensitivity

        if option.is_call():
            return np.exp(-q * T) * norm.cdf(d1)
        elif option.is_put():
            return np.exp(-q * T) * (norm.cdf(d1) - 1)
        else:
            raise ValueError("Invalid option type.")

    @staticmethod
    def gamma(option: "Option") -> float:
        """Calculates the Gamma of the option."""
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
        sigma = option.volatility
        T = option.time_to_maturity

        if T == 0 or sigma == 0:
            return 0.0

        sigma_sqrt_T = sigma * np.sqrt(T)
        if sigma_sqrt_T == 0:
            return 0.0

        pdf_d1 = norm.pdf(d1)
        gamma = (np.exp(-q * T) * pdf_d1) / (S * sigma_sqrt_T)
        return gamma

    @staticmethod
    def vega(option: "Option") -> float:
        """Calculates the Vega of the option."""
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

        if T == 0 or option.volatility == 0:
            return 0.0

        pdf_d1 = norm.pdf(d1)
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
        return vega

    @staticmethod
    def theta(option: "Option") -> float:
        """Calculates the Theta of the option."""
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

        if T == 0:
            return 0.0

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)

        term1 = -(S * np.exp(-q * T) * pdf_d1 * sigma) / (2 * np.sqrt(T))

        if option.is_call():
            theta = term1 - r * K * np.exp(-r * T) * cdf_d2 + q * S * np.exp(-q * T) * cdf_d1
        elif option.is_put():
            theta = (
                term1 + r * K * np.exp(-r * T) * cdf_neg_d2 - q * S * np.exp(-q * T) * cdf_neg_d1
            )
        else:
            raise ValueError("Invalid option type.")

        return theta

    @staticmethod
    def rho(option: "Option") -> float:
        """Calculates the Rho of the option."""
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

        if T == 0:
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
    def rho_dividend(option: "Option") -> float:
        """Calculates the sensitivity to dividend yield (Psi)."""
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


# Expose the functions at module level
def delta(option: "Option") -> float:
    """Calculates the Delta of the option."""
    return AnalyticalGreeks.delta(option)


def gamma(option: "Option") -> float:
    """Calculates the Gamma of the option."""
    return AnalyticalGreeks.gamma(option)


def vega(option: "Option") -> float:
    """Calculates the Vega of the option."""
    return AnalyticalGreeks.vega(option)


def theta(option: "Option") -> float:
    """Calculates the Theta of the option."""
    return AnalyticalGreeks.theta(option)


def rho(option: "Option") -> float:
    """Calculates the Rho of the option."""
    return AnalyticalGreeks.rho(option)


def rho_dividend(option: "Option") -> float:
    """Calculates the Rho with respect to dividend yield of the option."""
    return AnalyticalGreeks.rho_dividend(option)
