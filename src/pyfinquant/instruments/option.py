import enum
from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
from ..utils.types import Numeric
from ..models.black_scholes import BlackScholes


class OptionType(enum.Enum):
    """Enumeration for Call or Put option type."""

    CALL = "Call"
    PUT = "Put"


@dataclass(frozen=True)  # Use frozen=True for immutability, good for financial objects
class Option:
    """
    Represents a European financial option contract.

    Attributes:
        underlying_price (float): Current price of the underlying asset (S).
        strike_price (float): Price at which the option can be exercised (K).
        time_to_maturity (float): Time to expiration in years (T).
        risk_free_rate (float): Annualized risk-free interest rate (r), decimal form (e.g., 0.05 for 5%).
        volatility (float): Annualized volatility of the underlying asset's returns (sigma), decimal form (e.g., 0.2 for 20%).
        option_type (OptionType): Type of the option (CALL or PUT).
        dividend_yield (float): Annualized continuous dividend yield of the underlying asset (q), decimal form. Defaults to 0.0.
    """

    underlying_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    option_type: OptionType
    dividend_yield: float = 0.0  # Default dividend yield to 0

    def __post_init__(self):
        """Perform validation checks after initialization."""
        if self.underlying_price <= 0:
            raise ValueError("Underlying price must be positive.")
        if self.strike_price <= 0:
            raise ValueError("Strike price must be positive.")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")
        if self.time_to_maturity <= 0:
            raise ValueError("Time to maturity must be positive.")
        if self.dividend_yield < 0:
            raise ValueError("Dividend yield cannot be negative.")

    def is_call(self) -> bool:
        """Check if the option is a Call."""
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        """Check if the option is a Put."""
        return self.option_type == OptionType.PUT

    def price(self) -> float:
        """
        Calculate the option price.
        
        Returns
        -------
        float
            Option price
        """
        # Initialize Black-Scholes model
        model = BlackScholes(
            S=self.underlying_price,
            K=self.strike_price,
            T=self.time_to_maturity,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type=self.option_type.value
        )
        return model.price()

    def delta(self) -> float:
        """
        Calculate the option's delta.
        
        Returns
        -------
        float
            Option delta
        """
        from ..greeks.analytical import AnalyticalGreeks
        model = BlackScholes(
            S=self.underlying_price,
            K=self.strike_price,
            T=self.time_to_maturity,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type=self.option_type.value
        )
        return AnalyticalGreeks(model).delta()

    def gamma(self) -> float:
        """
        Calculate the option's gamma.
        
        Returns
        -------
        float
            Option gamma
        """
        from ..greeks.analytical import AnalyticalGreeks
        model = BlackScholes(
            S=self.underlying_price,
            K=self.strike_price,
            T=self.time_to_maturity,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type=self.option_type.value
        )
        return AnalyticalGreeks(model).gamma()

    def vega(self) -> float:
        """
        Calculate the option's vega.
        
        Returns
        -------
        float
            Option vega
        """
        from ..greeks.analytical import AnalyticalGreeks
        model = BlackScholes(
            S=self.underlying_price,
            K=self.strike_price,
            T=self.time_to_maturity,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type=self.option_type.value
        )
        return AnalyticalGreeks(model).vega()

    def rho(self) -> float:
        """
        Calculate the option's rho.
        
        Returns
        -------
        float
            Option rho
        """
        from ..greeks.analytical import AnalyticalGreeks
        model = BlackScholes(
            S=self.underlying_price,
            K=self.strike_price,
            T=self.time_to_maturity,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type=self.option_type.value
        )
        return AnalyticalGreeks(model).rho()

    def theta(self) -> float:
        """
        Calculate the option's theta.
        
        Returns
        -------
        float
            Option theta
        """
        from ..greeks.analytical import AnalyticalGreeks
        model = BlackScholes(
            S=self.underlying_price,
            K=self.strike_price,
            T=self.time_to_maturity,
            r=self.risk_free_rate,
            sigma=self.volatility,
            option_type=self.option_type.value
        )
        return AnalyticalGreeks(model).theta(self)

# Example of a helper function that could be in utils, but fits here too:
# This requires date calculations, which might add dependencies or complexity.
# For simplicity now, we assume time_to_maturity is directly provided in years.
# def calculate_time_to_maturity(evaluation_date: date, expiration_date: date) -> float:
#     """Calculates time to maturity in years between two dates."""
#     if evaluation_date >= expiration_date:
#         return 0.0 # Or raise error
#     delta = expiration_date - evaluation_date
#     # Approximate using average days in year
#     return delta.days / 365.25
