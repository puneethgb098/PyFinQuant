import enum
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


class OptionType(enum.Enum):
    """Enumeration for Call or Put option type."""

    CALL = "Call"
    PUT = "Put"


@dataclass(frozen=True)
class Option:
    """
    Represents a European financial option contract.

    Attributes:
        underlying_price (float): Current price of the underlying asset (S).
        strike_price (float): Price at which the option can be exercised (K).
        risk_free_rate (float): Annualized risk-free interest rate (r), decimal form (e.g., 0.05 for 5%).
        volatility (float): Annualized volatility of the underlying asset's returns (sigma), decimal form (e.g., 0.2 for 20%).
        time_to_maturity (float): Time to expiration in years (T).
        option_type (OptionType): Type of the option (CALL or PUT).
        dividend_yield (float): Annualized continuous dividend yield of the underlying asset (q), decimal form. Defaults to 0.0.
    """

    underlying_price: float
    strike_price: float
    risk_free_rate: float
    volatility: float
    time_to_maturity: float
    option_type: OptionType
    dividend_yield: float = 0.0

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
