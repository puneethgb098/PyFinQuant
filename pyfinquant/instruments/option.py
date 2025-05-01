"""
Option class implementation.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class OptionType(Enum):
    """Option type enumeration."""
    CALL = 'call'
    PUT = 'put'


@dataclass
class Option:
    """
    Class representing a financial option contract.
    """

    underlying_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: str = 'call'

    def __post_init__(self):
        """Validate inputs and convert option type to enum."""
        if isinstance(self.option_type, str):
            self.option_type = OptionType(self.option_type.lower())
        elif not isinstance(self.option_type, OptionType):
            raise ValueError("option_type must be a string or OptionType enum")

        if self.underlying_price <= 0:
            raise ValueError("underlying_price must be positive")
        if self.strike_price <= 0:
            raise ValueError("strike_price must be positive")
        if self.time_to_maturity < 0:
            raise ValueError("time_to_maturity must be non-negative")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.dividend_yield < 0:
            raise ValueError("dividend_yield must be non-negative")

    @property
    def underlying_price(self) -> float:
        """Get the current price of the underlying asset."""
        return self.underlying_price

    def is_call(self) -> bool:
        """Check if the option is a call option."""
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        """Check if the option is a put option."""
        return self.option_type == OptionType.PUT

    def is_in_the_money(self) -> bool:
        """Check if the option is in the money."""
        if self.is_call():
            return self.underlying_price > self.strike_price
        else:
            return self.underlying_price < self.strike_price

    def is_at_the_money(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the option is at the money.

        Args:
            tolerance: Tolerance for price comparison.
        """
        return abs(self.underlying_price - self.strike_price) <= tolerance

    def is_out_of_the_money(self) -> bool:
        """Check if the option is out of the money."""
        if self.is_call():
            return self.underlying_price < self.strike_price
        else:
            return self.underlying_price > self.strike_price

    def intrinsic_value(self) -> float:
        """Calculate the intrinsic value of the option."""
        if self.is_call():
            return max(0, self.underlying_price - self.strike_price)
        else:
            return max(0, self.strike_price - self.underlying_price)