"""
Internal utility functions for the PyFinQuant package.

These functions are generally intended for use by other modules within
this package and may not be part of the stable public API unless explicitly
exposed in pyfinquant.utils.__init__.py.
"""

from typing import Union

Numeric = Union[int, float]

def check_positive(value: Numeric, name: str = "Value") -> None:
    """
    Raises a ValueError if the provided numeric value is not strictly positive (> 0).

    Args:
        value: The numeric value to check (int or float).
        name: The name of the variable or parameter being checked, used in the error message.

    Raises:
        ValueError: If value is less than or equal to zero.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, but got {value}")

def check_non_negative(value: Numeric, name: str = "Value") -> None:
    """
    Raises a ValueError if the provided numeric value is negative (< 0).

    Args:
        value: The numeric value to check (int or float).
        name: The name of the variable or parameter being checked, used in the error message.

    Raises:
        ValueError: If value is less than zero.
    """
    if value < 0:
        raise ValueError(f"{name} cannot be negative, but got {value}")