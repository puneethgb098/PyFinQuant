"""Type definitions for PyFinQuant."""

from typing import TypeVar, Union

# Define Numeric as a type that can be either float or int
Numeric = Union[float, int]

# Type variable for numeric values
NumericType = TypeVar("NumericType", float, int)
