"""Utility functions for PyFinQuant."""

def check_positive(value: float, name: str) -> None:
    """
    Raises ValueError if the value is not positive.
    
    This function is used to validate that a numeric value is strictly greater than zero.
    It's commonly used for parameters like price, volatility, and time to maturity.
    
    Args:
        value: The numeric value to check.
        name: The name of the parameter being checked, used in the error message.
        
    Raises:
        ValueError: If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, but got {value}")

def check_non_negative(value: float, name: str) -> None:
    """
    Raises ValueError if the value is negative.
    
    This function is used to validate that a numeric value is greater than or equal to zero.
    It's commonly used for parameters like dividend yield.
    
    Args:
        value: The numeric value to check.
        name: The name of the parameter being checked, used in the error message.
        
    Raises:
        ValueError: If the value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} cannot be negative, but got {value}")

# You could centralize input validation logic here if desired,
# although the dataclass __post_init__ handles much of it for Option.
