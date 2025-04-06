# Expose key classes and functions at the top level of the package
from .greeks.analytical import AnalyticalGreeks
from .instruments.option import Option, OptionType
from .models.black_scholes import BlackScholes

# Define __all__ to specify the public API explicitly
__all__ = [
    "AnalyticalGreeks",
    "BlackScholes",
    "Option",
    "OptionType",
]

# Optionally set package version (can also be managed by build tools)
__version__ = "0.1.0"
