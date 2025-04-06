# Example utility function (can be expanded later)

def check_positive(value: float, name: str) -> None:
    """Raises ValueError if the value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, but got {value}")

# You could centralize input validation logic here if desired,
# although the dataclass __post_init__ handles much of it for Option.
