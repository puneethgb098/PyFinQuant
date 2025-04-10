import pytest
from pyfinquant.utils import check_positive, check_non_negative, Numeric

def test_check_positive():
    """Test the check_positive function."""
    # Test with positive values
    check_positive(1.0, "value")  # Should not raise an error
    check_positive(0.1, "value")  # Should not raise an error
    check_positive(1000, "value")  # Should not raise an error

    # Test with non-positive values
    with pytest.raises(ValueError):
        check_positive(0, "value")
    with pytest.raises(ValueError):
        check_positive(-1, "value")
    with pytest.raises(ValueError):
        check_positive(-0.1, "value")

def test_check_non_negative():
    """Test the check_non_negative function."""
    # Test with non-negative values
    check_non_negative(0, "value")  # Should not raise an error
    check_non_negative(0.1, "value")  # Should not raise an error
    check_non_negative(1000, "value")  # Should not raise an error

    # Test with negative values
    with pytest.raises(ValueError):
        check_non_negative(-1, "value")
    with pytest.raises(ValueError):
        check_non_negative(-0.1, "value") 