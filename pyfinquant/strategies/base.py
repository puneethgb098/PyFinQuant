"""
Base class for trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class Strategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the strategy.

        Args:
            parameters: Dictionary of strategy parameters.
        """
        self.parameters = parameters or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic.

        Args:
            data: DataFrame containing market data.

        Returns:
            Series of trading signals (1 for buy, -1 for sell, 0 for hold).
        """
        pass

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.

        Args:
            parameters: Dictionary of parameters to update.
        """
        self.parameters.update(parameters)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of current parameters.
        """
        return self.parameters.copy() 