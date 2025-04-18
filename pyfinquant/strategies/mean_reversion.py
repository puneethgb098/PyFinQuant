"""
Mean reversion trading strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import Strategy


class MeanReversion(Strategy):
    """
    Mean reversion trading strategy.
    
    This strategy generates trading signals based on deviations from a moving average,
    assuming that prices tend to revert to their mean.
    """

    def __init__(self, window: int = 20, std_dev: float = 2.0, parameters: Dict[str, Any] = None):
        """
        Initialize the mean reversion strategy.

        Args:
            window: Number of periods for calculating moving average.
            std_dev: Number of standard deviations for generating signals.
            parameters: Additional strategy parameters.
        """
        super().__init__(parameters)
        self.window = window
        self.std_dev = std_dev

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on mean reversion logic.

        Args:
            data: DataFrame with at least a 'close' price column.

        Returns:
            Series of trading signals (1 for buy, -1 for sell, 0 for hold).
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain a 'close' price column")

        # Calculate moving average and standard deviation
        ma = data['close'].rolling(window=self.window).mean()
        std = data['close'].rolling(window=self.window).std()

        # Calculate z-score
        z_score = (data['close'] - ma) / std

        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[z_score < -self.std_dev] = 1  # Buy when price is significantly below mean
        signals[z_score > self.std_dev] = -1  # Sell when price is significantly above mean

        return signals 