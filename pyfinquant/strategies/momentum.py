"""
Momentum trading strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import Strategy


class MomentumStrategy(Strategy):
    """
    Momentum trading strategy.
    
    This strategy generates trading signals based on price momentum,
    buying assets that have shown strong recent performance and
    selling those that have shown weak performance.
    """

    def __init__(
        self,
        lookback_period: int = 12,
        holding_period: int = 1,
        n_top: int = 10,
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the momentum strategy.

        Args:
            lookback_period: Number of periods to look back for momentum calculation.
            holding_period: Number of periods to hold positions.
            n_top: Number of top/bottom performing assets to trade.
            parameters: Additional strategy parameters.
        """
        super().__init__(parameters)
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.n_top = n_top

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on momentum.

        Args:
            data: DataFrame with at least a 'close' price column.

        Returns:
            Series of trading signals (1 for buy, -1 for sell, 0 for hold).
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain a 'close' price column")

        # Calculate returns
        returns = data['close'].pct_change(self.lookback_period)

        # Generate signals
        signals = pd.Series(0, index=data.index)

        # Identify top and bottom performers
        top_assets = returns.nlargest(self.n_top).index
        bottom_assets = returns.nsmallest(self.n_top).index

        # Generate signals
        signals[top_assets] = 1  # Buy top performers
        signals[bottom_assets] = -1  # Sell bottom performers

        return signals 