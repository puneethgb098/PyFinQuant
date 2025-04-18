import pandas as pd
from typing import Dict, Any, Union, Optional
from .base import Strategy
from ..data_fetcher import YahooDataFetcher


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategy that can work with Yahoo Finance data.
    """

    def __init__(
        self,
        ticker: Optional[str] = None,
        short_window: int = 50,
        long_window: int = 200,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        parameters: Dict[str, Any] = None
    ):
        """
        Initialize the Moving Average Crossover Strategy.
        
        Parameters:
        -----------
        ticker : str, optional
            Yahoo Finance ticker symbol
        short_window : int, default=50
            Window size for short moving average
        long_window : int, default=200
            Window size for long moving average
        start_date : str, optional
            Start date for data fetching (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for data fetching (format: 'YYYY-MM-DD')
        period : str, default="1y"
            Period to fetch data for if no dates provided
        parameters : Dict[str, Any], optional
            Additional strategy parameters
        """
        super().__init__(parameters)
        
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
            
        self.short_window = short_window
        self.long_window = long_window
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        
        if ticker is not None:
            self.data = self._fetch_data()
        else:
            self.data = None

    def _fetch_data(self) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        return YahooDataFetcher.fetch_data(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            period=self.period
        )

    def generate_signals(self, data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate trading signals based on moving average crossover.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Price data. If not provided, uses the data fetched during initialization.
            
        Returns:
        --------
        pd.Series
            Series containing trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data provided. Either provide data or initialize with a ticker.")
            data = self.data

        if 'close' not in data.columns:
            raise ValueError("Data must contain a 'close' price column")

        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()

        signals = pd.Series(0, index=data.index)
        
        signals[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1  # Buy signal
        signals[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1  # Sell signal

        return signals

    def get_ticker_info(self) -> Dict:
        """
        Get information about the ticker from Yahoo Finance.
        
        Returns:
        --------
        Dict
            Dictionary containing ticker information
        """
        if self.ticker is None:
            raise ValueError("No ticker provided")
        return YahooDataFetcher.get_ticker_info(self.ticker) 