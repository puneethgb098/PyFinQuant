import yfinance as yf
import pandas as pd
from typing import Union, List, Dict
from datetime import datetime, timedelta


class YahooDataFetcher:
    """
    Class to handle fetching OHLC data from Yahoo Finance.
    """
    
    @staticmethod
    def fetch_data(
        ticker: Union[str, List[str]],
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from Yahoo Finance for given ticker(s).
        
        Parameters:
        -----------
        ticker : Union[str, List[str]]
            Single ticker symbol or list of ticker symbols
        start_date : Union[str, datetime], optional
            Start date for data fetching (format: 'YYYY-MM-DD')
        end_date : Union[str, datetime], optional
            End date for data fetching (format: 'YYYY-MM-DD')
        period : str, optional
            Period to fetch data for (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing OHLC data
        """
        if start_date is None and end_date is None:
            data = yf.download(ticker, period=period)
        else:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                
            if end_date is None:
                end_date = datetime.now()
                
            data = yf.download(ticker, start=start_date, end=end_date)
            
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        data = data[ohlc_columns]
        
        # Rename columns to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]
        
        return data

    @staticmethod
    def get_ticker_info(ticker: str) -> Dict:
        """
        Get information about a ticker from Yahoo Finance.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        Dict
            Dictionary containing ticker information
        """
        ticker_obj = yf.Ticker(ticker)
        return {
            'name': ticker_obj.info.get('longName', ''),
            'sector': ticker_obj.info.get('sector', ''),
            'industry': ticker_obj.info.get('industry', ''),
            'market_cap': ticker_obj.info.get('marketCap', 0),
            'currency': ticker_obj.info.get('currency', ''),
        } 