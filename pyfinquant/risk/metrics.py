"""
Risk metrics module providing Value at Risk (VaR), Expected Shortfall (ES),
and Maximum Drawdown calculations.
"""

import numpy as np
from typing import Union, Optional, Tuple
from ..utils.helpers import calculate_returns


class ValueAtRisk:
    """Calculate Value at Risk using various methods."""
    
    def __init__(self, returns: np.ndarray):
        """
        Initialize ValueAtRisk calculator.
        
        Args:
            returns: Array of historical returns
        """
        self.returns = np.array(returns)
    
    def historical(self, confidence_level: float = 0.95) -> float:
        """
        Calculate historical VaR.
        
        Args:
            confidence_level: Confidence level (default: 0.95 for 95% VaR)
            
        Returns:
            Historical VaR at specified confidence level
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        return -np.percentile(self.returns, 100 * (1 - confidence_level))
    
    def parametric(self, confidence_level: float = 0.95) -> float:
        """
        Calculate parametric VaR assuming normal distribution.
        
        Args:
            confidence_level: Confidence level (default: 0.95 for 95% VaR)
            
        Returns:
            Parametric VaR at specified confidence level
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        z_score = -np.percentile(np.random.standard_normal(10000), 
                               100 * (1 - confidence_level))
        return -(np.mean(self.returns) + z_score * np.std(self.returns))


class ExpectedShortfall:
    """Calculate Expected Shortfall (Conditional VaR)."""
    
    def __init__(self, returns: np.ndarray):
        """
        Initialize ExpectedShortfall calculator.
        
        Args:
            returns: Array of historical returns
        """
        self.returns = np.array(returns)
    
    def historical(self, confidence_level: float = 0.95) -> float:
        """
        Calculate historical Expected Shortfall.
        
        Args:
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Historical Expected Shortfall at specified confidence level
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        var = ValueAtRisk(self.returns).historical(confidence_level)
        return -np.mean(self.returns[self.returns <= -var])


class MaxDrawdown:
    """Calculate Maximum Drawdown from a series of prices or returns."""
    
    @staticmethod
    def from_prices(prices: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate Maximum Drawdown from price series.
        
        Args:
            prices: Array of asset prices
            
        Returns:
            Tuple of (maximum drawdown, peak index, trough index)
        """
        prices = np.array(prices)
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        max_dd = drawdown.min()
        end_idx = drawdown.argmin()
        peak_idx = prices[:end_idx].argmax()
        
        return float(max_dd), int(peak_idx), int(end_idx)
    
    @staticmethod
    def from_returns(returns: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate Maximum Drawdown from returns series.
        
        Args:
            returns: Array of returns
            
        Returns:
            Tuple of (maximum drawdown, peak index, trough index)
        """
        prices = (1 + np.array(returns)).cumprod()
        return MaxDrawdown.from_prices(prices) 