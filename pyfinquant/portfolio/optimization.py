"""
Portfolio optimization module implementing Modern Portfolio Theory (MPT)
and other optimization strategies.
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory."""
    
    def __init__(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        constraints: Optional[List[dict]] = None
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            returns: Array of asset returns (n_samples, n_assets)
            risk_free_rate: Risk-free rate (default: 0.0)
            constraints: Additional optimization constraints
        """
        self.returns = np.array(returns)
        self.n_assets = self.returns.shape[1]
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or []
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = np.mean(self.returns, axis=0)
        self.cov_matrix = np.cov(self.returns, rowvar=False)
        
        # Default constraints
        self._add_default_constraints()
    
    def _add_default_constraints(self):
        """Add default constraints for portfolio optimization."""
        # Weights sum to 1
        self.constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        })
        
        # Non-negative weights (no short selling)
        self.constraints.append({
            'type': 'ineq',
            'fun': lambda x: x
        })
    
    def _portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio statistics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (returns, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def minimum_volatility(self) -> Tuple[np.ndarray, dict]:
        """
        Find the minimum volatility portfolio.
        
        Returns:
            Tuple of (optimal weights, optimization results)
        """
        n_assets = self.returns.shape[1]
        args = (self.cov_matrix,)
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        result = minimize(
            portfolio_vol,
            x0=np.array([1.0/n_assets] * n_assets),
            method='SLSQP',
            constraints=self.constraints,
            bounds=tuple((0, 1) for _ in range(n_assets))
        )
        
        return result.x, {
            'success': result.success,
            'message': result.message,
            'portfolio_return': np.sum(self.mean_returns * result.x),
            'portfolio_volatility': portfolio_vol(result.x)
        }
    
    def maximum_sharpe(self) -> Tuple[np.ndarray, dict]:
        """
        Find the maximum Sharpe ratio portfolio.
        
        Returns:
            Tuple of (optimal weights, optimization results)
        """
        n_assets = self.returns.shape[1]
        args = (self.mean_returns, self.cov_matrix, self.risk_free_rate)
        
        def neg_sharpe_ratio(weights):
            p_ret, p_std, p_sr = self._portfolio_stats(weights)
            return -p_sr
        
        result = minimize(
            neg_sharpe_ratio,
            x0=np.array([1.0/n_assets] * n_assets),
            method='SLSQP',
            constraints=self.constraints,
            bounds=tuple((0, 1) for _ in range(n_assets))
        )
        
        return result.x, {
            'success': result.success,
            'message': result.message,
            'portfolio_return': np.sum(self.mean_returns * result.x),
            'portfolio_volatility': np.sqrt(np.dot(result.x.T, np.dot(self.cov_matrix, result.x))),
            'sharpe_ratio': -neg_sharpe_ratio(result.x)
        }
    
    def efficient_frontier(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate efficient frontier points.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Tuple of (returns, volatilities) for efficient frontier
        """
        min_vol_weights, _ = self.minimum_volatility()
        max_sharpe_weights, _ = self.maximum_sharpe()
        
        min_ret = np.sum(self.mean_returns * min_vol_weights)
        max_ret = np.sum(self.mean_returns * max_sharpe_weights)
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        efficient_portfolios = []
        
        for target in target_returns:
            constraints = self.constraints + [{
                'type': 'eq',
                'fun': lambda x: np.sum(self.mean_returns * x) - target
            }]
            
            result = minimize(
                lambda x: np.sqrt(np.dot(x.T, np.dot(self.cov_matrix, x))),
                x0=np.array([1.0/self.n_assets] * self.n_assets),
                method='SLSQP',
                constraints=constraints,
                bounds=tuple((0, 1) for _ in range(self.n_assets))
            )
            
            if result.success:
                efficient_portfolios.append(result.x)
        
        efficient_portfolios = np.array(efficient_portfolios)
        returns = np.sum(self.mean_returns * efficient_portfolios, axis=1)
        volatilities = np.sqrt(np.sum(np.dot(efficient_portfolios, self.cov_matrix) * 
                                    efficient_portfolios, axis=1))
        
        return returns, volatilities 