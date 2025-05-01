import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional, Tuple
from ..strategies.base import Strategy
from ..data_fetcher import YahooDataFetcher
from ..utils.helpers import calculate_returns
from ..portfolio.metrics import SharpeRatio, SortinoRatio, InformationRatio


class Backtest:
    """
    Backtesting engine for trading strategies.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: Optional[pd.DataFrame] = None,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
        position_sizing: str = 'fixed',
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.2,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[float] = None
    ):
        """
        Initialize the backtesting engine.
        
        Parameters:
        -----------
        strategy : Strategy
            Trading strategy to backtest
        data : pd.DataFrame, optional
            Price data. If not provided, will use ticker to fetch data.
        ticker : str, optional
            Yahoo Finance ticker symbol
        start_date : str, optional
            Start date for data fetching (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for data fetching (format: 'YYYY-MM-DD')
        period : str, default="1y"
            Period to fetch data for if no dates provided
        initial_capital : float, default=100000.0
            Initial capital for backtesting
        commission : float, default=0.001
            Commission rate per trade
        slippage : float, default=0.0001
            Slippage rate per trade
        position_sizing : str, default='fixed'
            Position sizing method ('fixed', 'risk_based', 'volatility_based')
        risk_per_trade : float, default=0.02
            Risk per trade as a fraction of capital
        max_position_size : float, default=0.2
            Maximum position size as a fraction of capital
        stop_loss : float, optional
            Stop loss level as a fraction
        take_profit : float, optional
            Take profit level as a fraction
        trailing_stop : float, optional
            Trailing stop level as a fraction
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        
        self._validate_parameters()
        
        if data is not None:
            if isinstance(data, pd.DataFrame) and data.empty:
                raise ValueError("Data cannot be empty")
            self.data = data
        elif ticker is not None:
            self.data = YahooDataFetcher.fetch_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                period=period
            )
        else:
            raise ValueError("Either data or ticker must be provided")

    def _validate_parameters(self):
        """Validate backtest parameters."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission < 0:
            raise ValueError("Commission must be non-negative")
        if self.slippage < 0:
            raise ValueError("Slippage must be non-negative")
        if self.position_sizing not in ['fixed', 'risk_based', 'volatility_based']:
            raise ValueError("Invalid position sizing method")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 1:
            raise ValueError("Risk per trade must be between 0 and 1")
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
        if self.stop_loss is not None and (self.stop_loss <= 0 or self.stop_loss >= 1):
            raise ValueError("Stop loss must be between 0 and 1")
        if self.take_profit is not None and (self.take_profit <= 0 or self.take_profit >= 1):
            raise ValueError("Take profit must be between 0 and 1")
        if self.trailing_stop is not None and (self.trailing_stop <= 0 or self.trailing_stop >= 1):
            raise ValueError("Trailing stop must be between 0 and 1")

    def _calculate_position_sizes(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        prices_array = prices.values
        signals_array = signals.reindex(prices.index, method='ffill').fillna(0).values
        num_assets = prices_array.shape[1]
        target_positions = np.zeros_like(prices_array)
        
        if self.position_sizing == 'fixed':
            if 'positions' in signals.columns:
                target_weights = signals['positions'].values if isinstance(signals['positions'], pd.Series) else signals['positions'].values
                for col_idx in range(num_assets):
                    allocated_capital = self.initial_capital / num_assets
                    target_positions[:, col_idx] = target_weights * allocated_capital / prices_array[:, col_idx]
            elif 'signal' in signals.columns:
                signal_col = signals['signal'].values if isinstance(signals['signal'], pd.Series) else signals['signal'].values
                for col_idx in range(num_assets):
                    allocated_capital = self.initial_capital / num_assets
                    target_positions[:, col_idx] = signal_col * self.max_position_size * allocated_capital / prices_array[:, col_idx]
            else:
                print("Warning: 'fixed' sizing chosen but signal format ('positions' or 'signal' column) unclear. Defaulting to zero positions.")
        
        elif self.position_sizing == 'risk_based':
            if self.stop_loss is None:
                raise ValueError("Stop loss must be defined for risk-based position sizing.")
            
            for col_idx in range(num_assets):
                if 'signal' in signals.columns and isinstance(signals['signal'], pd.Series):
                    signal = signals['signal'].values
                else:
                    signal = signals_array[:, col_idx]
                
                price_col = prices_array[:, col_idx]
                
                stop_price = np.where(signal > 0, 
                                    price_col * (1 - self.stop_loss),
                                    np.where(signal < 0, 
                                            price_col * (1 + self.stop_loss), 
                                            np.nan))
                
                risk_per_share = np.abs(price_col - stop_price)
                risk_per_share = np.where(risk_per_share == 0, np.nan, risk_per_share)
                risk_per_share = np.where(signal == 0, np.nan, risk_per_share)
                
                capital_at_risk = (self.initial_capital / num_assets) * self.risk_per_trade
                position_size_shares = capital_at_risk / risk_per_share
                position_size_shares = position_size_shares * np.sign(signal)
                
                max_capital_per_asset = (self.initial_capital / num_assets) * self.max_position_size
                max_shares = np.where(price_col > 0, max_capital_per_asset / price_col, 0)
                position_size_shares = np.clip(position_size_shares, -max_shares, max_shares)
                
                target_positions[:, col_idx] = position_size_shares
        
        elif self.position_sizing == 'volatility_based':
            returns = np.diff(prices_array, axis=0) / prices_array[:-1]
            volatility = np.std(returns, axis=0) * np.sqrt(252)  
            target_risk_alloc = self.risk_per_trade
            
            for col_idx in range(num_assets):
                if 'signal' in signals.columns and isinstance(signals['signal'], pd.Series):
                    signal = signals['signal'].values
                else:
                    signal = signals_array[:, col_idx]
                
                price_col = prices_array[:, col_idx]
                vol_col = volatility[col_idx]
                
                position_size_frac_equity = np.where(vol_col > 0, target_risk_alloc / vol_col, 0)
                allocated_capital = self.initial_capital / num_assets
                position_size_shares = np.where(price_col > 0, position_size_frac_equity * allocated_capital / price_col, 0)
                position_size_shares *= np.sign(signal)
                
                max_capital_per_asset = (self.initial_capital / num_assets) * self.max_position_size
                max_shares = np.where(price_col > 0, max_capital_per_asset / price_col, 0)
                position_size_shares = np.clip(position_size_shares, -max_shares, max_shares)
                
                target_positions[:, col_idx] = position_size_shares
        
        return pd.DataFrame(target_positions, index=prices.index, columns=prices.columns).fillna(0)

    def _apply_risk_management(self, positions: np.ndarray, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices)
        positions = np.asarray(positions)

        if positions.ndim > 1 or prices.ndim > 1:
            raise ValueError("Multi-asset risk management not yet fully implemented in this refactored version.")

        entry_price_array = np.full_like(prices, np.nan)
        adjusted_positions = np.zeros_like(prices)
        trailing_high_array = np.full_like(prices, -np.inf)
        trailing_low_array = np.full_like(prices, np.inf)

        entry_mask = (positions != 0) & (np.concatenate([[0], positions[:-1]]) == 0)
        entry_price_array[entry_mask] = prices[entry_mask]
        adjusted_positions[entry_mask] = positions[entry_mask]

        trailing_high_array = np.maximum.accumulate(prices)
        trailing_low_array = np.minimum.accumulate(prices)

        if self.stop_loss is not None:
            stop_loss_mask = (adjusted_positions > 0) & (prices <= entry_price_array * (1 - self.stop_loss))
            adjusted_positions[stop_loss_mask] = 0

        if self.take_profit is not None:
            take_profit_mask = (adjusted_positions > 0) & (prices >= entry_price_array * (1 + self.take_profit))
            adjusted_positions[take_profit_mask] = 0

        if self.trailing_stop is not None:
            trailing_stop_mask = (adjusted_positions > 0) & (prices <= trailing_high_array * (1 - self.trailing_stop))
            adjusted_positions[trailing_stop_mask] = 0

        return adjusted_positions

    def _calculate_returns(self, positions: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(positions)
        prices = np.asarray(prices)

        if positions.ndim > 1 or prices.ndim > 1:
             raise ValueError("Multi-asset calculation not yet fully implemented in this refactored version.")

        equity = np.full(positions.shape[0], np.nan)
        equity[0] = self.initial_capital
        cash = np.full(positions.shape[0], np.nan)
        cash[0] = self.initial_capital
        current_holdings = np.zeros(positions.shape[0])

        position_changes = np.diff(positions, prepend=0)

        for i in range(1, len(prices)):
            mkt_value_holdings_start = current_holdings[i-1] * prices[i]

            trade_value = position_changes[i] * prices[i]
            transaction_costs = abs(trade_value) * (self.commission + self.slippage)

            cash[i] = cash[i-1] - trade_value - transaction_costs

            current_holdings[i] = current_holdings[i-1] + position_changes[i]

            equity[i] = cash[i] + current_holdings[i] * prices[i]

            if equity[i-1] <= 0:
                equity[i] = 0
                cash[i] = 0
                current_holdings[i] = 0

        returns = np.full(equity.shape[0], np.nan)
        valid_equity_idx = np.where(equity[:-1] > 0)[0] + 1
        returns[valid_equity_idx] = (equity[valid_equity_idx] / equity[valid_equity_idx - 1]) - 1
        returns[0] = 0

        returns = np.nan_to_num(returns, nan=0.0)

        return equity, returns

    def _calculate_equity_curve(self, returns: np.ndarray) -> np.ndarray:
        cumulative_returns = np.cumprod(1 + returns)
        equity_curve = self.initial_capital * cumulative_returns
        return equity_curve

    def _calculate_drawdowns(self, equity_curve: np.ndarray) -> Dict[str, float]:
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        drawdown_duration = np.argmax(running_max) - np.argmin(drawdowns)
        avg_drawdown = np.mean(drawdowns[drawdowns < 0])

        return {
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'avg_drawdown': avg_drawdown
        }

    def _calculate_trade_statistics(self, positions: pd.Series, equity_curve: pd.Series, prices: pd.Series) -> Dict[str, float]:
        positions_array = positions.fillna(0).values
        equity_curve_array = equity_curve.reindex(positions.index).values
        prices_array = prices.reindex(positions.index).values
        
        position_changes = np.diff(positions_array, prepend=0)
        entry_mask = (position_changes != 0) & (positions_array != 0)
        exit_mask = (position_changes != 0) & (np.roll(positions_array, 1) != 0)
        
        entry_indices = np.where(entry_mask)[0]
        exit_indices = np.where(exit_mask)[0]
        
        if len(entry_indices) > len(exit_indices):
            exit_indices = np.append(exit_indices, len(positions_array) - 1)
        
        trades = []
        for entry_idx, exit_idx in zip(entry_indices, exit_indices):
            trade = {
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_price': prices_array[entry_idx],
                'exit_price': prices_array[exit_idx],
                'position_size': positions_array[entry_idx],
                'pnl': equity_curve_array[exit_idx] - equity_curve_array[entry_idx]
            }
            trades.append(trade)
        
        if not trades:
            return {
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_win_pnl': 0.0,
                'avg_loss_pnl': 0.0,
                'profit_factor': np.inf
            }
        
        pnl_values = np.array([t['pnl'] for t in trades])
        winning_trades = np.sum(pnl_values > 0)
        losing_trades = np.sum(pnl_values < 0)
        gross_profit = np.sum(pnl_values[pnl_values > 0])
        gross_loss = np.abs(np.sum(pnl_values[pnl_values < 0]))
        
        num_trades = len(trades)
        win_rate = winning_trades / num_trades
        avg_trade_pnl = np.mean(pnl_values)
        avg_win_pnl = np.mean(pnl_values[pnl_values > 0]) if winning_trades > 0 else 0.0
        avg_loss_pnl = np.mean(pnl_values[pnl_values < 0]) if losing_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        
        return {
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win_pnl': avg_win_pnl,
            'avg_loss_pnl': avg_loss_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': gross_profit - gross_loss,
            'profit_factor': profit_factor
        }

    def run(self) -> Dict[str, Union[pd.Series, float]]:
        """
        Run the backtest.
        
        Returns:
        --------
        Dict[str, Union[pd.Series, float]]
            Dictionary containing backtest results
        """
        signals = self.strategy.generate_signals(self.data)

        positions = pd.Series(0, index=self.data.index)
        portfolio_value = pd.Series(self.initial_capital, index=self.data.index)
        
        if self.position_sizing == 'fixed':
            position_sizes = pd.Series(self.max_position_size, index=self.data.index)
        elif self.position_sizing == 'risk_based':
            position_sizes = pd.Series(self.risk_per_trade, index=self.data.index)
        else:
            volatility = self.data['close'].pct_change().rolling(window=20).std()
            position_sizes = (self.risk_per_trade / volatility).clip(0, self.max_position_size)
        
        current_position = 0
        entry_price = 0
        stop_price = 0
        take_profit_price = 0
        
        for i in range(1, len(self.data)):
            price = self.data['close'].iloc[i]
            
            if current_position != 0:
                if self.stop_loss is not None and price <= stop_price:
                    signals.iloc[i] = -current_position
                elif self.take_profit is not None and price >= take_profit_price:
                    signals.iloc[i] = -current_position
                elif self.trailing_stop is not None:
                    if current_position > 0 and price > entry_price * (1 + self.trailing_stop):
                        stop_price = price * (1 - self.trailing_stop)
                    elif current_position < 0 and price < entry_price * (1 - self.trailing_stop):
                        stop_price = price * (1 + self.trailing_stop)
            
            if signals.iloc[i] != 0 and current_position == 0:
                current_position = signals.iloc[i]
                entry_price = price
                positions.iloc[i] = current_position * position_sizes.iloc[i]
                
                if self.stop_loss is not None:
                    stop_price = entry_price * (1 - self.stop_loss) if current_position > 0 else entry_price * (1 + self.stop_loss)
                if self.take_profit is not None:
                    take_profit_price = entry_price * (1 + self.take_profit) if current_position > 0 else entry_price * (1 - self.take_profit)
            elif signals.iloc[i] == -current_position:
                positions.iloc[i] = 0
                current_position = 0
            else:
                positions.iloc[i] = positions.iloc[i-1]
            
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + positions.iloc[i] * (price / self.data['close'].iloc[i-1] - 1))
            
            if positions.iloc[i] != positions.iloc[i-1]:
                portfolio_value.iloc[i] *= (1 - self.commission - self.slippage)
        
        returns = portfolio_value.pct_change()
        total_return = (portfolio_value.iloc[-1] / self.initial_capital) - 1
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        trades = positions.diff().abs() > 0
        num_trades = trades.sum()
        winning_trades = (returns[trades] > 0).sum()
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        return {
            'returns': returns,
            'equity_curve': portfolio_value,
            'positions': positions,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate
        }

    def plot_results(self) -> None:
        if self.results is None:
            raise ValueError("Run backtest first using run() method")

        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        ax1.plot(self.results['equity_curve'], label='Strategy')
        if self.benchmark is not None:
            aligned_benchmark = self.benchmark.reindex(self.results['equity_curve'].index).fillna(0)
            benchmark_curve = self.initial_capital * (1 + aligned_benchmark).cumprod()
            ax1.plot(benchmark_curve, label='Benchmark')
        ax1.set_ylabel('Equity')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.results['positions'], label='Position')
        ax2.set_ylabel('Position')
        ax2.grid(True)

        equity = self.results['equity_curve']
        running_max = equity.cummax()
        drawdown = pd.Series(np.nan, index=equity.index)
        if running_max.iloc[0] > 0: 
            drawdown = (equity - running_max) / running_max
        else:
            first_positive_idx = (running_max > 0).idxmax()
            if first_positive_idx:
                 drawdown.loc[first_positive_idx:] = (equity.loc[first_positive_idx:] - running_max.loc[first_positive_idx:]) / running_max.loc[first_positive_idx:]

        drawdown = drawdown.fillna(0) 

        ax3.fill_between(equity.index, drawdown, 0, color='red', alpha=0.3)
        ax3.set_ylabel('Drawdown')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
