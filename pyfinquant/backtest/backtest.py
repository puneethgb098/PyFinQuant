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
        
        # Validate parameters
        self._validate_parameters()
        
        # Get data
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
        target_positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0)
        num_assets = len(prices.columns)

        signals = signals.reindex(prices.index, method='ffill').fillna(0)

        if self.position_sizing == 'fixed':
            if 'positions' in signals.columns:
                target_weights = signals['positions']
                for col in prices.columns:
                    weight_col = target_weights if isinstance(target_weights, pd.Series) else target_weights[col]
                    allocated_capital = self.initial_capital / num_assets
                    target_positions[col] = weight_col * allocated_capital / prices[col]

            elif 'signal' in signals.columns:
                 signal_col = signals['signal']
                 for col in prices.columns:
                    current_signal = signal_col if isinstance(signal_col, pd.Series) else signals.get(col, pd.Series(0, index=prices.index))
                    allocated_capital = self.initial_capital / num_assets
                    target_positions[col] = current_signal * self.max_position_size * allocated_capital / prices[col]
            else:
                 print("Warning: 'fixed' sizing chosen but signal format ('positions' or 'signal' column) unclear. Defaulting to zero positions.")

            return target_positions.fillna(0)

        elif self.position_sizing == 'risk_based':
            if self.stop_loss is None:
                raise ValueError("Stop loss must be defined for risk-based position sizing.")

            for col in prices.columns:
                if 'signal' in signals.columns and isinstance(signals['signal'], pd.Series):
                     signal = signals['signal']
                elif col in signals.columns:
                     signal = signals[col]
                else:
                     print(f"Warning: Cannot find signal for asset {col} in risk-based sizing. Skipping.")
                     continue

                price_col = prices[col]

                stop_price = np.where(signal > 0, price_col * (1 - self.stop_loss),
                                       np.where(signal < 0, price_col * (1 + self.stop_loss), np.nan))

                risk_per_share = abs(price_col - stop_price)
                risk_per_share = risk_per_share.replace(0, np.nan)
                risk_per_share = np.where(signal == 0, np.nan, risk_per_share)

                capital_at_risk = (self.initial_capital / num_assets) * self.risk_per_trade

                position_size_shares = capital_at_risk / risk_per_share
                position_size_shares *= signal.apply(np.sign)

                max_capital_per_asset = (self.initial_capital / num_assets) * self.max_position_size
                max_shares = np.where(price_col > 0, max_capital_per_asset / price_col, 0)
                position_size_shares = position_size_shares.clip(-max_shares, max_shares)

                target_positions[col] = position_size_shares

            return target_positions.fillna(0)

        elif self.position_sizing == 'volatility_based':
            returns = prices.pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            volatility = volatility.replace(0, np.nan)
            target_risk_alloc = self.risk_per_trade

            for col in prices.columns:
                if 'signal' in signals.columns and isinstance(signals['signal'], pd.Series):
                     signal = signals['signal']
                elif col in signals.columns:
                     signal = signals[col]
                else:
                     print(f"Warning: Cannot find signal for asset {col} in volatility-based sizing. Skipping.")
                     continue

                price_col = prices[col]
                vol_col = volatility[col]

                position_size_frac_equity = np.where(vol_col > 0, target_risk_alloc / vol_col, 0)

                allocated_capital = self.initial_capital / num_assets
                position_size_shares = np.where(price_col > 0, position_size_frac_equity * allocated_capital / price_col, 0)
                position_size_shares *= signal.apply(np.sign)

                max_capital_per_asset = (self.initial_capital / num_assets) * self.max_position_size
                max_shares = np.where(price_col > 0, max_capital_per_asset / price_col, 0)
                position_size_shares = position_size_shares.clip(-max_shares, max_shares)

                target_positions[col] = position_size_shares

            return target_positions.fillna(0)

        else:
            raise ValueError(f"Unknown position sizing method: {self.position_sizing}")

    def _apply_risk_management(self, positions: np.ndarray, prices: np.ndarray) -> np.ndarray:

        if self.stop_loss is None and self.take_profit is None and self.trailing_stop is None:
            return positions

        positions = np.asarray(positions)
        prices = np.asarray(prices)

        if positions.ndim > 1 or prices.ndim > 1:
            raise ValueError("Multi-asset risk management not yet fully implemented in this refactored version.")

        adjusted_positions = positions.copy()
        entry_price = np.nan
        current_position_size = 0.0
        trailing_high = -np.inf
        trailing_low = np.inf

        for i in range(len(prices)):
            if current_position_size == 0:
                if adjusted_positions[i] != 0:
                    entry_price = prices[i]
                    current_position_size = adjusted_positions[i]
                    if current_position_size > 0:
                        trailing_high = prices[i]
                    else:
                        trailing_low = prices[i]

            else:
                exit_signal = False
                exit_reason = ""

                if current_position_size > 0:
                    trailing_high = max(trailing_high, prices[i])
                else:
                    trailing_low = min(trailing_low, prices[i])

                if self.stop_loss is not None and not exit_signal:
                    if current_position_size > 0 and prices[i] <= entry_price * (1 - self.stop_loss):
                        exit_signal = True
                        exit_reason = "Stop Loss (Long)"
                    elif current_position_size < 0 and prices[i] >= entry_price * (1 + self.stop_loss):
                        exit_signal = True
                        exit_reason = "Stop Loss (Short)"

                if self.take_profit is not None and not exit_signal:
                    if current_position_size > 0 and prices[i] >= entry_price * (1 + self.take_profit):
                        exit_signal = True
                        exit_reason = "Take Profit (Long)"
                    elif current_position_size < 0 and prices[i] <= entry_price * (1 - self.take_profit):
                        exit_signal = True
                        exit_reason = "Take Profit (Short)"

                if self.trailing_stop is not None and not exit_signal:
                    if current_position_size > 0 and prices[i] <= trailing_high * (1 - self.trailing_stop):
                        exit_signal = True
                        exit_reason = "Trailing Stop (Long)"
                    elif current_position_size < 0 and prices[i] >= trailing_low * (1 + self.trailing_stop):
                        exit_signal = True
                        exit_reason = "Trailing Stop (Short)"

                if exit_signal:
                    adjusted_positions[i] = 0
                    current_position_size = 0
                    entry_price = np.nan
                    trailing_high = -np.inf
                    trailing_low = np.inf

                elif adjusted_positions[i] * current_position_size <= 0:
                     if adjusted_positions[i] == 0:
                         current_position_size = 0
                         entry_price = np.nan
                         trailing_high = -np.inf
                         trailing_low = np.inf
                     else:
                         entry_price = prices[i]
                         current_position_size = adjusted_positions[i]
                         if current_position_size > 0:
                             trailing_high = prices[i]
                             trailing_low = np.inf
                         else:
                             trailing_low = prices[i]
                             trailing_high = -np.inf
                else:
                     adjusted_positions[i] = current_position_size

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
        trades = []
        entry_idx = None
        entry_price = 0.0
        entry_equity = 0.0
        current_pos = 0.0

        positions = positions.fillna(0)
        equity_curve = equity_curve.reindex(positions.index)
        prices = prices.reindex(positions.index)

        for i in range(len(positions)):
            prev_pos = positions.iloc[i-1] if i > 0 else 0.0
            current_pos = positions.iloc[i]

            if (prev_pos == 0 and current_pos != 0) or (prev_pos * current_pos < 0):
                if prev_pos * current_pos < 0 and entry_idx is not None:
                    exit_idx = i
                    exit_price = prices.iloc[i]
                    exit_equity = equity_curve.iloc[i]
                    trade_pnl = equity_curve.iloc[i] - equity_curve.iloc[entry_idx]
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': exit_idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': prev_pos,
                        'pnl': trade_pnl
                    })
                    entry_idx = None

                if current_pos != 0:
                   entry_idx = i
                   entry_price = prices.iloc[i]
                   entry_equity = equity_curve.iloc[i-1] if i > 0 else self.initial_capital

            elif prev_pos != 0 and current_pos == 0 and entry_idx is not None:
                exit_idx = i
                exit_price = prices.iloc[i]
                exit_equity = equity_curve.iloc[i-1] if i > 0 else equity_curve.iloc[i]
                trade_pnl = exit_equity - entry_equity
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': prev_pos,
                    'pnl': trade_pnl
                })
                entry_idx = None

        if entry_idx is not None and current_pos != 0:
             exit_idx = len(positions) - 1
             exit_price = prices.iloc[-1]
             exit_equity = equity_curve.iloc[-1]
             trade_pnl = exit_equity - entry_equity
             trades.append({
                 'entry_idx': entry_idx,
                 'exit_idx': exit_idx,
                 'entry_price': entry_price,
                 'exit_price': exit_price,
                 'position_size': current_pos,
                 'pnl': trade_pnl
             })

        num_trades = len(trades)
        if num_trades == 0:
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

        pnl_values = [t['pnl'] for t in trades]
        winning_trades = sum(1 for pnl in pnl_values if pnl > 0)
        losing_trades = sum(1 for pnl in pnl_values if pnl < 0)
        gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_values if pnl < 0))

        win_rate = winning_trades / num_trades if num_trades > 0 else 0.0
        avg_trade_pnl = np.mean(pnl_values) if num_trades > 0 else 0.0
        avg_win_pnl = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss_pnl = gross_loss / losing_trades if losing_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

        return {
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win_pnl': avg_win_pnl,
            'avg_loss_pnl': avg_loss_pnl,
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
        # Generate signals
        signals = self.strategy.generate_signals(self.data)

        # Initialize position and portfolio value
        positions = pd.Series(0, index=self.data.index)
        portfolio_value = pd.Series(self.initial_capital, index=self.data.index)
        
        # Calculate position sizes
        if self.position_sizing == 'fixed':
            position_sizes = pd.Series(self.max_position_size, index=self.data.index)
        elif self.position_sizing == 'risk_based':
            position_sizes = pd.Series(self.risk_per_trade, index=self.data.index)
        else:  # volatility_based
            volatility = self.data['close'].pct_change().rolling(window=20).std()
            position_sizes = (self.risk_per_trade / volatility).clip(0, self.max_position_size)
        
        # Run backtest
        current_position = 0
        entry_price = 0
        stop_price = 0
        take_profit_price = 0
        
        for i in range(1, len(self.data)):
            price = self.data['close'].iloc[i]
            
            # Check for stop loss or take profit
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
            
            # Update position
            if signals.iloc[i] != 0 and current_position == 0:
                # Enter new position
                current_position = signals.iloc[i]
                entry_price = price
                positions.iloc[i] = current_position * position_sizes.iloc[i]
                
                # Set stop loss and take profit
                if self.stop_loss is not None:
                    stop_price = entry_price * (1 - self.stop_loss) if current_position > 0 else entry_price * (1 + self.stop_loss)
                if self.take_profit is not None:
                    take_profit_price = entry_price * (1 + self.take_profit) if current_position > 0 else entry_price * (1 - self.take_profit)
            elif signals.iloc[i] == -current_position:
                # Exit position
                positions.iloc[i] = 0
                current_position = 0
            else:
                # Maintain current position
                positions.iloc[i] = positions.iloc[i-1]
            
            # Calculate portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + positions.iloc[i] * (price / self.data['close'].iloc[i-1] - 1))
            
            # Apply commission and slippage
            if positions.iloc[i] != positions.iloc[i-1]:
                portfolio_value.iloc[i] *= (1 - self.commission - self.slippage)
        
        # Calculate returns and performance metrics
        returns = portfolio_value.pct_change()
        total_return = (portfolio_value.iloc[-1] / self.initial_capital) - 1
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calculate number of trades and win rate
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