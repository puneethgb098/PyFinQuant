"""
Tests for the backtesting engine.
"""

import numpy as np
import pandas as pd
import pytest
from pyfinquant.backtest.backtest import Backtest
from pyfinquant.strategies.moving_average import MovingAverageCrossover


def create_sample_data(n_days=365, n_assets=1, trend=0.1, volatility=0.2):
    """Create sample price data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    data = {}
    
    for i in range(n_assets):
        asset_name = f'Asset_{i+1}' if n_assets > 1 else 'Asset'
        returns = np.random.normal(trend/252, volatility/np.sqrt(252), n_days)
        prices = 100 * (1 + returns).cumprod()
        data[asset_name] = pd.DataFrame({
            'close': prices, 
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.random.normal(0.001, 0.001, n_days)),
            'low': prices * (1 - np.random.normal(0.001, 0.001, n_days))
        }, index=dates)
    
    if n_assets > 1:
        combined_data = pd.concat(data.values(), axis=1, keys=data.keys())
        return combined_data
    return data['Asset']


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for tests."""
    return create_sample_data()


@pytest.fixture
def strategy():
    """Fixture to provide strategy instance for tests."""
    return MovingAverageCrossover(short_window=5, long_window=20)


def test_backtest_initialization(sample_data, strategy):
    """Test backtest initialization with valid parameters."""
    backtest = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001
    )
    assert backtest.initial_capital == 100000
    assert backtest.commission == 0.001
    assert backtest.slippage == 0.0001


def test_single_asset_backtest(sample_data, strategy):
    """Test backtest with single asset."""
    backtest = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        position_sizing='fixed',
        risk_per_trade=0.02,
        max_position_size=0.2,
        stop_loss=0.05,
        take_profit=0.1
    )
    
    results = backtest.run()
    
    assert isinstance(results, dict)
    assert all(key in results for key in [
        'returns', 'equity_curve', 'positions', 'total_return',
        'sharpe_ratio', 'max_drawdown', 'num_trades', 'win_rate'
    ])
    
    assert isinstance(results['returns'], pd.Series)
    assert isinstance(results['equity_curve'], pd.Series)
    assert isinstance(results['positions'], pd.Series)
    assert isinstance(results['total_return'], float)
    assert isinstance(results['sharpe_ratio'], float)
    assert isinstance(results['max_drawdown'], float)
    assert results['max_drawdown'] <= 0


def test_position_sizing_methods(sample_data, strategy):
    """Test different position sizing methods."""
    sizing_methods = ['fixed', 'risk_based', 'volatility_based']
    
    for method in sizing_methods:
        backtest = Backtest(
            data=sample_data,
            strategy=strategy,
            initial_capital=100000,
            position_sizing=method,
            risk_per_trade=0.02,
            stop_loss=0.05 if method == 'risk_based' else None
        )
        results = backtest.run()
        assert 'positions' in results
        assert isinstance(results['positions'], pd.Series)
        
        positions = results['positions'].values
        prices = sample_data['close'].values
        
        if method == 'fixed':
            max_position = 100000 * 0.2 / prices  
            assert np.all(np.abs(positions) <= max_position)
            
        elif method == 'risk_based':
            max_position_value = 100000 * 0.2
            max_position_shares = np.where(prices > 0, max_position_value / prices, 0) 
            risk_per_share = np.abs(prices - prices * (1 - 0.05))
            expected_positions = max_position_value / risk_per_share
            expected_positions = np.clip(expected_positions, -max_position, max_position) 
            assert np.allclose(np.abs(positions), np.abs(expected_positions), rtol=1e-5)
            
        elif method == 'volatility_based':
            max_position = 100000 * 0.2 / prices 
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)
            target_risk_alloc = 0.02
            expected_positions = (target_risk_alloc / volatility) * 100000 / prices
            expected_positions = np.clip(expected_positions, -max_position, max_position)
            assert np.allclose(np.abs(positions), np.abs(expected_positions), rtol=1e-5)


def test_risk_management(sample_data, strategy):
    """Test risk management features."""
    backtest_sl = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        stop_loss=0.05
    )
    results_sl = backtest_sl.run()
    
    backtest_tp = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        take_profit=0.1
    )
    results_tp = backtest_tp.run()
    
    backtest_ts = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        trailing_stop=0.05
    )
    results_ts = backtest_ts.run()
    
    assert not np.array_equal(
        results_sl['positions'].values,
        results_tp['positions'].values
    )
    assert not np.array_equal(
        results_sl['positions'].values,
        results_ts['positions'].values
    )
    
    prices = sample_data['close'].values
    positions = results_sl['positions'].values
    entry_prices = np.full_like(prices, np.nan)
    entry_mask = (positions != 0) & (np.concatenate([[0], positions[:-1]]) == 0)
    entry_prices[entry_mask] = prices[entry_mask]
    
    stop_loss_mask = (positions > 0) & (prices <= entry_prices * (1 - 0.05))
    assert np.all(positions[stop_loss_mask] == 0)
    
    trailing_highs = np.maximum.accumulate(prices)
    trailing_stop_mask = (positions > 0) & (prices <= trailing_highs * (1 - 0.05))
    assert np.all(results_ts['positions'].values[trailing_stop_mask] == 0)


def test_edge_cases():
    """Test edge cases and error handling."""
    strategy = MovingAverageCrossover()
    
    with pytest.raises(ValueError):
        Backtest(
            data=pd.DataFrame(),
            strategy=strategy,
            initial_capital=100000
        )
    
    with pytest.raises(ValueError):
        Backtest(
            data=create_sample_data(),
            strategy=strategy,
            initial_capital=100000,
            position_sizing='invalid_method'
        )
    
    with pytest.raises(ValueError):
        Backtest(
            data=create_sample_data(),
            strategy=strategy,
            initial_capital=0
        )
    
    with pytest.raises(ValueError):
        Backtest(
            data=create_sample_data(),
            strategy=strategy,
            initial_capital=100000,
            commission=-0.01
        )


def test_trade_statistics(sample_data, strategy):
    """Test trade statistics calculation with vectorized operations."""
    backtest = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001
    )
    results = backtest.run()
    
    stats = backtest._calculate_trade_statistics(
        results['positions'],
        results['equity_curve'],
        sample_data['close']
    )
    
    assert isinstance(stats, dict)
    assert all(key in stats for key in [
        'num_trades', 'winning_trades', 'losing_trades',
        'win_rate', 'avg_trade_pnl', 'avg_win_pnl',
        'avg_loss_pnl', 'profit_factor'
    ])
    
    positions = results['positions'].values
    position_changes = np.diff(positions, prepend=0)
    entry_mask = (position_changes != 0) & (positions != 0)
    exit_mask = (position_changes != 0) & (np.roll(positions, 1) != 0)
    
    assert stats['num_trades'] == np.sum(entry_mask)
    
    if stats['num_trades'] > 0:
        assert 0 <= stats['win_rate'] <= 1
        assert stats['win_rate'] == stats['winning_trades'] / stats['num_trades']
    
    if stats['losing_trades'] > 0:
        assert stats['profit_factor'] >= 0
        assert stats['profit_factor'] == stats['gross_profit'] / abs(stats['gross_loss'])


if __name__ == "__main__":
    pytest.main([__file__, '-v']) 
