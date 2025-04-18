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
            'close': prices,  # Using lowercase to match strategy requirements
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.random.normal(0.001, 0.001, n_days)),
            'low': prices * (1 - np.random.normal(0.001, 0.001, n_days))
        }, index=dates)
    
    return data if n_assets > 1 else data['Asset']


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
    
    # Verify results structure
    assert isinstance(results, dict)
    assert all(key in results for key in [
        'returns', 'equity_curve', 'positions', 'total_return',
        'sharpe_ratio', 'max_drawdown', 'num_trades', 'win_rate'
    ])
    
    # Verify data types and constraints
    assert isinstance(results['returns'], pd.Series)
    assert isinstance(results['equity_curve'], pd.Series)
    assert isinstance(results['positions'], pd.Series)
    assert isinstance(results['total_return'], float)
    assert isinstance(results['sharpe_ratio'], float)
    assert isinstance(results['max_drawdown'], float)
    assert results['max_drawdown'] <= 0


def test_multi_asset_backtest():
    """Test backtest with multiple assets."""
    data = create_sample_data(n_assets=3)
    strategy = MovingAverageCrossover(short_window=5, long_window=20)
    
    backtest = Backtest(
        data=data,
        strategy=strategy,
        initial_capital=100000,
        commission=0.001,
        slippage=0.0001,
        position_sizing='fixed',
        risk_per_trade=0.02,
        max_position_size=0.2
    )
    
    results = backtest.run()
    
    # Verify multi-asset results
    assert isinstance(results, dict)
    assert all(key in results for key in [
        'returns', 'equity_curve', 'positions', 'total_return'
    ])


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


def test_risk_management(sample_data, strategy):
    """Test risk management features."""
    # Test with stop loss
    backtest_sl = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        stop_loss=0.05
    )
    results_sl = backtest_sl.run()
    
    # Test with take profit
    backtest_tp = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        take_profit=0.1
    )
    results_tp = backtest_tp.run()
    
    # Test with trailing stop
    backtest_ts = Backtest(
        data=sample_data,
        strategy=strategy,
        initial_capital=100000,
        trailing_stop=0.05
    )
    results_ts = backtest_ts.run()
    
    # Verify risk management affects positions
    assert not np.array_equal(
        results_sl['positions'].values,
        results_tp['positions'].values
    )
    assert not np.array_equal(
        results_sl['positions'].values,
        results_ts['positions'].values
    )


def test_edge_cases():
    """Test edge cases and error handling."""
    strategy = MovingAverageCrossover()
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        Backtest(
            data=pd.DataFrame(),
            strategy=strategy,
            initial_capital=100000
        )
    
    # Test with invalid position sizing
    with pytest.raises(ValueError):
        Backtest(
            data=create_sample_data(),
            strategy=strategy,
            initial_capital=100000,
            position_sizing='invalid_method'
        )
    
    # Test with zero initial capital
    with pytest.raises(ValueError):
        Backtest(
            data=create_sample_data(),
            strategy=strategy,
            initial_capital=0
        )
    
    # Test with negative commission
    with pytest.raises(ValueError):
        Backtest(
            data=create_sample_data(),
            strategy=strategy,
            initial_capital=100000,
            commission=-0.01
        )


if __name__ == "__main__":
    pytest.main([__file__, '-v']) 
