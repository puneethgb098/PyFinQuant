import pytest
import pandas as pd
import numpy as np
from pyfinquant.strategies.moving_average import MovingAverageCrossover

def test_moving_average_strategy_initialization():
    """Test moving average strategy initialization."""
    strategy = MovingAverageCrossover(short_window=10, long_window=20)
    
    assert strategy.short_window == 10
    assert strategy.long_window == 20
    assert isinstance(strategy.parameters, dict)

def test_moving_average_strategy_generate_signals():
    """Test signal generation for moving average strategy."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = np.linspace(100, 200, 100) 
    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    strategy = MovingAverageCrossover(short_window=10, long_window=20)
    signals = strategy.generate_signals(data)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(data)
    assert all(signal in [-1, 0, 1] for signal in signals)
    
    assert signals.sum() > 0

def test_moving_average_strategy_invalid_windows():
    """Test strategy with invalid window parameters."""
    with pytest.raises(ValueError):
        MovingAverageCrossover(short_window=0, long_window=20)
    
    with pytest.raises(ValueError):
        MovingAverageCrossover(short_window=10, long_window=5)
    
    with pytest.raises(ValueError):
        MovingAverageCrossover(short_window=-10, long_window=20)

def test_moving_average_strategy_edge_cases():
    """Test strategy with edge cases."""
    data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1000, 1000, 1000, 1000]
    })
    
    strategy = MovingAverageCrossover(short_window=2, long_window=3)
    signals = strategy.generate_signals(data)
    
    assert len(signals) == len(data)
    assert signals.iloc[0] == 0  
    
    # Test with constant price
    data = pd.DataFrame({
        'close': [100] * 100,
        'volume': [1000] * 100
    })
    
    strategy = MovingAverageCrossover(short_window=10, long_window=20)
    signals = strategy.generate_signals(data)
    
    assert all(signal == 0 for signal in signals)