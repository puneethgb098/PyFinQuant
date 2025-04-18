import pytest
import pandas as pd
import numpy as np
from pyfinquant.strategies.base import Strategy

class TestStrategy(Strategy):    
    def __init__(self, param1: float, param2: float, parameters: dict = None):
        super().__init__(parameters)
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate test signals."""
        return pd.Series(np.random.choice([-1, 0, 1], size=len(data)))

def test_base_strategy_initialization():
    strategy = TestStrategy(1.0, 2.0)
    
    assert strategy.param1 == 1.0
    assert strategy.param2 == 2.0
    assert isinstance(strategy.parameters, dict)

def test_base_strategy_generate_signals():
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    strategy = TestStrategy(1.0, 2.0)
    signals = strategy.generate_signals(data)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(data)
    assert all(signal in [-1, 0, 1] for signal in signals)

def test_base_strategy_invalid_data():
    strategy = TestStrategy(1.0, 2.0)
    
    with pytest.raises(ValueError):
        strategy.generate_signals(None)
    
    with pytest.raises(ValueError):
        strategy.generate_signals(pd.DataFrame())

def test_base_strategy_abstract_method():
    with pytest.raises(TypeError):
        Strategy() 