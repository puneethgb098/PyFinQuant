import pytest
import numpy as np
from scipy.stats import norm
from pyfinquant.models.black_scholes import BlackScholes
from pyfinquant.greeks.analytical import AnalyticalGreeks

# Test parameters
S = 100.0  # Current stock price
K = 105.0  # Strike price
T = 1.0    # Time to maturity in years
r = 0.05   # Risk-free rate
q = 0.01   # Dividend yield
sigma = 0.2  # Volatility

# Calculate d1 and d2 for reference
d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

# Calculate expected values
exp_neg_qt = np.exp(-q * T)
exp_neg_rt = np.exp(-r * T)

# Expected values based on manual calculations
EXPECTED = {
    'delta_call': exp_neg_qt * norm.cdf(d1),
    'delta_put': -exp_neg_qt * norm.cdf(-d1),
    'gamma': exp_neg_qt * norm.pdf(d1) / (S * sigma * np.sqrt(T)),
    'vega': S * exp_neg_qt * np.sqrt(T) * norm.pdf(d1) / 100.0,
    'theta_call': (-(S * sigma * exp_neg_qt * norm.pdf(d1)) / (2 * np.sqrt(T)) 
                  - r * K * exp_neg_rt * norm.cdf(d2) 
                  + q * S * exp_neg_qt * norm.cdf(d1)) / 365.0,
    'theta_put': (-(S * sigma * exp_neg_qt * norm.pdf(d1)) / (2 * np.sqrt(T)) 
                 + r * K * exp_neg_rt * norm.cdf(-d2) 
                 - q * S * exp_neg_qt * norm.cdf(-d1)) / 365.0,
    'rho_call': K * T * exp_neg_rt * norm.cdf(d2) / 100.0,
    'rho_put': -K * T * exp_neg_rt * norm.cdf(-d2) / 100.0,
    'psi_call': -S * T * exp_neg_qt * norm.cdf(d1) / 100.0,
    'psi_put': S * T * exp_neg_qt * norm.cdf(-d1) / 100.0
}

@pytest.fixture
def black_scholes_call():
    return BlackScholes(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type='call')

@pytest.fixture
def black_scholes_put():
    return BlackScholes(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type='put')

@pytest.fixture
def greeks_calculator_call(black_scholes_call):
    return AnalyticalGreeks(black_scholes_call)

@pytest.fixture
def greeks_calculator_put(black_scholes_put):
    return AnalyticalGreeks(black_scholes_put)

TOL = 1e-4  # Increased tolerance for floating point comparisons

def test_delta_call(greeks_calculator_call):
    delta = greeks_calculator_call.delta()
    expected_delta = EXPECTED['delta_call']
    assert isinstance(delta, float)
    assert abs(delta - expected_delta) < TOL

def test_delta_put(greeks_calculator_put):
    delta = greeks_calculator_put.delta()
    expected_delta = EXPECTED['delta_put']
    assert isinstance(delta, float)
    assert abs(delta - expected_delta) < TOL

def test_gamma(greeks_calculator_call, greeks_calculator_put):
    gamma_call = greeks_calculator_call.gamma()
    gamma_put = greeks_calculator_put.gamma()
    expected_gamma = EXPECTED['gamma']
    assert isinstance(gamma_call, float)
    assert gamma_call >= 0
    assert abs(gamma_call - expected_gamma) < TOL
    assert abs(gamma_call - gamma_put) < TOL  # Gamma should be equal for calls and puts

def test_theta(greeks_calculator_call, greeks_calculator_put):
    theta_call = greeks_calculator_call.theta()  # Daily theta
    theta_put = greeks_calculator_put.theta()  # Daily theta
    expected_theta_call = EXPECTED['theta_call']
    expected_theta_put = EXPECTED['theta_put']
    assert isinstance(theta_call, float)
    assert abs(theta_call - expected_theta_call) < TOL
    assert abs(theta_put - expected_theta_put) < TOL

def test_vega(greeks_calculator_call, greeks_calculator_put):
    vega_call = greeks_calculator_call.vega()
    vega_put = greeks_calculator_put.vega()
    expected_vega = EXPECTED['vega']
    assert isinstance(vega_call, float)
    assert vega_call >= 0
    assert abs(vega_call - expected_vega) < TOL
    assert abs(vega_call - vega_put) < TOL  # Vega should be equal for calls and puts

def test_rho(greeks_calculator_call, greeks_calculator_put):
    rho_call = greeks_calculator_call.rho()
    rho_put = greeks_calculator_put.rho()
    expected_rho_call = EXPECTED['rho_call']
    expected_rho_put = EXPECTED['rho_put']
    assert isinstance(rho_call, float)
    assert abs(rho_call - expected_rho_call) < TOL
    assert abs(rho_put - expected_rho_put) < TOL

def test_psi(greeks_calculator_call, greeks_calculator_put):
    psi_call = greeks_calculator_call.psi()
    psi_put = greeks_calculator_put.psi()
    expected_psi_call = EXPECTED['psi_call']
    expected_psi_put = EXPECTED['psi_put']
    assert abs(psi_call - expected_psi_call) < TOL
    assert abs(psi_put - expected_psi_put) < TOL 