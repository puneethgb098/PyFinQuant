"""Example demonstrating option pricing and Greeks calculation using PyFinQuant."""

import numpy as np
import pyfinquant as pfq

def main():
    """Runs the option pricing and Greeks example."""
    
    print(f'Using PyFinQuant version: {pfq.__version__}')
    print("--- Option Pricing and Greeks Example ---")

    # 1. Define Option Parameters
    # --------------------------
    S = 100.0  # Current underlying price
    K = 105.0  # Option strike price
    T = 0.5    # Time to maturity in years
    r = 0.04   # Annual risk-free interest rate
    sigma = 0.25 # Annual volatility of the underlying
    q = 0.015  # Annual continuous dividend yield
    print("\nParameters:")
    print(f"  S={S}, K={K}, T={T}, r={r}, sigma={sigma}, q={q}")

    # 2. Create Black-Scholes Models
    # ------------------------------
    try:
        call_model = pfq.BlackScholes(
            S=S, K=K, T=T, r=r, sigma=sigma, option_type='call', q=q
        )
        put_model = pfq.BlackScholes(
            S=S, K=K, T=T, r=r, sigma=sigma, option_type='put', q=q
        )
        print("\nModels Created Successfully:")
        # print(f"  Call Model: {call_model}") # __repr__ might be verbose
        # print(f"  Put Model: {put_model}")
    except ValueError as e:
        print(f"\nError creating models: {e}")
        return

    # 3. Calculate Option Prices
    # --------------------------
    call_price = call_model.price()
    put_price = put_model.price()
    print("\nOption Prices:")
    print(f"  Call Price: {call_price:.4f}")
    print(f"  Put Price:  {put_price:.4f}")

    # 4. Calculate Option Greeks
    # --------------------------
    call_greeks = pfq.AnalyticalGreeks(call_model)
    put_greeks = pfq.AnalyticalGreeks(put_model)

    print("\n--- Call Option Greeks ---")
    print(f"  Delta: {call_greeks.delta():.4f}")
    print(f"  Gamma: {call_greeks.gamma():.4f}")
    print(f"  Vega:  {call_greeks.vega():.4f} (per 1% vol change)")
    print(f"  Theta: {call_greeks.theta():.4f} (per day)")
    print(f"  Rho:   {call_greeks.rho():.4f} (per 1% rate change)")

    print("\n--- Put Option Greeks ---")
    print(f"  Delta: {put_greeks.delta():.4f}")
    print(f"  Gamma: {put_greeks.gamma():.4f}")
    print(f"  Vega:  {put_greeks.vega():.4f} (per 1% vol change)")
    print(f"  Theta: {put_greeks.theta():.4f} (per day)")
    print(f"  Rho:   {put_greeks.rho():.4f} (per 1% rate change)")

if __name__ == "__main__":
    main() 