"""Visualization module for PyFinQuant."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_price(data, title="Price Chart", xlabel="Date", ylabel="Price ($)"):
    """Plot price data.
    
    Args:
        data (pd.DataFrame): Price data with datetime index
        title (str): Chart title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig, ax

def plot_returns_distribution(returns, title="Returns Distribution", bins=50):
    """Plot returns distribution.
    
    Args:
        returns (pd.Series): Returns data
        title (str): Chart title
        bins (int): Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(returns.dropna(), bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Returns')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    return fig, ax

def plot_risk_metrics(metrics, title="Risk Metrics"):
    """Plot risk metrics.
    
    Args:
        metrics (pd.Series): Risk metrics data
        title (str): Chart title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.grid(True)
    return fig, ax

def plot_portfolio_performance(returns, title="Portfolio Performance"):
    """Plot portfolio performance.
    
    Args:
        returns (pd.Series): Portfolio returns
        title (str): Chart title
    """
    cumulative_returns = (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative_returns.index, cumulative_returns)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.grid(True)
    return fig, ax 