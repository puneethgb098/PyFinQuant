"""
Statistical functions for quantitative finance.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def mean(x: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the mean of a series."""
    return np.mean(x)


def std(x: Union[np.ndarray, pd.Series], ddof: int = 1) -> float:
    """Calculate the standard deviation of a series."""
    return np.std(x, ddof=ddof)


def skew(x: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the skewness of a series."""
    return pd.Series(x).skew()


def kurtosis(x: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the kurtosis of a series."""
    return pd.Series(x).kurtosis()


def correlation(
    x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]
) -> float:
    """Calculate the correlation between two series."""
    return np.corrcoef(x, y)[0, 1]


def covariance(
    x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series], ddof: int = 1
) -> float:
    """Calculate the covariance between two series."""
    return np.cov(x, y, ddof=ddof)[0, 1]


def rolling_mean(
    x: Union[np.ndarray, pd.Series], window: int
) -> Union[np.ndarray, pd.Series]:
    """Calculate the rolling mean of a series."""
    return pd.Series(x).rolling(window=window).mean()


def rolling_std(
    x: Union[np.ndarray, pd.Series], window: int, ddof: int = 1
) -> Union[np.ndarray, pd.Series]:
    """Calculate the rolling standard deviation of a series."""
    return pd.Series(x).rolling(window=window).std(ddof=ddof)


def rolling_correlation(
    x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series], window: int
) -> Union[np.ndarray, pd.Series]:
    """Calculate the rolling correlation between two series."""
    return pd.Series(x).rolling(window=window).corr(pd.Series(y))


def rolling_covariance(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    window: int,
    ddof: int = 1,
) -> Union[np.ndarray, pd.Series]:
    """Calculate the rolling covariance between two series."""
    return pd.Series(x).rolling(window=window).cov(pd.Series(y), ddof=ddof)
