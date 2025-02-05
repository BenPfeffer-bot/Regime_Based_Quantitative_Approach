"""
Technical Indicators Module

This module provides functions for calculating various technical indicators
used in financial analysis.
"""

import numpy as np
import pandas as pd


def compute_ewm_std(df: pd.DataFrame, span: int = 20) -> pd.Series:
    """Compute exponentially weighted moving standard deviation."""
    return df["Close"].ewm(span=span, adjust=False).std()


def compute_kama(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Compute Kaufman's Adaptive Moving Average."""
    close = df["Close"]
    change = abs(close.diff(window))
    volatility = close.diff().abs().rolling(window=window).sum()
    er = change / volatility
    er = er.fillna(0)
    
    fast_sc = 2.0 / (2 + 1)
    slow_sc = 2.0 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = pd.Series(index=df.index, dtype=float)
    kama.loc[df.index[0]] = close.loc[df.index[0]]
    
    for i in range(1, len(close)):
        current_idx = df.index[i]
        prev_idx = df.index[i-1]
        kama.loc[current_idx] = kama.loc[prev_idx] + sc.loc[current_idx] * (close.loc[current_idx] - kama.loc[prev_idx])
    
    return kama


def compute_moving_avg(df: pd.DataFrame, window_a: int = 50, window_b: int = 200) -> pd.DataFrame:
    """Compute various moving averages."""
    df = df.copy()
    df["SMA_50"] = df["Close"].rolling(window=window_a).mean()
    df["SMA_200"] = df["Close"].rolling(window=window_b).mean()
    df["EMA_50"] = df["Close"].ewm(span=window_a, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=window_b, adjust=False).mean()
    df["KAMA"] = compute_kama(df, window=10)
    return df


def compute_macd(
    df: pd.DataFrame,
    window_fast: int = 12,
    window_slow: int = 26,
    window_signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD indicator."""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=window_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=window_slow, adjust=False).mean()
    df["MACD_line"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD_line"].ewm(span=window_signal, adjust=False).mean()
    df["MACD_histogram"] = df["MACD_line"] - df["MACD_signal"]
    return df


def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index."""
    df = df.copy()
    delta = df["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    avg_gain = gains.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # Fill initial NaN with neutral value
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Average True Range."""
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=window).mean()
    df["ATR"] = df["ATR"].fillna(method="bfill")
    return df


def compute_keltner_channel(
    df: pd.DataFrame, window: int = 20, multiplier: float = 2.0
) -> pd.DataFrame:
    """Compute Keltner Channels."""
    df = df.copy()
    
    # Ensure KAMA is calculated
    if "KAMA" not in df.columns:
        df["KAMA"] = compute_kama(df)
    
    df["KC_middle"] = df["KAMA"]
    df["EWMSD"] = compute_ewm_std(df, span=window)
    
    # Handle NaN values
    df["EWMSD"] = df["EWMSD"].fillna(method="bfill")
    df["KC_middle"] = df["KC_middle"].fillna(method="bfill")
    
    for level in range(1, 4):
        mult = multiplier * (level * 0.5)
        df[f"KC_upper{level}"] = df["KC_middle"] + (mult * df["EWMSD"])
        df[f"KC_lower{level}"] = df["KC_middle"] - (mult * df["EWMSD"])
    
    return df


def compute_std_dev(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling standard deviation."""
    df = df.copy()
    df["RSTD"] = df["Close"].rolling(window=window).std()
    df["RSTD"] = df["RSTD"].fillna(method="bfill")
    return df


def compute_regime(
    df: pd.DataFrame, atr_quantile: float = 0.75, variance_quantile: float = 0.75
) -> pd.DataFrame:
    """Classify market regime based on ATR and standard deviation."""
    df = df.copy()
    
    # Ensure required columns exist
    if "ATR" not in df.columns:
        df = compute_atr(df)
    if "RSTD" not in df.columns:
        df = compute_std_dev(df)
    
    # Calculate thresholds using non-NaN values
    atr_threshold = df["ATR"].dropna().quantile(atr_quantile)
    var_threshold = df["RSTD"].dropna().quantile(variance_quantile)
    
    # Classify regimes
    df["Regime"] = "Range Bound"  # Default regime
    trending_mask = (df["ATR"] < atr_threshold) & (df["RSTD"] < var_threshold)
    df.loc[trending_mask, "Regime"] = "Trending"
    
    return df


def compile_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compile all technical indicators into a single DataFrame."""
    df = df.copy()
    
    # Compute indicators in sequence, handling NaN values at each step
    df = compute_atr(df)  # Compute ATR first since other indicators depend on it
    df = compute_moving_avg(df)
    df = compute_macd(df)
    df = compute_rsi(df)
    df = compute_std_dev(df)
    df = compute_keltner_channel(df)
    df = compute_regime(df)
    
    # Forward fill any remaining NaN values
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df
