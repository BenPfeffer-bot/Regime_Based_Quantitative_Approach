"""
Trend Following Strategy Module

This module implements a trend following strategy based on:
1. Moving Average crossovers
2. MACD momentum confirmation
3. Keltner Channel-based trailing stops
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class TrendFollowingStrategy:
    """Trend following strategy implementation."""
    
    def __init__(
        self,
        short_ma_window: int = 50,
        long_ma_window: int = 200,
        atr_multiple: float = 2.0,
        min_momentum: float = 0.0,
    ):
        """
        Initialize strategy parameters.
        
        Args:
            short_ma_window: Window for short-term moving average
            long_ma_window: Window for long-term moving average
            atr_multiple: Multiplier for ATR-based stops
            min_momentum: Minimum MACD histogram value for momentum confirmation
        """
        self.short_ma_window = short_ma_window
        self.long_ma_window = long_ma_window
        self.atr_multiple = atr_multiple
        self.min_momentum = min_momentum

    def generate_signals(self, df: pd.DataFrame, initial_capital: float = 100000.0) -> pd.DataFrame:
        """Generate trading signals based on moving average crossovers and MACD."""
        df = df.copy()
        
        # Handle MultiIndex columns by selecting the first level
        df.columns = df.columns.get_level_values(0)
        
        # Initialize signal columns
        df["Signal"] = 0
        df["Position"] = 0
        df["Position_Size"] = 0.0
        df["Stop_Loss"] = np.nan
        
        # Calculate moving average crossover
        crossover = (df["EMA_50"] > df["EMA_200"]).astype(int)
        prev_crossover = crossover.shift(1).fillna(0).astype(int)
        
        # Calculate MACD momentum
        macd_momentum = (df["MACD_histogram"] > 0).astype(int)
        
        # Calculate ATR-based stop loss
        df["Stop_Loss"] = df["Close"] - (self.atr_multiple * df["ATR"])
        
        position = 0
        stop_loss = np.nan
        capital = initial_capital
        
        # Create a copy of the index for iteration
        index_array = df.index.to_numpy()
        
        for i in range(1, len(df)):
            current_idx = index_array[i]
            prev_idx = index_array[i-1]
            
            # Check for entry signals
            if position == 0:
                # Long entry on MA crossover and MACD confirmation
                if (crossover.loc[current_idx] == 1 and 
                    crossover.loc[prev_idx] == 0 and 
                    macd_momentum.loc[current_idx] == 1):
                    position = 1
                    df.loc[current_idx, "Signal"] = 1
                    stop_loss = df.loc[current_idx, "Stop_Loss"]
            
            # Check for exit signals
            elif position == 1:
                # Exit on stop loss hit
                if df.loc[current_idx, "Low"] <= stop_loss:
                    position = 0
                    df.loc[current_idx, "Signal"] = -1
                    stop_loss = np.nan
                # Exit on trend reversal
                elif crossover.loc[current_idx] == 0 and crossover.loc[prev_idx] == 1:
                    position = 0
                    df.loc[current_idx, "Signal"] = -1
                    stop_loss = np.nan
                # Update trailing stop
                else:
                    stop_loss = max(stop_loss, df.loc[current_idx, "Stop_Loss"])
            
            # Update position and stop loss
            df.loc[current_idx, "Position"] = position
            df.loc[current_idx, "Stop_Loss"] = stop_loss
            
            # Calculate position size based on ATR
            if position != 0:
                df.loc[current_idx, "Position_Size"] = self.calculate_position_size(
                    capital,
                    df.loc[current_idx, "Close"],
                    df.loc[current_idx, "ATR"]
                )[0]  # Only take the position_size from the tuple
            
            # Update capital based on position and price change
            if i > 0:
                price_change = df.loc[current_idx, "Close"] - df.loc[prev_idx, "Close"]
                pnl = position * df.loc[prev_idx, "Position_Size"] * price_change
                capital += pnl
        
        return df

    def calculate_position_size(
        self, 
        capital: float,
        price: float,
        atr: float,
        max_risk_pct: float = 0.02
    ) -> Tuple[int, float]:
        """
        Calculate position size based on ATR and risk parameters.
        
        Args:
            capital: Available capital
            price: Current price
            atr: Average True Range
            max_risk_pct: Maximum risk per trade as percentage of capital
            
        Returns:
            Tuple of (position_size, stop_distance)
        """
        # Set stop distance based on ATR
        stop_distance = self.atr_multiple * atr
        
        # Calculate maximum loss allowed
        max_loss = capital * max_risk_pct
        
        # Calculate position size
        position_size = int(max_loss / stop_distance)
        
        return position_size, stop_distance

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_risk_pct: float = 0.02,
    ) -> pd.DataFrame:
        """
        Backtest the strategy.
        
        Args:
            df: DataFrame with price data and indicators
            initial_capital: Starting capital
            max_risk_pct: Maximum risk per trade as percentage of capital
            
        Returns:
            DataFrame with backtest results including:
            - Positions
            - Equity curve
            - Trade statistics
        """
        results = self.generate_signals(df)
        
        # Initialize backtest columns
        results["Capital"] = initial_capital
        results["Position_Size"] = 0
        results["Equity"] = initial_capital
        
        capital = initial_capital
        position = 0
        
        for i in range(1, len(results)):
            current_idx = results.index[i]
            prev_idx = results.index[i-1]
            
            price = results.loc[current_idx, "Close"]
            atr = results.loc[current_idx, "ATR"]
            signal = results.loc[current_idx, "Signal"]
            
            if signal == 1:  # Entry signal
                position_size, stop_distance = self.calculate_position_size(
                    capital, price, atr, max_risk_pct
                )
                position = position_size
                
            elif signal == -1:  # Exit signal
                position = 0
            
            # Update position size
            results.loc[current_idx, "Position_Size"] = position
            
            # Calculate P&L
            price_change = results.loc[current_idx, "Close"] - results.loc[prev_idx, "Close"]
            pnl = position * price_change
            
            # Update capital
            capital += pnl
            results.loc[current_idx, "Capital"] = capital
            
            # Calculate equity curve
            results.loc[current_idx, "Equity"] = capital + (position * price)
        
        return results
