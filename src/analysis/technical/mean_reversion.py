"""
Mean Reversion Strategy Module

This module implements a mean reversion strategy based on:
1. RSI oversold/overbought conditions
2. Keltner Channel breakouts
3. ATR-based position sizing
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class MeanReversionStrategy:
    """Mean reversion strategy implementation."""
    
    def __init__(
        self,
        rsi_window: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        keltner_window: int = 20,
        keltner_atr_multiple: float = 2.0,
        atr_window: int = 14,
    ):
        """
        Initialize strategy parameters.
        
        Args:
            rsi_window: Window for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            keltner_window: Window for Keltner Channel calculation
            keltner_atr_multiple: ATR multiplier for Keltner Channel width
            atr_window: Window for ATR calculation
        """
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.keltner_window = keltner_window
        self.keltner_atr_multiple = keltner_atr_multiple
        self.atr_window = atr_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI and Keltner Channels."""
        df = df.copy()
        
        # Handle MultiIndex columns by selecting the first level
        df.columns = df.columns.get_level_values(0)
        
        # Initialize signal columns
        df["Signal"] = 0
        df["Position"] = 0
        df["Position_Size"] = 0.0
        df["Stop_Loss"] = np.nan
        
        # Create a copy of the index for iteration
        index_array = df.index.to_numpy()
        
        position = 0
        stop_loss = np.nan
        
        for i in range(1, len(df)):
            current_idx = index_array[i]
            prev_idx = index_array[i-1]
            
            # Check for entry signals
            if position == 0:
                # Long entry on oversold RSI and price below lower Keltner
                if (df.loc[current_idx, "RSI"] < self.rsi_oversold and 
                    df.loc[current_idx, "Close"] < df.loc[current_idx, "KC_lower1"]):
                    position = 1
                    df.loc[current_idx, "Signal"] = 1
                    stop_loss = df.loc[current_idx, "Close"] - (2 * df.loc[current_idx, "ATR"])
                
                # Short entry on overbought RSI and price above upper Keltner
                elif (df.loc[current_idx, "RSI"] > self.rsi_overbought and 
                      df.loc[current_idx, "Close"] > df.loc[current_idx, "KC_upper1"]):
                    position = -1
                    df.loc[current_idx, "Signal"] = -1
                    stop_loss = df.loc[current_idx, "Close"] + (2 * df.loc[current_idx, "ATR"])
            
            # Check for exit signals
            elif position != 0:
                # Exit long position
                if position == 1:
                    # Take profit at middle Keltner
                    if df.loc[current_idx, "Close"] > df.loc[current_idx, "KC_middle"]:
                        position = 0
                        df.loc[current_idx, "Signal"] = -1
                        stop_loss = np.nan
                    # Stop loss hit
                    elif df.loc[current_idx, "Low"] < stop_loss:
                        position = 0
                        df.loc[current_idx, "Signal"] = -1
                        stop_loss = np.nan
                
                # Exit short position
                elif position == -1:
                    # Take profit at middle Keltner
                    if df.loc[current_idx, "Close"] < df.loc[current_idx, "KC_middle"]:
                        position = 0
                        df.loc[current_idx, "Signal"] = 1
                        stop_loss = np.nan
                    # Stop loss hit
                    elif df.loc[current_idx, "High"] > stop_loss:
                        position = 0
                        df.loc[current_idx, "Signal"] = 1
                        stop_loss = np.nan
            
            # Update position and stop loss
            df.loc[current_idx, "Position"] = position
            df.loc[current_idx, "Stop_Loss"] = stop_loss
            
            # Calculate position size based on ATR
            if position != 0:
                position_size = self.calculate_position_size(
                    df.loc[current_idx, "Close"],
                    df.loc[current_idx, "ATR"]
                )
                df.loc[current_idx, "Position_Size"] = position_size * abs(position)
        
        return df

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        max_risk_pct: float = 0.02
    ) -> float:
        """
        Calculate position size based on ATR and risk parameters.
        
        Args:
            price: Current price
            atr: Average True Range
            max_risk_pct: Maximum risk per trade as percentage
            
        Returns:
            Position size
        """
        # Set stop distance based on ATR
        stop_distance = 2 * atr
        
        # Calculate position size to risk max_risk_pct
        position_size = max_risk_pct / (stop_distance / price)
        
        return position_size

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_risk_pct: float = 0.02
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
        results["Equity"] = initial_capital
        
        capital = initial_capital
        
        for i in range(1, len(results)):
            price = results["Close"].iloc[i]
            position_size = results["Position_Size"].iloc[i]
            
            # Calculate P&L
            price_change = results["Close"].iloc[i] - results["Close"].iloc[i-1]
            pnl = position_size * price_change
            
            # Update capital
            capital += pnl
            results["Capital"].iloc[i] = capital
            
            # Calculate equity curve
            results["Equity"].iloc[i] = capital + (position_size * price)
        
        return results
