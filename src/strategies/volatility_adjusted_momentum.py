"""
Volatility-Adjusted Momentum Strategy Module

This module implements a momentum strategy that:
1. Ranks stocks based on momentum signals
2. Adjusts position sizes based on volatility
3. Uses dynamic entry thresholds based on market conditions
4. Implements risk parity for portfolio balancing
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import talib


class VolatilityAdjustedMomentum:
    """Implementation of volatility-adjusted momentum strategy."""
    
    def __init__(self, data):
        """Initialize strategy with data."""
        # Ensure column names are lowercase
        self.data = data.copy()
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        self.data.columns = [col.lower() for col in self.data.columns]
        
        # Strategy parameters
        self.min_momentum_threshold = 0.035  # Balanced threshold
        self.max_volatility_threshold = 0.022  # Slightly more conservative
        self.atr_multiplier = 2.75  # More balanced stop loss
        self.max_position_size = 0.15
        self.rsi_oversold = 30.0
        self.rsi_overbought = 65.0
        self.trend_confirmation_window = 20
        self.min_trend_strength = 0.65  # Increased trend strength requirement
        self.max_consecutive_losses = 2
        self.momentum_lookback = 14
        self.volatility_lookback = 20
        self.consecutive_losses = 0

    def calculate_momentum_score(self, index):
        """Calculate momentum score using multiple factors."""
        if index < self.momentum_lookback:
            return 0.0
        
        # Calculate price momentum
        returns = self.data['close'].pct_change(self.momentum_lookback)
        momentum = returns.iloc[index]
        
        # Calculate EMA momentum
        ema = self.data['close'].ewm(span=20, adjust=False).mean()
        ema_momentum = (ema.iloc[index] - ema.iloc[index - self.momentum_lookback]) / ema.iloc[index - self.momentum_lookback]
        
        # Calculate RSI
        rsi = talib.RSI(self.data['close'], timeperiod=14)
        rsi_momentum = (rsi.iloc[index] - 50) / 50  # Normalize around neutral level
        
        # Calculate volume momentum
        volume_sma = self.data['volume'].rolling(window=20).mean()
        volume_momentum = 1 if self.data['volume'].iloc[index] > volume_sma.iloc[index] else -1
        
        # Combine factors with weights
        momentum_score = (
            0.4 * momentum +
            0.3 * ema_momentum +
            0.2 * rsi_momentum +
            0.1 * volume_momentum
        )
        
        return momentum_score

    def calculate_trend_strength(self, index):
        """Calculate trend strength with modified weights."""
        if index < self.trend_confirmation_window:
            return 0.0
        
        # Calculate EMAs with shorter windows
        ema_short = self.data['close'].ewm(span=10, adjust=False).mean()
        ema_medium = self.data['close'].ewm(span=30, adjust=False).mean()
        ema_long = self.data['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate trend alignment score
        short_above_medium = ema_short.iloc[index] > ema_medium.iloc[index]
        medium_above_long = ema_medium.iloc[index] > ema_long.iloc[index]
        short_above_long = ema_short.iloc[index] > ema_long.iloc[index]
        
        trend_alignment = (short_above_medium + medium_above_long + short_above_long) / 3.0
        
        # Calculate shorter-term price momentum
        lookback = min(self.trend_confirmation_window, 10)  # Shorter lookback period
        price_momentum = (self.data['close'].iloc[index] - self.data['close'].iloc[index - lookback]) / self.data['close'].iloc[index - lookback]
        normalized_momentum = min(max(price_momentum, -1), 1)  # Clip between -1 and 1
        
        # Calculate volume trend with shorter window
        volume_sma = self.data['volume'].rolling(window=10).mean()
        volume_trend = self.data['volume'].iloc[index] > volume_sma.iloc[index]
        
        # Combine factors with modified weights
        trend_strength = (0.5 * trend_alignment + 
                         0.3 * abs(normalized_momentum) +
                         0.2 * volume_trend)
        
        return trend_strength

    def check_trend_confirmation(self, index):
        """Check for trend confirmation with more lenient conditions."""
        if index < self.trend_confirmation_window:
            return False
        
        # Calculate trend strength with more emphasis on recent price action
        trend_strength = self.calculate_trend_strength(index)
        
        # Calculate short-term moving averages
        sma_20 = self.data['close'].rolling(window=20).mean()
        sma_50 = self.data['close'].rolling(window=50).mean()
        
        # Check price momentum
        price_above_sma20 = self.data['close'].iloc[index] > sma_20.iloc[index]
        price_above_sma50 = self.data['close'].iloc[index] > sma_50.iloc[index]
        
        # Check RSI conditions - more lenient bounds
        rsi = talib.RSI(self.data['close'], timeperiod=14)
        rsi_conditions = (rsi.iloc[index] > 30 and rsi.iloc[index] < 75)
        
        # Check volume confirmation - more lenient
        volume_sma = self.data['volume'].rolling(window=20).mean()
        volume_increasing = self.data['volume'].iloc[index] > 0.8 * volume_sma.iloc[index]
        
        # Combined confirmation with more lenient requirements
        return ((trend_strength >= 0.5 or (price_above_sma20 and price_above_sma50)) and 
                rsi_conditions and 
                volume_increasing)

    def calculate_volatility(self, index):
        """Calculate volatility using ATR and standard deviation."""
        if index < self.volatility_lookback:
            return float('inf')
        
        # Calculate ATR-based volatility
        atr = talib.ATR(self.data['high'], self.data['low'], self.data['close'], 
                        timeperiod=self.volatility_lookback)
        atr_volatility = atr.iloc[index] / self.data['close'].iloc[index]
        
        # Calculate returns volatility
        returns = self.data['close'].pct_change()
        std_volatility = returns.iloc[max(0, index-self.volatility_lookback):index].std()
        
        # Combine volatility measures
        combined_volatility = (0.6 * atr_volatility + 0.4 * std_volatility)
        
        return combined_volatility

    def calculate_volatility_adjustment(self, index):
        """Calculate position size adjustment based on volatility."""
        if index < self.volatility_lookback:
            return 1.0
        
        volatility = self.calculate_volatility(index)
        if volatility == float('inf') or volatility == 0:
            return 1.0
        
        # Calculate adjustment (inverse relationship with volatility)
        volatility_adjustment = 1.0 / (volatility / self.calculate_volatility(index - 1))
        
        # Clip to reasonable range
        return np.clip(volatility_adjustment, 0.5, 1.5)

    def generate_signals(self, index):
        """Generate trading signals with position sizes."""
        if index < self.trend_confirmation_window:
            return 0, 0.0
        
        momentum_score = self.calculate_momentum_score(index)
        volatility = self.calculate_volatility(index)
        
        # Check entry conditions
        if (abs(momentum_score) > self.min_momentum_threshold and 
            volatility < self.max_volatility_threshold and 
            self.consecutive_losses < self.max_consecutive_losses):
            
            # Additional trend confirmation
            if self.check_trend_confirmation(index):
                position_size = self.max_position_size
                volatility_adj = self.calculate_volatility_adjustment(index)
                adjusted_size = position_size * volatility_adj
                
                if momentum_score > 0:
                    return 1, adjusted_size  # Long position
                else:
                    return -1, adjusted_size  # Short position
        
        return 0, 0.0  # No position

    def calculate_position_size(
        self,
        price: float,
        atr: float,
        max_risk_pct: float = 0.02
    ) -> float:
        """Calculate base position size using ATR for risk control."""
        # Set stop distance based on ATR
        stop_distance = self.atr_multiplier * atr
        
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
            DataFrame with backtest results
        """
        results = df.copy()
        
        # Initialize columns
        results['Signal'] = 0
        results['Position'] = 0
        results['Position_Size'] = 0.0
        results['Capital'] = initial_capital
        results['Equity'] = initial_capital
        
        capital = initial_capital
        position = 0
        position_size = 0.0
        entry_price = 0.0
        
        for i in range(len(results)):
            # Generate signal for current timestamp
            signal, size = self.generate_signals(i)
            
            # Record signal and position
            results.iloc[i, results.columns.get_loc('Signal')] = signal
            results.iloc[i, results.columns.get_loc('Position')] = position
            results.iloc[i, results.columns.get_loc('Position_Size')] = position_size
            
            # Process signal
            if signal != 0 and position == 0:  # Entry signal
                position = signal
                position_size = size
                entry_price = results.iloc[i]['close']
            elif position != 0:  # Check for exit
                current_price = results.iloc[i]['close']
                pnl = position * position_size * (current_price - entry_price) / entry_price
                
                # Exit on signal reversal or stop loss
                if (signal == -position) or (pnl < -max_risk_pct):
                    # Update capital
                    capital *= (1 + pnl)
                    
                    # Reset position
                    position = 0
                    position_size = 0.0
                    entry_price = 0.0
                    
                    # Update consecutive losses
                    if pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
            
            # Update capital and equity
            results.iloc[i, results.columns.get_loc('Capital')] = capital
            
            # Calculate equity (capital + open position value)
            if position != 0:
                current_price = results.iloc[i]['close']
                position_value = position * position_size * (current_price - entry_price)
                results.iloc[i, results.columns.get_loc('Equity')] = capital * (1 + position_value)
            else:
                results.iloc[i, results.columns.get_loc('Equity')] = capital
        
        return results 