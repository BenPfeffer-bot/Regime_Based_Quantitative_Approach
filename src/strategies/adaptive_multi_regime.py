"""
Adaptive Multi-Regime Strategy Module

This module implements an adaptive strategy that:
1. Detects market regimes using multiple metrics
2. Applies trend following in trending markets
3. Applies mean reversion in range-bound markets
4. Dynamically adjusts position sizing based on regime confidence
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from ..analysis.technical.trend_following import TrendFollowingStrategy
from ..analysis.technical.mean_reversion import MeanReversionStrategy


class RegimeDetector:
    """Market regime detection using multiple metrics."""
    
    def __init__(
        self,
        volatility_window: int = 20,
        atr_quantile: float = 0.75,
        variance_quantile: float = 0.75,
        trend_strength_threshold: float = 0.05,
    ):
        """Initialize regime detector."""
        self.volatility_window = volatility_window
        self.atr_quantile = atr_quantile
        self.variance_quantile = variance_quantile
        self.trend_strength_threshold = trend_strength_threshold
    
    def detect_regime(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Detect market regime using technical metrics only.
        """
        df = df.copy()
        
        # Handle MultiIndex columns by selecting the first level
        df.columns = df.columns.get_level_values(0)
        
        # Technical regime detection
        atr_threshold = df["ATR"].quantile(self.atr_quantile)
        var_threshold = df["RSTD"].quantile(self.variance_quantile)
        
        ma_trend = (df["EMA_50"] - df["EMA_200"]) / df["EMA_200"]
        macd_trend = df["MACD_histogram"].abs() > self.trend_strength_threshold
        
        low_volatility = (df["ATR"] < atr_threshold) & (df["RSTD"] < var_threshold)
        trending_ma = ma_trend.abs() > ma_trend.rolling(self.volatility_window).std()
        trending_macd = macd_trend & (df["MACD_line"].abs() > df["MACD_line"].rolling(self.volatility_window).std())
        
        trend_score = (
            0.4 * low_volatility.astype(float) +
            0.3 * trending_ma.astype(float) +
            0.3 * trending_macd.astype(float)
        )
        
        # Set regime confidence based on technical factors only
        df["Regime_Confidence"] = trend_score
        
        # Classify regime
        df["Regime"] = "Range_Bound"
        df.loc[trend_score > 0.5, "Regime"] = "Trending"
        
        # Determine trend direction
        df["Trend_Direction"] = 0
        df.loc[(df["Regime"] == "Trending") & (ma_trend > 0), "Trend_Direction"] = 1
        df.loc[(df["Regime"] == "Trending") & (ma_trend < 0), "Trend_Direction"] = -1
        
        return df


class AdaptiveMultiRegimeStrategy:
    """Adaptive strategy that switches between trend following and mean reversion."""
    
    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        trend_strategy: Optional[TrendFollowingStrategy] = None,
        mean_rev_strategy: Optional[MeanReversionStrategy] = None,
    ):
        """Initialize strategies and regime detector."""
        self.regime_detector = regime_detector or RegimeDetector()
        self.trend_strategy = trend_strategy or TrendFollowingStrategy()
        self.mean_rev_strategy = mean_rev_strategy or MeanReversionStrategy()
    
    def generate_signals(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Generate trading signals based on regime detection."""
        df = df.copy()
        
        # Detect regime
        df = self.regime_detector.detect_regime(df, ticker)
        
        # Get signals from both strategies
        trend_signals = self.trend_strategy.generate_signals(df)
        mean_rev_signals = self.mean_rev_strategy.generate_signals(df)
        
        # Initialize combined signal columns
        df["Signal"] = 0
        df["Position"] = 0
        df["Position_Size"] = 0.0
        df["Stop_Loss"] = np.nan
        
        # Apply signals based on regime
        for i in range(len(df)):
            current_idx = df.index[i]
            
            regime = df.loc[current_idx, "Regime"]
            confidence = df.loc[current_idx, "Regime_Confidence"]
            
            if regime == "Trending":
                df.loc[current_idx, "Signal"] = trend_signals.loc[current_idx, "Signal"]
                df.loc[current_idx, "Position"] = trend_signals.loc[current_idx, "Position"]
                df.loc[current_idx, "Position_Size"] = trend_signals.loc[current_idx, "Position_Size"] * confidence
                df.loc[current_idx, "Stop_Loss"] = trend_signals.loc[current_idx, "Stop_Loss"]
            else:
                df.loc[current_idx, "Signal"] = mean_rev_signals.loc[current_idx, "Signal"]
                df.loc[current_idx, "Position"] = mean_rev_signals.loc[current_idx, "Position"]
                df.loc[current_idx, "Position_Size"] = mean_rev_signals.loc[current_idx, "Position_Size"] * confidence
                df.loc[current_idx, "Stop_Loss"] = mean_rev_signals.loc[current_idx, "Stop_Loss"]
        
        return df
    
    def backtest(
        self,
        df: pd.DataFrame,
        ticker: str,
        initial_capital: float = 100000.0,
        max_risk_pct: float = 0.02
    ) -> pd.DataFrame:
        """Backtest with fundamental overlay."""
        results = self.generate_signals(df, ticker)
        
        # Rest of the backtest code remains the same
        results["Capital"] = initial_capital
        results["Equity"] = initial_capital
        results["Strategy"] = "None"
        
        capital = initial_capital
        
        for i in range(1, len(results)):
            current_idx = results.index[i]
            prev_idx = results.index[i-1]
            
            price = results.loc[current_idx, "Close"]
            position_size = results.loc[current_idx, "Position_Size"]
            regime = results.loc[current_idx, "Regime"]
            
            results.loc[current_idx, "Strategy"] = "Trend_Following" if regime == "Trending" else "Mean_Reversion"
            
            price_change = results.loc[current_idx, "Close"] - results.loc[prev_idx, "Close"]
            pnl = position_size * price_change
            
            capital += pnl
            results.loc[current_idx, "Capital"] = capital
            results.loc[current_idx, "Equity"] = capital + (position_size * price)
        
        return results
    
    def get_strategy_metrics(self, results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each strategy and regime.
        
        Returns:
            Dictionary containing performance metrics for each strategy/regime
        """
        metrics = {}
        
        # Calculate metrics for each strategy
        for strategy in ["Trend_Following", "Mean_Reversion"]:
            strategy_data = results[results["Strategy"] == strategy]
            
            if len(strategy_data) > 0:
                returns = strategy_data["Equity"].pct_change()
                metrics[strategy] = {
                    "Return": returns.mean() * 252,  # Annualized return
                    "Volatility": returns.std() * np.sqrt(252),  # Annualized volatility
                    "Sharpe": (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0,
                    "Win_Rate": (strategy_data["Capital"].diff() > 0).mean(),
                    "Trade_Count": (strategy_data["Signal"] != 0).sum()
                }
        
        return metrics
    
# if __name__ == "__main__":
    
#     # Initialize strategy   
#     strategy = AdaptiveMultiRegimeStrategy()

#     # Run backtest
#     results = strategy.backtest(df, initial_capital=100000)

#     # Get performance metrics
#     metrics = strategy.get_strategy_metrics(results)