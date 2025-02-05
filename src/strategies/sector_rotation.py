"""
Adaptive Sector Rotation Strategy Module

This module implements a sector rotation strategy that:
1. Groups stocks by sector and computes aggregated technical signals
2. Determines sector weights based on relative strength and regime
3. Adjusts individual stock positions based on sector allocation
4. Implements risk management through diversification and volatility adjustment
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
import talib
import logging


@dataclass
class SectorMetrics:
    """Container for sector-level metrics."""
    momentum_score: float
    volatility: float
    relative_strength: float
    ma_signal: float
    regime: str
    correlation: float
    trend_strength: float  # Added trend strength metric
    weight: float = 0.0


class AdaptiveSectorRotation:
    """Implementation of sector rotation strategy."""
    
    def __init__(
        self,
        momentum_window: int = 126,  # ~6 months
        volatility_window: int = 21,  # ~1 month
        ma_short: int = 50,
        ma_long: int = 200,
        min_sector_weight: float = 0.05,
        max_sector_weight: float = 0.30,
        regime_lookback: int = 63,  # ~3 months
        correlation_window: int = 63,
        risk_free_rate: float = 0.02,
        min_trend_strength: float = 0.4,  # Lowered threshold
        entry_confirmation_window: int = 3,  # Reduced confirmation window
        exit_confirmation_window: int = 2,  # Reduced confirmation window
        min_holding_period: int = 5,  # Reduced minimum holding period
        max_sector_correlation: float = 0.85,  # Increased correlation threshold
        position_sizing_atr_multiple: float = 1.5,  # Reduced ATR multiple
    ):
        """Initialize strategy parameters."""
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.min_sector_weight = min_sector_weight
        self.max_sector_weight = max_sector_weight
        self.regime_lookback = regime_lookback
        self.correlation_window = correlation_window
        self.risk_free_rate = risk_free_rate
        
        # New parameters for improved entry/exit
        self.min_trend_strength = min_trend_strength
        self.entry_confirmation_window = entry_confirmation_window
        self.exit_confirmation_window = exit_confirmation_window
        self.min_holding_period = min_holding_period
        self.max_sector_correlation = max_sector_correlation
        self.position_sizing_atr_multiple = position_sizing_atr_multiple
        
        # Strategy state
        self.sector_metrics: Dict[str, SectorMetrics] = {}
        self.sector_weights: Dict[str, float] = {}
        self.stock_weights: Dict[str, float] = {}
        self.position_entry_dates: Dict[str, pd.Timestamp] = {}  # Track entry dates
        self.consecutive_signals: Dict[str, Dict[str, int]] = {}  # Track signal consistency
    
    def calculate_sector_momentum(
        self,
        sector_prices: pd.Series,
        sector_volumes: pd.Series
    ) -> float:
        """Calculate sector momentum using price and volume data."""
        # Price momentum
        returns = sector_prices.pct_change(periods=self.momentum_window, fill_method=None)
        price_momentum = returns.iloc[-1] if not returns.empty else 0
        
        # Volume momentum
        volume_sma = sector_volumes.rolling(window=20).mean()
        volume_momentum = 1 if sector_volumes.iloc[-1] > volume_sma.iloc[-1] else -1
        
        # RSI
        rsi = talib.RSI(sector_prices.values, timeperiod=14)
        rsi_momentum = (rsi[-1] - 50) / 50 if not np.isnan(rsi[-1]) else 0
        
        # Combine signals
        momentum_score = (
            0.5 * price_momentum +
            0.3 * rsi_momentum +
            0.2 * volume_momentum
        )
        
        return momentum_score
    
    def calculate_sector_volatility(
        self,
        sector_prices: pd.Series
    ) -> float:
        """Calculate sector volatility."""
        returns = sector_prices.pct_change(fill_method=None)
        volatility = returns.rolling(window=self.volatility_window).std().iloc[-1]
        return volatility if not np.isnan(volatility) else 0
    
    def calculate_relative_strength(
        self,
        sector_prices: pd.Series,
        market_prices: pd.Series
    ) -> float:
        """Calculate sector relative strength compared to market."""
        sector_returns = sector_prices.pct_change(periods=self.momentum_window, fill_method=None).iloc[-1]
        market_returns = market_prices.pct_change(periods=self.momentum_window, fill_method=None).iloc[-1]
        
        relative_strength = sector_returns - market_returns
        return relative_strength if not np.isnan(relative_strength) else 0
    
    def calculate_ma_signal(
        self,
        sector_prices: pd.Series
    ) -> float:
        """Calculate moving average crossover signal."""
        ma_short = sector_prices.rolling(window=self.ma_short).mean()
        ma_long = sector_prices.rolling(window=self.ma_long).mean()
        
        # Calculate trend strength
        trend_strength = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        return trend_strength if not np.isnan(trend_strength) else 0
    
    def detect_sector_regime(
        self,
        sector_prices: pd.Series,
        sector_volumes: pd.Series
    ) -> str:
        """Detect regime for a sector."""
        # Calculate volatility
        volatility = self.calculate_sector_volatility(sector_prices)
        
        # Calculate trend strength
        ma_signal = self.calculate_ma_signal(sector_prices)
        
        # Calculate volume trend
        volume_sma = sector_volumes.rolling(window=20).mean()
        volume_trend = sector_volumes.iloc[-1] > volume_sma.iloc[-1]
        
        # Determine regime
        if abs(ma_signal) > 0.05 and volume_trend:
            return "Trending"
        elif volatility > sector_prices.pct_change().std() * 2:
            return "Volatile"
        else:
            return "Range_Bound"
    
    def calculate_sector_correlation(
        self,
        sector_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """Calculate correlation between sector and market."""
        correlation = sector_returns.rolling(window=self.correlation_window).corr(market_returns)
        return correlation.iloc[-1] if not np.isnan(correlation.iloc[-1]) else 0
    
    def calculate_trend_strength(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> float:
        """Calculate trend strength using multiple indicators."""
        # Price-based trend strength
        ema_20 = prices.ewm(span=20, adjust=False).mean()
        ema_50 = prices.ewm(span=50, adjust=False).mean()
        ema_200 = prices.ewm(span=200, adjust=False).mean()
        
        price_trend = (
            (prices > ema_20).astype(int) +
            (ema_20 > ema_50).astype(int) +
            (ema_50 > ema_200).astype(int)
        ) / 3
        
        # Volume trend confirmation
        volume_sma = volumes.rolling(window=20).mean()
        volume_trend = (volumes > volume_sma).astype(int)
        
        # ADX for trend strength
        high = prices * 1.001  # Approximate high/low from close
        low = prices * 0.999
        adx = talib.ADX(high.values, low.values, prices.values, timeperiod=14)
        adx_trend = (adx[-1] / 100) if not np.isnan(adx[-1]) else 0
        
        # Combine indicators
        trend_strength = (
            0.4 * price_trend.iloc[-1] +
            0.3 * adx_trend +
            0.3 * volume_trend.iloc[-1]
        )
        
        return trend_strength

    def check_entry_confirmation(
        self,
        sector: str,
        signal_type: str,
        current_date: pd.Timestamp
    ) -> bool:
        """Check if entry signal is confirmed by multiple indicators."""
        if sector not in self.consecutive_signals:
            self.consecutive_signals[sector] = {'long': 0, 'short': 0}
        
        metrics = self.sector_metrics[sector]
        
        # Check trend strength with lower threshold for initial entries
        if metrics.trend_strength < self.min_trend_strength * 0.8:  # More lenient for entries
            logging.debug(f"{current_date}: {sector} failed trend strength check: {metrics.trend_strength:.2f}")
            return False
        
        # Check correlation with higher threshold
        if metrics.correlation > self.max_sector_correlation:
            logging.debug(f"{current_date}: {sector} failed correlation check: {metrics.correlation:.2f}")
            return False
        
        # Check signal consistency
        if signal_type == 'long':
            self.consecutive_signals[sector]['long'] += 1
            self.consecutive_signals[sector]['short'] = 0
        else:
            self.consecutive_signals[sector]['short'] += 1
            self.consecutive_signals[sector]['long'] = 0
        
        # Require consistent signals with reduced threshold
        required_signals = max(2, self.entry_confirmation_window - 1)  # More lenient requirement
        confirmed = self.consecutive_signals[sector][signal_type] >= required_signals
        
        if confirmed:
            logging.debug(f"{current_date}: {sector} entry confirmed for {signal_type}")
        return confirmed

    def check_exit_confirmation(
        self,
        sector: str,
        position_type: str,
        current_date: pd.Timestamp
    ) -> bool:
        """Check if exit signal is confirmed."""
        # Check minimum holding period
        entry_date = self.position_entry_dates.get(sector)
        if entry_date and (current_date - entry_date).days < self.min_holding_period:
            return False
        
        metrics = self.sector_metrics[sector]
        
        # Exit on strong trend reversal
        if position_type == 'long':
            trend_reversal = metrics.momentum_score < -0.02  # Small negative threshold
        else:
            trend_reversal = metrics.momentum_score > 0.02  # Small positive threshold
        
        # Exit on clear regime change
        regime_exit = metrics.regime == "Volatile"  # Only exit on volatile regime
        
        # Exit on significant correlation breakdown
        correlation_exit = metrics.correlation < 0.2  # More extreme decorrelation threshold
        
        should_exit = trend_reversal or regime_exit or correlation_exit
        if should_exit:
            logging.debug(f"{current_date}: {sector} exit confirmed - trend:{trend_reversal}, regime:{regime_exit}, corr:{correlation_exit}")
        
        return should_exit

    def calculate_position_size(
        self,
        sector: str,
        price: float,
        atr: float
    ) -> float:
        """Calculate position size using ATR-based volatility adjustment."""
        metrics = self.sector_metrics[sector]
        base_weight = self.sector_weights[sector]
        
        # Adjust weight based on trend strength with higher base allocation
        trend_adjustment = max(0.5, metrics.trend_strength)  # Minimum 50% allocation
        
        # Adjust weight based on volatility with less penalization
        vol_adjustment = 1.0 / (metrics.volatility * 5 + 1)  # Reduced volatility impact
        
        # Calculate stop distance using ATR with tighter stops
        stop_distance = self.position_sizing_atr_multiple * atr
        
        # Final position size with higher base allocation
        position_size = base_weight * trend_adjustment * vol_adjustment * 1.2  # 20% higher allocation
        
        # Apply maximum position constraint
        return min(position_size, self.max_sector_weight)

    def calculate_sector_weights(
        self,
        sector_data: Dict[str, Dict[str, pd.DataFrame]],
        market_data: pd.DataFrame
    ) -> None:
        """Calculate optimal sector weights based on metrics and regime."""
        # Calculate market returns
        market_returns = market_data['close'].pct_change(fill_method=None)
        
        # Calculate metrics for each sector
        sector_metrics = {}
        total_score = 0
        
        for sector, data in sector_data.items():
            # Aggregate sector data
            sector_prices = pd.DataFrame({
                ticker: df['close'] for ticker, df in data.items()
            }).mean(axis=1)
            
            sector_volumes = pd.DataFrame({
                ticker: df['volume'] for ticker, df in data.items()
            }).mean(axis=1)
            
            sector_returns = sector_prices.pct_change(fill_method=None)
            
            # Calculate metrics
            momentum = self.calculate_sector_momentum(sector_prices, sector_volumes)
            volatility = self.calculate_sector_volatility(sector_prices)
            rel_strength = self.calculate_relative_strength(sector_prices, market_data['close'])
            ma_signal = self.calculate_ma_signal(sector_prices)
            regime = self.detect_sector_regime(sector_prices, sector_volumes)
            correlation = self.calculate_sector_correlation(sector_returns, market_returns)
            trend_strength = self.calculate_trend_strength(sector_prices, sector_volumes)
            
            # Store metrics
            metrics = SectorMetrics(
                momentum_score=momentum,
                volatility=volatility,
                relative_strength=rel_strength,
                ma_signal=ma_signal,
                regime=regime,
                correlation=correlation,
                trend_strength=trend_strength
            )
            
            # Calculate sector score
            if regime == "Trending":
                # In trending regime, favor momentum and relative strength
                score = (0.35 * momentum +
                        0.25 * rel_strength +
                        0.20 * ma_signal +
                        0.10 * (1 - correlation) +
                        0.10 * trend_strength)
            elif regime == "Range_Bound":
                # In range-bound regime, favor low volatility and low correlation
                score = (0.35 * (1 / volatility) +
                        0.25 * (1 - correlation) +
                        0.20 * momentum +
                        0.20 * trend_strength)
            else:  # Volatile regime
                # In volatile regime, focus on risk management
                score = (0.40 * (1 / volatility) +
                        0.30 * (1 - correlation) +
                        0.30 * abs(momentum))
            
            # Apply minimum score threshold
            if score > 0.2:  # Only consider sectors with meaningful scores
                sector_metrics[sector] = (metrics, max(0, score))
                total_score += max(0, score)
        
        # Normalize weights and apply diversification constraints
        min_sectors = 3  # Minimum number of sectors to hold
        max_sectors = 5  # Maximum number of sectors to hold
        
        # Sort sectors by score
        sorted_sectors = sorted(
            sector_metrics.items(),
            key=lambda x: x[1][1],
            reverse=True
        )
        
        # Select top sectors
        num_sectors = min(
            max(min_sectors, len(sorted_sectors)),
            max_sectors
        )
        
        selected_sectors = sorted_sectors[:num_sectors]
        total_selected_score = sum(score for _, (_, score) in selected_sectors)
        
        # Calculate initial weights for selected sectors
        weights = {}
        for sector, (metrics, score) in selected_sectors:
            weight = score / total_selected_score if total_selected_score > 0 else 0
            weights[sector] = weight
            
            # Store metrics
            self.sector_metrics[sector] = metrics
        
        # Apply minimum and maximum weight constraints
        weights = self._apply_weight_constraints(weights)
        
        # Store final weights
        self.sector_weights = weights
        
        # Update metrics with final weights
        for sector in self.sector_metrics:
            self.sector_metrics[sector].weight = weights.get(sector, 0)
    
    def _apply_weight_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        # First pass: apply maximum constraints
        excess_weight = 0
        for sector in weights:
            if weights[sector] > self.max_sector_weight:
                excess_weight += weights[sector] - self.max_sector_weight
                weights[sector] = self.max_sector_weight
        
        # Redistribute excess weight
        if excess_weight > 0:
            available_sectors = [s for s in weights if weights[s] < self.max_sector_weight]
            while excess_weight > 0.0001 and available_sectors:
                per_sector = excess_weight / len(available_sectors)
                for sector in available_sectors[:]:
                    new_weight = weights[sector] + per_sector
                    if new_weight > self.max_sector_weight:
                        weights[sector] = self.max_sector_weight
                        excess_weight -= (self.max_sector_weight - weights[sector])
                        available_sectors.remove(sector)
                    else:
                        weights[sector] = new_weight
                        excess_weight = 0
        
        # Second pass: apply minimum constraints
        required_weight = 0
        for sector in weights:
            if weights[sector] < self.min_sector_weight:
                required_weight += self.min_sector_weight - weights[sector]
                weights[sector] = self.min_sector_weight
        
        # Reduce weights of sectors above minimum to fund required weight
        if required_weight > 0:
            available_sectors = [s for s in weights if weights[s] > self.min_sector_weight]
            while required_weight > 0.0001 and available_sectors:
                per_sector = required_weight / len(available_sectors)
                for sector in available_sectors[:]:
                    reduction = min(per_sector, weights[sector] - self.min_sector_weight)
                    weights[sector] -= reduction
                    required_weight -= reduction
                    if weights[sector] <= self.min_sector_weight:
                        available_sectors.remove(sector)
        
        return weights
    
    def calculate_stock_weights(
        self,
        sector_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> None:
        """Calculate individual stock weights within sectors."""
        stock_weights = {}
        
        for sector, stocks in sector_data.items():
            sector_weight = self.sector_weights[sector]
            num_stocks = len(stocks)
            
            if num_stocks == 0:
                continue
            
            # Calculate stock-level metrics
            stock_metrics = {}
            for ticker, data in stocks.items():
                momentum = self.calculate_sector_momentum(data['close'], data['volume'])
                volatility = self.calculate_sector_volatility(data['close'])
                
                # Combine metrics into a score
                score = momentum / (volatility + 1e-6)  # Add small constant to avoid division by zero
                stock_metrics[ticker] = max(0, score)  # Ensure non-negative scores
            
            # Normalize stock scores within sector
            total_score = sum(stock_metrics.values())
            if total_score > 0:
                stock_weights_sector = {
                    ticker: (score / total_score) * sector_weight
                    for ticker, score in stock_metrics.items()
                }
            else:
                # Equal weight if all scores are zero
                stock_weights_sector = {
                    ticker: sector_weight / num_stocks
                    for ticker in stocks.keys()
                }
            
            stock_weights.update(stock_weights_sector)
        
        self.stock_weights = stock_weights
    
    def generate_signals(
        self,
        sector_data: Dict[str, Dict[str, pd.DataFrame]],
        market_data: pd.DataFrame,
        current_positions: Dict[str, float],
        current_date: pd.Timestamp
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Generate trading signals for individual stocks."""
        # Update sector and stock weights
        self.calculate_sector_weights(sector_data, market_data)
        self.calculate_stock_weights(sector_data)
        
        # Generate signals
        signals = {}
        
        # Track total position changes
        total_position_change = 0
        max_daily_position_changes = 5  # Limit number of position changes per day
        
        for sector, stocks in sector_data.items():
            if total_position_change >= max_daily_position_changes:
                break
                
            # Calculate sector ATR
            sector_prices = pd.DataFrame({
                ticker: df['close'] for ticker, df in stocks.items()
            }).mean(axis=1)
            
            high = sector_prices * 1.001
            low = sector_prices * 0.999
            close = sector_prices
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
            current_atr = atr[-1] if not np.isnan(atr[-1]) else sector_prices.std()
            
            # Sort stocks by opportunity score
            stock_scores = []
            for ticker in stocks:
                target_weight = self.stock_weights.get(ticker, 0)
                current_weight = current_positions.get(ticker, 0)
                weight_change = abs(target_weight - current_weight)
                
                if weight_change < 0.01:  # Ignore small weight changes
                    continue
                
                # Calculate opportunity score
                metrics = self.sector_metrics[sector]
                opportunity_score = (
                    0.4 * metrics.trend_strength +
                    0.3 * metrics.momentum_score +
                    0.2 * (1 - metrics.correlation) +
                    0.1 * (1 / metrics.volatility if metrics.volatility > 0 else 1)
                )
                
                stock_scores.append((ticker, opportunity_score, weight_change))
            
            # Sort by opportunity score
            stock_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Generate signals for top opportunities
            for ticker, score, weight_change in stock_scores:
                if total_position_change >= max_daily_position_changes:
                    break
                    
                target_weight = self.stock_weights.get(ticker, 0)
                current_weight = current_positions.get(ticker, 0)
                
                # Determine signal direction
                if target_weight > current_weight:
                    signal_type = 'long'
                    signal = 1
                else:
                    signal_type = 'short'
                    signal = -1
                
                # Check if we should enter position
                if current_weight == 0:
                    if not self.check_entry_confirmation(sector, signal_type, current_date):
                        continue
                    # Record entry date
                    self.position_entry_dates[ticker] = current_date
                # Check if we should exit position
                elif self.check_exit_confirmation(sector, 'long' if current_weight > 0 else 'short', current_date):
                    signal = -np.sign(current_weight)  # Exit signal
                    self.position_entry_dates.pop(ticker, None)
                else:
                    continue  # Hold position
                
                # Calculate position size
                size_adjustment = self.calculate_position_size(
                    sector,
                    stocks[ticker]['close'].iloc[-1],
                    current_atr
                )
                
                signals[ticker] = {
                    'signal': signal,
                    'size': size_adjustment,
                    'target_weight': target_weight
                }
                
                total_position_change += 1
        
        return signals
    
    def backtest(
        self,
        sector_data: Dict[str, Dict[str, pd.DataFrame]],
        market_data: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """Backtest the sector rotation strategy."""
        # Initialize results
        results = pd.DataFrame()
        capital = initial_capital
        current_positions = {}
        
        # Get common dates
        dates = sorted(set.intersection(
            *[set(df.index) for stocks in sector_data.values() for df in stocks.values()]
        ))
        
        # Initialize tracking variables
        trade_count = 0
        winning_trades = 0
        peak_capital = initial_capital
        total_long_exposure = 0
        total_short_exposure = 0
        
        for date in dates:
            # Get current prices
            prices = {}
            for sector, stocks in sector_data.items():
                for ticker, data in stocks.items():
                    if date in data.index:
                        prices[ticker] = data.loc[date, 'close']
            
            # Generate signals
            signals = self.generate_signals(
                sector_data,
                market_data,
                {ticker: pos['size'] for ticker, pos in current_positions.items()},
                date
            )
            
            # Execute trades
            daily_pnl = 0
            
            # First, calculate P&L for existing positions
            for ticker, position in list(current_positions.items()):
                if ticker in prices:
                    current_price = prices[ticker]
                    position_value = position['size'] * current_price
                    cost_basis_value = position['size'] * position['cost_basis']
                    position_pnl = position_value - cost_basis_value
                    daily_pnl += position_pnl
            
            # Then, process new signals
            for ticker, signal_data in signals.items():
                if signal_data['signal'] != 0:
                    trade_count += 1
                    current_price = prices[ticker]
                    
                    # Close existing position if any
                    if ticker in current_positions:
                        old_position = current_positions[ticker]
                        old_value = old_position['size'] * current_price
                        cost_basis_value = old_position['size'] * old_position['cost_basis']
                        position_pnl = old_value - cost_basis_value
                        
                        if position_pnl > 0:
                            winning_trades += 1
                        
                        daily_pnl += position_pnl
                        current_positions.pop(ticker)
                    
                    # Open new position if signal indicates
                    if abs(signal_data['size']) > 0:
                        current_positions[ticker] = {
                            'size': signal_data['size'] * signal_data['signal'],
                            'cost_basis': current_price
                        }
            
            # Update capital
            capital += daily_pnl
            peak_capital = max(peak_capital, capital)
            
            # Calculate exposures
            long_exposure = sum(pos['size'] * prices[ticker]
                              for ticker, pos in current_positions.items()
                              if pos['size'] > 0 and ticker in prices)
            short_exposure = sum(abs(pos['size'] * prices[ticker])
                               for ticker, pos in current_positions.items()
                               if pos['size'] < 0 and ticker in prices)
            
            total_long_exposure += long_exposure
            total_short_exposure += short_exposure
            
            # Record results
            results.loc[date, 'Capital'] = capital
            results.loc[date, 'Daily_PnL'] = daily_pnl
            results.loc[date, 'Returns'] = daily_pnl / capital if capital > 0 else 0
            results.loc[date, 'Drawdown'] = (peak_capital - capital) / peak_capital
            results.loc[date, 'Long_Exposure'] = long_exposure / capital if capital > 0 else 0
            results.loc[date, 'Short_Exposure'] = short_exposure / capital if capital > 0 else 0
            
            # Record sector weights
            for sector, weight in self.sector_weights.items():
                results.loc[date, f'Sector_Weight_{sector}'] = weight
        
        # Calculate performance metrics
        returns = results['Returns']
        volatility = returns.std() * np.sqrt(252)
        avg_long_exposure = total_long_exposure / len(dates) / initial_capital
        avg_short_exposure = total_short_exposure / len(dates) / initial_capital
        
        # Store summary statistics
        results.attrs['summary'] = {
            'total_trades': trade_count,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / trade_count if trade_count > 0 else 0,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital,
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': volatility,
            'sharpe_ratio': (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
            'sortino_ratio': (returns.mean() - self.risk_free_rate/252) / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
            'max_drawdown': results['Drawdown'].max(),
            'avg_long_exposure': avg_long_exposure,
            'avg_short_exposure': avg_short_exposure,
            'sector_correlations': {
                sector: self.sector_metrics[sector].correlation
                for sector in self.sector_metrics
            }
        }
        
        return results 