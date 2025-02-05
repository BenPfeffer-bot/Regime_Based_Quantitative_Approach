"""
Pairs Trading Strategy Module

This module implements a statistical arbitrage strategy that:
1. Identifies cointegrated pairs within the EUROSTOXX50
2. Generates trading signals based on spread mean reversion
3. Adjusts for market regimes
4. Implements risk management through position sizing and stop losses
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS


class PairsTrading:
    """Implementation of pairs trading strategy with regime awareness."""
    
    def __init__(
        self,
        lookback_period: int = 252,  # One year of trading days
        entry_threshold: float = 2.5,  # Increased from 2.0 for more conservative entries
        exit_threshold: float = 0.75,  # Increased from 0.5 for more conservative exits
        stop_loss_threshold: float = 3.5,  # Decreased from 4.0 for tighter risk control
        min_half_life: int = 5,
        max_half_life: int = 42,
        max_position_size: float = 0.08,  # Reduced from 0.1 for better risk management
        vol_lookback: int = 20,  # Window for volatility calculation
        regime_lookback: int = 50,  # Window for regime detection
        min_vol_threshold: float = 0.10,  # Minimum volatility for trading
        max_vol_threshold: float = 0.40,  # Maximum volatility for trading
        min_correlation: float = 0.5,  # Minimum correlation threshold
        correlation_lookback: int = 126,  # ~6 months
        correlation_stability_threshold: float = 0.7,  # Minimum correlation stability
        max_pairs_per_sector: int = 5,  # Maximum pairs per sector
        max_total_pairs: int = 20,  # Maximum total pairs to trade
    ):
        """Initialize strategy parameters."""
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.max_position_size = max_position_size
        self.vol_lookback = vol_lookback
        self.regime_lookback = regime_lookback
        self.min_vol_threshold = min_vol_threshold
        self.max_vol_threshold = max_vol_threshold
        
        # New correlation parameters
        self.min_correlation = min_correlation
        self.correlation_lookback = correlation_lookback
        self.correlation_stability_threshold = correlation_stability_threshold
        self.max_pairs_per_sector = max_pairs_per_sector
        self.max_total_pairs = max_total_pairs
        
        # Strategy state
        self.pairs = []
        self.current_positions = {}
        self.spread_means = {}
        self.spread_stds = {}
        self.hedge_ratios = {}
        self.pair_correlations = {}
        self.pair_sectors = {}
        self.correlation_metrics = {}
        self.pair_rankings = {}
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate the half-life of mean reversion for a spread."""
        lagged_spread = spread.shift(1)
        delta_spread = spread - lagged_spread
        
        # Remove NaN values
        spread = spread.dropna()
        lagged_spread = lagged_spread.dropna()
        delta_spread = delta_spread.dropna()
        
        # Regression of price changes against price levels
        model = OLS(delta_spread, lagged_spread).fit()
        half_life = -np.log(2) / model.params.iloc[0]
        
        return half_life
    
    def test_cointegration(
        self,
        price1: pd.Series,
        price2: pd.Series,
    ) -> Tuple[bool, float, float]:
        """
        Test for cointegration between two price series.
        
        Returns:
            Tuple of (is_cointegrated, p_value, hedge_ratio)
        """
        # Align price series and remove any NaN values
        df = pd.DataFrame({'price1': price1, 'price2': price2})
        df = df.dropna()
        
        if len(df) < self.lookback_period:
            return False, 1.0, 0.0
            
        # Calculate cointegration
        score, p_value, _ = coint(df['price1'], df['price2'])
        
        # Calculate hedge ratio using OLS
        model = OLS(df['price1'], df['price2']).fit()
        hedge_ratio = model.params.iloc[0]
        
        # Calculate spread
        spread = df['price1'] - hedge_ratio * df['price2']
        
        # Test for stationarity of spread
        adf_result = adfuller(spread)
        is_stationary = adf_result[1] < 0.05
        
        # Calculate half-life
        half_life = self.calculate_half_life(spread)
        
        # Check if pair meets our criteria
        is_cointegrated = (
            p_value < 0.05 and  # Cointegration test
            is_stationary and  # Spread stationarity
            self.min_half_life <= half_life <= self.max_half_life  # Mean reversion speed
        )
        
        return is_cointegrated, p_value, hedge_ratio
    
    def calculate_correlation_metrics(
        self,
        price1: pd.Series,
        price2: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive correlation metrics for a pair.
        
        Returns:
            Dictionary containing:
            - rolling_correlation: Current rolling correlation
            - correlation_stability: Stability of correlation over time
            - correlation_trend: Trend in correlation
            - beta: Beta between the two stocks
            - r_squared: R-squared of the relationship
        """
        # Calculate returns
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        
        # Calculate rolling correlation
        rolling_corr = returns1.rolling(window=self.correlation_lookback).corr(returns2)
        current_corr = rolling_corr.iloc[-1]
        
        # Calculate correlation stability (std dev of rolling correlation)
        corr_stability = 1 - rolling_corr.std()
        
        # Calculate correlation trend
        corr_trend = (
            rolling_corr.iloc[-int(self.correlation_lookback/4):].mean() -
            rolling_corr.iloc[-self.correlation_lookback:-int(self.correlation_lookback/4)].mean()
        )
        
        # Calculate beta and R-squared
        model = OLS(returns1.dropna(), returns2.dropna()).fit()
        beta = model.params.iloc[0]
        r_squared = model.rsquared
        
        return {
            'rolling_correlation': current_corr,
            'correlation_stability': corr_stability,
            'correlation_trend': corr_trend,
            'beta': beta,
            'r_squared': r_squared
        }

    def calculate_pair_score(
        self,
        correlation_metrics: Dict[str, float],
        cointegration_pvalue: float,
        half_life: float,
        sector_match: bool
    ) -> float:
        """Calculate overall score for a pair based on multiple factors."""
        # Base score from correlation metrics
        correlation_score = (
            correlation_metrics['rolling_correlation'] * 0.3 +
            correlation_metrics['correlation_stability'] * 0.2 +
            max(correlation_metrics['correlation_trend'], 0) * 0.1 +
            correlation_metrics['r_squared'] * 0.2
        )
        
        # Cointegration score (lower p-value is better)
        cointegration_score = 1 - cointegration_pvalue
        
        # Half-life score (prefer medium-term mean reversion)
        half_life_score = 1.0 - abs(half_life - (self.min_half_life + self.max_half_life) / 2) / self.max_half_life
        
        # Sector bonus
        sector_score = 0.2 if sector_match else 0.0
        
        # Combine scores with weights
        total_score = (
            correlation_score * 0.4 +
            cointegration_score * 0.3 +
            half_life_score * 0.2 +
            sector_score * 0.1
        )
        
        return total_score

    def identify_pairs(self, price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
        """
        Identify and rank cointegrated pairs from price data.
        
        Args:
            price_data: Dictionary of DataFrames with price data for each stock
            
        Returns:
            List of tuples containing pairs of stock symbols, ranked by quality
        """
        potential_pairs = []
        tickers = list(price_data.keys())
        
        # Convert all price series to the same index
        aligned_prices = {}
        for ticker in tickers:
            if 'close' in price_data[ticker].columns:
                if isinstance(price_data[ticker].columns, pd.MultiIndex):
                    close_col = [col for col in price_data[ticker].columns if col[1].lower() == 'close'][0]
                    aligned_prices[ticker] = price_data[ticker][close_col]
                else:
                    aligned_prices[ticker] = price_data[ticker]['close']
        
        # Create a common index
        common_index = None
        for price_series in aligned_prices.values():
            if common_index is None:
                common_index = price_series.index
            else:
                common_index = common_index.intersection(price_series.index)
        
        # Align all price series to the common index
        for ticker in aligned_prices:
            aligned_prices[ticker] = aligned_prices[ticker].reindex(common_index)
        
        # Extract sectors if available
        sectors = {}
        for ticker, data in price_data.items():
            if 'sector' in data.columns:
                sectors[ticker] = data['sector'].iloc[0]
            else:
                sectors[ticker] = 'Unknown'
        
        # Track pairs per sector
        sector_pair_counts = {}
        
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                ticker1, ticker2 = tickers[i], tickers[j]
                
                # Skip if either ticker is not in aligned_prices
                if ticker1 not in aligned_prices or ticker2 not in aligned_prices:
                    continue
                
                # Get aligned price series
                price1 = aligned_prices[ticker1]
                price2 = aligned_prices[ticker2]
                
                # Calculate correlation metrics first (faster than cointegration)
                corr_metrics = self.calculate_correlation_metrics(price1, price2)
                
                # Skip if correlation is too low or unstable
                if (corr_metrics['rolling_correlation'] < self.min_correlation or
                    corr_metrics['correlation_stability'] < self.correlation_stability_threshold):
                    continue
                
                # Test for cointegration
                is_cointegrated, p_value, hedge_ratio = self.test_cointegration(price1, price2)
                
                if is_cointegrated:
                    # Calculate spread and half-life
                    spread = price1 - hedge_ratio * price2
                    half_life = self.calculate_half_life(spread)
                    
                    # Check sector match
                    sector_match = sectors.get(ticker1) == sectors.get(ticker2)
                    
                    # Calculate pair score
                    pair_score = self.calculate_pair_score(
                        corr_metrics, p_value, half_life, sector_match
                    )
                    
                    potential_pairs.append({
                        'pair': (ticker1, ticker2),
                        'score': pair_score,
                        'sector': sectors.get(ticker1),
                        'metrics': corr_metrics,
                        'hedge_ratio': hedge_ratio,
                        'half_life': half_life
                    })
        
        # Sort pairs by score
        ranked_pairs = sorted(potential_pairs, key=lambda x: x['score'], reverse=True)
        
        # Select top pairs while respecting sector limits
        selected_pairs = []
        sector_counts = {}
        
        for pair_info in ranked_pairs:
            if len(selected_pairs) >= self.max_total_pairs:
                break
                
            ticker1, ticker2 = pair_info['pair']
            sector = pair_info['sector']
            
            # Check sector limits
            if sector_counts.get(sector, 0) >= self.max_pairs_per_sector:
                continue
            
            # Add pair
            selected_pairs.append(pair_info['pair'])
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # Store pair information
            self.hedge_ratios[pair_info['pair']] = pair_info['hedge_ratio']
            self.pair_correlations[pair_info['pair']] = pair_info['metrics']['rolling_correlation']
            self.correlation_metrics[pair_info['pair']] = pair_info['metrics']
            self.pair_rankings[pair_info['pair']] = pair_info['score']
            
            # Calculate spread statistics
            price1 = aligned_prices[ticker1]
            price2 = aligned_prices[ticker2]
            spread = price1 - pair_info['hedge_ratio'] * price2
            self.spread_means[pair_info['pair']] = spread.mean()
            self.spread_stds[pair_info['pair']] = spread.std()
        
        self.pairs = selected_pairs
        return selected_pairs
    
    def calculate_spread_zscore(
        self,
        price1: float,
        price2: float,
        pair: Tuple[str, str]
    ) -> float:
        """Calculate the z-score of the current spread."""
        hedge_ratio = self.hedge_ratios[pair]
        spread = price1 - hedge_ratio * price2
        zscore = (spread - self.spread_means[pair]) / self.spread_stds[pair]
        return zscore
    
    def calculate_position_sizes(
        self,
        zscore: float,
        price1: float,
        price2: float,
        capital: float,
        pair: Tuple[str, str]
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for both legs of the pair trade.
        
        Returns:
            Tuple of (size_1, size_2) representing position sizes in each stock
        """
        # Base position size on z-score magnitude and risk limits
        confidence = min(abs(zscore) / self.entry_threshold, 1.0)
        max_pair_exposure = self.max_position_size * capital * confidence
        
        # Calculate notional position sizes based on hedge ratio
        hedge_ratio = self.hedge_ratios[pair]
        total_exposure = price1 + hedge_ratio * price2
        
        # Adjust position sizes to maintain hedge ratio while respecting max exposure
        position_value = min(max_pair_exposure, capital * self.max_position_size)
        ratio_sum = 1 + abs(hedge_ratio)
        
        # Calculate number of shares for each leg
        size1 = (position_value / ratio_sum) / price1
        size2 = (position_value / ratio_sum * abs(hedge_ratio)) / price2
        
        return size1, size2
    
    def detect_regime(self, price_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> str:
        """
        Enhanced regime detection using multiple indicators.
        
        Returns:
            str: 'Range_Bound', 'Trending', or 'Volatile'
        """
        regime_scores = []
        
        for data in price_data.values():
            if date not in data.index:
                continue
                
            # Get relevant data
            returns = data['close'].pct_change()
            vol = returns.rolling(window=self.vol_lookback).std().loc[date]
            rsi = data.loc[date, 'rsi'] if 'rsi' in data.columns else 50
            
            # Check moving averages if available
            ma_trend = 0
            if all(col in data.columns for col in ['ema_50', 'ema_200']):
                ma_trend = 1 if data.loc[date, 'ema_50'] > data.loc[date, 'ema_200'] else -1
            
            # Calculate regime score
            regime_score = 0
            
            # Volatility check
            if vol < self.min_vol_threshold:
                regime_score += 1  # Favor range-bound
            elif vol > self.max_vol_threshold:
                regime_score -= 1  # Avoid volatile periods
            
            # RSI check
            if 40 <= rsi <= 60:
                regime_score += 1  # Range-bound
            elif rsi > 70 or rsi < 30:
                regime_score -= 1  # Trending/volatile
            
            # Moving average trend check
            if ma_trend != 0:
                regime_score -= abs(ma_trend)  # Trending
            
            regime_scores.append(regime_score)
        
        # Determine overall regime
        avg_score = np.mean(regime_scores) if regime_scores else 0
        
        if avg_score >= 0.5:
            return 'Range_Bound'
        elif avg_score <= -0.5:
            return 'Volatile'
        else:
            return 'Trending'

    def calculate_pair_correlation(self, price1: pd.Series, price2: pd.Series) -> float:
        """Calculate rolling correlation between two price series."""
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        correlation = returns1.rolling(window=self.regime_lookback).corr(returns2)
        return correlation.iloc[-1] if not pd.isna(correlation.iloc[-1]) else 0.0

    def generate_signals(
        self,
        current_prices: Dict[str, float],
        regime: str,
        capital: float,
        price_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Generate trading signals with enhanced regime awareness."""
        signals = {ticker: {'signal': 0, 'size': 0.0} for ticker in current_prices.keys()}
        
        # Only trade in range-bound regimes with acceptable volatility
        if regime != 'Range_Bound':
            return signals
        
        for pair in self.pairs:
            ticker1, ticker2 = pair
            
            # Skip if we don't have prices for both stocks
            if ticker1 not in current_prices or ticker2 not in current_prices:
                continue
            
            price1 = current_prices[ticker1]
            price2 = current_prices[ticker2]
            
            # Calculate spread z-score
            zscore = self.calculate_spread_zscore(price1, price2, pair)
            
            # Update correlation
            if date in price_data[ticker1].index and date in price_data[ticker2].index:
                correlation = self.calculate_pair_correlation(
                    price_data[ticker1]['close'],
                    price_data[ticker2]['close']
                )
                self.pair_correlations[pair] = correlation
            
            # Skip if correlation is too low
            if abs(self.pair_correlations.get(pair, 0)) < 0.5:
                continue
            
            # Check for entry signals with stricter criteria
            if abs(zscore) > self.entry_threshold:
                # Calculate volatility adjustment
                vol1 = price_data[ticker1]['close'].pct_change().rolling(window=self.vol_lookback).std().iloc[-1]
                vol2 = price_data[ticker2]['close'].pct_change().rolling(window=self.vol_lookback).std().iloc[-1]
                
                # Skip if either stock is too volatile
                if vol1 > self.max_vol_threshold or vol2 > self.max_vol_threshold:
                    continue
                
                # Calculate position sizes with volatility adjustment
                vol_adj = 1.0 - (max(vol1, vol2) / self.max_vol_threshold)
                size1, size2 = self.calculate_position_sizes(zscore, price1, price2, capital, pair)
                size1 *= vol_adj
                size2 *= vol_adj
                
                if zscore > self.entry_threshold:  # Spread is too high
                    signals[ticker1]['signal'] = -1
                    signals[ticker2]['signal'] = 1
                    signals[ticker1]['size'] = size1
                    signals[ticker2]['size'] = size2
                else:  # Spread is too low
                    signals[ticker1]['signal'] = 1
                    signals[ticker2]['signal'] = -1
                    signals[ticker1]['size'] = size1
                    signals[ticker2]['size'] = size2
            
            # Check for exit signals
            elif abs(zscore) < self.exit_threshold:
                if pair in self.current_positions:
                    signals[ticker1]['signal'] = -self.current_positions[pair][ticker1]
                    signals[ticker2]['signal'] = -self.current_positions[pair][ticker2]
                    signals[ticker1]['size'] = self.current_positions[pair][f'{ticker1}_size']
                    signals[ticker2]['size'] = self.current_positions[pair][f'{ticker2}_size']
            
            # Check stop loss with dynamic threshold
            elif abs(zscore) > self.stop_loss_threshold:
                if pair in self.current_positions:
                    signals[ticker1]['signal'] = -self.current_positions[pair][ticker1]
                    signals[ticker2]['signal'] = -self.current_positions[pair][ticker2]
                    signals[ticker1]['size'] = self.current_positions[pair][f'{ticker1}_size']
                    signals[ticker2]['size'] = self.current_positions[pair][f'{ticker2}_size']
        
        return signals
    
    def update_positions(
        self,
        executed_signals: Dict[str, Dict[str, Union[int, float]]]
    ) -> None:
        """Update current positions based on executed signals."""
        # Update positions for each pair
        for pair in self.pairs:
            ticker1, ticker2 = pair
            
            if ticker1 in executed_signals and ticker2 in executed_signals:
                signal1 = executed_signals[ticker1]['signal']
                signal2 = executed_signals[ticker2]['signal']
                
                if signal1 != 0 and signal2 != 0:  # New position or position update
                    self.current_positions[pair] = {
                        ticker1: signal1,
                        ticker2: signal2,
                        f'{ticker1}_size': executed_signals[ticker1]['size'],
                        f'{ticker2}_size': executed_signals[ticker2]['size']
                    }
                elif pair in self.current_positions:  # Position closed
                    del self.current_positions[pair]
    
    def backtest(
        self,
        price_data: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """
        Backtest the pairs trading strategy.
        
        Args:
            price_data: Dictionary of DataFrames with price data for each stock
            initial_capital: Starting capital
            
        Returns:
            DataFrame with backtest results
        """
        # First, identify pairs
        self.identify_pairs(price_data)
        
        # Initialize results
        results = pd.DataFrame()
        capital = initial_capital
        
        # Get common dates across all price data
        dates = sorted(set.intersection(*[set(df.index) for df in price_data.values()]))
        
        # Pre-calculate returns for all stocks
        returns = {
            ticker: data['close'].pct_change()
            for ticker, data in price_data.items()
        }
        
        # Initialize trade tracking
        trade_count = 0
        winning_trades = 0
        total_return = 0
        max_drawdown = 0
        peak_capital = initial_capital
        
        for i, date in enumerate(dates):
            # Get current prices
            current_prices = {
                ticker: data.loc[date, 'close']
                for ticker, data in price_data.items()
            }
            
            # Enhanced regime detection
            regime = self.detect_regime(price_data, date)
            
            # Generate signals with enhanced criteria
            signals = self.generate_signals(current_prices, regime, capital, price_data, date)
            
            # Track trades
            new_trades = sum(1 for s in signals.values() if s['signal'] != 0)
            if new_trades > 0:
                trade_count += new_trades
            
            # Update positions
            self.update_positions(signals)
            
            # Calculate P&L
            daily_pnl = 0.0
            for pair in self.current_positions:
                ticker1, ticker2 = pair
                pos = self.current_positions[pair]
                
                # Get returns for the day
                if date in returns[ticker1].index and date in returns[ticker2].index:
                    ret1 = returns[ticker1].loc[date]
                    ret2 = returns[ticker2].loc[date]
                    
                    # Skip if either return is NaN
                    if pd.isna(ret1) or pd.isna(ret2):
                        continue
                    
                    # Calculate P&L for each leg
                    pnl1 = pos[ticker1] * pos[f'{ticker1}_size'] * ret1 * current_prices[ticker1]
                    pnl2 = pos[ticker2] * pos[f'{ticker2}_size'] * ret2 * current_prices[ticker2]
                    
                    daily_pnl += pnl1 + pnl2
            
            # Update capital and track performance
            capital += daily_pnl
            if daily_pnl > 0:
                winning_trades += 1
            
            # Track drawdown
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
            
            # Record results
            results.loc[date, 'Capital'] = capital
            results.loc[date, 'Daily_PnL'] = daily_pnl
            results.loc[date, 'Regime'] = regime
            results.loc[date, 'Active_Pairs'] = len(self.current_positions)
            results.loc[date, 'Drawdown'] = drawdown
            results.loc[date, 'Returns'] = daily_pnl / capital if capital > 0 else 0
        
        # Calculate rolling metrics
        window = 20
        returns_series = pd.Series(results['Returns'].values, index=results.index)
        results['Volatility'] = returns_series.rolling(window=window).std()
        
        # Calculate rolling Sharpe ratio (annualized)
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        results['Rolling_Sharpe'] = np.sqrt(252) * (rolling_mean / rolling_std)
        
        # Add summary statistics
        results.attrs['summary'] = {
            'total_trades': trade_count,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / trade_count if trade_count > 0 else 0,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital,
            'sharpe_ratio': np.sqrt(252) * results['Returns'].mean() / results['Returns'].std()
            if results['Returns'].std() != 0 else 0,
            'pairs_identified': len(self.pairs),
            'avg_correlation': np.mean(list(self.pair_correlations.values()))
        }
        
        return results 