#!/usr/bin/env python3
"""
Pairs Trading Backtesting Module

This module provides functionality to:
1. Run backtests using the pairs trading strategy
2. Generate performance reports
3. Create visualization plots
4. Save results in standardized format
"""

import sys
from pathlib import Path
import os

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.strategies.pairs_trading import PairsTrading
from src.config import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)


class PairsBacktestRunner:
    """Class to run backtests and generate reports for pairs trading strategy."""
    
    def __init__(
        self,
        strategy: Optional[PairsTrading] = None,
        indicators_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        initial_capital: float = 100000.0,
    ):
        """Initialize backtest runner."""
        self.strategy = strategy or PairsTrading()
        self.indicators_dir = indicators_dir or DATA_DIR / "with_indicators"
        self.output_dir = output_dir or DATA_DIR / "backtest_results"
        self.initial_capital = initial_capital
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory with timestamp
        self.run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        """Load price data for all stocks."""
        price_data = {}
        
        # Get list of stock data files
        stock_files = list(self.indicators_dir.glob("*_indicators.parquet"))
        
        for file_path in stock_files:
            ticker = file_path.stem.replace("_indicators", "")
            df = pd.read_parquet(file_path)
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(col).lower() for col in df.columns]
            
            price_data[ticker] = df
        
        return price_data
    
    def plot_pair_results(
        self,
        pair: Tuple[str, str],
        price_data: Dict[str, pd.DataFrame],
        signals: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        """Plot results for a single pair."""
        ticker1, ticker2 = pair
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), height_ratios=[2, 1, 1])
        
        # Plot normalized prices
        price1 = price_data[ticker1]['close']
        price2 = price_data[ticker2]['close']
        
        norm_price1 = price1 / price1.iloc[0]
        norm_price2 = price2 / price2.iloc[0]
        
        ax1.plot(norm_price1.index, norm_price1, label=ticker1)
        ax1.plot(norm_price2.index, norm_price2, label=ticker2)
        ax1.set_title(f'Normalized Prices - {ticker1} vs {ticker2}')
        ax1.legend()
        ax1.grid(True)
        
        # Plot spread
        hedge_ratio = self.strategy.hedge_ratios.get(pair)
        if hedge_ratio is not None:
            spread = price1 - hedge_ratio * price2
            zscore = (spread - spread.mean()) / spread.std()
            
            ax2.plot(zscore.index, zscore, label='Z-Score', color='blue')
            ax2.axhline(y=self.strategy.entry_threshold, color='r', linestyle='--')
            ax2.axhline(y=-self.strategy.entry_threshold, color='r', linestyle='--')
            ax2.axhline(y=self.strategy.exit_threshold, color='g', linestyle='--')
            ax2.axhline(y=-self.strategy.exit_threshold, color='g', linestyle='--')
            ax2.set_title('Spread Z-Score')
            ax2.legend()
            ax2.grid(True)
        
        # Plot P&L
        if 'Daily_PnL' in signals.columns:
            cumulative_pnl = signals['Daily_PnL'].cumsum()
            ax3.plot(cumulative_pnl.index, cumulative_pnl, label='Cumulative P&L', color='green')
            ax3.set_title('Cumulative P&L')
            ax3.legend()
            ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"pair_{ticker1}_{ticker2}_backtest.png")
        plt.close()
    
    def calculate_pair_metrics(
        self,
        results: pd.DataFrame,
        pair: Tuple[str, str]
    ) -> Dict[str, float]:
        """Calculate performance metrics for a pair."""
        returns = results['Daily_PnL'] / self.initial_capital
        
        metrics = {
            'pair': f"{pair[0]}_{pair[1]}",
            'total_return': float(((results['Capital'].iloc[-1] / self.initial_capital) - 1) * 100),
            'annualized_return': float(returns.mean() * 252 * 100),
            'annualized_volatility': float(returns.std() * np.sqrt(252) * 100),
            'sharpe_ratio': float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() != 0 else 0,
            'max_drawdown': float(self._calculate_max_drawdown(results['Capital']) * 100),
            'win_rate': float((returns > 0).mean() * 100),
            'avg_profit_per_trade': float(returns[returns > 0].mean() * 100) if len(returns[returns > 0]) > 0 else 0,
            'avg_loss_per_trade': float(returns[returns < 0].mean() * 100) if len(returns[returns < 0]) > 0 else 0,
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from peak."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1.0
        return abs(drawdowns.min())
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save backtest results to disk."""
        # Create results directory if it doesn't exist
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save summary report
        report_path = os.path.join(self.run_dir, 'report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Log summary statistics
        logging.info("\nBacktest Summary:")
        logging.info(f"Number of pairs identified: {results['pairs']}")
        logging.info(f"Total number of trades: {results['total_trades']}")
        logging.info(f"Win rate: {results['win_rate']:.2f}%")
        logging.info(f"Maximum drawdown: {results['max_drawdown']:.2f}%")
        logging.info(f"Final capital: ${results['final_capital']:.2f}")
        logging.info(f"Total return: {results['total_return']:.2f}%")
        logging.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logging.info(f"Sortino ratio: {results['sortino_ratio']:.2f}")
        logging.info(f"Annualized volatility: {results['annualized_volatility']:.2f}%")
        logging.info(f"Average pair correlation: {results['avg_correlation']:.2f}")
        
        # Log regime distribution
        logging.info("\nRegime Distribution:")
        for regime, count in results['regime_distribution'].items():
            total_days = sum(results['regime_distribution'].values())
            percentage = (count / total_days) * 100
            logging.info(f"{regime}: {count} days ({percentage:.1f}%)")
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run backtest on all pairs."""
        # Load price data
        price_data = self.load_price_data()
        logging.info(f"Loaded data for {len(price_data)} stocks")
        
        # Initialize strategy and identify pairs
        pairs = self.strategy.identify_pairs(price_data)
        logging.info(f"Identified {len(pairs)} cointegrated pairs")
        
        # Run backtest
        results = self.strategy.backtest(price_data, self.initial_capital)
        
        # Get summary statistics
        summary = results.attrs.get('summary', {})
        
        # Calculate additional metrics
        volatility = results['Returns'].std() * np.sqrt(252)  # Annualized volatility
        sortino_ratio = (
            np.sqrt(252) * results['Returns'].mean() / 
            results['Returns'][results['Returns'] < 0].std()
            if len(results['Returns'][results['Returns'] < 0]) > 0 else 0
        )
        
        # Aggregate results
        aggregate_results = {
            "pairs": len(pairs),
            "total_trades": summary.get('total_trades', 0),
            "win_rate": summary.get('win_rate', 0) * 100,  # Convert to percentage
            "max_drawdown": summary.get('max_drawdown', 0) * 100,  # Convert to percentage
            "final_capital": summary.get('final_capital', self.initial_capital),
            "total_return": summary.get('total_return', 0) * 100,  # Convert to percentage
            "sharpe_ratio": summary.get('sharpe_ratio', 0),
            "sortino_ratio": sortino_ratio,
            "annualized_volatility": volatility * 100,  # Convert to percentage
            "avg_correlation": summary.get('avg_correlation', 0),
            "regime_distribution": results['Regime'].value_counts().to_dict()
        }
        
        # Log detailed results
        logging.info("\nBacktest Summary:")
        logging.info(f"Number of pairs identified: {aggregate_results['pairs']}")
        logging.info(f"Total number of trades: {aggregate_results['total_trades']}")
        logging.info(f"Win rate: {aggregate_results['win_rate']:.2f}%")
        logging.info(f"Maximum drawdown: {aggregate_results['max_drawdown']:.2f}%")
        logging.info(f"Final capital: ${aggregate_results['final_capital']:.2f}")
        logging.info(f"Total return: {aggregate_results['total_return']:.2f}%")
        logging.info(f"Sharpe ratio: {aggregate_results['sharpe_ratio']:.2f}")
        logging.info(f"Sortino ratio: {aggregate_results['sortino_ratio']:.2f}")
        logging.info(f"Annualized volatility: {aggregate_results['annualized_volatility']:.2f}%")
        logging.info(f"Average pair correlation: {aggregate_results['avg_correlation']:.2f}")
        logging.info("\nRegime Distribution:")
        for regime, count in aggregate_results['regime_distribution'].items():
            logging.info(f"{regime}: {count} days ({count/len(results)*100:.1f}%)")
        
        # Save results
        self._save_results(aggregate_results)
        
        return aggregate_results


def main():
    """Main function to run pairs trading backtest."""
    # Initialize and run backtest
    runner = PairsBacktestRunner()
    results = runner.run_backtest()
    
    # Log summary
    logging.info(f"\nCompleted pairs trading backtest")
    logging.info(f"Number of pairs identified: {results['pairs']}")
    logging.info(f"Final capital: ${results['final_capital']:.2f}")
    logging.info(f"Total return: {results['total_return']:.2f}%")
    logging.info(f"Results saved in: {runner.run_dir}")


if __name__ == "__main__":
    main() 