#!/usr/bin/env python3
"""
Sector Rotation Backtesting Module

This module provides functionality to:
1. Run backtests using the sector rotation strategy
2. Generate performance reports and visualizations
3. Save results in standardized format
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

from src.strategies.sector_rotation import AdaptiveSectorRotation
from src.config import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)


class SectorRotationBacktestRunner:
    """Class to run backtests and generate reports for sector rotation strategy."""
    
    def __init__(
        self,
        strategy: Optional[AdaptiveSectorRotation] = None,
        indicators_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        initial_capital: float = 100000.0,
    ):
        """Initialize backtest runner."""
        self.strategy = strategy or AdaptiveSectorRotation()
        self.indicators_dir = indicators_dir or DATA_DIR / "with_indicators"
        self.output_dir = output_dir or DATA_DIR / "backtest_results"
        self.initial_capital = initial_capital
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory with timestamp
        self.run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
        """
        Load price data for all stocks and market index.
        
        Returns:
            Tuple of (sector_data, market_data)
        """
        # Get list of stock data files
        stock_files = list(self.indicators_dir.glob("*_indicators.parquet"))
        
        # Initialize sector data structure
        sector_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Load stock data and organize by sector
        for file_path in stock_files:
            ticker = file_path.stem.replace("_indicators", "")
            df = pd.read_parquet(file_path)
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(col).lower() for col in df.columns]
            
            # Get sector information
            sector = df['sector'].iloc[0] if 'sector' in df.columns else 'Unknown'
            
            # Initialize sector if not exists
            if sector not in sector_data:
                sector_data[sector] = {}
            
            # Add stock data to sector
            sector_data[sector][ticker] = df
        
        # Load or create market index data
        market_data = self._create_market_index(sector_data)
        
        return sector_data, market_data
    
    def _create_market_index(
        self,
        sector_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        """Create market index from constituent stocks."""
        # Get all unique dates
        all_dates = set()
        for sector in sector_data.values():
            for df in sector.values():
                all_dates.update(df.index)
        
        # Create index DataFrame
        market_index = pd.DataFrame(index=sorted(all_dates))
        
        # Calculate market-cap weighted index
        total_mcap = pd.Series(0.0, index=market_index.index)
        weighted_prices = pd.Series(0.0, index=market_index.index)
        
        for sector in sector_data.values():
            for df in sector.values():
                if 'market_cap' in df.columns and 'close' in df.columns:
                    mcap = df['market_cap']
                    price = df['close']
                    total_mcap = total_mcap.add(mcap, fill_value=0)
                    weighted_prices = weighted_prices.add(price * mcap, fill_value=0)
        
        # Calculate index values
        market_index['close'] = weighted_prices / total_mcap
        market_index['volume'] = 0  # Placeholder for volume
        
        return market_index
    
    def plot_sector_weights(
        self,
        results: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Plot sector weight evolution over time."""
        # Get sector weight columns
        weight_cols = [col for col in results.columns if col.startswith('Sector_Weight_')]
        
        if not weight_cols:
            return
        
        # Create stacked area plot
        plt.figure(figsize=(15, 8))
        plt.stackplot(
            results.index,
            [results[col] for col in weight_cols],
            labels=[col.replace('Sector_Weight_', '') for col in weight_cols]
        )
        
        plt.title('Sector Weight Evolution')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / "sector_weights.png")
        plt.close()
    
    def plot_performance(
        self,
        results: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Plot strategy performance metrics."""
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot equity curve
        results['Capital'].plot(ax=ax1)
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital')
        ax1.grid(True)
        
        # Plot drawdown
        results['Drawdown'].plot(ax=ax2, color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True)
        
        # Plot rolling Sharpe ratio
        window = 126  # ~6 months
        returns = results['Returns']
        rolling_sharpe = (
            returns.rolling(window=window).mean() /
            returns.rolling(window=window).std()
        ) * np.sqrt(252)
        
        rolling_sharpe.plot(ax=ax3, color='green')
        ax3.set_title('Rolling Sharpe Ratio (6-month)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_metrics.png")
        plt.close()
    
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
        logging.info(f"Total trades: {results['total_trades']}")
        logging.info(f"Win rate: {results['win_rate']:.2f}%")
        logging.info(f"Maximum drawdown: {results['max_drawdown']:.2f}%")
        logging.info(f"Final capital: ${results['final_capital']:.2f}")
        logging.info(f"Total return: {results['total_return']:.2f}%")
        logging.info(f"Annualized return: {results['annualized_return']:.2f}%")
        logging.info(f"Annualized volatility: {results['annualized_volatility']:.2f}%")
        logging.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logging.info(f"Sortino ratio: {results['sortino_ratio']:.2f}")
        
        # Log sector correlations
        logging.info("\nSector Correlations:")
        for sector, corr in results['sector_correlations'].items():
            logging.info(f"{sector}: {corr:.2f}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run backtest on all sectors."""
        # Load data
        sector_data, market_data = self.load_data()
        logging.info(f"Loaded data for {sum(len(stocks) for stocks in sector_data.values())} stocks in {len(sector_data)} sectors")
        
        # Run backtest
        results = self.strategy.backtest(sector_data, market_data, self.initial_capital)
        
        # Get summary statistics
        summary = results.attrs.get('summary', {})
        
        # Create plots
        self.plot_sector_weights(results, self.run_dir)
        self.plot_performance(results, self.run_dir)
        
        # Save results
        self._save_results(summary)
        
        return summary


def main():
    """Main function to run sector rotation backtest."""
    # Initialize and run backtest
    runner = SectorRotationBacktestRunner()
    results = runner.run_backtest()
    
    # Log summary
    logging.info(f"\nCompleted sector rotation backtest")
    logging.info(f"Final capital: ${results['final_capital']:.2f}")
    logging.info(f"Total return: {results['total_return']:.2f}%")
    logging.info(f"Results saved in: {runner.run_dir}")


if __name__ == "__main__":
    main() 