#!/usr/bin/env python3
"""
Adaptive Backtesting Module

A comprehensive backtesting framework for financial trading strategies that provides:
1. Automated backtesting of trading strategies across multiple instruments
2. Performance analysis and metric calculation
3. Visualization of trading results and market regimes
4. Standardized reporting and result storage

Key Features:
- Multi-instrument backtesting support
- Regime-aware strategy testing
- Performance visualization
- Automated report generation
- Risk management integration

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- logging

Version: 1.0.0
Last Updated: 2025-02-05
"""


import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.strategies.adaptive_multi_regime import AdaptiveMultiRegimeStrategy
from src.config import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)


class BacktestRunner:
    """
    A comprehensive backtesting framework for evaluating trading strategies.
    
    This class handles the complete backtesting workflow including:
    - Data loading and preprocessing
    - Strategy execution
    - Performance measurement
    - Result visualization
    - Report generation
    
    Attributes:
        strategy (AdaptiveMultiRegimeStrategy): Trading strategy instance
        indicators_dir (Path): Directory containing pre-computed indicator files
        output_dir (Path): Directory for storing backtest results
        initial_capital (float): Starting capital for each backtest
        max_risk_pct (float): Maximum risk percentage per trade
        run_dir (Path): Unique directory for current backtest run
    
    Example:
        ```python
        # Basic usage
        runner = BacktestRunner(
            initial_capital=100000.0,
            max_risk_pct=0.2
        )
        results = runner.run_all_backtests()
        
        # Custom strategy and directories
        custom_strategy = AdaptiveMultiRegimeStrategy(
            lookback_period=50,
            volatility_window=20
        )
        runner = BacktestRunner(
            strategy=custom_strategy,
            indicators_dir=Path("./data/indicators"),
            output_dir=Path("./results"),
            initial_capital=200000.0
        )
        results = runner.run_all_backtests()
        ```
    """
    
    def __init__(
        self,
        strategy: Optional[AdaptiveMultiRegimeStrategy] = None,
        indicators_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        initial_capital: float = 100000.0,
        max_risk_pct: float = 0.2,
    ):
        """Initialize backtest runner."""
        self.strategy = strategy or AdaptiveMultiRegimeStrategy()
        self.indicators_dir = indicators_dir or DATA_DIR / "with_indicators"
        self.output_dir = output_dir or DATA_DIR / "backtest_results"
        self.initial_capital = initial_capital
        self.max_risk_pct = max_risk_pct
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory with timestamp
        self.run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run_single_backtest(
    self, 
    df: pd.DataFrame, 
    ticker: str
) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Execute backtest for a single financial instrument.
        
        Performs a complete backtest including:
        1. Strategy execution
        2. Performance tracking
        3. Metrics calculation
        
        Args:
            df (pd.DataFrame): DataFrame containing price data and indicators
                Required columns: ['Close', 'Open', 'High', 'Low', technical indicators...]
            ticker (str): Instrument identifier
            
        Returns:
            Tuple containing:
                - pd.DataFrame: Detailed backtest results including:
                    - Signal: Trading signals (-1, 0, 1)
                    - Equity: Portfolio value
                    - Regime: Market regime classification
                - Dict[str, float]: Performance metrics including:
                    - final_net_worth: Final portfolio value ratio
                    - total_return: Percentage return
                    - sharpe_ratio: Risk-adjusted return metric
                    
        Raises:
            ValueError: If required columns are missing from input DataFrame
            Exception: Strategy-specific execution errors
            
        Example:
            ```python
            # Load data
            df = pd.read_parquet("AAPL_indicators.parquet")
            
            # Run backtest
            results, metrics = runner.run_single_backtest(df, "AAPL")
            
            # Access results
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            ```
        """
        try:
            # Run backtest
            results = self.strategy.backtest(
                df=df,
                ticker=ticker,
                initial_capital=self.initial_capital,
                max_risk_pct=self.max_risk_pct
            )
            
            # Calculate metrics
            metrics = self._calculate_backtest_metrics(results)
            
            return results, metrics
            
        except Exception as e:
            logging.error(f"Error in run_single_backtest for {ticker}: {str(e)}")
            raise

    def _calculate_backtest_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a single backtest."""
        returns = results["Equity"].pct_change()
        
        metrics = {
            "final_net_worth": float(results["Equity"].iloc[-1] / self.initial_capital),
            "total_return": float(((results["Equity"].iloc[-1] / self.initial_capital) - 1) * 100),
            "num_steps": int(len(results)),
            "avg_profit_per_trade": float(returns[results["Signal"] != 0].mean()),
            "profitable_trades_ratio": float((returns[results["Signal"] != 0] > 0).mean()),
            "num_trades": int((results["Signal"] != 0).sum()),
            "annualized_return": float(returns.mean() * 252),
            "annualized_volatility": float(returns.std() * np.sqrt(252)),
            "sharpe_ratio": float((returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0),
            "max_drawdown": float(self._calculate_max_drawdown(results["Equity"])),
            "regime_distribution": {k: int(v) for k, v in results["Regime"].value_counts().to_dict().items()}
        }
        
        return metrics

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from peak."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1.0
        return abs(drawdowns.min())

    def plot_backtest_results(
        self, 
        results: pd.DataFrame, 
        ticker: str, 
        output_path: Path
    ) -> None:
        """Create visualization of backtest results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        fig.suptitle(f"Backtest Results - {ticker}", y=0.95)
        
        # Price chart with entries/exits
        ax1.plot(results.index, results["Close"], label="Price", color="black", alpha=0.7)
        
        # Plot long entries and exits
        long_entries = results[results["Signal"] == 1].index
        long_exits = results[results["Signal"] == -1].index
        
        ax1.scatter(long_entries, results.loc[long_entries, "Close"], 
                   marker="^", color="green", s=100, label="Long Entry")
        ax1.scatter(long_exits, results.loc[long_exits, "Close"], 
                   marker="x", color="red", s=100, label="Exit")
        
        # Add regime background
        for regime in results["Regime"].unique():
            mask = results["Regime"] == regime
            ax1.fill_between(results.index, results["Close"].min(), results["Close"].max(),
                           where=mask, alpha=0.2,
                           color="green" if regime == "Trending" else "orange",
                           label=f"{regime} Regime")
        
        ax1.set_title("Price Chart with Long and Short Entries")
        ax1.legend()
        ax1.grid(True)
        
        # Net worth chart
        ax2.plot(results.index, results["Equity"], label="Net Worth", color="blue")
        ax2.plot(results.index, results["Capital"], label="Unrealized Net Worth", 
                color="orange", alpha=0.7)
        
        ax2.set_title("Net Worth Chart")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / f"chart_{ticker}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def run_all_backtests(self) -> Dict:
        """Run backtests on all pre-computed indicator files."""
        all_results = {
            "experiment_name": self.run_dir.name,
            "num_runs": 0,
            "average_final_net_worth": 0.0,
            "num_trades": 0,
            "runs": []
        }
        
        # Find all parquet files with indicators
        data_files = list(self.indicators_dir.glob("*_indicators.parquet"))
        all_results["num_runs"] = len(data_files)
        
        logging.info(f"Found {len(data_files)} stocks to backtest")
        
        total_trades = 0
        
        for i, file_path in enumerate(data_files):
            # Extract ticker from filename (remove _indicators.parquet)
            ticker = file_path.stem.replace("_indicators", "")
            logging.info(f"Running backtest {i+1}/{len(data_files)}: {ticker}")
            
            try:
                # Load pre-computed indicators
                df = pd.read_parquet(file_path)
                
                # Run backtest
                results, metrics = self.run_single_backtest(df, ticker)
                
                # Save individual results
                metrics["ticker"] = ticker
                all_results["runs"].append(metrics)
                
                # Update total trades
                total_trades += metrics["num_trades"]
                
                # Plot results
                self.plot_backtest_results(results, ticker, self.run_dir)
                
                # # Save detailed results
                # results.to_parquet(self.run_dir / f"results_{ticker}.parquet")
                
            except Exception as e:
                logging.error(f"Error processing {ticker}: {str(e)}")
                continue  # Skip to next file on error
        
        # Update total number of trades
        all_results["num_trades"] = total_trades
        
        # Calculate aggregate metrics
        if all_results["runs"]:
            all_results["average_final_net_worth"] = np.mean(
                [run["final_net_worth"] for run in all_results["runs"]]
            )
            
            # Add additional aggregate metrics
            all_results["summary"] = {
                "total_trades": total_trades,
                "average_trades_per_stock": total_trades / len(all_results["runs"]),
                "average_profit_per_trade": np.mean([run["avg_profit_per_trade"] for run in all_results["runs"]]),
                "average_win_rate": np.mean([run["profitable_trades_ratio"] for run in all_results["runs"]]),
                "average_sharpe": np.mean([run["sharpe_ratio"] for run in all_results["runs"]]),
                "best_performing_ticker": max(all_results["runs"], key=lambda x: x["total_return"])["ticker"],
                "worst_performing_ticker": min(all_results["runs"], key=lambda x: x["total_return"])["ticker"]
            }
        
        # Save summary report
        with open(self.run_dir / "report.json", "w") as f:
            json.dump(all_results, f, indent=4)
        
        # Log summary statistics
        if all_results["runs"]:
            logging.info("\nBacktest Summary:")
            logging.info(f"Total number of trades: {all_results['summary']['total_trades']}")
            logging.info(f"Average trades per stock: {all_results['summary']['average_trades_per_stock']:.1f}")
            logging.info(f"Average profit per trade: {all_results['summary']['average_profit_per_trade']:.4%}")
            logging.info(f"Average win rate: {all_results['summary']['average_win_rate']:.2%}")
            logging.info(f"Average Sharpe ratio: {all_results['summary']['average_sharpe']:.2f}")
            logging.info(f"Best performing ticker: {all_results['summary']['best_performing_ticker']}")
            logging.info(f"Worst performing ticker: {all_results['summary']['worst_performing_ticker']}")
        
        return all_results


def main():
    """Main function to run backtests."""
    # Initialize and run backtests
    runner = BacktestRunner()
    results = runner.run_all_backtests()
    
    # Log summary
    logging.info(f"\nCompleted {results['num_runs']} backtests")
    logging.info(f"Total trades: {results['num_trades']}")
    logging.info(f"Average final net worth: {results['average_final_net_worth']:.4f}")
    logging.info(f"Results saved in: {runner.run_dir}")


if __name__ == "__main__":
    main()