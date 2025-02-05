#!/usr/bin/env python3
"""
Volatility-Adjusted Momentum Backtesting Module

This module provides functionality to:
1. Run backtests using the volatility-adjusted momentum strategy
2. Generate performance reports
3. Create visualization plots
4. Save results in standardized format
"""

import sys
from pathlib import Path

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
import talib

from src.strategies.volatility_adjusted_momentum import VolatilityAdjustedMomentum
from src.config import DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.2)


class VolatilityMomentumBacktestRunner:
    """Class to run backtests and generate reports for volatility-adjusted momentum strategy."""
    
    def __init__(
        self,
        strategy: Optional[VolatilityAdjustedMomentum] = None,
        indicators_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        initial_capital: float = 100000.0,
        max_risk_pct: float = 0.02,
    ):
        """Initialize backtest runner."""
        self.strategy = None  # Will be initialized per stock
        self.indicators_dir = indicators_dir or Path("data/with_indicators")
        self.output_dir = output_dir or Path("data/backtest_results")
        self.initial_capital = initial_capital
        self.max_risk_pct = max_risk_pct
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory with timestamp
        self.run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_strategy(self, data: pd.DataFrame):
        """Initialize strategy with current stock data."""
        self.strategy = VolatilityAdjustedMomentum(data)
    
    def run_single_backtest(
        self,
        stock_data: pd.DataFrame,
        ticker: str,
    ) -> Dict[str, Any]:
        """Run backtest for a single stock."""
        # Initialize strategy with current stock data
        self._initialize_strategy(stock_data)
        
        # Initialize results
        position = 0
        position_size = 0.0
        entry_price = 0.0
        net_worth = self.initial_capital
        trades = []
        
        for i in range(len(stock_data)):
            # Generate signal for current timestamp
            signal, size = self.strategy.generate_signals(i)
            
            # Process signal
            if signal != 0 and position == 0:  # Entry signal
                position = signal
                position_size = size
                entry_price = stock_data['close'].iloc[i]
                trades.append({
                    'entry_date': stock_data.index[i],
                    'entry_price': entry_price,
                    'position': position,
                    'size': position_size
                })
            elif position != 0:  # Check for exit
                current_price = stock_data['close'].iloc[i]
                pnl = position * (current_price - entry_price) / entry_price
                
                # Exit on signal reversal or stop loss
                if (signal == -position) or (pnl < -self.max_risk_pct):
                    # Update last trade with exit details
                    trades[-1].update({
                        'exit_date': stock_data.index[i],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl * 100
                    })
                    
                    # Update net worth
                    net_worth *= (1 + pnl * position_size)
                    
                    # Reset position
                    position = 0
                    position_size = 0.0
                    entry_price = 0.0
        
        # Close any open position at the end
        if position != 0:
            current_price = stock_data['close'].iloc[-1]
            pnl = position * (current_price - entry_price) / entry_price
            trades[-1].update({
                'exit_date': stock_data.index[-1],
                'exit_price': current_price,
                'pnl': pnl,
                'pnl_pct': pnl * 100
            })
            net_worth *= (1 + pnl * position_size)
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_profit = sum(t.get('pnl_pct', 0) for t in trades) / total_trades if total_trades > 0 else 0
        
        return {
            'ticker': ticker,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_profit_pct': avg_profit,
            'final_net_worth': net_worth,
            'trades': trades
        }

    def plot_backtest_results(
        self,
        df: pd.DataFrame,
        ticker: str,
        output_dir: Path,
    ) -> None:
        """Plot backtest results for a single stock."""
        try:
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(col).lower() for col in df.columns]
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), height_ratios=[2, 1, 1])
            
            # Plot price and signals
            df['close'].plot(ax=ax1, label='Price')
            ax1.set_title(f'{ticker} - Price and Signals')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Plot RSI
            rsi = talib.RSI(df['close'].values, timeperiod=14)
            pd.Series(rsi, index=df.index).plot(ax=ax2, label='RSI')
            ax2.axhline(y=30, color='r', linestyle='--')
            ax2.axhline(y=70, color='r', linestyle='--')
            ax2.set_title('RSI')
            ax2.set_ylabel('RSI')
            ax2.grid(True)
            ax2.legend()
            
            # Plot ATR
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            pd.Series(atr, index=df.index).plot(ax=ax3, label='ATR')
            ax3.set_title('ATR')
            ax3.set_ylabel('ATR')
            ax3.grid(True)
            ax3.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_dir / f"{ticker}_backtest.png")
            plt.close()
            
        except Exception as e:
            logging.error(f"Error plotting results for {ticker}: {str(e)}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save backtest results to file."""
        # Convert Timestamp objects to string format
        def convert_timestamps(obj):
            if isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            return obj
        
        # Convert timestamps in results
        results_json = convert_timestamps(results)
        
        # Save summary report
        with open(self.run_dir / "report.json", "w") as f:
            json.dump(results_json, f, indent=4)
        
        # Log summary statistics
        if results["runs"]:
            logging.info("\nBacktest Summary:")
            logging.info(f"Total number of trades: {results['total_trades']}")
            logging.info(f"Average trades per stock: {results['average_trades_per_stock']:.1f}")
            logging.info(f"Average profit per trade: {results['average_profit_per_trade']:.4%}")
            logging.info(f"Average win rate: {results['average_win_rate']:.2%}")
            logging.info(f"Best performing ticker: {results['best_performing_ticker']}")
            logging.info(f"Worst performing ticker: {results['worst_performing_ticker']}")

    def run_all_backtests(self) -> Dict[str, Any]:
        """Run backtests on all stocks in the data directory."""
        all_results = {"runs": []}
        total_trades = 0
        
        # Get list of stock data files
        stock_files = list(self.indicators_dir.glob("*_indicators.parquet"))
        logging.info(f"Found {len(stock_files)} stocks to backtest")
        
        for i, file_path in enumerate(stock_files, 1):
            ticker = file_path.stem.replace("_indicators", "")
            logging.info(f"Running backtest {i}/{len(stock_files)}: {ticker}")
            
            try:
                # Load and prepare data
                df = pd.read_parquet(file_path)
                
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [str(col).lower() for col in df.columns]
                
                # Run backtest
                results = self.run_single_backtest(df, ticker)
                
                # Save individual results
                results["ticker"] = ticker
                all_results["runs"].append(results)
                
                # Update total trades
                total_trades += results["total_trades"]
                
                # Plot results
                self.plot_backtest_results(df, ticker, self.run_dir)
                
            except Exception as e:
                logging.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Calculate aggregate metrics
        if all_results["runs"]:
            all_results.update({
                "total_trades": total_trades,
                "average_trades_per_stock": total_trades / len(all_results["runs"]),
                "average_profit_per_trade": np.mean([run["avg_profit_pct"] for run in all_results["runs"]]),
                "average_win_rate": np.mean([run["win_rate"] for run in all_results["runs"]]),
                "average_final_net_worth": np.mean([run["final_net_worth"] for run in all_results["runs"]]),
                "best_performing_ticker": max(all_results["runs"], key=lambda x: x["final_net_worth"])["ticker"],
                "worst_performing_ticker": min(all_results["runs"], key=lambda x: x["final_net_worth"])["ticker"]
            })
        else:
            all_results.update({
                "total_trades": 0,
                "average_trades_per_stock": 0,
                "average_profit_per_trade": 0,
                "average_win_rate": 0,
                "average_final_net_worth": 0,
                "best_performing_ticker": None,
                "worst_performing_ticker": None
            })
        
        # Save results
        self._save_results(all_results)
        
        return all_results


def main():
    """Main function to run backtests."""
    # Initialize and run backtests
    runner = VolatilityMomentumBacktestRunner()
    results = runner.run_all_backtests()
    
    # Log summary
    logging.info(f"\nCompleted {len(results['runs'])} backtests")
    logging.info(f"Total trades: {results['total_trades']}")
    logging.info(f"Average final net worth: {results['average_final_net_worth']:.4f}")
    logging.info(f"Results saved in: {runner.run_dir}")


if __name__ == "__main__":
    main() 