"""
ML Ensemble Strategy Backtest

This script runs a backtest of the ML ensemble strategy on historical data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from dvclive import Live
import json
import sys

from src.strategies.ml_ensemble import MLEnsembleStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(data_dir: str = 'data/with_indicators') -> tuple:
    """Load historical price and volume data."""
    data_path = Path(data_dir)
    stock_data = {}
    
    # Load all stock data files
    for file_path in data_path.glob('*_indicators.parquet'):
        ticker = file_path.stem.replace('_indicators', '')
        df = pd.read_parquet(file_path)
        
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.lower() for col in df.columns]
        
        stock_data[ticker] = df
    
    if not stock_data:
        raise ValueError(f"No data files found in {data_path}")
    
    # Create market index from stock data
    market_data = pd.DataFrame()
    
    # Get common dates across all stocks
    common_dates = None
    for df in stock_data.values():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(df.index)
    
    common_dates = sorted(common_dates)
    
    if not common_dates:
        raise ValueError("No common dates found across stocks")
    
    # Calculate market-cap weighted index
    market_data = pd.DataFrame(index=common_dates)
    total_mcap = pd.Series(0.0, index=common_dates)
    weighted_prices = pd.Series(0.0, index=common_dates)
    total_volume = pd.Series(0.0, index=common_dates)
    
    for df in stock_data.values():
        df = df.loc[common_dates]
        if 'market_cap' in df.columns:
            mcap = df['market_cap']
        else:
            # If market cap not available, use equal weights
            mcap = pd.Series(1.0, index=common_dates)
        
        total_mcap = total_mcap.add(mcap)
        weighted_prices = weighted_prices.add(df['close'] * mcap)
        total_volume = total_volume.add(df['volume'])
    
    market_data['close'] = weighted_prices / total_mcap
    market_data['volume'] = total_volume
    market_data['high'] = market_data['close'] * 1.001  # Approximate high/low
    market_data['low'] = market_data['close'] * 0.999
    
    logging.info(f"Loaded data for {len(stock_data)} stocks")
    logging.info(f"Market data spans from {market_data.index[0]} to {market_data.index[-1]}")
    
    return stock_data, market_data

def plot_performance_metrics(results: pd.DataFrame, save_dir: Path, live: Live) -> None:
    """Plot and save performance metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot equity curve
    plt.subplot(2, 2, 1)
    results['Capital'].plot()
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    
    # Plot returns distribution
    plt.subplot(2, 2, 2)
    returns = results['Returns'].dropna()
    sns.histplot(returns, kde=True)
    plt.title('Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    
    # Plot drawdown
    plt.subplot(2, 2, 3)
    drawdown = (results['Capital'].cummax() - results['Capital']) / results['Capital'].cummax()
    drawdown.plot()
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    
    # Plot position sizes
    plt.subplot(2, 2, 4)
    results['Position'].plot()
    plt.title('Position Sizes')
    plt.xlabel('Date')
    plt.ylabel('Position Size')
    
    plt.tight_layout()
    
    # Save plot using DVCLive
    live.log_image('performance_metrics.png', plt.gcf())
    plt.close()

def plot_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    live: Live
) -> None:
    """Plot and save model evaluation metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot confusion matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot ROC curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    
    # Save plot using DVCLive
    live.log_image('model_metrics.png', plt.gcf())
    plt.close()

def save_results(results: pd.DataFrame, metrics: dict, save_dir: Path, live: Live) -> None:
    """Save backtest results and metrics."""
    # Save results DataFrame
    results.to_csv(save_dir / 'results.csv')
    
    # Log metrics using DVCLive
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            live.log_metric(f'backtest/{metric_name}', value)

def main():
    # Get parameters from command line or use defaults
    lookback_window = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    prediction_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    min_train_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    retrain_frequency = int(sys.argv[4]) if len(sys.argv) > 4 else 21
    tune_hyperparameters = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    
    # Initialize DVCLive
    with Live(save_dvc_exp=True) as live:
        try:
            # Log parameters
            live.log_param("lookback_window", lookback_window)
            live.log_param("prediction_threshold", prediction_threshold)
            live.log_param("min_train_size", min_train_size)
            live.log_param("retrain_frequency", retrain_frequency)
            live.log_param("tune_hyperparameters", tune_hyperparameters)
            
            # Create results directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = Path(f'data/backtest_results/ml_ensemble_{timestamp}')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Load data
            logging.info('Loading data...')
            stock_data, market_data = load_data()
            
            # Log data information
            live.log_param("data_start_date", str(market_data.index[0]))
            live.log_param("data_end_date", str(market_data.index[-1]))
            live.log_param("num_stocks", len(stock_data))
            live.log_param("num_trading_days", len(market_data))
            
            # Initialize strategy
            strategy = MLEnsembleStrategy(
                lookback_window=lookback_window,
                prediction_threshold=prediction_threshold,
                min_train_size=min_train_size,
                retrain_frequency=retrain_frequency
            )
            
            # Run backtest
            logging.info('Running backtest...')
            
            # Split data for walk-forward optimization
            train_end = market_data.index[int(len(market_data) * 0.7)]
            validation_end = market_data.index[int(len(market_data) * 0.85)]
            
            # Train and tune on training data
            if tune_hyperparameters:
                logging.info('Tuning hyperparameters on training data...')
                strategy.train_models(
                    prices=market_data.loc[:train_end, 'close'],
                    volumes=market_data.loc[:train_end, 'volume'],
                    current_date=train_end,
                    tune_hyperparameters=True
                )
                
                # Validate on validation set
                logging.info('Validating on validation data...')
                validation_results = strategy.backtest(
                    prices=market_data.loc[train_end:validation_end, 'close'],
                    volumes=market_data.loc[train_end:validation_end, 'volume'],
                    initial_capital=100000.0
                )
                
                # Log validation metrics
                validation_returns = validation_results['Returns'].fillna(0)
                validation_metrics = {
                    'validation_return': float(validation_returns.sum()),
                    'validation_sharpe': float(validation_returns.mean() / validation_returns.std() * np.sqrt(252)) if validation_returns.std() != 0 else 0,
                    'validation_win_rate': float((validation_returns > 0).sum() / len(validation_returns))
                }
                
                for metric_name, value in validation_metrics.items():
                    live.log_metric(f'validation/{metric_name}', value)
            
            # Run final backtest on test data
            logging.info('Running final backtest on test data...')
            results = strategy.backtest(
                prices=market_data.loc[validation_end:, 'close'],
                volumes=market_data.loc[validation_end:, 'volume'],
                initial_capital=100000.0
            )
            
            # Calculate performance metrics
            returns = results['Returns'].fillna(0)
            capital_series = results['Capital'].fillna(method='ffill')
            
            # Calculate drawdown
            peak = capital_series.expanding(min_periods=1).max()
            drawdown = (capital_series - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            metrics = {
                'total_return': float(returns.sum()),
                'annualized_return': float(returns.mean() * 252),
                'annualized_volatility': float(returns.std() * np.sqrt(252)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
                'max_drawdown': float(max_drawdown),
                'win_rate': float((returns > 0).sum() / len(returns)),
                'avg_win': float(returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 0,
                'avg_loss': float(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0,
                'total_trades': int((results['Signal'] != 0).sum()),
                'avg_position_size': float(abs(results['Position']).mean()),
                'max_position_size': float(abs(results['Position']).max()),
            }
            
            # Plot metrics
            logging.info('Generating plots...')
            plot_performance_metrics(results, results_dir, live)
            
            # Save results
            logging.info('Saving results...')
            save_results(results, metrics, results_dir, live)
            
            # Log summary
            logging.info('\nBacktest Summary:')
            for metric, value in metrics.items():
                if metric in ['total_trades', 'max_position_size', 'avg_position_size']:
                    logging.info(f'{metric}: {value:.2f}')
                else:
                    logging.info(f'{metric}: {value:.2%}')
            
            logging.info(f'\nResults saved in: {results_dir}')
            
            # Track model training metrics
            if hasattr(strategy, 'training_history'):
                for epoch, history in enumerate(strategy.training_history):
                    for metric_name, value in history.items():
                        # Skip dictionary metrics (like feature importance)
                        if isinstance(value, (int, float)):
                            live.log_metric(f'training/{metric_name}', value)
                        elif isinstance(value, dict):
                            # For feature importance, log each feature separately
                            for feature, importance in value.items():
                                if isinstance(importance, (int, float)):
                                    live.log_metric(f'feature_importance/{metric_name}/{feature}', importance)
                    live.next_step()
                    
        except Exception as e:
            logging.error(f"Error during backtest: {str(e)}")
            raise

if __name__ == '__main__':
    main() 