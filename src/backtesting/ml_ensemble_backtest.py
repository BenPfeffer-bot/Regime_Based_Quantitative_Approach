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

from src.strategies.ml_ensemble import MLEnsembleStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(data_dir: str = 'data/raw') -> tuple:
    """Load historical price and volume data."""
    # Load your data here
    # For example:
    data_path = Path(data_dir)
    stock_data = {}
    
    # Load market data (e.g., EUROSTOXX 50 index)
    market_data = pd.DataFrame()  # Load your market data here
    
    return stock_data, market_data

def plot_performance_metrics(results: pd.DataFrame, save_dir: Path) -> None:
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
    plt.savefig(save_dir / 'performance_metrics.png')
    plt.close()

def plot_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Path
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
    plt.savefig(save_dir / 'model_metrics.png')
    plt.close()

def save_results(results: pd.DataFrame, metrics: dict, save_dir: Path) -> None:
    """Save backtest results and metrics."""
    # Save results DataFrame
    results.to_csv(save_dir / 'results.csv')
    
    # Save metrics as JSON
    import json
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f'data/backtest_results/ml_ensemble_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info('Loading data...')
    stock_data, market_data = load_data()
    
    # Initialize strategy
    strategy = MLEnsembleStrategy(
        lookback_window=20,
        prediction_threshold=0.7,
        min_train_size=252,
        retrain_frequency=63
    )
    
    # Run backtest
    logging.info('Running backtest...')
    results = strategy.backtest(
        prices=market_data['close'],
        volumes=market_data['volume'],
        initial_capital=100000.0
    )
    
    # Calculate performance metrics
    returns = results['Returns']
    metrics = {
        'total_return': float(returns.iloc[-1]),
        'annualized_return': float(returns.mean() * 252),
        'annualized_volatility': float(returns.std() * np.sqrt(252)),
        'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0,
        'max_drawdown': float((results['Capital'].cummax() - results['Capital']) / results['Capital'].cummax()).max(),
        'win_rate': float((returns > 0).sum() / len(returns)),
        'avg_win': float(returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 0,
        'avg_loss': float(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0,
    }
    
    # Plot metrics
    logging.info('Generating plots...')
    plot_performance_metrics(results, results_dir)
    
    # Save results
    logging.info('Saving results...')
    save_results(results, metrics, results_dir)
    
    # Log summary
    logging.info('\nBacktest Summary:')
    for metric, value in metrics.items():
        logging.info(f'{metric}: {value:.2%}')
    
    logging.info(f'\nResults saved in: {results_dir}')

if __name__ == '__main__':
    main() 