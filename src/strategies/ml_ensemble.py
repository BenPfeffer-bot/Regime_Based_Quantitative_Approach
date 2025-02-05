"""
Machine Learning Ensemble Forecasting Strategy

This module implements an ensemble of machine learning models for predicting
market movements and generating trading signals. It combines multiple models
including Random Forests, XGBoost, and Neural Networks with traditional
technical indicators for robust market forecasting.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
import talib
from dataclasses import dataclass
import logging

@dataclass
class ModelPrediction:
    """Container for model predictions and confidence scores."""
    direction: int  # 1 for up, -1 for down, 0 for neutral
    probability: float
    model_name: str

@dataclass
class TrainingMetrics:
    """Container for model training metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    validation_metrics: Optional[Dict[str, float]] = None

class MLEnsembleStrategy:
    """Implementation of ML ensemble forecasting strategy."""
    
    def __init__(
        self,
        lookback_window: int = 20,
        prediction_threshold: float = 0.7,
        ensemble_weights: Optional[Dict[str, float]] = None,
        feature_windows: List[int] = [5, 10, 20, 50],
        min_train_size: int = 252,  # One year of data
        retrain_frequency: int = 63,  # Retrain every quarter
        risk_free_rate: float = 0.02,
    ):
        """Initialize strategy parameters and ML models."""
        self.lookback_window = lookback_window
        self.prediction_threshold = prediction_threshold
        self.feature_windows = feature_windows
        self.min_train_size = min_train_size
        self.retrain_frequency = retrain_frequency
        self.risk_free_rate = risk_free_rate
        
        # Set default ensemble weights if none provided
        self.ensemble_weights = ensemble_weights or {
            'random_forest': 0.35,
            'xgboost': 0.35,
            'neural_network': 0.30
        }
        
        # Initialize models
        self.models = {
            'random_forest': None,
            'xgboost': None,
            'neural_network': None
        }
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.last_train_date = None
        
        # Strategy state
        self.current_predictions: Dict[str, ModelPrediction] = {}
        self.feature_columns: List[str] = []
        
        # Training history
        self.training_history: List[Dict[str, float]] = []
        self.model_metrics: Dict[str, TrainingMetrics] = {}
        
    def _create_technical_features(
        self,
        prices: pd.Series,
        volumes: pd.Series
    ) -> pd.DataFrame:
        """Create technical indicators as features for ML models."""
        features = pd.DataFrame(index=prices.index)
        
        # Price-based features
        for window in self.feature_windows:
            # Moving averages
            features[f'sma_{window}'] = prices.rolling(window=window).mean()
            features[f'ema_{window}'] = prices.ewm(span=window, adjust=False).mean()
            
            # Price momentum
            features[f'momentum_{window}'] = prices.pct_change(periods=window)
            
            # Volatility
            features[f'volatility_{window}'] = prices.pct_change().rolling(window=window).std()
            
            # Volume momentum
            features[f'volume_momentum_{window}'] = volumes.pct_change(periods=window)
        
        # Technical indicators
        high = prices * 1.001  # Approximate high prices
        low = prices * 0.999   # Approximate low prices
        close = prices
        
        # RSI
        features['rsi'] = talib.RSI(close.values, timeperiod=14)
        
        # MACD
        macd, signal, _ = talib.MACD(close.values)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close.values, timeperiod=20)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        
        # ATR
        features['atr'] = talib.ATR(high.values, low.values, close.values, timeperiod=14)
        
        # Store feature columns
        self.feature_columns = features.columns.tolist()
        
        return features
    
    def _prepare_training_data(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        forward_returns: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for model training."""
        # Remove rows with NaN values
        valid_data = features.dropna()
        valid_returns = forward_returns[valid_data.index]
        
        # Create binary labels (1 for positive returns, 0 for negative)
        labels = (valid_returns > 0).astype(int)
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(valid_data)
        
        return scaled_features, labels.values
    
    def _create_random_forest(self) -> RandomForestClassifier:
        """Create and configure Random Forest model."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    
    def _create_xgboost(self) -> xgb.XGBClassifier:
        """Create and configure XGBoost model."""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def _create_neural_network(self) -> Sequential:
        """Create and configure Neural Network model."""
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=len(self.feature_columns)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _calculate_model_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> TrainingMetrics:
        """Calculate performance metrics for a model."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = TrainingMetrics(
            model_name=model_name,
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred)),
            recall=float(recall_score(y_true, y_pred)),
            f1_score=float(f1_score(y_true, y_pred)),
            validation_metrics=validation_metrics
        )
        
        return metrics
    
    def _update_training_history(
        self,
        metrics: TrainingMetrics,
        epoch: Optional[int] = None
    ) -> None:
        """Update training history with new metrics."""
        history_entry = {
            f'{metrics.model_name}/accuracy': metrics.accuracy,
            f'{metrics.model_name}/precision': metrics.precision,
            f'{metrics.model_name}/recall': metrics.recall,
            f'{metrics.model_name}/f1_score': metrics.f1_score,
        }
        
        if metrics.validation_metrics:
            for metric_name, value in metrics.validation_metrics.items():
                history_entry[f'{metrics.model_name}/val_{metric_name}'] = value
        
        if epoch is not None:
            history_entry['epoch'] = epoch
        
        self.training_history.append(history_entry)
        self.model_metrics[metrics.model_name] = metrics
    
    def train_models(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        current_date: pd.Timestamp
    ) -> None:
        """Train all models in the ensemble."""
        # Check if retraining is needed
        if (self.last_train_date is not None and
            (current_date - self.last_train_date).days < self.retrain_frequency):
            return
        
        # Create features
        features = self._create_technical_features(prices, volumes)
        
        # Calculate forward returns
        forward_returns = prices.pct_change().shift(-1)
        
        # Prepare training data
        X, y = self._prepare_training_data(features, prices, forward_returns)
        
        if len(X) < self.min_train_size:
            logging.debug(f"Insufficient data for training: {len(X)} < {self.min_train_size}")
            return
        
        # Reset training history
        self.training_history = []
        
        try:
            # Train Random Forest
            rf_model = self._create_random_forest()
            rf_model.fit(X, y)
            self.models['random_forest'] = rf_model
            
            # Calculate and log RF metrics
            rf_pred = rf_model.predict(X)
            rf_metrics = self._calculate_model_metrics(y, rf_pred, 'random_forest')
            self._update_training_history(rf_metrics)
            
            # Train XGBoost with corrected parameters
            xgb_model = self._create_xgboost()
            eval_set = [(X, y)]
            xgb_model.fit(
                X, y,
                eval_set=eval_set,
                verbose=False
            )
            self.models['xgboost'] = xgb_model
            
            # Calculate and log XGB metrics
            xgb_pred = xgb_model.predict(X)
            xgb_metrics = self._calculate_model_metrics(
                y, xgb_pred, 'xgboost',
                validation_metrics={'validation_error': 1 - xgb_model.score(X, y)}
            )
            self._update_training_history(xgb_metrics)
            
            # Train Neural Network
            nn_model = self._create_neural_network()
            history = nn_model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                verbose=0,
                validation_split=0.2
            )
            self.models['neural_network'] = nn_model
            
            # Log NN training history
            for epoch, metrics in enumerate(zip(
                history.history['accuracy'],
                history.history['val_accuracy'],
                history.history['loss'],
                history.history['val_loss']
            )):
                train_acc, val_acc, train_loss, val_loss = metrics
                nn_metrics = TrainingMetrics(
                    model_name='neural_network',
                    accuracy=float(train_acc),
                    precision=0.0,  # Not available from Keras history
                    recall=0.0,     # Not available from Keras history
                    f1_score=0.0,   # Not available from Keras history
                    validation_metrics={
                        'accuracy': float(val_acc),
                        'loss': float(val_loss)
                    }
                )
                self._update_training_history(nn_metrics, epoch)
            
            self.last_train_date = current_date
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            # Keep existing models if training fails
            return
        
    def generate_predictions(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        current_date: pd.Timestamp
    ) -> Dict[str, ModelPrediction]:
        """Generate predictions from all models in the ensemble."""
        # Create features for prediction
        features = self._create_technical_features(prices, volumes)
        
        # Get latest feature values
        latest_features = features.iloc[-1:].copy()
        
        # Scale features
        scaled_features = self.feature_scaler.transform(latest_features)
        
        predictions = {}
        
        # Get predictions from each model
        if self.models['random_forest'] is not None:
            rf_prob = self.models['random_forest'].predict_proba(scaled_features)[0, 1]
            predictions['random_forest'] = ModelPrediction(
                direction=1 if rf_prob > 0.5 else -1,
                probability=rf_prob,
                model_name='random_forest'
            )
        
        if self.models['xgboost'] is not None:
            xgb_prob = self.models['xgboost'].predict_proba(scaled_features)[0, 1]
            predictions['xgboost'] = ModelPrediction(
                direction=1 if xgb_prob > 0.5 else -1,
                probability=xgb_prob,
                model_name='xgboost'
            )
        
        if self.models['neural_network'] is not None:
            nn_prob = self.models['neural_network'].predict(scaled_features)[0, 0]
            predictions['neural_network'] = ModelPrediction(
                direction=1 if nn_prob > 0.5 else -1,
                probability=float(nn_prob),
                model_name='neural_network'
            )
        
        self.current_predictions = predictions
        return predictions
    
    def calculate_ensemble_signal(
        self,
        predictions: Dict[str, ModelPrediction]
    ) -> Tuple[int, float]:
        """Calculate weighted ensemble prediction and confidence."""
        if not predictions:
            return 0, 0.0
        
        weighted_prob = 0.0
        total_weight = 0.0
        
        for model_name, pred in predictions.items():
            weight = self.ensemble_weights.get(model_name, 0)
            if pred.direction == 1:
                weighted_prob += weight * pred.probability
            else:
                weighted_prob += weight * (1 - pred.probability)
            total_weight += weight
        
        if total_weight > 0:
            ensemble_prob = weighted_prob / total_weight
            
            # Generate signal based on threshold
            if ensemble_prob > self.prediction_threshold:
                return 1, ensemble_prob
            elif ensemble_prob < (1 - self.prediction_threshold):
                return -1, (1 - ensemble_prob)
        
        return 0, 0.5
    
    def calculate_position_size(
        self,
        signal: int,
        confidence: float,
        volatility: float,
        max_position: float = 1.0
    ) -> float:
        """Calculate position size based on signal confidence and volatility."""
        if signal == 0:
            return 0.0
        
        # Scale confidence to [0.5, 1.0] range
        scaled_confidence = 0.5 + (confidence - self.prediction_threshold) / (1 - self.prediction_threshold) * 0.5
        
        # Adjust position size based on volatility
        vol_adjustment = 1.0 / (1 + volatility * 10)  # Reduce size in high volatility
        
        # Calculate base position size
        position_size = signal * scaled_confidence * vol_adjustment * max_position
        
        return max(min(position_size, max_position), -max_position)
    
    def generate_signals(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        current_date: pd.Timestamp,
        current_positions: Dict[str, float]
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """Generate trading signals based on ensemble predictions."""
        # Train models if needed
        self.train_models(prices, volumes, current_date)
        
        # Generate predictions
        predictions = self.generate_predictions(prices, volumes, current_date)
        
        # Calculate ensemble signal
        signal, confidence = self.calculate_ensemble_signal(predictions)
        
        # Calculate recent volatility
        volatility = prices.pct_change().rolling(window=20).std().iloc[-1]
        
        # Calculate position size
        size = self.calculate_position_size(signal, confidence, volatility)
        
        return {
            'ML_ENSEMBLE': {
                'signal': signal,
                'size': size,
                'confidence': confidence
            }
        }
    
    def backtest(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """Backtest the ML ensemble strategy."""
        results = pd.DataFrame(index=prices.index)
        capital = initial_capital
        position = 0
        entry_price = 0
        
        for i in range(self.min_train_size, len(prices)):
            current_date = prices.index[i]
            
            # Get historical data up to current date
            historical_prices = prices[:i+1]
            historical_volumes = volumes[:i+1]
            
            # Generate signals
            signals = self.generate_signals(
                historical_prices,
                historical_volumes,
                current_date,
                {'ML_ENSEMBLE': position}
            )
            
            signal_data = signals.get('ML_ENSEMBLE', {})
            new_signal = signal_data.get('signal', 0)
            new_size = signal_data.get('size', 0)
            
            # Calculate P&L
            current_price = prices.iloc[i]
            if position != 0:
                pnl = position * (current_price - entry_price)
                capital += pnl
            
            # Update position
            if new_size != position:
                position = new_size
                entry_price = current_price
            
            # Record results
            results.loc[current_date, 'Capital'] = capital
            results.loc[current_date, 'Position'] = position
            results.loc[current_date, 'Signal'] = new_signal
            results.loc[current_date, 'Price'] = current_price
            results.loc[current_date, 'Returns'] = (capital - initial_capital) / initial_capital
        
        return results 