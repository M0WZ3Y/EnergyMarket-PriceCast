"""
Linear Regression Model for Electricity Price Forecasting

This module implements linear regression models for electricity price forecasting,
including standard linear regression, ridge regression, and lasso regression.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from typing import Dict, Any, Tuple, Optional
import logging

try:
    from ...utils.config.config_loader import get_model_config
except ImportError:
    # Fallback for direct execution and IDE resolution
    import sys
    from pathlib import Path
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Try importing from project root
    try:
        from src.electricity_price_forecast.utils.config.config_loader import get_model_config
    except ImportError:
        # Final fallback - try direct import
        from src.electricity_price_forecast.utils.config.config_loader import get_model_config


class LinearRegressionModel:
    """
    Linear Regression model class for electricity price forecasting.
    
    This class provides a unified interface for training and evaluating
    linear regression models with different regularization techniques.
    """
    
    def __init__(self, model_type: str = 'linear', random_state: int = 42):
        """
        Initialize the linear regression model.
        
        Args:
            model_type (str): Type of linear model ('linear', 'ridge', 'lasso')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get model configuration
        self.config = get_model_config()
        self.model_config = self.config['models'].get(f'{model_type}_regression', {})
        
    def _create_model(self) -> None:
        """Create the appropriate linear regression model based on model_type."""
        if self.model_type == 'linear':
            self.model = LinearRegression(**self.model_config)
        elif self.model_type == 'ridge':
            self.model = Ridge(random_state=self.random_state, **self.model_config)
        elif self.model_type == 'lasso':
            self.model = Lasso(random_state=self.random_state, **self.model_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Fit the linear regression model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame, optional): Validation features
            y_val (pd.Series, optional): Validation target
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        self.logger.info(f"Training {self.model_type} regression model...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Create model if not already created
        if self.model is None:
            self._create_model()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit the model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train_scaled)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        # Calculate validation metrics if validation data is provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_pred = self.model.predict(X_val_scaled)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        results = {
            'model_type': self.model_type,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_names': self.feature_names,
            'coefficients': self.model.coef_.tolist() if hasattr(self.model, 'coef_') else None,
            'intercept': self.model.intercept_ if hasattr(self.model, 'intercept_') else None
        }
        
        self.logger.info(f"Training completed. Train MAE: {train_metrics['mae']:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predictions
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        predictions = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred))
        }
        
        # Add R² score
        from sklearn.metrics import r2_score
        metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            param_grid: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            param_grid (Dict[str, Any], optional): Parameter grid for tuning
            
        Returns:
            Dict[str, Any]: Tuning results
        """
        if param_grid is None:
            # Use default parameter grid from config
            param_grid = self.config['hyperparameter_tuning']['param_grids'].get(
                f'{self.model_type}', {}
            )
        
        if not param_grid:
            self.logger.warning(f"No parameter grid provided for {self.model_type}")
            return {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create model
        if self.model is None:
            self._create_model()
        
        # Perform grid search
        cv_config = self.config['hyperparameter_tuning']
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv_config['cv_folds'],
            scoring=cv_config['scoring'],
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (coefficients) for linear models.
        
        Returns:
            Dict[str, float]: Feature importance scores
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model does not have coefficients")
        
        importance = dict(zip(self.feature_names, np.abs(self.model.coef_)))
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }
        
        if self.is_fitted and hasattr(self.model, 'coef_'):
            info['n_nonzero_coeffs'] = np.sum(self.model.coef_ != 0)
            info['coeff_norm'] = np.linalg.norm(self.model.coef_)
        
        return info


# Convenience functions for creating different linear models
def create_linear_model(random_state: int = 42) -> LinearRegressionModel:
    """Create a standard linear regression model."""
    return LinearRegressionModel(model_type='linear', random_state=random_state)

def create_ridge_model(random_state: int = 42) -> LinearRegressionModel:
    """Create a ridge regression model."""
    return LinearRegressionModel(model_type='ridge', random_state=random_state)

def create_lasso_model(random_state: int = 42) -> LinearRegressionModel:
    """Create a lasso regression model."""
    return LinearRegressionModel(model_type='lasso', random_state=random_state)