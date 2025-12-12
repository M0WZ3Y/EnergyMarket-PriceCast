"""
Machine Learning Models package for electricity price forecasting.

This package contains implementations of various machine learning models
including baseline models, tree-based models, deep learning models, and other ML approaches.
"""

from .baseline_models.linear_regression import LinearRegressionModel, create_linear_model, create_ridge_model, create_lasso_model

__all__ = [
    'LinearRegressionModel',
    'create_linear_model',
    'create_ridge_model', 
    'create_lasso_model'
]