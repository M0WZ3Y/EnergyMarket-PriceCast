"""
Baseline Models package for electricity price forecasting.

This package contains baseline machine learning models including linear regression,
ridge regression, and lasso regression models.
"""

from .linear_regression import LinearRegressionModel, create_linear_model, create_ridge_model, create_lasso_model

__all__ = [
    'LinearRegressionModel',
    'create_linear_model',
    'create_ridge_model',
    'create_lasso_model'
]