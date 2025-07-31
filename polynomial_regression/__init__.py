"""
Polynomial Regression from Scratch

A complete implementation of polynomial regression with regularization options,
feature engineering, and comprehensive evaluation metrics.
"""

from .models import PolynomialRegressionMetrics, PolynomialRegression
from .utils import generate_polynomial_data, generate_sinusoidal_data, generate_exponential_data,\
                    train_test_split, plot_polynomial_fit, plot_learning_curves, compare_polynomial_degrees,\
                    normalize_features, apply_normalization, plot_residuals, cross_validate


__version__ = "1.0.0"
__author__ = "M. Hossein Ghaemi"
__email__ = "h.ghaemi.2003@gmail.com"

__all__ = [
    "PolynomialRegressionMetrics",
    "PolynomialRegression",
    "generate_polynomial_data",
    "generate_sinusoidal_data",
    "generate_exponential_data",
    "train_test_split",
    "plot_polynomial_fit",
    "plot_learning_curves",
    "compare_polynomial_degrees",
    "normalize_features",
    "apply_normalization",
    "plot_residuals",
    "cross_validate"
]
