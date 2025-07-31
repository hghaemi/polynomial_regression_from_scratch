# Polynomial Regression from Scratch

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, educational implementation of polynomial regression built from scratch using only NumPy and Matplotlib. This project demonstrates machine learning fundamentals including gradient descent, regularization techniques, feature engineering, and comprehensive model evaluation.

## 🚀 Features

### Core Implementation
- **Pure NumPy Implementation**: No scikit-learn dependencies for core algorithms
- **Flexible Polynomial Degrees**: Support for any polynomial degree with automatic feature generation
- **Multiple Regularization Options**: L1 (Lasso), L2 (Ridge), and no regularization
- **Gradient Descent Optimization**: Implemented from scratch with convergence monitoring
- **Feature Normalization**: Standard, Min-Max, and Robust scaling methods

### Advanced Functionality
- **Cross-Validation**: K-fold cross-validation for robust model evaluation
- **Hyperparameter Tuning**: Grid search capabilities for optimal parameter selection
- **Comprehensive Metrics**: R², Adjusted R², MSE, RMSE, MAE with detailed reporting
- **Overfitting Analysis**: Tools to detect and visualize overfitting vs. underfitting
- **Feature Importance**: Analysis of polynomial coefficient significance

### Visualization & Analysis
- **Learning Curves**: Training cost evolution and convergence analysis
- **Model Comparison**: Side-by-side comparison of different polynomial degrees
- **Residual Analysis**: Q-Q plots, residual distributions, and diagnostic plots
- **Interactive Fitting**: Visual polynomial curve fitting with confidence regions

## 📦 Installation

### From Source
```bash
git clone https://github.com/hghaemi/polynomial_regression_from_scratch.git
cd polynomial_regression_from_scratch
pip install -e .
```

### Dependencies
```bash
# Core dependencies
pip install numpy>=1.19.0 matplotlib>=3.3.0 scipy>=1.16.1

# Development dependencies (optional)
pip install pytest>=8.4.1 pytest-cov>=4.0.0 black>=22.0.0 flake8>=4.0.0
```

## 🔧 Quick Start

### Basic Usage
```python
from polynomial_regression import PolynomialRegression, generate_polynomial_data
from polynomial_regression import train_test_split, plot_polynomial_fit

# Generate sample data
X, y = generate_polynomial_data(n_samples=200, degree=3, noise=0.2)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = PolynomialRegression(
    degree=3,
    learning_rate=0.01,
    max_iterations=1000,
    regularization='l2',
    reg_lambda=0.001
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
r2_score = model.score(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")

# Visualize results
plot_polynomial_fit(model, X_test, y_test)
```

### Advanced Example: Regularization Comparison
```python
from polynomial_regression import PolynomialRegression, PolynomialRegressionMetrics
import numpy as np

# Compare different regularization methods
regularizations = [None, 'l1', 'l2']
results = {}

for reg in regularizations:
    model = PolynomialRegression(
        degree=4,
        regularization=reg,
        reg_lambda=0.01 if reg else 0
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[reg] = {
        'r2': PolynomialRegressionMetrics.r_squared(y_test, y_pred),
        'rmse': PolynomialRegressionMetrics.root_mean_squared_error(y_test, y_pred),
        'weights_norm': np.linalg.norm(model.weights)
    }

# Print comparison
for reg, metrics in results.items():
    reg_name = reg if reg else 'None'
    print(f"{reg_name:4s}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
```

## 🎯 Examples

The project includes comprehensive examples demonstrating various use cases:

### 1. Basic Polynomial Regression (`examples/basic_example.py`)
- Simple polynomial data fitting
- Model evaluation and metrics
- Cost function visualization
- Feature importance analysis

### 2. Multivariate Regression (`examples/multivariate_example.py`)
- Multi-feature polynomial regression
- Hyperparameter tuning with grid search
- Cross-validation analysis
- Overfitting detection and prevention


## 📊 Key Algorithms Implemented

### 1. Polynomial Feature Generation
```python
# Automatic generation of polynomial features
# For degree=3: [x, x², x³, xy, x²y, xy², etc.]
def _create_polynomial_features(self, X, normalize=True):
    # Creates interaction terms and higher-order features
    # Includes feature normalization for numerical stability
```

### 2. Gradient Descent with Regularization
```python
# Cost function with L1/L2 regularization
def _compute_cost(self, y_true, y_pred):
    mse = np.mean((y_pred - y_true) ** 2)
    
    if self.regularization == 'l1':
        mse += self.reg_lambda * np.sum(np.abs(self.weights))
    elif self.regularization == 'l2':
        mse += self.reg_lambda * np.sum(self.weights ** 2)
    
    return mse
```

### 3. Cross-Validation Implementation
```python
# K-fold cross-validation from scratch
def cross_validate(model_class, X, y, cv_folds=5, **model_params):
    # Implements stratified k-fold validation
    # Returns comprehensive performance metrics
```

## 🔍 Technical Details

### Model Architecture
- **Feature Engineering**: Automatic polynomial feature generation with interaction terms
- **Optimization**: Batch gradient descent with adaptive learning rates
- **Regularization**: L1 (Lasso) and L2 (Ridge) penalty terms
- **Normalization**: Multiple scaling methods to ensure numerical stability

### Performance Optimizations
- **Vectorized Operations**: Efficient NumPy operations for matrix computations
- **Memory Management**: Optimized feature matrix generation
- **Early Stopping**: Convergence detection to prevent unnecessary iterations
- **Numerical Stability**: Regularization and normalization to handle ill-conditioned problems

### Evaluation Metrics
- **R² Score**: Coefficient of determination
- **Adjusted R²**: Penalized R² for model complexity
- **MSE/RMSE**: Mean squared error metrics
- **MAE**: Mean absolute error for robust evaluation
- **Cross-Validation**: K-fold validation with confidence intervals

## 📈 Performance Benchmarks

| Dataset Size | Features | Degree | Training Time | Memory Usage |
|-------------|----------|---------|---------------|--------------|
| 1,000       | 1        | 3       | 0.05s        | ~2MB         |
| 10,000      | 3        | 4       | 0.3s         | ~15MB        |
| 50,000      | 5        | 3       | 1.2s         | ~60MB        |

*Benchmarks run on Intel i7-8750H CPU with 16GB RAM*

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ -v --cov=polynomial_regression --cov-report=html

# Run specific test file
python -m pytest tests/test_models.py -v
```

The test suite includes:
- Unit tests for all core functions
- Integration tests for end-to-end workflows
- Performance regression tests
- Numerical accuracy validation

## 🛠️ Project Structure

```
polynomial_regression_from_scratch/
├── polynomial_regression/          # Main package
│   ├── __init__.py                # Package initialization
│   ├── models.py                  # Core PolynomialRegression class
│   └── utils.py                   # Utility functions and data generation
├── examples/                      # Usage examples
│   ├── basic_example.py          # Simple polynomial fitting
│   ├── multivariate_example.py   # Advanced multivariate analysis
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_models.py           # Comprehensive unit tests
├── setup.py                     # Package installation script
├── requirements.txt             # Project dependencies
├── requirements-dev.txt         # Development dependencies
├── README.md                    # This file
└── LICENSE                      # MIT license
```

## 🎓 Educational Value

This project serves as an excellent learning resource for:

### Machine Learning Concepts
- **Gradient Descent**: Understanding optimization from first principles
- **Regularization**: Preventing overfitting with L1/L2 penalties
- **Feature Engineering**: Creating polynomial and interaction features
- **Model Evaluation**: Comprehensive metrics and cross-validation

### Software Engineering Practices
- **Clean Code**: Well-documented, readable implementation
- **Testing**: Comprehensive test suite with >90% coverage
- **Package Structure**: Professional Python package organization

### Mathematical Understanding
- **Linear Algebra**: Matrix operations and vectorization
- **Calculus**: Gradient computation and optimization
- **Statistics**: Model evaluation and significance testing

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`python -m pytest`)
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/hghaemi/polynomial_regression_from_scratch.git
cd polynomial_regression_from_scratch

# Install in development mode with all dependencies
pip install -e .
pip install -r requirements-dev.txt

# Optional: Set up pre-commit hooks for code quality
pip install pre-commit
pre-commit install
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NumPy Community**: For providing the foundational numerical computing tools
- **Matplotlib**: For powerful visualization capabilities
- **Educational Resources**: Inspired by Andrew Ng's Machine Learning Course
- **Open Source Community**: For continuous inspiration and best practices

## 📬 Contact

**M. Hossein Ghaemi**
- Email: h.ghaemi.2003@gmail.com
- GitHub: [@hghaemi](https://github.com/hghaemi)

##
*Happy Learning! 🚀*