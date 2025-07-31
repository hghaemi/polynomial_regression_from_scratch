import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Callable


def generate_polynomial_data(n_samples: int = 1000,
                           degree: int = 2,
                           n_features: int = 1,
                           noise: float = 0.1,
                           coeff_range: Tuple[float, float] = (-2, 2),
                           x_range: Tuple[float, float] = (-2, 2),
                           random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, n_features))
    
    n_poly_features = 0
    for d in range(1, degree + 1):
        n_poly_features += n_features ** d
    
    coefficients = np.random.uniform(coeff_range[0], coeff_range[1], n_poly_features)
    
    y = np.zeros(n_samples)
    coeff_idx = 0
    
    for i in range(n_features):
        y += coefficients[coeff_idx] * X[:, i]
        coeff_idx += 1
    
    for d in range(2, degree + 1):
        for i in range(n_features):
            if coeff_idx < len(coefficients):
                y += coefficients[coeff_idx] * (X[:, i] ** d)
                coeff_idx += 1
    
    y += np.random.normal(0, noise, n_samples)
    
    return X, y


def generate_sinusoidal_data(n_samples: int = 1000,
                           frequency: float = 1.0,
                           amplitude: float = 1.0,
                           phase: float = 0.0,
                           noise: float = 0.1,
                           x_range: Tuple[float, float] = (0, 4*np.pi),
                           random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    y = amplitude * np.sin(frequency * X.flatten() + phase)
    y += np.random.normal(0, noise, n_samples)
    
    return X, y


def generate_exponential_data(n_samples: int = 1000,
                            base: float = np.e,
                            scale: float = 1.0,
                            noise: float = 0.1,
                            x_range: Tuple[float, float] = (0, 2),
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    y = scale * (base ** X.flatten())
    y += np.random.normal(0, noise * np.mean(y), n_samples)
    
    return X, y


def train_test_split(X: np.ndarray, 
                    y: np.ndarray, 
                    test_size: float = 0.2,
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * test_size)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def plot_polynomial_fit(model, X: np.ndarray, y: np.ndarray, 
                       title: str = "Polynomial Regression Fit",
                       figsize: Tuple[int, int] = (12, 8),
                       n_points: int = 300):
    if X.shape[1] != 1:
        raise ValueError("This function only works with 1D input data")
    
    plt.figure(figsize=figsize)
    
    plt.scatter(X.flatten(), y, alpha=0.6, color='blue', label='Data points', s=30)
    
    x_min, x_max = X.min(), X.max()
    x_range = x_max - x_min
    x_smooth = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, n_points).reshape(-1, 1)
    y_smooth = model.predict(x_smooth)
    
    plt.plot(x_smooth.flatten(), y_smooth, color='red', linewidth=2, 
             label=f'Polynomial fit (degree {model.degree})')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_learning_curves(model, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        figsize: Tuple[int, int] = (12, 5)):
    if not hasattr(model, 'cost_history') or not model.cost_history:
        raise ValueError("Model must be trained and have cost history")
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    from .models import PolynomialRegressionMetrics
    
    train_r2 = PolynomialRegressionMetrics.r_squared(y_train, train_pred)
    val_r2 = PolynomialRegressionMetrics.r_squared(y_val, val_pred)
    train_rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_train, train_pred)
    val_rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_val, val_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    iterations = range(1, len(model.cost_history) + 1)
    ax1.plot(iterations, model.cost_history, 'b-', linewidth=2)
    ax1.set_title('Training Cost Over Time')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    metrics = ['R²', 'RMSE']
    train_scores = [train_r2, train_rmse]
    val_scores = [val_r2, val_rmse]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x_pos - width/2, train_scores, width, label='Training', alpha=0.7, color='blue')
    ax2.bar(x_pos + width/2, val_scores, width, label='Validation', alpha=0.7, color='orange')
    
    ax2.set_title('Model Performance Comparison')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for i, (train_val, val_val) in enumerate(zip(train_scores, val_scores)):
        ax2.text(i - width/2, train_val + 0.01, f'{train_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, val_val + 0.01, f'{val_val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def compare_polynomial_degrees(X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             degrees: list = [1, 2, 3, 4, 5],
                             figsize: Tuple[int, int] = (15, 10)):
    from .models import PolynomialRegression, PolynomialRegressionMetrics
    
    results = []
    models = []
    
    print("Comparing Polynomial Degrees")
    print("=" * 50)
    print("Degree\tTrain R2\tTest R2\t\tTrain RMSE\tTest RMSE")
    print("-" * 60)
    
    for degree in degrees:
        model = PolynomialRegression(
            degree=degree,
            learning_rate=0.01,
            max_iterations=1000,
            regularization='l2',
            reg_lambda=0.001,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = PolynomialRegressionMetrics.r_squared(y_train, train_pred)
        test_r2 = PolynomialRegressionMetrics.r_squared(y_test, test_pred)
        train_rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_train, train_pred)
        test_rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_test, test_pred)
        
        results.append({
            'degree': degree,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        })
        
        models.append(model)
        
        print(f"{degree}\t{train_r2:.4f}\t\t{test_r2:.4f}\t\t{train_rmse:.4f}\t\t{test_rmse:.4f}")
    
    degrees_list = [r['degree'] for r in results]
    train_r2_scores = [r['train_r2'] for r in results]
    test_r2_scores = [r['test_r2'] for r in results]
    train_rmse_scores = [r['train_rmse'] for r in results]
    test_rmse_scores = [r['test_rmse'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    ax1.plot(degrees_list, train_r2_scores, 'o-', label='Training', linewidth=2, markersize=8)
    ax1.plot(degrees_list, test_r2_scores, 's-', label='Test', linewidth=2, markersize=8)
    ax1.set_title('R² Score vs Polynomial Degree')
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('R² Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(degrees_list, train_rmse_scores, 'o-', label='Training', linewidth=2, markersize=8)
    ax2.plot(degrees_list, test_rmse_scores, 's-', label='Test', linewidth=2, markersize=8)
    ax2.set_title('RMSE vs Polynomial Degree')
    ax2.set_xlabel('Polynomial Degree')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if X_train.shape[1] == 1:
        n_params = []
        for degree in degrees_list:
            n_params.append(degree + 1)  # degree + intercept
        
        ax3.bar(degrees_list, n_params, alpha=0.7, color='green')
        ax3.set_title('Model Complexity (Number of Parameters)')
        ax3.set_xlabel('Polynomial Degree')
        ax3.set_ylabel('Number of Parameters')
        ax3.grid(True, alpha=0.3)
    
    overfitting_scores = [train - test for train, test in zip(train_r2_scores, test_r2_scores)]
    ax4.bar(degrees_list, overfitting_scores, alpha=0.7, color='red')
    ax4.set_title('Overfitting Analysis (Train R² - Test R²)')
    ax4.set_xlabel('Polynomial Degree')
    ax4.set_ylabel('R² Difference')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results, models


def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    X = np.array(X)
    
    if method == 'standard':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std, 'method': 'standard'}
        
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
    elif method == 'robust':
        median = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        X_normalized = (X - median) / (iqr + 1e-8)
        params = {'median': median, 'iqr': iqr, 'method': 'robust'}
        
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    return X_normalized, params


def apply_normalization(X: np.ndarray, params: dict) -> np.ndarray:

    X = np.array(X)
    method = params['method']
    
    if method == 'standard':
        return (X - params['mean']) / (params['std'] + 1e-8)
    elif method == 'minmax':
        return (X - params['min']) / (params['max'] - params['min'] + 1e-8)
    elif method == 'robust':
        return (X - params['median']) / (params['iqr'] + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def plot_residuals(model, X: np.ndarray, y: np.ndarray, 
                  title: str = "Residual Analysis",
                  figsize: Tuple[int, int] = (12, 8)):
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y, residuals, alpha=0.6)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs True Values')
    ax2.grid(True, alpha=0.3)
    
    ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    from scipy import stats
    try:
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot')
        ax4.grid(True, alpha=0.3)
    except ImportError:
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
        ax4.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
        ax4.plot(theoretical_quantiles, theoretical_quantiles, 'r--')
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Sample Quantiles')
        ax4.set_title('Q-Q Plot (approximate)')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def cross_validate(model_class, X: np.ndarray, y: np.ndarray, 
                  cv_folds: int = 5, **model_params) -> dict:
    from .models import PolynomialRegressionMetrics
    
    n_samples = len(X)
    fold_size = n_samples // cv_folds
    
    cv_scores = {
        'r2': [],
        'mse': [],
        'rmse': [],
        'mae': []
    }
    
    for fold in range(cv_folds):

        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < cv_folds - 1 else n_samples
        
        val_indices = list(range(start_idx, end_idx))
        train_indices = list(range(0, start_idx)) + list(range(end_idx, n_samples))
        
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]
        
        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred_fold = model.predict(X_val_fold)
        
        cv_scores['r2'].append(PolynomialRegressionMetrics.r_squared(y_val_fold, y_pred_fold))
        cv_scores['mse'].append(PolynomialRegressionMetrics.mean_squared_error(y_val_fold, y_pred_fold))
        cv_scores['rmse'].append(PolynomialRegressionMetrics.root_mean_squared_error(y_val_fold, y_pred_fold))
        cv_scores['mae'].append(PolynomialRegressionMetrics.mean_absolute_error(y_val_fold, y_pred_fold))
    
    results = {}
    for metric, scores in cv_scores.items():
        results[f'{metric}_mean'] = np.mean(scores)
        results[f'{metric}_std'] = np.std(scores)
        results[f'{metric}_scores'] = scores
    
    return results