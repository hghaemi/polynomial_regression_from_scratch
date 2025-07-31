import numpy as np
import matplotlib.pyplot as plt 
from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class PolynomialRegression:
    def __init__(self,
                 degree: int = 2,
                 learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 regularization: Optional[str] = None,
                 reg_lambda: float = 0.01,
                 tolerance: float = 1e-6,
                 fit_intercept: bool = True,
                 verbose: bool = False):

        self.degree = degree
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        self.weights = None
        self.bias = None
        self.cost_history = []
        self.n_iterations = 0
        self.feature_means = None
        self.feature_stds = None

        if regularization and regularization not in ['l1', 'l2']:
            raise ValueError("Regularization must be 'l1', 'l2', or None")
        
        if degree < 1:
            raise ValueError("Degree must be at least 1")
    

    def _create_polynomial_features(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        poly_features = [X]
        
        for d in range(2, self.degree + 1):
            for i in range(n_features):
                poly_features.append((X[:, i] ** d).reshape(-1, 1))
        
        if n_features > 1:
            for d in range(2, self.degree + 1):
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        for power_i in range(1, d):
                            power_j = d - power_i
                            if power_j > 0:
                                interaction = (X[:, i] ** power_i) * (X[:, j] ** power_j)
                                poly_features.append(interaction.reshape(-1, 1))
        
        poly_X = np.hstack(poly_features)
        
        if normalize:
            if self.feature_means is None:
                self.feature_means = np.mean(poly_X, axis=0)
                self.feature_stds = np.std(poly_X, axis=0) + 1e-8
            
            poly_X = (poly_X - self.feature_means) / self.feature_stds
        
        return poly_X
    

    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        m = len(y_true)
        
        mse = np.mean((y_pred - y_true) ** 2)
        
        # add regularization
        if self.regularization == 'l1':
            mse += self.reg_lambda * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            mse += self.reg_lambda * np.sum(self.weights ** 2)
        
        return mse


    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        m = X.shape[0]

        dw = (2/m) * np.dot(X.T, (y_pred - y))
        db = (2/m) * np.sum(y_pred - y) if self.fit_intercept else 0
    
        if self.regularization == 'l1':
            dw += self.reg_lambda * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.reg_lambda * self.weights
        
        return dw, db
    
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PolynomialRegression':

        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_poly = self._create_polynomial_features(X, normalize=True)
        
        n_samples, n_features = X_poly.shape
        
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0 if self.fit_intercept else None
        self.cost_history = []

        prev_cost = float('inf')
        
        for i in range(self.max_iterations):
            y_pred = np.dot(X_poly, self.weights)
            if self.fit_intercept:
                y_pred += self.bias
            
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            dw, db = self._compute_gradients(X_poly, y, y_pred)
            
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db
            
            if abs(prev_cost - cost) < self.tolerance:
                if self.verbose:
                    print(f"Converged after {i+1} iterations")
                break
            
            prev_cost = cost
            
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.max_iterations}, Cost: {cost:.6f}")
        
        self.n_iterations = i + 1
        return self
    

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # create polynomial features (using stored normalization parameters)
        X_poly = self._create_polynomial_features(X, normalize=True)
        
        y_pred = np.dot(X_poly, self.weights)
        if self.fit_intercept:
            y_pred += self.bias
        
        return y_pred
    

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    

    def plot_cost_history(self, figsize: Tuple[int, int] = (10, 6)):
        if not self.cost_history:
            raise ValueError("Model must be fitted first")
        
        plt.figure(figsize=figsize)
        plt.plot(self.cost_history, 'b-', linewidth=2)
        plt.title('Cost Function Over Iterations', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
    
    def get_params(self) -> dict:
        return {
            'degree': self.degree,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'regularization': self.regularization,
            'reg_lambda': self.reg_lambda,
            'tolerance': self.tolerance,
            'fit_intercept': self.fit_intercept,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self


class PolynomialRegressionMetrics:
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(PolynomialRegressionMetrics.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        n = len(y_true)
        r2 = PolynomialRegressionMetrics.r_squared(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    @staticmethod
    def regression_report(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = None) -> dict:
        return {
            'mse': PolynomialRegressionMetrics.mean_squared_error(y_true, y_pred),
            'rmse': PolynomialRegressionMetrics.root_mean_squared_error(y_true, y_pred),
            'mae': PolynomialRegressionMetrics.mean_absolute_error(y_true, y_pred),
            'r2': PolynomialRegressionMetrics.r_squared(y_true, y_pred),
            'adj_r2': PolynomialRegressionMetrics.adjusted_r_squared(y_true, y_pred, n_features) if n_features else None
        }
    
    @staticmethod
    def print_regression_report(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = None):
        report = PolynomialRegressionMetrics.regression_report(y_true, y_pred, n_features)
        
        print("Regression Report")
        print("=" * 40)
        print(f"Mean Squared Error:     {report['mse']:.6f}")
        print(f"Root Mean Squared Error: {report['rmse']:.6f}")
        print(f"Mean Absolute Error:    {report['mae']:.6f}")
        print(f"R-squared:              {report['r2']:.6f}")
        if report['adj_r2'] is not None:
            print(f"Adjusted R-squared:     {report['adj_r2']:.6f}")