import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polynomial_regression import PolynomialRegression, PolynomialRegressionMetrics
from polynomial_regression import generate_polynomial_data, generate_sinusoidal_data
from polynomial_regression import train_test_split, plot_polynomial_fit, plot_residuals


def main():
    
    print("=" * 60)
    print("BASIC POLYNOMIAL REGRESSION EXAMPLE")
    print("=" * 60)
    
    print("\n" + "=" * 50)
    print("EXAMPLE 1: POLYNOMIAL DATA")
    print("=" * 50)
    
    print("\n1. Generating polynomial data...")
    X, y = generate_polynomial_data(
        n_samples=200, 
        degree=3, 
        noise=0.2,
        x_range=(-2, 2),
        random_state=42
    )
    print(f"    - Generated {len(X)} samples")
    print(f"    - True polynomial degree: 3")
    print(f"    - X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"    - y range: [{y.min():.2f}, {y.max():.2f}]")
    
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    - Training set: {len(X_train)} samples")
    print(f"    - Test set: {len(X_test)} samples")
    
    print("\n3. Training polynomial regression models...")
    degrees_to_test = [1, 2, 3, 4, 5]
    models = {}
    
    for degree in degrees_to_test:
        print(f"\n   Training degree {degree} model...")
        model = PolynomialRegression(
            degree=degree,
            learning_rate=0.01,
            max_iterations=1000,
            regularization='l2',
            reg_lambda=0.001,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        models[degree] = model
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred_test = model.predict(X_test)
        rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_test, y_pred_test)
        
        print(f"       Train R2: {train_score:.4f}")
        print(f"       Test R2:  {test_score:.4f}")
        print(f"       Test RMSE: {rmse:.4f}")
    
    print("\n4. Detailed evaluation of best model (degree 3)...")
    best_model = models[3]
    y_pred = best_model.predict(X_test)
    
    print("    Performance Metrics:")
    print("    " + "-" * 30)
    PolynomialRegressionMetrics.print_regression_report(y_test, y_pred, n_features=len(best_model.weights))
    
    print("\n5. Visualizing results...")
    plot_polynomial_fit(best_model, X_test, y_test, 
                       title="Polynomial Regression Fit (Degree 3)")
    
    best_model.plot_cost_history()
    
    plot_residuals(best_model, X_test, y_test, title="Residual Analysis - Degree 3")
    
    print("\n" + "=" * 50)
    print("EXAMPLE 2: SINUSOIDAL DATA APPROXIMATION")
    print("=" * 50)
    
    print("\n1. Generating sinusoidal data...")
    X_sin, y_sin = generate_sinusoidal_data(
        n_samples=300,
        frequency=1.0,
        amplitude=2.0,
        noise=0.1,
        x_range=(0, 2*np.pi),
        random_state=42
    )
    print(f"    - Generated {len(X_sin)} samples")
    print(f"    - Sine wave: amplitude=2.0, frequency=1.0")
    
    print("\n2. Splitting data...")
    X_sin_train, X_sin_test, y_sin_train, y_sin_test = train_test_split(
        X_sin, y_sin, test_size=0.2, random_state=42
    )
    
    print("\n3. Training high-degree polynomial to approximate sine...")
    degrees_sine = [3, 5, 7, 9, 11]
    sine_models = {}
    
    print("    Degree\tTrain R2\tTest R2\t\tRMSE")
    print("    " + "-" * 45)
    
    for degree in degrees_sine:
        model = PolynomialRegression(
            degree=degree,
            learning_rate=0.01,
            max_iterations=1500,
            regularization='l2',
            reg_lambda=0.01,
            verbose=False
        )
        
        model.fit(X_sin_train, y_sin_train)
        sine_models[degree] = model
        
        train_score = model.score(X_sin_train, y_sin_train)
        test_score = model.score(X_sin_test, y_sin_test)
        
        y_pred_sine = model.predict(X_sin_test)
        rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_sin_test, y_pred_sine)
        
        print(f"    {degree}\t{train_score:.4f}\t\t{test_score:.4f}\t\t{rmse:.4f}")
    
    print("\n4. Visualizing sine approximation...")
    best_sine_model = sine_models[9]
    
    plot_polynomial_fit(best_sine_model, X_sin_test, y_sin_test,
                       title="Polynomial Approximation of Sine Wave (Degree 9)")
    
    print("\n5. Comparing different regularization strengths...")
    reg_lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    
    print("    Lambda\t\tTrain R2\tTest R2\t\tRMSE")
    print("    " + "-" * 50)
    
    for reg_lambda in reg_lambdas:
        model = PolynomialRegression(
            degree=9,
            learning_rate=0.01,
            max_iterations=1500,
            regularization='l2',
            reg_lambda=reg_lambda,
            verbose=False
        )
        
        model.fit(X_sin_train, y_sin_train)
        
        train_score = model.score(X_sin_train, y_sin_train)
        test_score = model.score(X_sin_test, y_sin_test)
        
        y_pred_reg = model.predict(X_sin_test)
        rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_sin_test, y_pred_reg)
        
        print(f"    {reg_lambda:<10.3f}\t{train_score:.4f}\t\t{test_score:.4f}\t\t{rmse:.4f}")
    
    print("\n" + "=" * 50)
    print("EXAMPLE 3: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    print("\n1. Analyzing polynomial coefficients...")
    analysis_model = models[3]  
    
    print(f"    Model weights: {analysis_model.weights}")
    print(f"    Model bias: {analysis_model.bias}")
    print(f"    Weights magnitude: {np.linalg.norm(analysis_model.weights):.4f}")
    
    feature_importance = np.abs(analysis_model.weights)
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    print("\n    Feature importance (by absolute weight):")
    print("    " + "-" * 40)
    for i, idx in enumerate(sorted_indices):
        print(f"    {i+1}. Feature {idx+1}: {feature_importance[idx]:.4f} (weight: {analysis_model.weights[idx]:+.4f})")
    
    print("\n6. Model complexity analysis...")
    print("    " + "-" * 35)
    for degree, model in models.items():
        n_params = len(model.weights) + (1 if model.fit_intercept else 0)
        print(f"    Degree {degree}: {n_params} parameters")
    
    print("\n" + "=" * 60)
    print("BASIC EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return models, sine_models


if __name__ == "__main__":
    models, sine_models = main()