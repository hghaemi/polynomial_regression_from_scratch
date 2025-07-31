import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polynomial_regression import PolynomialRegression, PolynomialRegressionMetrics
from polynomial_regression import generate_polynomial_data, train_test_split
from polynomial_regression import normalize_features, apply_normalization
from polynomial_regression import compare_polynomial_degrees, cross_validate


def compare_regularization_methods(X_train, y_train, X_test, y_test):

    print("\n" + "=" * 50)
    print("COMPARING REGULARIZATION METHODS")
    print("=" * 50)
    
    configs = [
        {'name': 'No Regularization', 'regularization': None, 'reg_lambda': 0},
        {'name': 'L1 Regularization', 'regularization': 'l1', 'reg_lambda': 0.01},
        {'name': 'L2 Regularization', 'regularization': 'l2', 'reg_lambda': 0.01},
        {'name': 'Strong L2 Reg', 'regularization': 'l2', 'reg_lambda': 0.1},
    ]
    
    results = []
    models = []
    
    for config in configs:
        print(f"\nTraining {config['name']}...")
        
        model = PolynomialRegression(
            degree=3,
            learning_rate=0.01,
            max_iterations=1500,
            regularization=config['regularization'],
            reg_lambda=config['reg_lambda'],
            verbose=False
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = PolynomialRegressionMetrics.regression_report(y_test, y_pred, len(model.weights))
        
        results.append({
            'name': config['name'],
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'weights_norm': np.linalg.norm(model.weights)
        })
        
        models.append(model)
        
        print(f"   R² Score: {metrics['r2']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   Weights L2 Norm: {np.linalg.norm(model.weights):.4f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    names = [r['name'] for r in results]
    r2_scores = [r['r2'] for r in results]
    plt.bar(names, r2_scores, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    plt.subplot(1, 3, 2)
    rmse_scores = [r['rmse'] for r in results]
    plt.bar(names, rmse_scores, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    weights_norms = [r['weights_norm'] for r in results]
    plt.bar(names, weights_norms, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.title('Weights L2 Norm')
    plt.ylabel('L2 Norm')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results, models


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    
    print("\n" + "=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)
    
    learning_rates = [0.001, 0.01, 0.1]
    reg_lambdas = [0.001, 0.01, 0.1]
    degrees = [2, 3, 4]
    
    best_score = -float('inf')
    best_params = {}
    results = []
    
    print("Testing hyperparameter combinations...")
    print("Degree\tLR\tLambda\tR2 Score\tRMSE")
    print("-" * 50)
    
    for degree in degrees:
        for lr in learning_rates:
            for reg_lambda in reg_lambdas:
                model = PolynomialRegression(
                    degree=degree,
                    learning_rate=lr,
                    max_iterations=1500,
                    regularization='l2',
                    reg_lambda=reg_lambda,
                    verbose=False
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                r2 = PolynomialRegressionMetrics.r_squared(y_val, y_pred)
                rmse = PolynomialRegressionMetrics.root_mean_squared_error(y_val, y_pred)
                
                results.append({
                    'degree': degree,
                    'learning_rate': lr,
                    'reg_lambda': reg_lambda,
                    'r2': r2,
                    'rmse': rmse
                })
                
                print(f"{degree}\t{lr}\t{reg_lambda}\t{r2:.4f}\t\t{rmse:.4f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_params = {
                        'degree': degree,
                        'learning_rate': lr, 
                        'reg_lambda': reg_lambda
                    }
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation R2: {best_score:.4f}")
    
    return best_params, results


def analyze_feature_importance(model, feature_names=None):
    
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    if feature_names is None:
        feature_names = [f"Feature_{i+1}" for i in range(len(model.weights))]
    
    importance = np.abs(model.weights)
    sorted_indices = np.argsort(importance)[::-1]
    
    print("Feature Importance (based on absolute weights):")
    print("-" * 45)
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1:2d}. {feature_names[idx]:15s}: {importance[idx]:.4f} (weight: {model.weights[idx]:+.4f})")
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance[sorted_indices], alpha=0.7)
    plt.title('Feature Importance (Absolute Weights)')
    plt.xlabel('Features (sorted by importance)')
    plt.ylabel('Absolute Weight Value')
    plt.xticks(range(len(importance)), 
               [feature_names[i] for i in sorted_indices], 
               rotation=45)
    plt.tight_layout()
    plt.show()


def generate_complex_polynomial_data(n_samples=1000, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.uniform(-2, 2, (n_samples, 3))
    
    y = (2 * X[:, 0] + 
         1.5 * X[:, 1] - 
         0.8 * X[:, 2] +
         0.5 * X[:, 0]**2 + 
         0.3 * X[:, 1]**2 +
         0.7 * X[:, 0] * X[:, 1] +
         0.2 * X[:, 0]**3 +
         0.1 * X[:, 1] * X[:, 2])
    
    y += np.random.normal(0, 0.3, n_samples)
    
    return X, y


def main():
    
    print("=" * 60)
    print("MULTIVARIATE POLYNOMIAL REGRESSION EXAMPLE")
    print("=" * 60)
    
    print("\n1. Generating complex multivariate polynomial data...")
    X, y = generate_complex_polynomial_data(n_samples=2000, random_state=42)
    print(f"   - Generated {len(X)} samples with {X.shape[1]} features")
    print(f"   - Feature ranges:")
    for i in range(X.shape[1]):
        print(f"     Feature {i+1}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    print(f"   - Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    print("\n2. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    print(f"   - Training set: {len(X_train)} samples")
    print(f"   - Validation set: {len(X_val)} samples")
    print(f"   - Test set: {len(X_test)} samples")
    
    print("\n3. Normalizing features...")
    X_train_norm, norm_params = normalize_features(X_train, method='standard')
    X_val_norm = apply_normalization(X_val, norm_params)
    X_test_norm = apply_normalization(X_test, norm_params)
    
    print("   - Applied standard normalization")
    print(f"   - Training set mean: {np.mean(X_train_norm, axis=0)}")
    print(f"   - Training set std: {np.std(X_train_norm, axis=0)}")
    
    reg_results, reg_models = compare_regularization_methods(
        X_train_norm, y_train, X_test_norm, y_test
    )
    
    best_params, tuning_results = hyperparameter_tuning(
        X_train_norm, y_train, X_val_norm, y_val
    )
    
    print("\n" + "=" * 50)
    print("COMPARING POLYNOMIAL DEGREES")
    print("=" * 50)
    
    degree_results, degree_models = compare_polynomial_degrees(
        X_train_norm, y_train, X_test_norm, y_test,
        degrees=[1, 2, 3, 4, 5]
    )
    
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 50)
    
    print("\nPerforming 5-fold cross-validation...")
    cv_results = cross_validate(
        PolynomialRegression, 
        X_train_norm, y_train,
        cv_folds=5,
        degree=best_params['degree'],
        learning_rate=best_params['learning_rate'],
        regularization='l2',
        reg_lambda=best_params['reg_lambda'],
        max_iterations=1500,
        verbose=False
    )
    
    print("Cross-validation results:")
    print("-" * 30)
    for metric in ['r2', 'mse', 'rmse', 'mae']:
        mean_score = cv_results[f'{metric}_mean']
        std_score = cv_results[f'{metric}_std']
        print(f"{metric.upper():4s}: {mean_score:.4f} ± {std_score:.4f}")
    
    print("\n" + "=" * 50)
    print("TRAINING FINAL MODEL")
    print("=" * 50)
    
    final_model = PolynomialRegression(
        degree=best_params['degree'],
        learning_rate=best_params['learning_rate'],
        max_iterations=2000,
        regularization='l2',
        reg_lambda=best_params['reg_lambda'],
        verbose=True
    )
    
    X_train_full = np.vstack([X_train_norm, X_val_norm])
    y_train_full = np.hstack([y_train, y_val])
    
    final_model.fit(X_train_full, y_train_full)
    
    print("\n7. Final Model Evaluation:")
    print("-" * 30)
    y_pred_final = final_model.predict(X_test_norm)
    PolynomialRegressionMetrics.print_regression_report(
        y_test, y_pred_final, n_features=len(final_model.weights)
    )
    
    n_original_features = X.shape[1]
    feature_names = []
    
    for i in range(n_original_features):
        feature_names.append(f"X{i+1}")
    
    if best_params['degree'] >= 2:
        for i in range(n_original_features):
            feature_names.append(f"X{i+1}²")
        for i in range(n_original_features):
            for j in range(i+1, n_original_features):
                feature_names.append(f"X{i+1}*X{j+1}")
    
    if best_params['degree'] >= 3:
        for i in range(n_original_features):
            feature_names.append(f"X{i+1}³")
    
    while len(feature_names) < len(final_model.weights):
        feature_names.append(f"Feature_{len(feature_names)+1}")
    
    analyze_feature_importance(final_model, feature_names[:len(final_model.weights)])
    
    final_model.plot_cost_history()
    
    print("\n" + "=" * 50)
    print("OVERFITTING ANALYSIS")
    print("=" * 50)
    
    print("\nAnalyzing overfitting across different degrees...")
    degrees_analysis = [1, 2, 3, 4, 5, 6, 7]
    train_scores = []
    test_scores = []
    
    for degree in degrees_analysis:
        model = PolynomialRegression(
            degree=degree,
            learning_rate=0.01,
            max_iterations=1500,
            regularization='l2',
            reg_lambda=0.01,
            verbose=False
        )
        
        model.fit(X_train_norm, y_train)
        
        train_r2 = model.score(X_train_norm, y_train)
        test_r2 = model.score(X_test_norm, y_test)
        
        train_scores.append(train_r2)
        test_scores.append(test_r2)
        
        print(f"Degree {degree}: Train R2={train_r2:.4f}, Test R2={test_r2:.4f}, "
              f"Gap={train_r2-test_r2:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(degrees_analysis, train_scores, 'o-', label='Training R2', linewidth=2, markersize=8)
    plt.plot(degrees_analysis, test_scores, 's-', label='Test R2', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R2 Score')
    plt.title('Overfitting Analysis: Training vs Test Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n" + "=" * 50)
    print("PREDICTION ANALYSIS")
    print("=" * 50)
    
    y_pred_final = final_model.predict(X_test_norm)
    residuals = y_test - y_pred_final
    
    print(f"Prediction statistics:")
    print(f"  Mean absolute error: {np.mean(np.abs(residuals)):.4f}")
    print(f"  Mean squared error: {np.mean(residuals**2):.4f}")
    print(f"  Standard deviation of residuals: {np.std(residuals):.4f}")
    print(f"  Max absolute error: {np.max(np.abs(residuals)):.4f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_final, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n8. Model interpretability:")
    print("-" * 25)
    print(f"Final model degree: {final_model.degree}")
    print(f"Number of features: {len(final_model.weights)}")
    print(f"Regularization strength: {final_model.reg_lambda}")
    print(f"Converged after: {final_model.n_iterations} iterations")
    print(f"Final cost: {final_model.cost_history[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("MULTIVARIATE EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'final_model': final_model,
        'best_params': best_params,
        'cv_results': cv_results,
        'reg_results': reg_results,
        'degree_results': degree_results
    }


if __name__ == "__main__":
    results = main()