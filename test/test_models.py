import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polynomial_regression.models import PolynomialRegression, PolynomialRegressionMetrics
from polynomial_regression.utils import generate_polynomial_data, generate_sinusoidal_data


class TestPolynomialRegression(unittest.TestCase):
    
    def setUp(self):
        self.X_simple, self.y_simple = generate_polynomial_data(
            n_samples=100, degree=2, n_features=1, noise=0.1, random_state=42
        )
        self.X_multi, self.y_multi = generate_polynomial_data(
            n_samples=200, degree=3, n_features=2, noise=0.1, random_state=42
        )
    
    def test_initialization(self):

        model = PolynomialRegression()
        
        self.assertEqual(model.degree, 2)
        self.assertEqual(model.learning_rate, 0.01)
        self.assertEqual(model.max_iterations, 1000)
        self.assertIsNone(model.regularization)
        self.assertEqual(model.reg_lambda, 0.01)
        self.assertEqual(model.tolerance, 1e-6)
        self.assertTrue(model.fit_intercept)
        self.assertFalse(model.verbose)
        self.assertIsNone(model.weights)
        self.assertIsNone(model.bias)
        self.assertEqual(model.cost_history, [])
    
    def test_initialization_with_custom_params(self):
        model = PolynomialRegression(
            degree=4,
            learning_rate=0.1,
            max_iterations=500,
            regularization='l2',
            reg_lambda=0.1,
            tolerance=1e-5,
            fit_intercept=False,
            verbose=True
        )
        
        self.assertEqual(model.degree, 4)
        self.assertEqual(model.learning_rate, 0.1)
        self.assertEqual(model.max_iterations, 500)
        self.assertEqual(model.regularization, 'l2')
        self.assertEqual(model.reg_lambda, 0.1)
        self.assertEqual(model.tolerance, 1e-5)
        self.assertFalse(model.fit_intercept)
        self.assertTrue(model.verbose)
    
    def test_invalid_regularization(self):
        with self.assertRaises(ValueError):
            PolynomialRegression(regularization='invalid')
    
    def test_invalid_degree(self):
        with self.assertRaises(ValueError):
            PolynomialRegression(degree=0)
        
        with self.assertRaises(ValueError):
            PolynomialRegression(degree=-1)
    
    def test_polynomial_feature_creation(self):
        model = PolynomialRegression(degree=2)
        X = np.array([[1], [2], [3]])
        
        # test feature creation without normalization first
        model.feature_means = None
        model.feature_stds = None
        poly_features = model._create_polynomial_features(X, normalize=False)
        
        # for degree 2 with 1 feature, we should have X and X^2
        expected_features = 2
        self.assertEqual(poly_features.shape[1], expected_features)
        
        # Test with normalization
        poly_features_norm = model._create_polynomial_features(X, normalize=True)
        self.assertEqual(poly_features_norm.shape[1], expected_features)
        self.assertIsNotNone(model.feature_means)
        self.assertIsNotNone(model.feature_stds)
    
    def test_fit_simple_data(self):

        model = PolynomialRegression(degree=2, learning_rate=0.1, max_iterations=1000)
        model.fit(self.X_simple, self.y_simple)
        
        self.assertIsNotNone(model.weights)
        self.assertIsNotNone(model.bias)
        self.assertGreater(len(model.weights), 0)
        self.assertGreater(len(model.cost_history), 0)
        
        # cost should generally decrease
        self.assertLess(model.cost_history[-1], model.cost_history[0])
    
    def test_fit_with_regularization(self):

        # Test L1 regularization
        model_l1 = PolynomialRegression(degree=2, regularization='l1', reg_lambda=0.1)
        model_l1.fit(self.X_simple, self.y_simple)
        
        self.assertIsNotNone(model_l1.weights)
        self.assertGreater(len(model_l1.cost_history), 0)
        
        # Test L2 regularization
        model_l2 = PolynomialRegression(degree=2, regularization='l2', reg_lambda=0.1)
        model_l2.fit(self.X_simple, self.y_simple)
        
        self.assertIsNotNone(model_l2.weights)
        self.assertGreater(len(model_l2.cost_history), 0)
    
    def test_fit_without_intercept(self):
        model = PolynomialRegression(degree=2, fit_intercept=False)
        model.fit(self.X_simple, self.y_simple)
        
        self.assertIsNotNone(model.weights)
        self.assertIsNone(model.bias)
    
    def test_predict(self):
        model = PolynomialRegression(degree=2)
        model.fit(self.X_simple, self.y_simple)
        
        predictions = model.predict(self.X_simple)
        
        self.assertEqual(len(predictions), len(self.X_simple))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_predict_before_fit_error(self):
        model = PolynomialRegression(degree=2)
        
        with self.assertRaises(ValueError):
            model.predict(self.X_simple)
    
    def test_score(self):
        model = PolynomialRegression(degree=2, learning_rate=0.1)
        model.fit(self.X_simple, self.y_simple)
        
        score = model.score(self.X_simple, self.y_simple)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_1d_input_handling(self):
        X_1d = np.array([1, 2, 3, 4, 5])
        y_1d = X_1d ** 2 + np.random.normal(0, 0.1, len(X_1d))
        
        model = PolynomialRegression(degree=2)
        model.fit(X_1d, y_1d)
        
        self.assertIsNotNone(model.weights)
        
        pred = model.predict(X_1d[:3])
        self.assertEqual(len(pred), 3)
    
    def test_multivariate_fitting(self):

        model = PolynomialRegression(degree=2, learning_rate=0.01, max_iterations=1500)
        model.fit(self.X_multi, self.y_multi)
        
        self.assertIsNotNone(model.weights)
        self.assertGreater(len(model.weights), self.X_multi.shape[1])
        
        predictions = model.predict(self.X_multi)
        self.assertEqual(len(predictions), len(self.X_multi))
    
    def test_get_set_params(self):
        model = PolynomialRegression()
        
        params = model.get_params()
        expected_keys = ['degree', 'learning_rate', 'max_iterations', 'regularization', 
                        'reg_lambda', 'tolerance', 'fit_intercept', 'verbose']
        for key in expected_keys:
            self.assertIn(key, params)
        
        model.set_params(degree=4, learning_rate=0.1)
        self.assertEqual(model.degree, 4)
        self.assertEqual(model.learning_rate, 0.1)
        
        with self.assertRaises(ValueError):
            model.set_params(invalid_param=0.1)
    
    def test_cost_computation(self):
        model = PolynomialRegression(degree=2)
        model.fit(self.X_simple, self.y_simple)
        
        y_pred = model.predict(self.X_simple)
        cost_no_reg = model._compute_cost(self.y_simple, y_pred)
        
        # Test L2 regularization cost
        model_l2 = PolynomialRegression(degree=2, regularization='l2', reg_lambda=0.1)
        model_l2.fit(self.X_simple, self.y_simple)
        y_pred_l2 = model_l2.predict(self.X_simple)
        cost_l2 = model_l2._compute_cost(self.y_simple, y_pred_l2)
        
        self.assertGreater(cost_l2, cost_no_reg * 0.8)
    
    def test_convergence(self):
        model = PolynomialRegression(degree=2, learning_rate=0.1, tolerance=1e-4, verbose=False)
        model.fit(self.X_simple, self.y_simple)
        
        self.assertLess(model.n_iterations, model.max_iterations)


class TestPolynomialRegressionMetrics(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_good = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        self.y_pred_poor = np.array([2.0, 1.0, 4.0, 3.0, 6.0])
    
    def test_mean_squared_error(self):
        mse_perfect = PolynomialRegressionMetrics.mean_squared_error(
            self.y_true, self.y_pred_perfect
        )
        self.assertAlmostEqual(mse_perfect, 0.0, places=5)
        
        mse_good = PolynomialRegressionMetrics.mean_squared_error(
            self.y_true, self.y_pred_good
        )
        self.assertGreater(mse_good, 0)
        self.assertLess(mse_good, 0.1)
    
    def test_root_mean_squared_error(self):
        rmse_perfect = PolynomialRegressionMetrics.root_mean_squared_error(
            self.y_true, self.y_pred_perfect
        )
        self.assertAlmostEqual(rmse_perfect, 0.0, places=5)
        
        rmse_good = PolynomialRegressionMetrics.root_mean_squared_error(
            self.y_true, self.y_pred_good
        )
        mse_good = PolynomialRegressionMetrics.mean_squared_error(
            self.y_true, self.y_pred_good
        )
        self.assertAlmostEqual(rmse_good, np.sqrt(mse_good), places=5)
    
    def test_mean_absolute_error(self):
        mae_perfect = PolynomialRegressionMetrics.mean_absolute_error(
            self.y_true, self.y_pred_perfect
        )
        self.assertAlmostEqual(mae_perfect, 0.0, places=5)
        
        mae_good = PolynomialRegressionMetrics.mean_absolute_error(
            self.y_true, self.y_pred_good
        )
        expected_mae = np.mean(np.abs(self.y_true - self.y_pred_good))
        self.assertAlmostEqual(mae_good, expected_mae, places=5)
    
    def test_r_squared(self):
        r2_perfect = PolynomialRegressionMetrics.r_squared(
            self.y_true, self.y_pred_perfect
        )
        self.assertAlmostEqual(r2_perfect, 1.0, places=5)
        
        r2_good = PolynomialRegressionMetrics.r_squared(
            self.y_true, self.y_pred_good
        )
        self.assertGreater(r2_good, 0.8)
        self.assertLessEqual(r2_good, 1.0)
        
        r2_poor = PolynomialRegressionMetrics.r_squared(
            self.y_true, self.y_pred_poor
        )
        self.assertLess(r2_poor, r2_good)
    
    def test_adjusted_r_squared(self):
        r2 = PolynomialRegressionMetrics.r_squared(self.y_true, self.y_pred_good)
        adj_r2 = PolynomialRegressionMetrics.adjusted_r_squared(
            self.y_true, self.y_pred_good, n_features=2
        )
        
        self.assertLess(adj_r2, r2)
    
    def test_regression_report(self):
        report = PolynomialRegressionMetrics.regression_report(
            self.y_true, self.y_pred_good, n_features=2
        )
        
        required_keys = ['mse', 'rmse', 'mae', 'r2', 'adj_r2']
        for key in required_keys:
            self.assertIn(key, report)
        
        self.assertAlmostEqual(report['rmse'], np.sqrt(report['mse']), places=5)
        
        self.assertIsNotNone(report['adj_r2'])
    
    def test_regression_report_without_features(self):
        report = PolynomialRegressionMetrics.regression_report(
            self.y_true, self.y_pred_good
        )
        
        self.assertIsNone(report['adj_r2'])


class TestPolynomialRegressionIntegration(unittest.TestCase):
    """Integration tests combining model and metrics."""
    
    def test_end_to_end_workflow(self):
        X, y = generate_polynomial_data(
            n_samples=200, degree=3, n_features=1, noise=0.1, random_state=42
        )
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = PolynomialRegression(degree=3, learning_rate=0.01, max_iterations=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        report = PolynomialRegressionMetrics.regression_report(
            y_test, y_pred, n_features=len(model.weights)
        )
        
        self.assertGreater(report['r2'], 0.8)
        self.assertLess(report['rmse'], 1.0)
    

    def test_overfitting_detection(self):
        X, y = generate_sinusoidal_data(
            n_samples=50, noise=0.1, random_state=42
        )
        
        split_idx = 40
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model_overfit = PolynomialRegression(
            degree=10, learning_rate=0.01, max_iterations=2000, regularization=None
        )
        model_overfit.fit(X_train, y_train)
        
        model_reg = PolynomialRegression(
            degree=10, learning_rate=0.01, max_iterations=2000, 
            regularization='l2', reg_lambda=0.1
        )
        model_reg.fit(X_train, y_train)
        
        train_score_overfit = model_overfit.score(X_train, y_train)
        test_score_overfit = model_overfit.score(X_test, y_test)
        
        train_score_reg = model_reg.score(X_train, y_train)
        test_score_reg = model_reg.score(X_test, y_test)
        
        overfit_gap = abs(train_score_overfit - test_score_overfit)
        reg_gap = abs(train_score_reg - test_score_reg)
        
        self.assertLessEqual(reg_gap, overfit_gap + 0.05, 
                            f"Regularization should reduce overfitting. "
                            f"Overfit gap: {overfit_gap:.4f}, Reg gap: {reg_gap:.4f}")
        
        self.assertGreaterEqual(test_score_reg, test_score_overfit - 0.2,
                            "Regularized model test performance shouldn't be much worse")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    
    suite.addTest(unittest.makeSuite(TestPolynomialRegression))
    suite.addTest(unittest.makeSuite(TestPolynomialRegressionMetrics))
    suite.addTest(unittest.makeSuite(TestPolynomialRegressionIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed.")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")