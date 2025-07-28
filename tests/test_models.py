"""
Unit tests for GARCH models and data loading functionality.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import StockDataLoader
from src.garch_models import GARCHModel, EGARCHModel, GJR_GARCHModel
from src.utils import validate_ticker, validate_period, validate_model_type


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = StockDataLoader()
    
    def test_validate_ticker(self):
        """Test ticker validation."""
        self.assertTrue(validate_ticker("AAPL"))
        self.assertTrue(validate_ticker("SPY"))
        self.assertTrue(validate_ticker("TSLA"))
        self.assertFalse(validate_ticker(""))
        self.assertFalse(validate_ticker("INVALID"))
        self.assertFalse(validate_ticker("TOOLONG"))
    
    def test_validate_period(self):
        """Test period validation."""
        self.assertTrue(validate_period("1y"))
        self.assertTrue(validate_period("5y"))
        self.assertTrue(validate_period("max"))
        self.assertFalse(validate_period("invalid"))
        self.assertFalse(validate_period(""))
    
    def test_validate_model_type(self):
        """Test model type validation."""
        self.assertTrue(validate_model_type("GARCH"))
        self.assertTrue(validate_model_type("EGARCH"))
        self.assertTrue(validate_model_type("GJR-GARCH"))
        self.assertFalse(validate_model_type("invalid"))
        self.assertFalse(validate_model_type(""))


class TestGARCHModels(unittest.TestCase):
    """Test cases for GARCH model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic return data
        np.random.seed(42)
        n = 1000
        self.returns = pd.Series(
            np.random.normal(0, 0.02, n),
            index=pd.date_range('2020-01-01', periods=n, freq='D')
        )
    
    def test_garch_model_creation(self):
        """Test GARCH model creation."""
        model = GARCHModel()
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, "GARCH")
    
    def test_egarch_model_creation(self):
        """Test EGARCH model creation."""
        model = EGARCHModel()
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, "EGARCH")
    
    def test_gjr_garch_model_creation(self):
        """Test GJR-GARCH model creation."""
        model = GJR_GARCHModel()
        self.assertIsNotNone(model)
        self.assertEqual(model.model_type, "GJR-GARCH")
    
    def test_garch_model_fitting(self):
        """Test GARCH model fitting."""
        model = GARCHModel()
        diagnostics = model.fit(self.returns)
        
        self.assertTrue(model.fitted)
        self.assertIsNotNone(diagnostics)
        self.assertIn('aic', diagnostics)
        self.assertIn('bic', diagnostics)
        self.assertIn('log_likelihood', diagnostics)
        self.assertIn('parameters', diagnostics)
    
    def test_egarch_model_fitting(self):
        """Test EGARCH model fitting."""
        model = EGARCHModel()
        diagnostics = model.fit(self.returns)
        
        self.assertTrue(model.fitted)
        self.assertIsNotNone(diagnostics)
        self.assertIn('aic', diagnostics)
        self.assertIn('bic', diagnostics)
        self.assertIn('log_likelihood', diagnostics)
        self.assertIn('parameters', diagnostics)
    
    def test_gjr_garch_model_fitting(self):
        """Test GJR-GARCH model fitting."""
        model = GJR_GARCHModel()
        diagnostics = model.fit(self.returns)
        
        self.assertTrue(model.fitted)
        self.assertIsNotNone(diagnostics)
        self.assertIn('aic', diagnostics)
        self.assertIn('bic', diagnostics)
        self.assertIn('log_likelihood', diagnostics)
        self.assertIn('parameters', diagnostics)
    
    def test_forecasting(self):
        """Test forecasting functionality."""
        model = GARCHModel()
        diagnostics = model.fit(self.returns)
        
        forecast = model.forecast(horizon=5)
        
        self.assertIsNotNone(forecast)
        self.assertIn('volatility_forecast', forecast)
        self.assertIn('variance_forecast', forecast)
        self.assertIn('forecast_dates', forecast)
        self.assertEqual(forecast['horizon'], 5)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        from src.garch_models import GARCHModelComparison
        
        comparison = GARCHModelComparison()
        results = comparison.fit_models(self.returns, ['GARCH', 'EGARCH'])
        
        self.assertIsNotNone(results)
        self.assertIn('GARCH', results)
        self.assertIn('EGARCH', results)
        self.assertIn('best_aic', results)
        self.assertIn('best_bic', results)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_format_number(self):
        """Test number formatting."""
        from src.utils import format_number
        
        self.assertEqual(format_number(3.14159, 2), "3.14")
        self.assertEqual(format_number(0.001, 4), "0.0010")
        self.assertEqual(format_number(1000, 0), "1000")
    
    def test_calculate_percentage_change(self):
        """Test percentage change calculation."""
        from src.utils import calculate_percentage_change
        
        self.assertEqual(calculate_percentage_change(110, 100), 10.0)
        self.assertEqual(calculate_percentage_change(90, 100), -10.0)
        self.assertEqual(calculate_percentage_change(0, 100), -100.0)
        self.assertEqual(calculate_percentage_change(100, 0), 0)
    
    def test_generate_timestamp(self):
        """Test timestamp generation."""
        from src.utils import generate_timestamp
        
        timestamp = generate_timestamp()
        self.assertIsInstance(timestamp, str)
        self.assertEqual(len(timestamp), 15)  # YYYYMMDD_HHMMSS format


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 