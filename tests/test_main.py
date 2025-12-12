"""
Main test file for electricity price forecasting project.

This file contains integration tests and basic functionality tests
to ensure the project works correctly after reorganization.
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from electricity_price_forecast import __version__
    from electricity_price_forecast.models.baseline_models.linear_regression import LinearRegressionModel
    from electricity_price_forecast.utils.config.config_loader import get_model_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the package is properly installed or src is in PYTHONPATH")


class TestBasicFunctionality(unittest.TestCase):
    """Test basic project functionality."""

    def test_package_import(self):
        """Test that the main package can be imported."""
        try:
            import electricity_price_forecast
            self.assertIsNotNone(electricity_price_forecast.__version__)
        except ImportError:
            self.fail("Could not import electricity_price_forecast package")

    def test_version_exists(self):
        """Test that version is defined."""
        try:
            from electricity_price_forecast import __version__
            self.assertIsInstance(__version__, str)
            self.assertTrue(len(__version__) > 0)
        except ImportError:
            self.skipTest("Package not importable")

    def test_linear_regression_import(self):
        """Test that LinearRegressionModel can be imported."""
        try:
            from electricity_price_forecast.models.baseline_models.linear_regression import LinearRegressionModel
            self.assertIsNotNone(LinearRegressionModel)
        except ImportError:
            self.fail("Could not import LinearRegressionModel")

    def test_config_loader_import(self):
        """Test that config loader can be imported."""
        try:
            from electricity_price_forecast.utils.config.config_loader import get_model_config
            self.assertIsNotNone(get_model_config)
        except ImportError:
            self.fail("Could not import config loader")

    def test_linear_regression_instantiation(self):
        """Test that LinearRegressionModel can be instantiated."""
        try:
            from electricity_price_forecast.models.baseline_models.linear_regression import LinearRegressionModel
            model = LinearRegressionModel()
            self.assertIsNotNone(model)
        except ImportError:
            self.skipTest("LinearRegressionModel not importable")
        except Exception as e:
            self.fail(f"Could not instantiate LinearRegressionModel: {e}")


class TestProjectStructure(unittest.TestCase):
    """Test that project structure is correct."""

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        self.assertTrue(os.path.exists(src_path))

    def test_main_package_exists(self):
        """Test that main package directory exists."""
        package_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'electricity_price_forecast')
        self.assertTrue(os.path.exists(package_path))

    def test_init_files_exist(self):
        """Test that __init__.py files exist."""
        init_files = [
            os.path.join(os.path.dirname(__file__), '..', 'src', 'electricity_price_forecast', '__init__.py'),
            os.path.join(os.path.dirname(__file__), '..', 'tests', '__init__.py'),
        ]
        
        for init_file in init_files:
            self.assertTrue(os.path.exists(init_file), f"{init_file} does not exist")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)