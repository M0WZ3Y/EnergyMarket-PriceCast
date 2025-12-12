#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Setup Validation Script for Electricity Price Forecasting Thesis Project

This script validates:
1. All required directories exist
2. All critical files are present
3. All Python libraries can be imported
4. Configuration files are valid
5. Model imports work correctly
6. Jupyter notebooks are accessible
"""

import sys
import os
import importlib
import yaml
from pathlib import Path
import traceback

# Set console encoding to handle Unicode characters
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


class SetupValidator:
    def __init__(self):
        self.project_root = Path.cwd()
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        
    def log_success(self, message):
        print(f"✅ {message}")
        self.success_count += 1
        
    def log_error(self, message):
        print(f"❌ {message}")
        self.errors.append(message)
        
    def log_warning(self, message):
        print(f"⚠️  {message}")
        self.warnings.append(message)
        
    def log_info(self, message):
        print(f"ℹ️  {message}")
        
    def check_directories(self):
        """Check if all required directories exist."""
        print("\n" + "="*60)
        print("📁 CHECKING DIRECTORY STRUCTURE")
        print("="*60)
        
        required_dirs = [
            "01_documentation",
            "01_documentation/proposal",
            "01_documentation/literature_review",
            "01_documentation/thesis_draft",
            "02_data",
            "02_data/raw",
            "02_data/processed",
            "03_code",
            "03_code/data_pipeline",
            "03_code/models",
            "03_code/models/baseline_models",
            "03_code/models/tree_models",
            "03_code/models/deep_learning",
            "03_code/evaluation",
            "03_code/utils",
            "04_experiments",
            "05_outputs",
            "06_deployment",
            "06_deployment/config",
            "06_deployment/notebooks",
            "06_deployment/scripts",
            "06_deployment/requirements",
            "07_admin"
        ]
        
        for dir_path in required_dirs:
            self.total_checks += 1
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.log_success(f"Directory exists: {dir_path}")
            else:
                self.log_error(f"Missing directory: {dir_path}")
                
    def check_critical_files(self):
        """Check if all critical files are present."""
        print("\n" + "="*60)
        print("📄 CHECKING CRITICAL FILES")
        print("="*60)
        
        critical_files = [
            "README.md",
            ".gitignore",
            "06_deployment/requirements/requirements.txt",
            "06_deployment/requirements/environment.yml",
            "06_deployment/config/data_config.yaml",
            "06_deployment/config/model_config.yaml",
            "06_deployment/scripts/setup_environment.py",
            "03_code/utils/config/config_loader.py",
            "03_code/models/baseline_models/linear_regression.py",
            "06_deployment/notebooks/data_exploration/01_data_overview.ipynb",
            "setup.bat",
            "START_SETUP.bat",
            "quick_setup.py"
        ]
        
        for file_path in critical_files:
            self.total_checks += 1
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                self.log_success(f"File exists: {file_path}")
            else:
                self.log_error(f"Missing file: {file_path}")
                
    def check_python_libraries(self):
        """Check if all required Python libraries can be imported."""
        print("\n" + "="*60)
        print("🐍 CHECKING PYTHON LIBRARIES")
        print("="*60)
        
        required_libraries = [
            "pandas",
            "numpy", 
            "scipy",
            "sklearn",
            "matplotlib",
            "seaborn",
            "plotly",
            "yaml",
            "joblib",
            "tqdm",
            "requests",
            "beautifulsoup4",
            "pytest"
        ]
        
        optional_libraries = [
            ("xgboost", "XGBoost for gradient boosting"),
            ("tensorflow", "TensorFlow for deep learning"),
            ("keras", "Keras for neural networks"),
            ("shap", "SHAP for model explainability"),
            ("statsmodels", "Statsmodels for time series"),
            ("numba", "Numba for performance"),
            ("jupyter", "Jupyter for notebooks"),
            ("ipykernel", "Jupyter kernel support")
        ]
        
        # Check required libraries
        for lib in required_libraries:
            self.total_checks += 1
            try:
                importlib.import_module(lib)
                self.log_success(f"Library importable: {lib}")
            except ImportError as e:
                self.log_error(f"Cannot import {lib}: {e}")
                
        # Check optional libraries
        for lib, description in optional_libraries:
            self.total_checks += 1
            try:
                importlib.import_module(lib)
                self.log_success(f"Optional library available: {lib} ({description})")
            except ImportError:
                self.log_warning(f"Optional library missing: {lib} ({description})")
                
    def check_configuration_files(self):
        """Check if configuration files are valid YAML."""
        print("\n" + "="*60)
        print("⚙️  CHECKING CONFIGURATION FILES")
        print("="*60)
        
        config_files = [
            "06_deployment/config/data_config.yaml",
            "06_deployment/config/model_config.yaml"
        ]
        
        for config_file in config_files:
            self.total_checks += 1
            config_path = self.project_root / config_file
            
            if not config_path.exists():
                self.log_error(f"Config file missing: {config_file}")
                continue
                
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    
                # Basic validation
                if isinstance(config_data, dict) and config_data:
                    self.log_success(f"Valid YAML config: {config_file}")
                    
                    # Specific validations
                    if "model_config.yaml" in config_file:
                        if "models" in config_data:
                            self.log_success("Model config has 'models' section")
                        else:
                            self.log_error("Model config missing 'models' section")
                            
                    elif "data_config.yaml" in config_file:
                        if "data_sources" in config_data:
                            self.log_success("Data config has 'data_sources' section")
                        else:
                            self.log_error("Data config missing 'data_sources' section")
                else:
                    self.log_error(f"Invalid or empty config: {config_file}")
                    
            except yaml.YAMLError as e:
                self.log_error(f"YAML syntax error in {config_file}: {e}")
            except Exception as e:
                self.log_error(f"Error reading {config_file}: {e}")
                
    def check_project_imports(self):
        """Check if project modules can be imported."""
        print("\n" + "="*60)
        print("🔧 CHECKING PROJECT IMPORTS")
        print("="*60)
        
        # Add project root to Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
            
        # Check config loader
        self.total_checks += 1
        try:
            # Add the 03_code directory to Python path
            code_path = str(self.project_root / "03_code")
            if code_path not in sys.path:
                sys.path.insert(0, code_path)
            # Import using the correct path from 03_code
            from utils.config.config_loader import ConfigLoader
            self.log_success("Config loader importable")
            
            # Test config loading
            try:
                config_loader = ConfigLoader()
                self.log_success("Config loader instantiable")
            except Exception as e:
                self.log_error(f"Config loader instantiation failed: {e}")
                
        except ImportError as e:
            self.log_error(f"Cannot import config loader: {e}")
            
        # Check model imports
        # Ensure 03_code is in the path (already added above)
        model_imports = [
            ("models.baseline_models.linear_regression", "LinearRegressionModel"),
        ]
        
        for module_name, class_name in model_imports:
            self.total_checks += 1
            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                self.log_success(f"Model importable: {class_name}")
                
                # Test instantiation
                try:
                    if class_name == "LinearRegressionModel":
                        model = model_class(model_type='linear')
                        self.log_success(f"Model instantiable: {class_name}")
                except Exception as e:
                    self.log_warning(f"Model instantiation issue: {class_name} - {e}")
                    
            except ImportError as e:
                self.log_error(f"Cannot import {class_name}: {e}")
            except AttributeError as e:
                self.log_error(f"Class {class_name} not found: {e}")
                
    def check_jupyter_notebooks(self):
        """Check if Jupyter notebooks are accessible."""
        print("\n" + "="*60)
        print("📓 CHECKING JUPYTER NOTEBOOKS")
        print("="*60)
        
        notebook_files = [
            "06_deployment/notebooks/data_exploration/01_data_overview.ipynb"
        ]
        
        for notebook_file in notebook_files:
            self.total_checks += 1
            notebook_path = self.project_root / notebook_file
            
            if notebook_path.exists():
                try:
                    import json
                    with open(notebook_path, 'r', encoding='utf-8') as f:
                        notebook_data = json.load(f)
                        
                    if "cells" in notebook_data and isinstance(notebook_data["cells"], list):
                        self.log_success(f"Valid Jupyter notebook: {notebook_file}")
                    else:
                        self.log_error(f"Invalid notebook structure: {notebook_file}")
                        
                except json.JSONDecodeError as e:
                    self.log_error(f"Invalid JSON in notebook {notebook_file}: {e}")
                except Exception as e:
                    self.log_error(f"Error reading notebook {notebook_file}: {e}")
            else:
                self.log_error(f"Notebook missing: {notebook_file}")
                
    def check_virtual_environment(self):
        """Check if virtual environment is properly set up."""
        print("\n" + "="*60)
        print("🐍 CHECKING VIRTUAL ENVIRONMENT")
        print("="*60)
        
        self.total_checks += 1
        
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.log_success(f"Virtual environment active: {sys.prefix}")
        else:
            self.log_warning("No virtual environment detected")
            
        # Check for venv directory
        venv_path = self.project_root / "venv"
        if venv_path.exists():
            self.log_success("Virtual environment directory exists")
        else:
            self.log_warning("Virtual environment directory not found")
            
    def run_validation(self):
        """Run all validation checks."""
        print("🔍 ELECTRICITY PRICE FORECASTING - SETUP VALIDATION")
        print(f"📁 Project root: {self.project_root}")
        print(f"🐍 Python version: {sys.version}")
        print()
        
        # Run all checks
        self.check_virtual_environment()
        self.check_directories()
        self.check_critical_files()
        self.check_python_libraries()
        self.check_configuration_files()
        self.check_project_imports()
        self.check_jupyter_notebooks()
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0
        
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        success_rate = (self.success_count / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        print(f"✅ Successful checks: {self.success_count}/{self.total_checks} ({success_rate:.1f}%)")
        print(f"❌ Errors: {len(self.errors)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n🚨 ERRORS FOUND:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
                
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        print("\n" + "="*60)
        
        if len(self.errors) == 0:
            print("🎉 SETUP VALIDATION PASSED!")
            print("Your environment is ready for electricity price forecasting! 🚀")
        else:
            print("❌ SETUP VALIDATION FAILED!")
            print("Please fix the errors above before proceeding.")
            
        print("="*60)


def main():
    """Main validation function."""
    validator = SetupValidator()
    success = validator.run_validation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())