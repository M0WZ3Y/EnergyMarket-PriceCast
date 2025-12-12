#!/usr/bin/env python3
"""
Setup Validation Script for Electricity Price Forecasting Thesis Project
"""

import sys
import os
import importlib
import yaml
from pathlib import Path


def check_directories():
    """Check if all required directories exist."""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    required_dirs = [
        "01_documentation",
        "02_data",
        "03_code",
        "04_experiments", 
        "05_outputs",
        "06_deployment",
        "07_admin",
        "03_code/models",
        "03_code/utils",
        "06_deployment/config",
        "06_deployment/notebooks"
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists() and full_path.is_dir():
            print(f"[OK] Directory exists: {dir_path}")
        else:
            print(f"[ERROR] Missing directory: {dir_path}")
            missing.append(dir_path)
    
    return len(missing) == 0


def check_critical_files():
    """Check if all critical files are present."""
    print("\n" + "="*60)
    print("CHECKING CRITICAL FILES")
    print("="*60)
    
    critical_files = [
        "README.md",
        ".gitignore",
        "06_deployment/requirements/requirements.txt",
        "06_deployment/config/data_config.yaml",
        "06_deployment/config/model_config.yaml",
        "03_code/utils/config/config_loader.py",
        "03_code/models/baseline_models/linear_regression.py"
    ]
    
    missing = []
    for file_path in critical_files:
        full_path = Path(file_path)
        if full_path.exists() and full_path.is_file():
            print(f"[OK] File exists: {file_path}")
        else:
            print(f"[ERROR] Missing file: {file_path}")
            missing.append(file_path)
    
    return len(missing) == 0


def check_python_libraries():
    """Check if all required Python libraries can be imported."""
    print("\n" + "="*60)
    print("CHECKING PYTHON LIBRARIES")
    print("="*60)
    
    required_libraries = [
        "pandas",
        "numpy", 
        "sklearn",
        "matplotlib",
        "yaml",
        "joblib"
    ]
    
    optional_libraries = [
        "xgboost",
        "tensorflow",
        "shap",
        "jupyter"
    ]
    
    missing_required = []
    missing_optional = []
    
    for lib in required_libraries:
        try:
            importlib.import_module(lib)
            print(f"[OK] Library importable: {lib}")
        except ImportError:
            print(f"[ERROR] Cannot import {lib}")
            missing_required.append(lib)
    
    for lib in optional_libraries:
        try:
            importlib.import_module(lib)
            print(f"[OK] Optional library available: {lib}")
        except ImportError:
            print(f"[WARNING] Optional library missing: {lib}")
            missing_optional.append(lib)
    
    return len(missing_required) == 0


def check_configuration_files():
    """Check if configuration files are valid YAML."""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION FILES")
    print("="*60)
    
    config_files = [
        "06_deployment/config/data_config.yaml",
        "06_deployment/config/model_config.yaml"
    ]
    
    invalid = []
    for config_file in config_files:
        config_path = Path(config_file)
        
        if not config_path.exists():
            print(f"[ERROR] Config file missing: {config_file}")
            invalid.append(config_file)
            continue
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                
            if isinstance(config_data, dict) and config_data:
                print(f"[OK] Valid YAML config: {config_file}")
            else:
                print(f"[ERROR] Invalid or empty config: {config_file}")
                invalid.append(config_file)
                
        except yaml.YAMLError as e:
            print(f"[ERROR] YAML syntax error in {config_file}: {e}")
            invalid.append(config_file)
        except Exception as e:
            print(f"[ERROR] Error reading {config_file}: {e}")
            invalid.append(config_file)
    
    return len(invalid) == 0


def check_project_imports():
    """Check if project modules can be imported."""
    print("\n" + "="*60)
    print("CHECKING PROJECT IMPORTS")
    print("="*60)
    
    # Add project root to Python path
    project_root = Path.cwd()
    code_path = project_root / "03_code"
    
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))
    
    # Check config loader
    try:
        from utils.config.config_loader import ConfigLoader
        print("[OK] Config loader importable")
        
        # Test instantiation
        try:
            config_loader = ConfigLoader()
            print("[OK] Config loader instantiable")
        except Exception as e:
            print(f"[WARNING] Config loader instantiation issue: {e}")
            
    except ImportError as e:
        print(f"[ERROR] Cannot import config loader: {e}")
        return False
        
    # Check model imports
    try:
        from models.baseline_models.linear_regression import LinearRegressionModel
        print("[OK] Linear regression model importable")
        
        # Test instantiation
        try:
            model = LinearRegressionModel(model_type='linear')
            print("[OK] Linear regression model instantiable")
        except Exception as e:
            print(f"[WARNING] Model instantiation issue: {e}")
            
    except ImportError as e:
        print(f"[ERROR] Cannot import linear regression model: {e}")
        return False
    
    return True


def main():
    """Main validation function."""
    print("ELECTRICITY PRICE FORECASTING - SETUP VALIDATION")
    print(f"Project root: {Path.cwd()}")
    print(f"Python version: {sys.version}")
    
    # Run all checks
    checks = [
        ("Directories", check_directories),
        ("Critical Files", check_critical_files),
        ("Python Libraries", check_python_libraries),
        ("Configuration Files", check_configuration_files),
        ("Project Imports", check_project_imports)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"[ERROR] {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\nSUCCESS: Setup validation passed!")
        print("Your environment is ready for electricity price forecasting!")
        return 0
    else:
        print(f"\nFAILED: {total - passed} check(s) failed.")
        print("Please fix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())