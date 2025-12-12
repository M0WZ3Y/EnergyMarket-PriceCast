#!/usr/bin/env python3
"""
Test script to verify import resolution is working correctly.
This script tests all the import patterns used in the project.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all import patterns to ensure they work correctly."""
    
    print("Testing Import Resolution")
    print("=" * 50)
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    tests_passed = 0
    tests_failed = 0
    
    def run_test(test_name, test_func):
        """Run a single import test."""
        nonlocal tests_passed, tests_failed
        
        try:
            test_func()
            print(f"[PASS] {test_name}")
            tests_passed += 1
        except Exception as e:
            print(f"[FAIL] {test_name}")
            print(f"       Error: {e}")
            tests_failed += 1
    
    # Test 1: Direct import using importlib
    run_test(
        "Direct import from 03_code.models.baseline_models.linear_regression",
        lambda: __import__('03_code.models.baseline_models.linear_regression', fromlist=['LinearRegressionModel'])
    )
    
    # Test 2: Config loader import
    run_test(
        "Direct import from 03_code.utils.config.config_loader",
        lambda: __import__('03_code.utils.config.config_loader', fromlist=['get_model_config'])
    )
    
    # Test 3: Package imports
    run_test(
        "Package import - 03_code.models",
        lambda: __import__('03_code.models')
    )
    
    run_test(
        "Package import - 03_code.utils",
        lambda: __import__('03_code.utils')
    )
    
    # Test 4: From package imports
    run_test(
        "From package import - models.baseline_models",
        lambda: __import__('03_code.models', fromlist=['baseline_models'])
    )
    
    run_test(
        "From package import - utils.config",
        lambda: __import__('03_code.utils', fromlist=['config'])
    )
    
    # Test 5: Test the actual functionality with regular imports
    def test_model_functionality():
        """Test model instantiation and functionality."""
        # Import the modules
        linear_module = __import__('03_code.models.baseline_models.linear_regression', fromlist=['LinearRegressionModel'])
        config_module = __import__('03_code.utils.config.config_loader', fromlist=['get_model_config'])
        
        # Create model instance
        model = linear_module.LinearRegressionModel()
        
        # Load config
        config = config_module.get_model_config()
        
        # Verify linear regression config exists
        if 'models' not in config or 'linear_regression' not in config['models']:
            raise ValueError("Linear regression config not found")
    
    run_test(
        "Model instantiation and config loading",
        test_model_functionality
    )
    
    # Test 6: Test relative import fallback mechanism
    def test_fallback_import():
        """Test the fallback import mechanism in linear_regression.py."""
        # Read the file to check if fallback mechanism exists
        linear_file = project_root / "03_code" / "models" / "baseline_models" / "linear_regression.py"
        if not linear_file.exists():
            raise FileNotFoundError("linear_regression.py not found")
        
        with open(linear_file, 'r') as f:
            content = f.read()
            
        # Check for fallback import patterns
        if "try:" not in content or "except ImportError:" not in content:
            raise ValueError("Fallback import mechanism not found")
        
        # Check for the main import
        if "from ...utils.config.config_loader import get_model_config" not in content:
            raise ValueError("Main relative import not found")
    
    run_test(
        "Fallback import mechanism verification",
        test_fallback_import
    )
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results:")
    print(f"   Passed: {tests_passed}")
    print(f"   Failed: {tests_failed}")
    print(f"   Total:  {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n[SUCCESS] All imports are working correctly!")
        return True
    else:
        print(f"\n[WARNING] {tests_failed} import(s) failed. Please check the errors above.")
        return False

def test_package_structure():
    """Test that the package structure is correct."""
    print("\nTesting Package Structure")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    
    # Check for __init__.py files
    init_files = [
        "03_code/__init__.py",
        "03_code/utils/__init__.py",
        "03_code/utils/config/__init__.py",
        "03_code/models/__init__.py",
        "03_code/models/baseline_models/__init__.py"
    ]
    
    all_exist = True
    for init_file in init_files:
        full_path = project_root / init_file
        if full_path.exists():
            print(f"[PASS] {init_file}")
        else:
            print(f"[FAIL] {init_file} - Missing!")
            all_exist = False
    
    # Check for main Python files
    py_files = [
        "03_code/models/baseline_models/linear_regression.py",
        "03_code/utils/config/config_loader.py"
    ]
    
    for py_file in py_files:
        full_path = project_root / py_file
        if full_path.exists():
            print(f"[PASS] {py_file}")
        else:
            print(f"[FAIL] {py_file} - Missing!")
            all_exist = False
    
    return all_exist

def test_vscode_config():
    """Test VS Code configuration."""
    print("\nTesting VS Code Configuration")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    vscode_settings = project_root / ".vscode" / "settings.json"
    
    if vscode_settings.exists():
        print("[PASS] .vscode/settings.json exists")
        
        # Check content
        import json
        try:
            with open(vscode_settings, 'r') as f:
                settings = json.load(f)
            
            required_keys = [
                "python.analysis.extraPaths",
                "python.analysis.autoImportCompletions"
            ]
            
            for key in required_keys:
                if key in settings:
                    print(f"[PASS] {key} configured")
                else:
                    print(f"[WARN] {key} not configured")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Error reading settings.json: {e}")
            return False
    else:
        print("[FAIL] .vscode/settings.json missing")
        return False

if __name__ == "__main__":
    print("Starting Import Resolution Tests")
    print("Project Root:", Path(__file__).parent.parent)
    print("Python Path:", sys.path[:3])  # Show first 3 paths
    print()
    
    # Run tests
    imports_ok = test_imports()
    structure_ok = test_package_structure()
    vscode_ok = test_vscode_config()
    
    # Final status
    print("\n" + "=" * 50)
    print("FINAL STATUS:")
    print(f"  Import Resolution: {'PASS' if imports_ok else 'FAIL'}")
    print(f"  Package Structure: {'PASS' if structure_ok else 'FAIL'}")
    print(f"  VS Code Config:    {'PASS' if vscode_ok else 'FAIL'}")
    
    if imports_ok and structure_ok and vscode_ok:
        print("\n[SUCCESS] All tests passed! Import resolution is fully working.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed. Please check the output above.")
        sys.exit(1)