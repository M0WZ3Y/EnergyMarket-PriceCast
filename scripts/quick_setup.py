#!/usr/bin/env python3
"""
Quick Setup Script for Electricity Price Forecasting Thesis Project
This script can be run from anywhere and will set up the environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def find_project_root():
    """Find the project root by looking for key files/directories."""
    current = Path.cwd()
    
    # Look for project indicators
    indicators = [
        "01_documentation",
        "02_data", 
        "03_code",
        "06_deployment",
        "README.md"
    ]
    
    # Search up to 5 levels up
    for _ in range(5):
        if all((current / indicator).exists() for indicator in indicators):
            return current
        current = current.parent
    
    # If not found, try the script's directory
    script_dir = Path(__file__).parent
    if all((script_dir / indicator).exists() for indicator in indicators):
        return script_dir
    
    return None

def main():
    print("[SETUP] Electricity Price Forecasting - Quick Setup")
    print("=" * 50)
    
    # Find project root
    project_root = find_project_root()
    if not project_root:
        print("[ERROR] Could not find project root directory")
        print("Please run this script from within the project directory")
        return 1
    
    print(f"[INFO] Project root: {project_root}")
    print()
    
    # Change to project root
    os.chdir(project_root)
    
    # Check if setup script exists
    setup_script = project_root / "06_deployment" / "scripts" / "setup_environment.py"
    if not setup_script.exists():
        print(f"[ERROR] Setup script not found: {setup_script}")
        return 1
    
    print("[INFO] Running setup script...")
    print()
    
    # Run the setup script
    try:
        result = subprocess.run([sys.executable, str(setup_script)],
                              cwd=project_root, check=True)
        print()
        print("[SUCCESS] Setup completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Setup failed with exit code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"[ERROR] Setup error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())