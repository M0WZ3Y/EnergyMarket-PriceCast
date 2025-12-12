# Scripts Directory

This directory contains utility scripts for setting up, testing, and maintaining the thesis project.

## Setup Scripts

### Installation & Setup
- `setup.sh` - Linux/macOS setup script
- `setup.bat` - Windows setup script  
- `START_SETUP.bat` - Quick setup launcher for Windows
- `quick_setup.py` - Python-based quick setup script

### Validation & Testing
- `check_setup.py` - Verify installation and configuration
- `validate_setup.py` - Comprehensive setup validation
- `test_imports.py` - Test import resolution functionality

## Usage

### Quick Start (Windows)
```bash
# Run the quick setup launcher
START_SETUP.bat
```

### Quick Start (Linux/macOS)
```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

### Manual Setup
```bash
# Install dependencies
pip install -r 06_deployment/requirements/requirements.txt

# Validate setup
python scripts/check_setup.py

# Test imports
python scripts/test_imports.py
```

## Script Descriptions

### setup.sh / setup.bat
Automated setup scripts that:
- Create virtual environment
- Install dependencies
- Configure environment variables
- Set up project structure

### quick_setup.py
Python-based setup script with cross-platform compatibility:
- Detects operating system
- Installs required packages
- Creates necessary directories
- Validates configuration

### check_setup.py
Basic setup verification that checks:
- Python version compatibility
- Required package installation
- Directory structure
- Configuration files

### validate_setup.py
Comprehensive validation script that:
- Tests all import paths
- Validates configuration files
- Checks model loading
- Verifies data access

### test_imports.py
Import resolution testing that:
- Tests all import patterns
- Verifies package structure
- Validates VS Code configuration
- Provides detailed error reporting

## Troubleshooting

If scripts fail to run:

1. **Permission Issues** (Linux/macOS):
   ```bash
   chmod +x scripts/*.sh
   ```

2. **Python Path Issues**:
   ```bash
   python -m scripts.check_setup
   ```

3. **Virtual Environment**:
   ```bash
   # Activate virtual environment first
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

## Notes

- All scripts are designed to be run from the project root directory
- Scripts will automatically detect the project structure
- Error messages provide guidance for fixing common issues
- Logs are saved to the `05_outputs/logs/` directory