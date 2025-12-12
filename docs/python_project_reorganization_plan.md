# Python Project Reorganization Plan

## 🎯 Objective
Reorganize the thesis project to follow standard Python project structure while maintaining all academic functionality and thesis-specific components.

## 📋 Target Structure
```
electricity-price-forecasting/
│
├── src/                          # All source code
│   └── electricity_price_forecast/
│       ├── __init__.py
│       ├── data_pipeline/
│       ├── models/
│       ├── utils/
│       └── evaluation/
│
├── tests/                        # All test files
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   └── test_utils.py
│
├── docs/                         # Documentation (from 01_documentation)
│   ├── project_organization.md
│   ├── data_collection_*.md
│   ├── eia_api_*.md
│   ├── project_roadmap.md
│   ├── literature_review/
│   ├── presentations/
│   ├── proposal/
│   └── thesis_draft/
│
├── data/                         # Data storage (from 02_data)
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── experiments/                  # Experimental results (from 04_experiments)
│   ├── model_comparison/
│   ├── feature_analysis/
│   └── volatility_studies/
│
├── outputs/                      # Final outputs (from 05_outputs)
│   ├── models/
│   ├── predictions/
│   ├── reports/
│   └── visualizations/
│
├── config/                       # Configuration (from 06_deployment/config)
│   ├── data_config.yaml
│   └── model_config.yaml
│
├── notebooks/                    # Jupyter notebooks (from 06_deployment/notebooks)
│   ├── data_exploration/
│   ├── model_development/
│   └── results_analysis/
│
├── scripts/                      # Utility scripts (existing)
│   ├── setup.sh
│   ├── setup.bat
│   ├── START_SETUP.bat
│   ├── quick_setup.py
│   ├── check_setup.py
│   ├── validate_setup.py
│   └── test_imports.py
│
├── admin/                        # Project management (from 07_admin)
│   ├── project_management/
│   └── collaboration/
│
├── venv/                         # Virtual environment
│
├── requirements.txt              # Dependencies (from 06_deployment/requirements)
├── setup.py                      # Package setup
├── README.md                     # Main documentation
├── .gitignore                    # Git ignore rules
└── Thesis Proposal.docx          # Thesis document
```

## 🔄 File Migration Plan

### Phase 1: Create New Structure
1. Create `src/` directory
2. Create `tests/` directory
3. Create `docs/` directory
4. Create `venv/` directory
5. Create `config/` directory
6. Create `notebooks/` directory

### Phase 2: Move Source Code
**From `03_code/` to `src/electricity_price_forecast/`**
- Move all Python files and subdirectories
- Update `__init__.py` files
- Update import paths

### Phase 3: Move Documentation
**From `01_documentation/` to `docs/`**
- Move all `.md` files
- Move subdirectories (literature_review, presentations, proposal, thesis_draft)
- Update internal references

### Phase 4: Reorganize Data
**From `02_data/` to `data/`**
- Move entire directory structure
- No changes needed to internal structure

### Phase 5: Move Experimental Results
**From `04_experiments/` to `experiments/`**
- Move entire directory structure
- No changes needed to internal structure

### Phase 6: Move Outputs
**From `05_outputs/` to `outputs/`**
- Move entire directory structure
- No changes needed to internal structure

### Phase 7: Extract Configuration
**From `06_deployment/config/` to `config/`**
- Move YAML configuration files
- Update paths in configuration

### Phase 8: Move Notebooks
**From `06_deployment/notebooks/` to `notebooks/`**
- Move all Jupyter notebooks
- Update import paths in notebooks

### Phase 9: Move Admin
**From `07_admin/` to `admin/`**
- Move entire directory structure
- No changes needed to internal structure

### Phase 10: Extract Requirements
**From `06_deployment/requirements/` to root**
- Move `requirements.txt` to root
- Move Docker files to root if needed

## 📝 Files to Create

### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="electricity-price-forecasting",
    version="0.1.0",
    description="Daily and Hourly Electricity Price Forecasting Using Machine Learning",
    author="Thesis Author",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # Dependencies from requirements.txt
    ],
)
```

### `tests/__init__.py`
```python
"""
Test suite for electricity price forecasting project.
"""
```

### `tests/test_main.py`
```python
"""
Main test file for electricity price forecasting project.
"""

def test_basic_functionality():
    """Test basic project functionality."""
    pass
```

### Updated `requirements.txt`
- Consolidate from `06_deployment/requirements/requirements.txt`
- Add any missing dependencies

### Updated `.gitignore`
- Add `venv/`
- Add `__pycache__/`
- Add `*.pyc`
- Add `.pytest_cache/`
- Keep existing rules

### Updated `README.md`
- Reflect new project structure
- Update installation instructions
- Update usage examples

## 🔧 Import Path Updates

### Before (Current)
```python
from 03_code.models.baseline_models.linear_regression import LinearRegressionModel
from 03_code.utils.config.config_loader import get_model_config
```

### After (New)
```python
from src.electricity_price_forecast.models.baseline_models.linear_regression import LinearRegressionModel
from src.electricity_price_forecast.utils.config.config_loader import get_model_config
```

### Alternative (with setup.py install)
```python
from electricity_price_forecast.models.baseline_models.linear_regression import LinearRegressionModel
from electricity_price_forecast.utils.config.config_loader import get_model_config
```

## 📋 Configuration Updates

### `config/data_config.yaml`
- Update paths from `02_data/` to `data/`
- Update paths from `03_code/` to `src/`

### `config/model_config.yaml`
- No changes needed (model configurations)

## 🧪 Testing Strategy

### Test Structure
- `tests/test_data_pipeline.py` - Test data collection and processing
- `tests/test_models.py` - Test all ML models
- `tests/test_utils.py` - Test utility functions
- `tests/test_main.py` - Integration tests

### Test Execution
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 🚀 Installation and Usage

### Development Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# Run data collection
python -m electricity_price_forecast.data_pipeline.collect_data

# Run model training
python -m electricity_price_forecast.models.train_models

# Run predictions
python -m electricity_price_forecast.models.predict
```

## ✅ Verification Checklist

### Post-Migration Verification
- [ ] All source code moved to `src/`
- [ ] All tests moved to `tests/`
- [ ] All documentation moved to `docs/`
- [ ] Import paths updated in all Python files
- [ ] Configuration files updated with new paths
- [ ] `setup.py` created and functional
- [ ] `requirements.txt` updated
- [ ] `.gitignore` updated
- [ ] `README.md` updated
- [ ] Virtual environment created
- [ ] All tests pass
- [ ] Project can be installed with `pip install -e .`
- [ ] Jupyter notebooks can access modules
- [ ] Data pipeline functions correctly
- [ ] Model training works
- [ ] No broken imports or references

## 🔄 Rollback Plan

If reorganization causes issues:
1. Keep backup of current structure
2. Document all changes made
3. Test thoroughly before committing
4. Use git to track changes for easy rollback

## 📈 Benefits of New Structure

### Python Standards Compliance
- Follows standard Python project layout
- Easier to install and distribute
- Better IDE support and code completion
- Standard testing framework integration

### Improved Maintainability
- Clear separation of concerns
- Standard import patterns
- Easier dependency management
- Better collaboration support

### Academic Compatibility
- Maintains all thesis-specific directories
- Preserves documentation structure
- Keeps experimental results organized
- Maintains academic workflow

---

**Status**: 📋 **PLANNING COMPLETED**  
**Next Step**: 🔄 **SWITCH TO CODE MODE FOR IMPLEMENTATION**