# Local Setup Instructions

## 1. GitHub Repository
```bash
git init
git remote add origin https://github.com/[username]/EnergyMarket-PriceCast.git
git add .
git commit -m "feat: initialise project structure, requirements, MLflow config"
git branch -M main
git push -u origin main
```

## 2. Python Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. MLflow Server
```bash
bash configs/start_mlflow.sh
# Open http://localhost:5000
```

## 4. Verify
```bash
python -c "import mlflow; print('MLflow:', mlflow.__version__)"
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
pytest tests/ -v
```
