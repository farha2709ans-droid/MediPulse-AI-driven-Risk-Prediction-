import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

# Database configuration
DB_CONFIG = {
    'drivername': 'postgresql',
    'username': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'healthcare_analytics')
}

# Model parameters
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'random_state': 42
    },
    'lstm': {
        'units': 64,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'epochs': 50,
        'batch_size': 32
    }
}

# Feature engineering parameters
FEATURE_PARAMS = {
    'window_size': 24,  # hours
    'step_size': 1,     # hours
    'vital_signs': [
        'heart_rate',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'blood_glucose',
        'oxygen_saturation',
        'respiratory_rate',
        'skin_temperature'
    ]
}

# Risk thresholds
RISK_THRESHOLDS = {
    'diabetes': 0.7,
    'arrhythmia': 0.65,
    'hypertension': 0.6
}

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
