# Healthcare Analytics: Early Disease Detection

This project focuses on early detection of diseases like diabetes and cardiac arrhythmias using wearable device data and clinical records.

## Project Structure

```
healthcare_analytics/
├── data/                    # Data storage
│   ├── raw/                 # Raw data (EHR, wearable data)
│   └── processed/           # Processed and cleaned data
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature engineering
│   ├── models/              # ML model definitions
│   └── visualization/       # Visualization utilities
├── config/                  # Configuration files
└── docs/                    # Documentation
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Sources

- Electronic Health Records (EHR)
- Wearable device data (heart rate, activity, sleep patterns)
- Lab test results
- Patient demographics

## Models

- XGBoost for tabular data classification
- LSTM for time-series analysis
- SHAP/LIME for model explainability

## Usage

1. Place raw data in `data/raw/`
2. Run preprocessing scripts
3. Train models using provided notebooks
4. Generate visualizations and reports

## License

MIT
