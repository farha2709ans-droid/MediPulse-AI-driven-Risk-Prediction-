"""
Data preprocessing module for healthcare analytics.
Handles cleaning, normalization, and feature engineering for EHR and wearable data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing healthcare data from multiple sources."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.feature_columns = self.config.get('feature_columns', [])
        self.target_columns = self.config.get('target_columns', [])
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various file formats.
        
        Supported formats: CSV, Excel, Parquet, JSON
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(file_path, low_memory=False)
            elif file_ext in ('.xlsx', '.xls'):
                return pd.read_excel(file_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(file_path)
            elif file_ext == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform data cleaning operations."""
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Convert data types
        df_clean = self._convert_dtypes(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Standardize column names
        df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column types."""
        df_clean = df.copy()
        
        # Numerical columns: fill with median
        num_cols = df_clean.select_dtypes(include=['number']).columns
        if not num_cols.empty:
            df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
        
        # Categorical columns: fill with mode
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if not df_clean[col].empty:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
        
        return df_clean
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types appropriately."""
        df_clean = df.copy()
        
        # Convert date columns
        date_columns = [col for col in df_clean.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert column {col} to datetime: {str(e)}")
        
        # Convert categorical columns
        cat_columns = df_clean.select_dtypes(include=['object']).columns
        for col in cat_columns:
            if df_clean[col].nunique() < 100:  # Only convert if reasonable number of categories
                df_clean[col] = df_clean[col].astype('category')
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers using z-score method."""
        df_clean = df.copy()
        
        # Only process numerical columns
        num_cols = df_clean.select_dtypes(include=['number']).columns
        
        for col in num_cols:
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean[col] = np.where(z_scores > threshold, np.nan, df_clean[col])
        
        return df_clean
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize numerical features.
        
        Args:
            df: Input DataFrame
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Normalized DataFrame
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        df_norm = df.copy()
        num_cols = df_norm.select_dtypes(include=['number']).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        df_norm[num_cols] = scaler.fit_transform(df_norm[num_cols])
        return df_norm
    
    def process_time_series(self, df: pd.DataFrame, timestamp_col: str, value_col: str, 
                          freq: str = '1H', agg_method: str = 'mean') -> pd.DataFrame:
        """Process time series data by resampling and aggregating.
        
        Args:
            df: Input DataFrame with time series data
            timestamp_col: Name of the timestamp column
            value_col: Name of the value column to aggregate
            freq: Resampling frequency (e.g., '1H' for hourly, '1D' for daily)
            agg_method: Aggregation method ('mean', 'sum', 'max', 'min', 'first', 'last')
            
        Returns:
            Resampled and aggregated DataFrame
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")
        
        # Ensure timestamp column is datetime
        df_ts = df.copy()
        df_ts[timestamp_col] = pd.to_datetime(df_ts[timestamp_col])
        
        # Set timestamp as index
        df_ts = df_ts.set_index(timestamp_col)
        
        # Resample and aggregate
        if agg_method == 'mean':
            df_resampled = df_ts.resample(freq).mean()
        elif agg_method == 'sum':
            df_resampled = df_ts.resample(freq).sum()
        elif agg_method == 'max':
            df_resampled = df_ts.resample(freq).max()
        elif agg_method == 'min':
            df_resampled = df_ts.resample(freq).min()
        elif agg_method == 'first':
            df_resampled = df_ts.resample(freq).first()
        elif agg_method == 'last':
            df_resampled = df_ts.resample(freq).last()
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")
        
        return df_resampled.reset_index()
    
    def extract_features(self, df: pd.DataFrame, timestamp_col: str, value_cols: List[str], 
                        window_size: int = 24, step_size: int = 1) -> pd.DataFrame:
        """Extract time-series features using sliding window approach.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of the timestamp column
            value_cols: List of column names to extract features from
            window_size: Size of the sliding window in hours
            step_size: Step size for sliding window in hours
            
        Returns:
            DataFrame with extracted features
        """
        import warnings
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import make_forecast_frame
        
        df_ts = df.copy()
        
        # Ensure timestamp is datetime and sort
        df_ts[timestamp_col] = pd.to_datetime(df_ts[timestamp_col])
        df_ts = df_ts.sort_values(timestamp_col)
        
        # Create features for each value column
        all_features = []
        
        for col in value_cols:
            if col not in df_ts.columns:
                logger.warning(f"Column {col} not found in DataFrame, skipping")
                continue
                
            # Prepare data for tsfresh
            df_col = df_ts[[timestamp_col, col]].copy()
            df_col = df_col.rename(columns={col: 'value'})
            
            # Extract features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features = extract_features(
                    df_col, 
                    column_id="id",  # Not used but required
                    column_sort=timestamp_col,
                    column_value='value',
                    default_fc_parameters={
                        "mean": None,
                        "standard_deviation": None,
                        "minimum": None,
                        "maximum": None,
                        "variance": None,
                        "median": None,
                    }
                )
            
            # Rename features to include column name
            features.columns = [f"{col}_{c}" for c in features.columns]
            all_features.append(features)
        
        # Combine all features
        if all_features:
            return pd.concat(all_features, axis=1)
        return pd.DataFrame()
    
    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path], 
                          format: str = 'parquet', **kwargs):
        """Save processed data to disk.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            format: Output format ('csv', 'parquet', 'feather', 'pickle')
            **kwargs: Additional arguments to pass to the writer
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False, **kwargs)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False, **kwargs)
        elif format == 'feather':
            df.to_feather(output_path, **kwargs)
        elif format == 'pickle':
            df.to_pickle(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        logger.info(f"Processed data saved to {output_path}")


def main():
    """Example usage of the DataPreprocessor class."""
    # Example configuration
    config = {
        'feature_columns': ['age', 'gender', 'heart_rate', 'blood_glucose'],
        'target_columns': ['diabetes_risk', 'arrhythmia_risk']
    }
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Example: Load and process data
    try:
        # This is just an example - replace with actual data loading
        sample_data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'heart_rate': np.random.normal(75, 10, 100),
            'blood_glucose': np.random.normal(100, 20, 100)
        }
        df = pd.DataFrame(sample_data)
        
        # Process time series data
        df_processed = preprocessor.clean_data(df)
        df_resampled = preprocessor.process_time_series(
            df_processed, 
            timestamp_col='timestamp', 
            value_col='heart_rate',
            freq='1H',
            agg_method='mean'
        )
        
        # Extract features
        features = preprocessor.extract_features(
            df_processed,
            timestamp_col='timestamp',
            value_cols=['heart_rate', 'blood_glucose'],
            window_size=24,
            step_size=1
        )
        
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Resampled data shape: {df_resampled.shape}")
        print(f"Extracted features shape: {features.shape}")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
