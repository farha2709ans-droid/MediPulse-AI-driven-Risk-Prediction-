""
Machine learning models for healthcare analytics.
Includes XGBoost for tabular data and LSTM for time-series predictions.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

# Machine learning imports
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Dense, LSTM, Dropout, Input, 
        BatchNormalization, Concatenate, MultiHeadAttention,
        LayerNormalization, MultiHeadAttention as MHA
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, 
        ReduceLROnPlateau, TensorBoard
    )
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow not available. LSTM models will not be available.")

# Explainability
import shap
import lime
import lime.lime_tabular

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Custom feature preprocessor for healthcare data."""
    
    def __init__(self, numeric_features: List[str], 
                 categorical_features: List[str] = None,
                 datetime_features: List[str] = None):
        """Initialize the preprocessor.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            datetime_features: List of datetime feature names
        """
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.datetime_features = datetime_features or []
        self.feature_names_out_ = None
        
        # Initialize transformers
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.datetime_transformer = Pipeline(steps=[
            ('extractor', DateTimeExtractor())
        ])
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the preprocessor."""
        # Create column transformer
        transformers = []
        
        if self.numeric_features:
            transformers.append(('num', self.numeric_transformer, self.numeric_features))
            
        if self.categorical_features:
            transformers.append(('cat', self.categorical_transformer, self.categorical_features))
            
        if self.datetime_features:
            transformers.append(('dt', self.datetime_transformer, self.datetime_features))
        
        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop other columns
        )
        
        # Fit the preprocessor
        self.preprocessor_.fit(X)
        
        # Get feature names after transformation
        self._set_feature_names(X)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the input data."""
        return self.preprocessor_.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        if not hasattr(self, 'feature_names_out_'):
            raise RuntimeError("Transformer has not been fitted yet.")
        return self.feature_names_out_
    
    def _set_feature_names(self, X: pd.DataFrame):
        """Set feature names after transformation."""
        self.feature_names_out_ = []
        
        # Process numeric features
        if self.numeric_features:
            self.feature_names_out_.extend(self.numeric_features)
        
        # Process categorical features
        if self.categorical_features and hasattr(self.categorical_transformer.named_steps['onehot'], 'get_feature_names_out'):
            cat_features = self.categorical_transformer.named_steps['onehot'].get_feature_names_out(self.categorical_features)
            self.feature_names_out_.extend(cat_features)
        
        # Process datetime features
        if self.datetime_features:
            dt_features = [f"{feat}_{comp}" for feat in self.datetime_features 
                          for comp in ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'is_weekend']]
            self.feature_names_out_.extend(dt_features)


class DateTimeExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract datetime features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        result = []
        
        for col in X.columns:
            # Convert to datetime if not already
            dt_series = pd.to_datetime(X[col], errors='coerce')
            
            # Extract datetime components
            dt_df = pd.DataFrame({
                f"{col}_year": dt_series.dt.year,
                f"{col}_month": dt_series.dt.month,
                f"{col}_day": dt_series.dt.day,
                f"{col}_hour": dt_series.dt.hour,
                f"{col}_minute": dt_series.dt.minute,
                f"{col}_second": dt_series.dt.second,
                f"{col}_dayofweek": dt_series.dt.dayofweek,
                f"{col}_is_weekend": dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            })
            
            # Fill any remaining NaNs
            dt_df = dt_df.fillna(-1)
            result.append(dt_df)
            
        return pd.concat(result, axis=1).values


class DiseasePredictor:
    """Base class for disease prediction models."""
    
    def __init__(self, model_type: str = 'xgboost', target_disease: str = 'diabetes',
                 model_dir: str = 'models', **kwargs):
        """Initialize the disease predictor.
        
        Args:
            model_type: Type of model ('xgboost' or 'lstm')
            target_disease: Target disease for prediction
            model_dir: Directory to save/load models
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type.lower()
        self.target_disease = target_disease.lower()
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.classes_ = None
        self.model_version = "1.0.0"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model based on type
        if self.model_type == 'xgboost':
            self._init_xgboost(**kwargs)
        elif self.model_type == 'lstm' and TF_AVAILABLE:
            self._init_lstm(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _init_xgboost(self, **kwargs):
        """Initialize XGBoost model."""
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with any user-provided parameters
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)
        self.model_file = self.model_dir / f"{self.target_disease}_xgb_model.json"
    
    def _init_lstm(self, **kwargs):
        """Initialize LSTM model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models.")
            
        # Default LSTM parameters
        self.sequence_length = kwargs.get('sequence_length', 24)
        self.n_features = kwargs.get('n_features', 10)
        self.n_classes = kwargs.get('n_classes', 2)
        
        # Build the model
        self._build_lstm_model(**kwargs)
        self.model_file = self.model_dir / f"{self.target_disease}_lstm_model.h5"
    
    def _build_lstm_model(self, **kwargs):
        """Build LSTM model architecture."""
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        x = LSTM(units=64, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units=32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        # Dense layers
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        output_layer = Dense(
            self.n_classes if self.n_classes > 2 else 1,
            activation='softmax' if self.n_classes > 2 else 'sigmoid'
        )(x)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        optimizer = Adam(learning_rate=kwargs.get('learning_rate', 0.001))
        loss = 'categorical_crossentropy' if self.n_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy']
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            validation_data: Optional[Tuple] = None, **fit_params):
        """Train the model on the given data."""
        # Preprocess data if preprocessor is available
        if hasattr(self, 'preprocessor_') and self.preprocessor_ is not None:
            X_transformed = self.preprocessor_.transform(X)
            self.feature_names = self.preprocessor_.get_feature_names()
            
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_transformed = self.preprocessor_.transform(X_val)
                validation_data = (X_val_transformed, y_val)
        else:
            X_transformed = X
        
        # Convert y to numpy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values
        
        # Fit the model
        if self.model_type == 'xgboost':
            return self._fit_xgboost(X_transformed, y, validation_data, **fit_params)
        elif self.model_type == 'lstm':
            return self._fit_lstm(X_transformed, y, validation_data, **fit_params)
    
    def _fit_xgboost(self, X, y, validation_data, **fit_params):
        """Train XGBoost model."""
        eval_set = None
        if validation_data is not None:
            eval_set = [(validation_data[0], validation_data[1])]
        
        return self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=fit_params.get('verbose', True),
            early_stopping_rounds=fit_params.get('early_stopping_rounds', 10),
            **fit_params
        )
    
    def _fit_lstm(self, X, y, validation_data, **fit_params):
        """Train LSTM model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models.")
        
        # Reshape data for LSTM if needed
        if len(X.shape) == 2:  # If not already in sequence format
            X = self._create_sequences(X, self.sequence_length)
        
        # Convert y to one-hot encoding if needed
        if self.n_classes > 2 and len(y.shape) == 1:
            y = to_categorical(y, num_classes=self.n_classes)
        
        # Prepare validation data
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if len(X_val.shape) == 2:
                X_val = self._create_sequences(X_val, self.sequence_length)
            if self.n_classes > 2 and len(y_val.shape) == 1:
                y_val = to_categorical(y_val, num_classes=self.n_classes)
            val_data = (X_val, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=fit_params.get('patience', 10),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=str(self.model_file),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_data=val_data,
            epochs=fit_params.get('epochs', 50),
            batch_size=fit_params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=fit_params.get('verbose', 1)
        )
        
        return history
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        # Preprocess data if preprocessor is available
        if hasattr(self, 'preprocessor_') and self.preprocessor_ is not None:
            X = self.preprocessor_.transform(X)
        
        # Reshape for LSTM if needed
        if self.model_type == 'lstm' and len(X.shape) == 2:
            X = self._create_sequences(X, self.sequence_length)
        
        # Make predictions
        if self.model_type == 'xgboost':
            return self.model.predict_proba(X)
        elif self.model_type == 'lstm':
            return self.model.predict(X)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        
        if proba.shape[1] > 1:  # Multi-class
            return np.argmax(proba, axis=1)
        else:  # Binary
            return (proba > 0.5).astype(int)
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance."""
        # Preprocess data if preprocessor is available
        if hasattr(self, 'preprocessor_') and self.preprocessor_ is not None:
            X = self.preprocessor_.transform(X)
        
        # Convert y to numpy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values
        
        # Reshape for LSTM if needed
        if self.model_type == 'lstm' and len(X.shape) == 2:
            X = self._create_sequences(X, self.sequence_length)
        
        # Make predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if binary classification
        if y_pred_proba.shape[1] == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
        
        return metrics
    
    def explain_prediction(self, X: Union[pd.DataFrame, np.ndarray], 
                         feature_names: List[str] = None, 
                         class_names: List[str] = None) -> Dict:
        """Explain model predictions using SHAP and LIME."""
        if feature_names is None and hasattr(self, 'feature_names'):
            feature_names = self.feature_names
        
        if class_names is None and hasattr(self, 'classes_'):
            class_names = self.classes_
        
        explanations = {}
        
        # SHAP explanation
        try:
            if self.model_type == 'xgboost':
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)
                
                # For binary classification, SHAP returns a list with one element
                if isinstance(shap_values, list) and len(shap_values) == 1:
                    shap_values = shap_values[0]
                
                explanations['shap'] = {
                    'values': shap_values.tolist(),
                    'base_value': float(explainer.expected_value),
                    'feature_names': feature_names,
                    'class_names': class_names
                }
                
            elif self.model_type == 'lstm' and TF_AVAILABLE:
                # For LSTM, we'll use KernelExplainer as a general approach
                def model_predict(x):
                    return self.model.predict(x)
                
                explainer = shap.KernelExplainer(model_predict, X[:100])  # Use first 100 samples as background
                shap_values = explainer.shap_values(X[:10], nsamples=100)  # Explain first 10 samples
                
                explanations['shap'] = {
                    'values': [v.tolist() for v in shap_values],
                    'base_value': explainer.expected_value.tolist(),
                    'feature_names': feature_names,
                    'class_names': class_names
                }
                
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {str(e)}")
        
        # LIME explanation
        try:
            if len(X.shape) == 2:  # Tabular data
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X,
                    feature_names=feature_names,
                    class_names=class_names,
                    mode='classification',
                    verbose=True
                )
                
                # Explain first sample
                exp = explainer.explain_instance(
                    X[0], 
                    self.predict_proba, 
                    num_features=min(10, X.shape[1]),
                    top_labels=1
                )
                
                explanations['lime'] = {
                    'as_list': exp.as_list(),
                    'as_map': exp.as_map(),
                    'predicted_class': int(exp.predict_proba.argmax()),
                    'prediction_probability': float(exp.predict_proba.max())
                }
                
        except Exception as e:
            logger.warning(f"LIME explanation failed: {str(e)}")
        
        return explanations
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the model to disk."""
        if filepath is None:
            filepath = self.model_file
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'xgboost':
            self.model.save_model(str(filepath))
        elif self.model_type == 'lstm' and TF_AVAILABLE:
            self.model.save(str(filepath))
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, model_type: str, **kwargs):
        """Load a saved model from disk."""
        filepath = Path(filepath)
        
        # Create model instance
        model = cls(model_type=model_type, **kwargs)
        
        # Load the model weights/parameters
        if model_type == 'xgboost':
            model.model = xgb.XGBClassifier()
            model.model.load_model(str(filepath))
        elif model_type == 'lstm' and TF_AVAILABLE:
            model.model = load_model(str(filepath))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """Convert time series data into sequences for LSTM."""
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:(i + seq_length)])
        return np.array(sequences)


class DiabetesPredictor(DiseasePredictor):
    """Specialized predictor for diabetes risk assessment."""
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """Initialize diabetes predictor."""
        super().__init__(
            model_type=model_type,
            target_disease='diabetes',
            **kwargs
        )
        
        # Define feature columns
        self.numeric_features = [
            'age', 'bmi', 'glucose', 'blood_pressure',
            'skin_thickness', 'insulin', 'diabetes_pedigree'
        ]
        
        self.categorical_features = ['gender']
        
        # Initialize preprocessor
        self.preprocessor_ = FeaturePreprocessor(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features
        )


class ArrhythmiaPredictor(DiseasePredictor):
    """Specialized predictor for cardiac arrhythmia detection."""
    
    def __init__(self, model_type: str = 'lstm', **kwargs):
        """Initialize arrhythmia predictor."""
        if model_type != 'lstm':
            logger.warning("Arrhythmia detection typically works best with LSTM models. Consider using model_type='lstm'.")
        
        super().__init__(
            model_type=model_type,
            target_disease='arrhythmia',
            **kwargs
        )
        
        # Define feature columns for time-series data
        self.n_features = 12  # Typical number of ECG leads
        self.sequence_length = 256  # Typical window size for ECG analysis
        
        # Rebuild model with correct input shape
        if model_type == 'lstm':
            self._build_lstm_model()


def main():
    """Example usage of the DiseasePredictor class."""
    # Example with synthetic data
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Convert to DataFrame for better column handling
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    print("Training XGBoost model...")
    model = DiseasePredictor(model_type='xgboost')
    
    # Set up preprocessor
    model.preprocessor_ = FeaturePreprocessor(numeric_features=feature_names)
    
    # Fit the model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        early_stopping_rounds=10,
        verbose=True
    )
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nModel performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Explain predictions
    print("\nGenerating explanations...")
    explanations = model.explain_prediction(X_test[:5], feature_names=feature_names)
    
    # Save model
    model.save_model("diabetes_predictor.xgb")
    
    # Example of loading the model
    # loaded_model = DiseasePredictor.load_model("diabetes_predictor.xgb", model_type='xgboost')


if __name__ == "__main__":
    main()
