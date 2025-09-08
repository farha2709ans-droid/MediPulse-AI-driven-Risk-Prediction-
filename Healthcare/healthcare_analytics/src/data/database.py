""
Database module for healthcare analytics project.
Handles database connections, schema management, and data operations.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy base class for declarative models
Base = declarative_base()

# Define database models
class Patient(Base):
    """Patient information model."""
    __tablename__ = 'patients'
    
    patient_id = Column(String(50), primary_key=True)
    gender = Column(String(10))
    date_of_birth = Column(DateTime)
    height_cm = Column(Float)
    weight_kg = Column(Float)
    blood_type = Column(String(5))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VitalSigns(Base):
    """Vital signs measurements model."""
    __tablename__ = 'vital_signs'
    
    measurement_id = Column(String(50), primary_key=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'))
    timestamp = Column(DateTime, index=True)
    heart_rate = Column(Float)
    blood_pressure_systolic = Column(Float)
    blood_pressure_diastolic = Column(Float)
    oxygen_saturation = Column(Float)
    respiratory_rate = Column(Float)
    temperature_c = Column(Float)
    source = Column(String(50))  # 'wearable', 'clinic', 'hospital', etc.
    created_at = Column(DateTime, default=datetime.utcnow)


class LabResults(Base):
    """Laboratory test results model."""
    __tablename__ = 'lab_results'
    
    result_id = Column(String(50), primary_key=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'))
    test_date = Column(DateTime)
    test_type = Column(String(100))
    test_name = Column(String(100))
    value = Column(Float)
    unit = Column(String(20))
    reference_range = Column(String(50))
    abnormal_flag = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)


class RiskScores(Base):
    """Disease risk scores model."""
    __tablename__ = 'risk_scores'
    
    score_id = Column(String(50), primary_key=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'))
    timestamp = Column(DateTime, index=True)
    risk_type = Column(String(50))  # 'diabetes', 'arrhythmia', 'hypertension', etc.
    risk_score = Column(Float)
    risk_category = Column(String(20))  # 'low', 'medium', 'high'
    features_used = Column(String)  # JSON string of features and their importance
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Class for managing database connections and operations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database connection.
        
        Args:
            connection_string: SQLAlchemy connection string. If None, will use environment variables.
        """
        self.connection_string = connection_string or self._get_connection_string()
        self.engine = create_engine(self.connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        
    def _get_connection_string(self) -> str:
        """Get database connection string from environment variables."""
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', 'postgres')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'healthcare_analytics')
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection successful")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def create_tables(self):
        """Create all tables defined in the models."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("Dropped all database tables")
        except SQLAlchemyError as e:
            logger.error(f"Error dropping tables: {str(e)}")
            raise
    
    def reset_database(self):
        """Reset the database by dropping and recreating all tables."""
        self.drop_tables()
        self.create_tables()
    
    def insert_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """Insert data from a pandas DataFrame into a database table.
        
        Args:
            table_name: Name of the target table
            df: DataFrame containing data to insert
            if_exists: What to do if table exists: 'fail', 'replace', 'append'
            
        Returns:
            Number of rows inserted
        """
        try:
            with self.engine.begin() as connection:
                rows_inserted = df.to_sql(
                    name=table_name,
                    con=connection,
                    if_exists=if_exists,
                    index=False,
                    method='multi'
                )
            logger.info(f"Inserted {rows_inserted} rows into {table_name}")
            return rows_inserted
        except SQLAlchemyError as e:
            logger.error(f"Error inserting data into {table_name}: {str(e)}")
            raise
    
    def query_to_dataframe(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Parameters for the query (optional)
            
        Returns:
            DataFrame containing query results
        """
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(query, connection, params=params)
            return df
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_patient_data(self, patient_id: str) -> Dict[str, pd.DataFrame]:
        """Get all data for a specific patient.
        
        Args:
            patient_id: ID of the patient
            
        Returns:
            Dictionary with DataFrames for each data type
        """
        try:
            with self.Session() as session:
                # Get patient info
                patient_query = f"""
                    SELECT * FROM patients 
                    WHERE patient_id = :patient_id
                """
                patient_df = self.query_to_dataframe(patient_query, {'patient_id': patient_id})
                
                # Get vital signs
                vitals_query = """
                    SELECT * FROM vital_signs 
                    WHERE patient_id = :patient_id
                    ORDER BY timestamp DESC
                """
                vitals_df = self.query_to_dataframe(vitals_query, {'patient_id': patient_id})
                
                # Get lab results
                labs_query = """
                    SELECT * FROM lab_results 
                    WHERE patient_id = :patient_id
                    ORDER BY test_date DESC
                """
                labs_df = self.query_to_dataframe(labs_query, {'patient_id': patient_id})
                
                # Get risk scores
                risks_query = """
                    SELECT * FROM risk_scores 
                    WHERE patient_id = :patient_id
                    ORDER BY timestamp DESC
                """
                risks_df = self.query_to_dataframe(risks_query, {'patient_id': patient_id})
                
                return {
                    'patient': patient_df,
                    'vital_signs': vitals_df,
                    'lab_results': labs_df,
                    'risk_scores': risks_df
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving data for patient {patient_id}: {str(e)}")
            raise
    
    def get_latest_risk_scores(self, limit: int = 100) -> pd.DataFrame:
        """Get the latest risk scores for all patients.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with the latest risk scores
        """
        query = """
            WITH ranked_scores AS (
                SELECT 
                    patient_id,
                    risk_type,
                    risk_score,
                    risk_category,
                    timestamp,
                    ROW_NUMBER() OVER (PARTITION BY patient_id, risk_type ORDER BY timestamp DESC) as rn
                FROM risk_scores
            )
            SELECT 
                p.patient_id,
                p.gender,
                p.date_of_birth,
                rs.risk_type,
                rs.risk_score,
                rs.risk_category,
                rs.timestamp as last_updated
            FROM ranked_scores rs
            JOIN patients p ON rs.patient_id = p.patient_id
            WHERE rs.rn = 1
            ORDER BY rs.risk_score DESC
            LIMIT :limit
        """
        return self.query_to_dataframe(query, {'limit': limit})
    
    def add_risk_score(self, patient_id: str, risk_type: str, risk_score: float, 
                      features_used: Dict[str, float], model_version: str) -> bool:
        """Add a new risk score to the database.
        
        Args:
            patient_id: ID of the patient
            risk_type: Type of risk ('diabetes', 'arrhythmia', etc.)
            risk_score: Calculated risk score (0-1)
            features_used: Dictionary of features and their importance
            model_version: Version of the model used
            
        Returns:
            True if successful, False otherwise
        """
        # Determine risk category
        if risk_score < 0.3:
            risk_category = 'low'
        elif risk_score < 0.7:
            risk_category = 'medium'
        else:
            risk_category = 'high'
        
        try:
            with self.Session() as session:
                # Create new risk score record
                score = RiskScores(
                    score_id=f"rs_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{patient_id}",
                    patient_id=patient_id,
                    timestamp=datetime.utcnow(),
                    risk_type=risk_type,
                    risk_score=float(risk_score),
                    risk_category=risk_category,
                    features_used=json.dumps(features_used),
                    model_version=model_version
                )
                
                session.add(score)
                session.commit()
                logger.info(f"Added {risk_type} risk score for patient {patient_id}: {risk_score:.2f} ({risk_category})")
                return True
                
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error adding risk score: {str(e)}")
            return False


def main():
    """Example usage of the DatabaseManager class."""
    # Initialize database manager
    db = DatabaseManager()
    
    # Test connection
    if not db.test_connection():
        print("Failed to connect to database. Please check your connection settings.")
        return
    
    # Create tables (only needed once)
    try:
        db.create_tables()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating tables: {str(e)}")
    
    # Example: Insert sample data
    try:
        # Sample patient data
        patients_data = [
            {
                'patient_id': 'P1001',
                'gender': 'Male',
                'date_of_birth': '1980-05-15',
                'height_cm': 175.0,
                'weight_kg': 75.0,
                'blood_type': 'A+'
            },
            {
                'patient_id': 'P1002',
                'gender': 'Female',
                'date_of_birth': '1990-09-22',
                'height_cm': 165.0,
                'weight_kg': 62.5,
                'blood_type': 'O+'
            }
        ]
        
        # Convert to DataFrame and insert
        patients_df = pd.DataFrame(patients_data)
        db.insert_dataframe('patients', patients_df)
        print(f"Inserted {len(patients_df)} patients")
        
    except Exception as e:
        print(f"Error inserting sample data: {str(e)}")
    
    # Example: Query data
    try:
        # Get all patients
        query = "SELECT * FROM patients"
        patients = db.query_to_dataframe(query)
        print("\nPatients in database:")
        print(patients)
        
    except Exception as e:
        print(f"Error querying data: {str(e)}")


if __name__ == "__main__":
    main()
