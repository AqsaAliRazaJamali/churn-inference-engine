import pandas as pd
from typing import Tuple
from src.utils import get_logger

logger = get_logger(__name__)

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads the expanded IBM Telco Excel file and standardizes column schemas."""
    logger.info(f"Loading raw data from {filepath}")
    
    df = pd.read_excel(filepath)
    
    # Strip any accidental hidden spaces around column headers
    df.columns = df.columns.str.strip()
    
    # Map the specific expanded IBM column headers to the expected pipeline schema
    column_mapping = {
        'Gender': 'Gender',
        'Senior Citizen': 'SeniorCitizen',
        'Partner': 'Partner',
        'Dependents': 'Dependents',
        'Tenure Months': 'tenure',  # Map 'Tenure Months' to 'tenure'
        'Tenure': 'tenure',         # Fallback if named 'Tenure'
        'Phone Service': 'PhoneService',
        'Multiple Lines': 'MultipleLines',
        'Internet Service': 'InternetService',
        'Online Security': 'OnlineSecurity',
        'Online Backup': 'OnlineBackup',
        'Device Protection': 'DeviceProtection',
        'Tech Support': 'TechSupport',
        'Streaming TV': 'StreamingTV',
        'Streaming Movies': 'StreamingMovies',
        'Contract': 'Contract',
        'Paperless Billing': 'PaperlessBilling',
        'Payment Method': 'PaymentMethod',
        'Monthly Charges': 'MonthlyCharges',
        'Total Charges': 'TotalCharges',
        'Churn Label': 'Churn'      # Map 'Churn Label' (Yes/No) to our target column 'Churn'
    }
    
    # Rename columns that exist in the dataset
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Drop structural geolocation and advanced IBM metrics that aren't in the project baseline
    columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
                       'Lat Long', 'Latitude', 'Longitude', 'Churn Value', 
                       'Churn Score', 'CLTV', 'Churn Reason']
    
    existing_drops = [col for col in columns_to_drop if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        
    # Standardize TotalCharges clean-up logic
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', None, regex=True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    return df

def split_features_target(df: pd.DataFrame, target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:
    """Separates features from the target prediction variable."""
    if target_col not in df.columns:
        raise KeyError(
            f"\n\n[CRITICAL SCHEMA ERROR] Expected target '{target_col}' was not generated. "
            f"Available columns are: {list(df.columns)}\n"
        )
        
    X = df.drop(columns=[target_col])
    # Convert 'Yes'/'No' string from Churn Label directly into binary 1s and 0s
    y = df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1'] else 0)
    return X, y