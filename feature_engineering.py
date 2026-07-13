import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer implementing targeted business logic for telecom churn."""
    def __init__(self) -> None:
        pass
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Total Services Subscribed
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Verify columns exist before computing service totals
        existing_services = [col for col in service_cols if col in X_out.columns]
        if existing_services:
            X_out['TotalServices'] = X_out[existing_services].apply(
                lambda row: sum(1 for val in row if str(val).strip().lower() not in ['no', 'no phone service', 'no internet service']),
                axis=1
            )
        else:
            X_out['TotalServices'] = 0
            
        # 2. Average Monthly Spending Ratio
        if 'TotalCharges' in X_out.columns and 'tenure' in X_out.columns:
            X_out['AvgMonthlySpending'] = np.where(
                X_out['tenure'] > 0, 
                X_out['TotalCharges'] / X_out['tenure'], 
                X_out['MonthlyCharges']
            )
            
        # 3. Automated Payment Indicator
        if 'PaymentMethod' in X_out.columns:
            X_out['IsAutoPayment'] = X_out['PaymentMethod'].str.contains('automatic', case=False).astype(int)
            
        return X_out