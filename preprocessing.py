import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.feature_engineering import TelecomFeatureEngineer

def build_preprocessing_pipeline(X: pd.DataFrame) -> Pipeline:
    """Assembles engineering, scaling, and categorical encoding into a robust pipeline."""
    
    # Identify basic baseline columns dynamically
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in X.columns if col not in num_cols and col != 'customerID']
    
    # Sub-pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine baseline components before adding engineered mutations
    base_transformer = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='passthrough')
    
    # Master structural pipeline
    full_pipeline = Pipeline([
        ('engineer', TelecomFeatureEngineer()),
        # Re-apply transform strategies to newly engineered float/int columns if necessary
        ('preprocessor', ColumnTransformer([
            ('num_pass', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpending']),
            ('cat_pass', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('engineered_pass', 'passthrough', ['TotalServices', 'IsAutoPayment'])
        ]))
    ])
    
    return full_pipeline