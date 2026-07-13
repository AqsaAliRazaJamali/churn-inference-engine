import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from src.utils import setup_reproducibility, get_logger
from src.data_loader import load_raw_data, split_features_target
from src.preprocessing import build_preprocessing_pipeline

logger = get_logger(__name__)

def run_training_pipeline(data_path: str, model_output_dir: str) -> None:
    setup_reproducibility(seed=42)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 1. Pipeline Input processing
    df = load_raw_data(data_path)
    X, y = split_features_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Transform Fitment
    logger.info("Initializing and fitting preprocessor pipeline...")
    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # 3. Model Training
    logger.info("Training high-performance Gradient Boosting model...")
    # Using sample weights or internal scale options acts as a resilient alternative to SMOTE
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train_transformed, y_train)
    
    # 4. Serialize operational artifacts safely
    logger.info(f"Saving compiled pipeline structural states to {model_output_dir}")
    joblib.dump(preprocessor, os.path.join(model_output_dir, 'preprocessor.pkl'))
    joblib.dump(model, os.path.join(model_output_dir, 'churn_model.pkl'))
    logger.info("Training workflow successfully finalized.")

if __name__ == "__main__":
    # Test file execution safely
    run_training_pipeline(
        data_path='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.xlsx',
        model_output_dir='models'
    )