#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def train_model(data_path='dataset.csv', model_path='mastertrend_ml_model.pkl', scaler_path='mastertrend_ml_scaler.pkl', optimize=False):
    # Load dataset
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    print(f"Original dataset shape: {df.shape}")
    
    # Drop rows with NaN in target
    df = df.dropna(subset=['target'])
    print(f"After dropping NaN targets: {df.shape}")
    
    # Features and target
    X = df.drop(columns=['supertrend', 'prev_trend', 'target', 'basic_up', 'basic_dn', 'tr1', 'tr2', 'tr3', 'tr', 'prev_close'])
    y = df['target'].map({-1:0, 1:1})  # binary targets: 1=up flip, 0=down flip
    
    # Drop rows where y is NaN (target values of 0 become NaN after mapping)
    mask_y = y.notna()
    if (~mask_y).any():
        print(f"Dropping {(~mask_y).sum()} rows with no flip target (NaN y)")
    X = X[mask_y]
    y = y[mask_y]
    
    # Drop any remaining NaN in features
    mask = X.isna().any(axis=1)
    if mask.any():
        print(f"Dropping {mask.sum()} rows with NaN in features")
        X = X[~mask]
        y = y[~mask]
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")

    # Split train/test by time (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Model
    if optimize:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2,5,10]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train_scaled, y_train)
        model = grid.best_estimator_
        print("Best params:", grid.best_params_)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance.head(10))  # Top 10 most important features

    # Save
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}\nScaler saved to {scaler_path}")
    return model, scaler


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train ML model for MasterTrend flips')
    parser.add_argument('--data', default='dataset.csv', help='Path to dataset CSV')
    parser.add_argument('--model', default='mastertrend_ml_model.pkl', help='Path to save model')
    parser.add_argument('--scaler', default='mastertrend_ml_scaler.pkl', help='Path to save scaler')
    parser.add_argument('--optimize', action='store_true', help='Use GridSearchCV to optimize hyperparameters')
    args = parser.parse_args()

    train_model(data_path=args.data, model_path=args.model, scaler_path=args.scaler, optimize=args.optimize) 