#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply trained ML model to dataset')
    parser.add_argument('--data', default='dataset.csv', help='Path to dataset CSV')
    parser.add_argument('--model', default='mastertrend_ml_model.pkl', help='Path to trained model pickle')
    parser.add_argument('--scaler', default='mastertrend_ml_scaler.pkl', help='Path to scaler pickle')
    parser.add_argument('--output', default='predictions.csv', help='Path to save predictions CSV')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data, parse_dates=['datetime'], index_col='datetime')
    
    # Prepare features
    drop_cols = ['supertrend', 'prev_trend', 'target', 'basic_up', 'basic_dn', 'tr1', 'tr2', 'tr3', 'tr', 'prev_close']
    X = df.drop(columns=drop_cols)

    # Load scaler and model
    scaler = joblib.load(args.scaler)
    model = joblib.load(args.model)

    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Map predictions back to flip labels
    df['predicted_flip'] = np.where(y_pred == 1, 1, -1)

    # Evaluate if true target exists
    if 'target' in df.columns:
        y_true = df['target']
        # Map true flips to binary and filter only flip events
        y_true_bin = y_true.map({-1:0, 1:1})
        # Convert predictions to a Series aligned with df index
        y_pred_series = pd.Series(y_pred, index=df.index)
        # Keep only rows where y_true_bin is not NaN (flip events only)
        mask = y_true_bin.notna()
        if mask.sum() == 0:
            print("No flip labels to evaluate (all non-flip events)")
        else:
            y_true_eval = y_true_bin[mask].astype(int)
            y_pred_eval = y_pred_series[mask].astype(int)
            print('Classification Report:')
            print(classification_report(y_true_eval, y_pred_eval, target_names=['Down Flip','Up Flip']))
            print('Confusion Matrix:')
            print(confusion_matrix(y_true_eval, y_pred_eval))

    # Save predictions
    df.to_csv(args.output)
    print(f"Predictions saved to {args.output}") 