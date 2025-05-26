#!/usr/bin/env python3
"""
Script pour déboguer les features du modèle ML
"""
import os
import joblib
import pandas as pd
import numpy as np
from mastertrend_ml import FeatureGenerator

def main():
    # Charger les données
    data_file = "EURUSD_data_1M.csv"
    if not os.path.exists(data_file):
        print(f"Fichier {data_file} non trouvé.")
        return
    
    # Charger les données
    df = pd.read_csv(data_file)
    df_tail = df.tail(1000).reset_index(drop=True)
    
    # Générer les features comme lors de l'entraînement
    print("Calcul des indicateurs...")
    df_features = FeatureGenerator.compute_indicators(df_tail)
    
    print("Création des labels...")
    df_labeled = FeatureGenerator.create_labels(df_features, forward_window=5, threshold=0.0)
    
    print("Préparation des features...")
    X = FeatureGenerator.prepare_features(df_labeled)
    
    # Afficher les noms des colonnes
    print("\nNoms de colonnes entraînement:")
    for col in X.columns:
        print(f"- {col}")
    
    print("\nNombre de features:", len(X.columns))
    
    # Pendant la prédiction, les features créées par la stratégie
    print("\nSimulation des features de prédiction...")
    from mastertrend_ml import MasterTrendML
    
    # Créer un dictionnaire simulant les features de prédiction
    pred_features = {
        'hlc3': 1.2345,
        'returns': 0.001,
        'log_returns': 0.0009,
        'volatility_5': 0.002,
        'volatility_10': 0.003,
        'volatility_20': 0.004,
        'rsi_14': 60,
        'momentum_5': 0.01,
        'momentum_10': 0.02,
        'momentum_20': 0.03,
        'macd': 0.001,
        'macd_signal': 0.0005,
        'macd_hist': 0.0005,
        'bb_width_20': 0.01,
        'atr_14': 0.005,
        'sma_5': 1.02,
        'sma_10': 1.01,
        'sma_20': 1.005,
        'sma_50': 1.001,
        'ema_5': 1.02,
        'ema_10': 1.01,
        'ema_20': 1.005,
        'ema_50': 1.001,
    }
    
    # Créer un DataFrame
    df_pred = pd.DataFrame([pred_features])
    print("\nNoms de colonnes prédiction:")
    for col in df_pred.columns:
        print(f"- {col}")
    
    print("\nNombre de features prédiction:", len(df_pred.columns))
    
    # Les différences
    train_cols = set(X.columns)
    pred_cols = set(df_pred.columns)
    
    print("\nColonnes dans train mais pas dans pred:")
    for col in sorted(train_cols - pred_cols):
        print(f"- {col}")
    
    print("\nColonnes dans pred mais pas dans train:")
    for col in sorted(pred_cols - train_cols):
        print(f"- {col}")

if __name__ == "__main__":
    main() 