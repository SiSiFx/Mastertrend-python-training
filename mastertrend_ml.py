#!/usr/bin/env python3
"""
Stratégie MasterTrend améliorée avec Machine Learning
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
import pytz
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Importer la stratégie MasterTrend
from mastertrend_strategy import MasterTrendStrategy, JMA, JMACD, WilliamsFractal, SessionIndicator

# Classe pour générer des features à partir des données
class FeatureGenerator:
    """Génère des caractéristiques (features) pour le ML à partir des données market"""
    
    @staticmethod
    def compute_indicators(df, window_sizes=[5, 10, 20, 50]):
        """Calcule divers indicateurs techniques sur les données"""
        result = df.copy()
        
        # Prix moyens
        result['hlc3'] = (result['high'] + result['low'] + result['close']) / 3
        
        # Rendements
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Volatilité
        for window in window_sizes:
            result[f'volatility_{window}'] = result['returns'].rolling(window=window).std()
        
        # Moyennes Mobiles
        for window in window_sizes:
            result[f'sma_{window}'] = result['close'].rolling(window=window).mean()
            result[f'ema_{window}'] = result['close'].ewm(span=window, adjust=False).mean()
        
        # Ajouter explicitement ema_12 et ema_26 pour le MACD si non présents
        if 'ema_12' not in result.columns:
            result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
        if 'ema_26' not in result.columns:
            result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # Momentum
        for window in window_sizes:
            result[f'momentum_{window}'] = result['close'] / result['close'].shift(window) - 1
        
        # RSI
        for window in window_sizes:
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in window_sizes:
            result[f'bb_middle_{window}'] = result['close'].rolling(window=window).mean()
            result[f'bb_std_{window}'] = result['close'].rolling(window=window).std()
            result[f'bb_upper_{window}'] = result[f'bb_middle_{window}'] + 2 * result[f'bb_std_{window}']
            result[f'bb_lower_{window}'] = result[f'bb_middle_{window}'] - 2 * result[f'bb_std_{window}']
            result[f'bb_width_{window}'] = (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']) / result[f'bb_middle_{window}']
        
        # ATR
        for window in window_sizes:
            tr1 = result['high'] - result['low']
            tr2 = abs(result['high'] - result['close'].shift(1))
            tr3 = abs(result['low'] - result['close'].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result[f'atr_{window}'] = tr.rolling(window=window).mean()
        
        # Supprimer les lignes avec NaN
        result.dropna(inplace=True)
        
        return result
    
    @staticmethod
    def create_labels(df, forward_window=5, threshold=0.0):
        """Crée des labels pour les données (1 pour hausse, 0 pour baisse)"""
        # Calculer la variation de prix sur les N prochaines barres
        df['forward_returns'] = df['close'].shift(-forward_window) / df['close'] - 1
        
        # Créer des labels binaires basés sur la direction future
        df['target'] = np.where(df['forward_returns'] > threshold, 1, 0)
        
        # Supprimer la dernière ligne car elle n'a pas de label
        df = df.iloc[:-forward_window].copy()
        
        return df
    
    @staticmethod
    def prepare_features(df, drop_cols=['datetime', 'open', 'high', 'low', 'close', 'volume', 'forward_returns', 'target']):
        """Prépare les features pour l'entraînement ML en supprimant les colonnes non nécessaires"""
        features = df.drop(columns=[col for col in drop_cols if col in df.columns])
        return features


# Classe pour entraîner un modèle ML
class MLModelTrainer:
    """Entraîne un modèle de ML pour prédire les mouvements de prix"""
    
    def __init__(self, model_type='RandomForest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X, y, optimize=False):
        """Entraîne le modèle ML avec les données fournies"""
        # Séparation train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Standardisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Sélection et entraînement du modèle
        if self.model_type == 'RandomForest':
            if optimize:
                # Optimisation des hyperparamètres avec GridSearchCV
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                                         param_grid=param_grid, 
                                         cv=5, 
                                         scoring='accuracy')
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
                print(f"Meilleurs paramètres: {grid_search.best_params_}")
            else:
                # Entraînement simple
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.model.fit(X_train_scaled, y_train)
        
        # Évaluation du modèle
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Précision du modèle: {accuracy:.4f}")
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred))
        
        return self.model, self.scaler
    
    def save_model(self, model_path='mastertrend_ml_model.pkl', scaler_path='mastertrend_ml_scaler.pkl'):
        """Sauvegarde le modèle et le scaler pour une utilisation future"""
        if self.model is not None:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Modèle sauvegardé dans {model_path}")
            print(f"Scaler sauvegardé dans {scaler_path}")
        else:
            print("Aucun modèle à sauvegarder. Entraînez d'abord un modèle.")
    
    def load_model(self, model_path='mastertrend_ml_model.pkl', scaler_path='mastertrend_ml_scaler.pkl'):
        """Charge un modèle et un scaler préentraînés"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Modèle chargé depuis {model_path}")
            print(f"Scaler chargé depuis {scaler_path}")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False


# Stratégie MasterTrend améliorée avec ML
class MasterTrendML(MasterTrendStrategy):
    """Stratégie MasterTrend améliorée avec prédictions de Machine Learning"""
    
    params = (
        # Paramètres standards de MasterTrend
        ('line1_macd', 13),
        ('line2_macd', 26),
        ('supertrend_period', 10),
        ('supertrend_multiplier', 3),
        ('jma_length', 7),
        ('jma_phase', 50),
        ('jma_power', 2),
        ('pivot_length', 10),
        ('left_range', 2),
        ('right_range', 2),
        ('london_session', True),
        ('london_hours', '0300-1200'),
        ('ny_session', True),
        ('ny_hours', '0800-1700'),
        ('tokyo_session', True),
        ('tokyo_hours', '2000-0400'),
        ('sydney_session', False),
        ('sydney_hours', '1700-0200'),
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('rsi_length', 14),
        ('volume_mult', 1.5),
        ('atr_length', 14),
        ('williams_stop_buffer', 0.0),
        
        # Paramètres ML
        ('ml_enabled', True),              # Activer/Désactiver le ML
        ('ml_confidence_threshold', 0.6),  # Seuil de confiance pour les prédictions ML
        ('ml_model_path', 'mastertrend_ml_model.pkl'),  # Chemin du modèle ML
        ('ml_scaler_path', 'mastertrend_ml_scaler.pkl'), # Chemin du scaler
        ('filter_signals', True),          # Filtrer les signaux MasterTrend avec ML
        ('feature_window', 20),            # Fenêtre pour calculer les features
    )
    
    def __init__(self):
        # Appel du constructeur parent
        MasterTrendStrategy.__init__(self)
        
        # Charger le modèle ML s'il est activé
        self.ml_model_ready = False
        if self.p.ml_enabled:
            self.ml_trainer = MLModelTrainer()
            
            try:
                self.ml_trainer.load_model(
                    model_path=self.p.ml_model_path,
                    scaler_path=self.p.ml_scaler_path
                )
                self.ml_model_ready = True
                self.log("Modèle ML chargé avec succès")
            except Exception as e:
                self.log(f"Erreur lors du chargement du modèle ML: {e}")
                self.log("Les signaux ML ne seront pas utilisés")
        
        # Ajouter des indicateurs supplémentaires pour les features ML
        self.add_ml_indicators()
        
        # Variables pour stocker les prédictions ML
        self.ml_prediction = 0  # 0: neutre, 1: haussier
        self.ml_confidence = 0.0
        
    def add_ml_indicators(self):
        """Ajouter des indicateurs supplémentaires utilisés comme features ML"""
        # Momentum
        self.momentum5 = self.data.close / self.data.close(-5) - 1
        self.momentum10 = self.data.close / self.data.close(-10) - 1
        self.momentum20 = self.data.close / self.data.close(-20) - 1
        
        # Volatilité
        self.volatility5 = bt.indicators.StdDev(self.data.close, period=5)
        self.volatility10 = bt.indicators.StdDev(self.data.close, period=10)
        self.volatility20 = bt.indicators.StdDev(self.data.close, period=20)
        
        # Bollinger Bands Width
        self.bb20 = bt.indicators.BollingerBands(self.data, period=20)
        self.bbwidth20 = (self.bb20.lines.top - self.bb20.lines.bot) / self.bb20.lines.mid
        
        # Moyenne mobile ratio
        self.sma_ratio = self.ema_fast / self.ema_slow
    
    def get_features(self):
        """Extraire les features actuelles pour le ML"""
        # Si on n'a pas assez de données, retourner None
        if len(self) < self.p.feature_window:
            return None
        
        # Créer un dictionnaire de features avec des noms correspondant à ceux utilisés lors de l'entraînement
        features = {
            # Features de base
            'hlc3': (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3,
            'returns': self.data.close[0] / self.data.close[-1] - 1,
            'log_returns': math.log(self.data.close[0] / self.data.close[-1]) if self.data.close[-1] > 0 else 0,
            
            # Volatilité
            'volatility_5': self.volatility5[0],
            'volatility_10': self.volatility10[0],
            'volatility_20': self.volatility20[0],
            'volatility_50': self.volatility10[0] * 1.2,  # Approximation
            
            # Moyennes mobiles
            'sma_5': self.datas[0].close[0] / self.datas[0].close[-5],
            'ema_5': self.ema_fast[0] / self.ema_fast[-5],
            'sma_10': self.datas[0].close[0] / self.datas[0].close[-10],
            'ema_10': self.ema_fast[0] / self.ema_fast[-10],
            'sma_20': self.datas[0].close[0] / self.datas[0].close[-20],
            'ema_20': self.ema_fast[0] / self.ema_fast[-20],
            'sma_50': self.ema_slow[0] / self.ema_slow[-1],
            'ema_50': self.ema_slow[0] / self.ema_slow[-1],
            
            # MACD
            'ema_12': self.macd1[0] + self.macd2[0],  # Approximation
            'ema_26': self.macd2[0],                  # Approximation
            'macd': self.macd1[0],
            'macd_signal': self.macd2[0],
            'macd_hist': self.macd1[0] - self.macd2[0],
            
            # Momentum
            'momentum_5': self.momentum5[0],
            'momentum_10': self.momentum10[0],
            'momentum_20': self.momentum20[0],
            'momentum_50': self.momentum20[0] * 0.8,  # Approximation
            
            # RSI (les périodes disponibles dans la stratégie)
            'rsi_5': self.rsi[0] * 0.9,               # Approximation
            'rsi_10': self.rsi[0] * 0.95,             # Approximation
            'rsi_20': self.rsi[0],                    # Approximation
            'rsi_50': self.rsi[0] * 1.05,             # Approximation
            
            # Bollinger Bands
            'bb_middle_5': self.datas[0].close[0],                      # Approximation
            'bb_std_5': self.volatility5[0],                            # Approximation
            'bb_upper_5': self.datas[0].close[0] + 2 * self.volatility5[0],  # Approximation
            'bb_lower_5': self.datas[0].close[0] - 2 * self.volatility5[0],  # Approximation
            'bb_width_5': 4 * self.volatility5[0] / self.datas[0].close[0],  # Approximation
            
            'bb_middle_10': self.datas[0].close[0],                       # Approximation
            'bb_std_10': self.volatility10[0],                            # Approximation
            'bb_upper_10': self.datas[0].close[0] + 2 * self.volatility10[0],  # Approximation
            'bb_lower_10': self.datas[0].close[0] - 2 * self.volatility10[0],  # Approximation
            'bb_width_10': 4 * self.volatility10[0] / self.datas[0].close[0],  # Approximation
            
            'bb_middle_20': self.bb20.lines.mid[0],
            'bb_std_20': self.bb20.lines.top[0] - self.bb20.lines.mid[0],
            'bb_upper_20': self.bb20.lines.top[0],
            'bb_lower_20': self.bb20.lines.bot[0],
            'bb_width_20': self.bbwidth20[0],
            
            'bb_middle_50': self.datas[0].close[0],                       # Approximation
            'bb_std_50': self.volatility20[0] * 1.2,                      # Approximation
            'bb_upper_50': self.datas[0].close[0] + 2 * self.volatility20[0] * 1.2,  # Approximation
            'bb_lower_50': self.datas[0].close[0] - 2 * self.volatility20[0] * 1.2,  # Approximation
            'bb_width_50': 4 * self.volatility20[0] * 1.2 / self.datas[0].close[0],  # Approximation
            
            # ATR
            'atr_5': self.atr[0] * 0.7,    # Approximation
            'atr_10': self.atr[0] * 0.8,   # Approximation
            'atr_20': self.atr[0] * 0.9,   # Approximation
            'atr_50': self.atr[0] * 1.1,   # Approximation
        }
        
        return features
    
    def predict_with_ml(self):
        """Utiliser le modèle ML pour prédire la direction du marché"""
        if not self.ml_model_ready:
            return 0, 0.0
        
        # Obtenir les features
        features = self.get_features()
        if features is None:
            return 0, 0.0
        
        # Convertir en DataFrame
        features_df = pd.DataFrame([features])
        
        # Standardiser les features
        try:
            features_scaled = self.ml_trainer.scaler.transform(features_df)
            
            # Faire la prédiction
            prediction_proba = self.ml_trainer.model.predict_proba(features_scaled)[0]
            prediction = 1 if prediction_proba[1] > self.p.ml_confidence_threshold else 0
            confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            
            return prediction, confidence
        except Exception as e:
            self.log(f"Erreur lors de la prédiction ML: {e}")
            return 0, 0.0
    
    def next(self):
        # Mettre à jour la prédiction ML
        if self.p.ml_enabled and self.ml_model_ready and len(self) > self.p.feature_window:
            self.ml_prediction, self.ml_confidence = self.predict_with_ml()
        
        # Appeler la méthode next de la stratégie parent
        if not self.p.ml_enabled or not self.p.filter_signals:
            # Si ML est désactivé ou qu'on ne filtre pas les signaux, utiliser la stratégie originale
            super().next()
        else:
            # Si ML est activé et qu'on filtre les signaux, on adapte la stratégie
            
            # 1. Mise à jour de SuperTrend et Williams Stops (comme dans MasterTrendStrategy)
            if len(self) > 1:
                up1 = self.up[-1]
                dn1 = self.dn[-1]
                
                if self.data.close[-1] > up1:
                    up = max(self.up[0], up1)
                else:
                    up = self.up[0]
                
                if self.data.close[-1] < dn1:
                    dn = min(self.dn[0], dn1)
                else:
                    dn = self.dn[0]
                
                if self.trend == -1 and self.data.close[0] > dn1:
                    self.trend = 1
                elif self.trend == 1 and self.data.close[0] < up1:
                    self.trend = -1
            
            self.update_williams_stops()
            
            # 2. Vérifier si un ordre est déjà en attente
            if self.order:
                return
            
            # 3. Vérifier si nous sommes dans une session de trading
            if not self.is_in_session():
                return
            
            # 4. Variables de décision avec filtrage ML
            # Conditions pour un signal long
            macd_cross_up = self.macd2[0] > 0 and self.macd2[-1] <= 0  # Croisement vers le haut
            macd_condition = self.macd1[0] < self.macd2[0] and self.macd1[-1] < self.macd1[0] and self.macd1[0] < 0
            supertrend_up = self.trend == 1
            jma_up = self.data.close[0] > self.jma[0]
            rsi_not_overbought = self.rsi[0] < 70
            volume_surge = self.data.volume[0] > self.vol_ma[0] * self.p.volume_mult
            
            # Ajouter la condition ML pour long
            ml_long_signal = bool(self.ml_prediction == 1 and self.ml_confidence >= self.p.ml_confidence_threshold)
            
            # Conditions pour un signal short
            macd_cross_down = self.macd2[0] < 0 and self.macd2[-1] >= 0  # Croisement vers le bas
            macd_condition_down = self.macd1[0] > self.macd2[0] and self.macd1[-1] > self.macd1[0] and self.macd1[0] > 0
            supertrend_down = self.trend == -1
            jma_down = self.data.close[0] < self.jma[0]
            rsi_not_oversold = self.rsi[0] > 30
            
            # Ajouter la condition ML pour short
            ml_short_signal = bool(self.ml_prediction == 0 and self.ml_confidence >= self.p.ml_confidence_threshold)
            
            # 5. Signaux combinés avec ML
            long_signal = (macd_cross_up and macd_condition and supertrend_up and 
                          self.williams_long_stop > 0 and self.data.low[0] > self.williams_long_stop and
                          ml_long_signal)  # Inclure le signal ML
            
            short_signal = (macd_cross_down and macd_condition_down and supertrend_down and 
                           self.williams_short_stop > 0 and self.data.high[0] < self.williams_short_stop and
                           ml_short_signal)  # Inclure le signal ML
            
            # 6. Exécution des ordres (identique à MasterTrendStrategy)
            if not self.position:  # Pas de position
                if long_signal:
                    self.log(f'BUY CREATE (ML conf: {self.ml_confidence:.2f}) {self.data.close[0]:.5f}')
                    self.order = self.buy()
                    self.in_long = True
                    self.in_short = False
                    self.current_count = 0
                    self.start_price = self.data.close[0]
                    
                elif short_signal:
                    self.log(f'SELL CREATE (ML conf: {self.ml_confidence:.2f}) {self.data.close[0]:.5f}')
                    self.order = self.sell()
                    self.in_short = True
                    self.in_long = False
                    self.current_count = 0
                    self.start_price = self.data.close[0]
                    
            else:  # Position ouverte (gestion identique à MasterTrendStrategy)
                # Gérer les stops et les profits
                if self.position.size > 0:  # Position longue
                    if self.williams_long_stop > 0 and self.data.low[0] <= self.williams_long_stop:
                        self.log(f'SELL STOP HIT {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif macd_cross_down:
                        self.log(f'SELL PROFIT {self.data.close[0]:.5f}')
                        self.order = self.close()
                    
                    # Calculer les statistiques
                    if self.data.close[0] > self.data.open[0]:
                        self.current_count += 1
                    else:
                        self.sum_long_bars += self.current_count
                        self.sum_long_distance += (self.data.high[0] - self.start_price)
                        self.count_long += 1
                        self.in_long = False
                        
                else:  # Position courte
                    if self.williams_short_stop > 0 and self.data.high[0] >= self.williams_short_stop:
                        self.log(f'BUY STOP HIT {self.data.close[0]:.5f}')
                        self.order = self.close()
                    elif macd_cross_up:
                        self.log(f'BUY PROFIT {self.data.close[0]:.5f}')
                        self.order = self.close()
                    
                    # Calculer les statistiques
                    if self.data.close[0] < self.data.open[0]:
                        self.current_count += 1
                    else:
                        self.sum_short_bars += self.current_count
                        self.sum_short_distance += (self.start_price - self.data.low[0])
                        self.count_short += 1
                        self.in_short = False


# Fonction pour entraîner un modèle ML sur les données historiques
def train_ml_model(data_file="EURUSD_data_1M.csv", lookback_period=10000, optimize=False):
    """Entraîne un modèle ML sur les données historiques"""
    print(f"Entraînement du modèle ML sur {data_file}...")
    
    # Charger les données
    if not os.path.exists(data_file):
        print(f"Erreur: Le fichier {data_file} n'existe pas.")
        return None, None
    
    # Charger les N dernières barres
    df = pd.read_csv(data_file)
    if lookback_period and lookback_period < len(df):
        df = df.tail(lookback_period).reset_index(drop=True)
    
    # Générer les features
    print("Calcul des indicateurs et features...")
    df_features = FeatureGenerator.compute_indicators(df)
    
    # Créer les labels
    print("Création des labels...")
    df_labeled = FeatureGenerator.create_labels(df_features, forward_window=5, threshold=0.0)
    
    # Préparer les features
    X = FeatureGenerator.prepare_features(df_labeled)
    y = df_labeled['target']
    
    # Entraîner le modèle
    print("Entraînement du modèle...")
    trainer = MLModelTrainer()
    model, scaler = trainer.train(X, y, optimize=optimize)
    
    # Sauvegarder le modèle
    trainer.save_model()
    
    return model, scaler


# Fonction principale pour exécuter la stratégie MasterTrendML
def run_mastertrend_ml(data_file="EURUSD_data_1M.csv", train_model=False, optimize=False, plot=False):
    """Exécute la stratégie MasterTrendML avec ou sans entraînement préalable"""
    
    # 1. Entraîner le modèle si demandé
    if train_model:
        train_ml_model(data_file, optimize=optimize)
    
    # 2. Exécuter la stratégie
    cerebro = bt.Cerebro()
    
    # Ajouter les données
    if not os.path.exists(data_file):
        print(f"Erreur: Le fichier {data_file} n'existe pas.")
        return
    
    data = bt.feeds.GenericCSVData(
        dataname=data_file,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    cerebro.adddata(data)
    
    # Paramètres de la stratégie ML
    ml_params = {
        'ml_enabled': True,
        'ml_confidence_threshold': 0.3,  # Réduire le seuil de confiance pour avoir plus de signaux
        'filter_signals': True
    }
    
    # Ajouter la stratégie
    cerebro.addstrategy(MasterTrendML, **ml_params)
    
    # Configurer le capital initial et les commissions
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    # Ajouter des analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Exécuter la stratégie
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    strat = results[0]
    
    # Afficher les résultats
    print('Capital final: %.2f' % cerebro.broker.getvalue())
    print('Rendement: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
    
    # Statistiques des trades
    trades = strat.analyzers.trades.get_analysis()
    
    print('\nAnalyse des transactions:')
    try:
        total_trades = trades.total.closed if hasattr(trades, 'total') else 0
        print('Total des trades:', total_trades)
        
        if hasattr(trades, 'won'):
            print('Trades gagnants:', trades.won.total)
            if hasattr(trades.won, 'pnl') and hasattr(trades.won.pnl, 'average'):
                print('Gain moyen des trades gagnants:', trades.won.pnl.average)
        else:
            print('Trades gagnants: 0')
        
        if hasattr(trades, 'lost'):
            print('Trades perdants:', trades.lost.total)
            if hasattr(trades.lost, 'pnl') and hasattr(trades.lost.pnl, 'average'):
                print('Perte moyenne des trades perdants:', trades.lost.pnl.average)
        else:
            print('Trades perdants: 0')
    except Exception as e:
        print('Pas assez de trades pour générer des statistiques:', e)
    
    # Sharpe Ratio
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_value = sharpe['sharperatio'] if 'sharperatio' in sharpe else "N/A"
        print('\nSharpe Ratio:', sharpe_value)
    except Exception as e:
        print('\nErreur lors du calcul du Sharpe Ratio:', e)
    
    # Afficher le graphique si demandé
    if plot:
        cerebro.plot(style='candle')


# Exécuter le script
if __name__ == '__main__':
    import sys
    import os
    
    # Déterminer les arguments
    train = '--train' in sys.argv
    optimize = '--optimize' in sys.argv
    plot = '--plot' in sys.argv
    
    # Fichier de données par défaut
    data_file = "EURUSD_data_1M.csv"
    
    # Chercher un fichier de données spécifié
    for arg in sys.argv:
        if arg.endswith('.csv') and os.path.exists(arg):
            data_file = arg
            break
    
    # Exécuter la stratégie
    print(f"Exécution de MasterTrendML sur {data_file}")
    print(f"Entraînement du modèle: {'Oui' if train else 'Non'}")
    print(f"Optimisation des hyperparamètres: {'Oui' if optimize else 'Non'}")
    print(f"Affichage du graphique: {'Oui' if plot else 'Non'}")
    
    run_mastertrend_ml(
        data_file=data_file,
        train_model=train,
        optimize=optimize,
        plot=plot
    ) 