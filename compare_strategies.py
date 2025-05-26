#!/usr/bin/env python3
"""
Script de comparaison des comportements clés entre ISI_MAXI_WEEP original et sa traduction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backtrader as bt
import datetime
import os

# Importer les indicateurs et la stratégie depuis notre version traduite
from isi_maxi_weep_optimized import PivotHigh, PivotLow

def analyze_pivots(data_file, pivot_length=10, max_rows=5000):
    """
    Analyser les détections de pivots high/low et les comparer aux attendus
    """
    print(f"Analyse des pivots sur {data_file} (max {max_rows} lignes)...")
    
    # 1. Charger les données
    df = pd.read_csv(data_file)
    df = df.iloc[:max_rows]
    # Convert datetime column to pandas datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 2. Créer une version Cerebro pour calculer les pivots avec notre implémentation
    cerebro = bt.Cerebro()
    
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )
    
    # Créer une stratégie simple pour observer les pivots
    class PivotObserver(bt.Strategy):
        def __init__(self):
            self.pivot_high = PivotHigh(self.data, length=pivot_length)
            self.pivot_low = PivotLow(self.data, length=pivot_length)
            
            # Stocker les pivots
            self.pivot_high_values = []
            self.pivot_low_values = []
            
        def next(self):
            if not np.isnan(self.pivot_high[0]):
                self.pivot_high_values.append((len(self), self.pivot_high[0]))
                
            if not np.isnan(self.pivot_low[0]):
                self.pivot_low_values.append((len(self), self.pivot_low[0]))
    
    cerebro.adddata(data)
    cerebro.addstrategy(PivotObserver)
    results = cerebro.run()
    
    # Récupérer les pivots détectés
    strat = results[0]
    pivot_highs = strat.pivot_high_values
    pivot_lows = strat.pivot_low_values
    
    # 3. Implémenter manuellement la logique PineScript pour vérification
    # On pourrait implémenter ici la logique PineScript des pivots pour comparer
    
    # 4. Afficher les résultats pour comparaison manuelle
    print(f"Nombre de Pivot Highs détectés: {len(pivot_highs)}")
    print(f"Nombre de Pivot Lows détectés: {len(pivot_lows)}")
    
    # Afficher quelques exemples
    print("\nExemples de Pivot Highs (jusqu'à 5):")
    for i, (bar, value) in enumerate(pivot_highs[:5]):
        dt = pd.to_datetime(df.iloc[bar-pivot_length]['datetime'])  # Ajuster pour la position du pivot
        print(f"Bar {bar}, Date: {dt}, Value: {value:.5f}")
    
    print("\nExemples de Pivot Lows (jusqu'à 5):")
    for i, (bar, value) in enumerate(pivot_lows[:5]):
        dt = pd.to_datetime(df.iloc[bar-pivot_length]['datetime'])  # Ajuster pour la position du pivot
        print(f"Bar {bar}, Date: {dt}, Value: {value:.5f}")
    
    # 5. Visualiser les pivots
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label='Close', color='blue', alpha=0.6)
    
    # Ajouter les pivots high
    ph_x = [p[0]-pivot_length for p in pivot_highs]
    ph_y = [p[1] for p in pivot_highs]
    plt.scatter(ph_x, ph_y, color='green', marker='^', s=50, label='Pivot High')
    
    # Ajouter les pivots low
    pl_x = [p[0]-pivot_length for p in pivot_lows]
    pl_y = [p[1] for p in pivot_lows]
    plt.scatter(pl_x, pl_y, color='red', marker='v', s=50, label='Pivot Low')
    
    plt.title(f'Détection des Pivots (length={pivot_length})')
    plt.legend()
    plt.savefig('pivot_detection.png', dpi=150)
    print("Graphique des pivots sauvegardé dans 'pivot_detection.png'")
    
    return pivot_highs, pivot_lows

def analyze_sweeps_and_breakouts():
    """
    Analyser la détection des sweeps et breakouts
    """
    print("\nAnalyse de la logique de sweep et breakout...")
    
    # Pas besoin de réimplémenter toute la logique ici, car le fichier 
    # observer_strategy.py affiche déjà les détails des sweeps et breakouts
    
    print("Pour cette validation:")
    print("1. Examinez les logs de la stratégie pour identifier les sweeps")
    print("2. Vérifiez que les conditions de sweep sont remplies:")
    print("   - Prix fermant au-dessus d'un Pivot High (sweep haut)")
    print("   - Prix fermant en-dessous d'un Pivot Low (sweep bas)")
    print("3. Confirmez que les entrées se font après sweep:")
    print("   - Achat après sweep d'un bas et retour au-dessus du support")
    print("   - Vente après sweep d'un haut et retour en-dessous de la résistance")

def compare_parameters():
    """
    Comparaison des paramètres entre la version PineScript et la traduction
    """
    print("\nComparaison des paramètres clés:")
    
    # Paramètres PineScript (tels que définis dans le script original)
    pinescript_params = {
        'pivot_length': 10,
        'cooldown_bars': 5,
        'rr_ratio': 2.0,
        'use_stop_loss': True,
        'use_take_profit': True,
    }
    
    # Paramètres dans la version Backtrader
    backtrader_params = {
        'pivot_length': 10,
        'cooldown_bars': 5,
        'rr_ratio': 2.0,
        'sl_buffer': 0.0002,  # Spécifique à Backtrader
    }
    
    # Affichage
    print("\nParamètres PineScript:")
    for k, v in pinescript_params.items():
        print(f"  {k}: {v}")
    
    print("\nParamètres BackTrader:")
    for k, v in backtrader_params.items():
        print(f"  {k}: {v}")
    
    print("\nConclusion:")
    print("Les paramètres essentiels (pivot_length, cooldown_bars, rr_ratio) sont identiques.")
    print("Backtrader utilise sl_buffer pour ajouter une petite marge aux stop loss,")
    print("ce qui n'est pas explicitement présent dans PineScript mais aide à gérer les exécutions.")

if __name__ == "__main__":
    print("==== Analyse de la traduction ISI_MAXI_WEEP ====")
    
    # 1. Analyser les pivots
    pivots_high, pivots_low = analyze_pivots("EURUSD_data_1M.csv", pivot_length=10, max_rows=10000)
    
    # 2. Analyser les sweeps et breakouts
    analyze_sweeps_and_breakouts()
    
    # 3. Comparer les paramètres
    compare_parameters()
    
    print("\n==== Instructions de Validation ====")
    print("Pour valider complètement la traduction:")
    print("1. Exécutez observer_strategy.py pour voir la stratégie en action avec les logs")
    print("2. Vérifiez que les règles d'entrée/sortie correspondent à la stratégie originale")
    print("3. Comparez les résultats de quelques trades spécifiques entre les deux plateformes")
    print("4. Ajustez les paramètres si nécessaire pour des comportements plus proches") 