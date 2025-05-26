#!/usr/bin/env python3
"""
Script de test pour la stratégie MasterTrend
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Mode sans interface graphique

# Importer la stratégie MasterTrend
from mastertrend_strategy import MasterTrendStrategy, JMA, JMACD, WilliamsFractal

def test_indicators():
    """Test des indicateurs personnalisés"""
    print("Test des indicateurs personnalisés...")
    
    # Créer une petite série temporelle pour le test
    data = pd.DataFrame({
        'datetime': pd.date_range(start='2020-01-01', periods=100, freq='1min'),
        'open': np.random.normal(100, 1, 100),
        'high': np.random.normal(101, 1, 100),
        'low': np.random.normal(99, 1, 100),
        'close': np.random.normal(100, 1, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Sauvegarder pour test
    test_file = "test_data.csv"
    data.to_csv(test_file, index=False)
    
    # Créer un Cerebro pour tester les indicateurs
    cerebro = bt.Cerebro()
    
    # Ajouter les données
    datafeed = bt.feeds.GenericCSVData(
        dataname=test_file,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    cerebro.adddata(datafeed)
    
    # Ajouter une stratégie de test pour observer les indicateurs
    class TestStrategy(bt.Strategy):
        def __init__(self):
            # Tester JMA
            self.jma = JMA(self.data.close, period=7, phase=50, power=2)
            
            # Tester JMACD
            self.jmacd = JMACD(self.data.close, fastlen=12, slowlen=26, siglen=9)
            
            # Tester WilliamsFractal
            self.fractals = WilliamsFractal(self.data, left_bars=2, right_bars=2)
            
        def next(self):
            if len(self) == 50:  # Afficher les valeurs à mi-chemin
                print(f"JMA: {self.jma[0]:.5f}")
                print(f"JMACD (macd, signal, histo): {self.jmacd.macd[0]:.5f}, {self.jmacd.signal[0]:.5f}, {self.jmacd.histo[0]:.5f}")
                print(f"WilliamsFractal (top, bottom): {self.fractals.fractal_top[0]}, {self.fractals.fractal_bottom[0]}")
    
    cerebro.addstrategy(TestStrategy)
    cerebro.run()
    
    # Nettoyer
    try:
        os.remove(test_file)
    except:
        pass
    
    print("Test des indicateurs terminé.")

def run_strategy_test(data_file="EURUSD_data_1M.csv", plot=False):
    """Exécuter un test de la stratégie complète"""
    print(f"Test de la stratégie MasterTrend sur {data_file}...")
    
    # Vérifier si le fichier existe
    if not os.path.exists(data_file):
        print(f"Erreur: Le fichier {data_file} n'existe pas.")
        print("Vous devez d'abord préparer un fichier CSV avec les données de marché au format:")
        print("datetime,open,high,low,close,volume")
        return
    
    # Utiliser un échantillon limité pour le test rapide
    sample_size = 10000  # Nombre de barres à utiliser pour le test
    
    # Charger et préparer les données
    full_data = pd.read_csv(data_file)
    sample_data = full_data.iloc[:sample_size].copy()
    sample_file = "sample_data.csv"
    sample_data.to_csv(sample_file, index=False)
    
    # Créer le Cerebro
    cerebro = bt.Cerebro()
    
    # Ajouter les données
    datafeed = bt.feeds.GenericCSVData(
        dataname=sample_file,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        openinterest=-1
    )
    
    cerebro.adddata(datafeed)
    
    # Paramètres de test
    test_params = {
        # Paramètres MACD
        'line1_macd': 13,
        'line2_macd': 26,
        
        # Paramètres SuperTrend
        'supertrend_period': 10,
        'supertrend_multiplier': 3,
        
        # Paramètres JMA
        'jma_length': 7,
        'jma_phase': 50,
        'jma_power': 2,
        
        # Paramètres des sessions
        'london_session': True,
        'ny_session': True,
        'tokyo_session': True,
        'sydney_session': False,
    }
    
    # Ajouter la stratégie
    cerebro.addstrategy(MasterTrendStrategy, **test_params)
    
    # Configurer le capital initial et les commissions
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    # Ajouter des analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    # Exécuter la stratégie
    print(f"Exécution de la stratégie sur {sample_size} barres...")
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    strat = results[0]
    
    # Afficher les résultats
    print('Capital final: %.2f' % cerebro.broker.getvalue())
    print('Rendement: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
    
    # Statistiques des trades
    trades = strat.analyzers.trades.get_analysis()
    
    print('\nAnalyse des transactions:')
    print('Total des trades:', trades.total.closed if hasattr(trades, 'total') else 0)
    print('Trades gagnants:', trades.won.total if hasattr(trades, 'won') else 0)
    print('Trades perdants:', trades.lost.total if hasattr(trades, 'lost') else 0)
    
    if hasattr(trades, 'won') and trades.won.total > 0:
        print('Gain moyen des trades gagnants:', trades.won.pnl.average)
    if hasattr(trades, 'lost') and trades.lost.total > 0:
        print('Perte moyenne des trades perdants:', trades.lost.pnl.average)
    
    # Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    print('\nSharpe Ratio:', sharpe['sharperatio'] if 'sharperatio' in sharpe else "N/A")
    
    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    print('Drawdown max: %.2f%%' % drawdown.max.drawdown if hasattr(drawdown, 'max') else "N/A")
    
    # Tracer et sauvegarder le graphique
    if plot:
        print('\nGénération du graphique...')
        try:
            fig = cerebro.plot(style='candlestick', barup='green', bardown='red',
                         volup='green', voldown='red',
                         plotdist=0.1, width=16, height=9)
            
            # Sauvegarder le graphique
            fig[0][0].savefig('mastertrend_test_result.png', dpi=150)
            print('Graphique sauvegardé sous mastertrend_test_result.png')
        except Exception as e:
            print(f'Erreur lors de la génération du graphique: {e}')
    
    # Nettoyer
    try:
        os.remove(sample_file)
    except:
        pass
    
    print("Test de la stratégie terminé.")

if __name__ == '__main__':
    print("=== Test de la stratégie MasterTrend ===")
    
    # Tester les indicateurs personnalisés
    test_indicators()
    
    # Tester la stratégie complète
    # Note: Vous devez avoir un fichier CSV de données pour EURUSD_data_1M.csv
    data_file = "EURUSD_data_1M.csv"
    if os.path.exists(data_file):
        run_strategy_test(data_file, plot=True)
    else:
        print(f"\nFichier de données {data_file} non trouvé.")
        print("Pour exécuter le test complet, placez un fichier CSV de données EURUSD 1 minute")
        print("au format datetime,open,high,low,close,volume dans le répertoire courant.")
    
    print("\nInstallation recommandée:")
    print("pip install backtrader pandas numpy matplotlib pytz")
    print("\nPour un affichage correct des graphiques, assurez-vous d'avoir matplotlib correctement installé.") 