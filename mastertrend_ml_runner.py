#!/usr/bin/env python3
"""
Script pour exécuter la stratégie MasterTrend avec ou sans ML
"""
import sys
import argparse
from mastertrend_ml import run_mastertrend_ml, train_ml_model, MasterTrendStrategy
import backtrader as bt
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Exécuter la stratégie MasterTrend avec ML')
    
    parser.add_argument('--data', type=str, 
                        default='EURUSD_data_1M.csv',
                        help='Fichier CSV de données')
    
    parser.add_argument('--train', action='store_true',
                        help='Entraîner le modèle ML avant exécution')
    
    parser.add_argument('--train-only', action='store_true',
                        help='Uniquement entraîner le modèle ML sans exécuter la stratégie')
    
    parser.add_argument('--optimize', action='store_true',
                        help='Optimiser les hyperparamètres du modèle ML (plus lent)')
    
    parser.add_argument('--no-ml', action='store_true',
                        help='Désactiver ML (utilise MasterTrendStrategy standard)')
    
    parser.add_argument('--plot', action='store_true',
                        help='Afficher les graphiques')
    
    parser.add_argument('--save-plot', action='store_true',
                        help='Sauvegarder les graphiques au lieu de les afficher')
    
    parser.add_argument('--lookback', type=int, default=20000,
                        help='Nombre de barres à considérer pour l\'entraînement')
    
    parser.add_argument('--cash', type=float, default=10000.0,
                        help='Capital initial')
    
    parser.add_argument('--commission', type=float, default=0.0001,
                        help='Commission de trading (en %)')
    
    strategy_params = parser.add_argument_group('Paramètres de stratégie')
    
    strategy_params.add_argument('--macd1', type=int, default=13,
                                help='Période ligne 1 MACD')
    
    strategy_params.add_argument('--macd2', type=int, default=26,
                                help='Période ligne 2 MACD')
    
    strategy_params.add_argument('--supertrend-period', type=int, default=10,
                                help='Période SuperTrend')
    
    strategy_params.add_argument('--supertrend-mult', type=float, default=3.0,
                                help='Multiplicateur SuperTrend')
    
    strategy_params.add_argument('--jma-length', type=int, default=7,
                                help='Longueur JMA')
    
    strategy_params.add_argument('--jma-phase', type=int, default=50,
                                help='Phase JMA')
    
    strategy_params.add_argument('--pivot-length', type=int, default=10,
                                help='Longueur Pivot')
    
    return parser.parse_args()


def run_classic_mastertrend(data_file, plot=False):
    """Exécute la stratégie MasterTrend classique sans ML"""
    print(f"Exécution de MasterTrend classique sur {data_file}...")
    
    # 1. Initialiser Cerebro
    cerebro = bt.Cerebro()
    
    # 2. Ajouter les données
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
    
    # 3. Ajouter la stratégie classique
    cerebro.addstrategy(MasterTrendStrategy)
    
    # 4. Configurer le capital initial et les commissions
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 0.01%
    
    # 5. Ajouter des analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 6. Exécuter la stratégie
    print('Capital initial: %.2f' % cerebro.broker.getvalue())
    
    results = cerebro.run()
    strat = results[0]
    
    # 7. Afficher les résultats
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
    
    # 8. Afficher le graphique si demandé
    if plot:
        cerebro.plot(style='candle')


def main():
    args = parse_args()
    
    # Paramètres de la stratégie
    strategy_params = {
        'line1_macd': args.macd1,
        'line2_macd': args.macd2,
        'supertrend_period': args.supertrend_period,
        'supertrend_multiplier': args.supertrend_mult,
        'jma_length': args.jma_length,
        'jma_phase': args.jma_phase,
        'pivot_length': args.pivot_length,
    }
    
    # Si on veut uniquement entraîner le modèle
    if args.train_only:
        print(f"Entraînement du modèle ML sur {args.data} (lookback: {args.lookback} barres)...")
        train_ml_model(
            data_file=args.data,
            lookback_period=args.lookback,
            optimize=args.optimize
        )
        return 0
    
    # Si on veut exécuter sans ML, utiliser la stratégie classique
    if args.no_ml:
        print(f"Exécution de MasterTrend sur {args.data}...")
        print(f"Machine Learning: Désactivé")
        run_classic_mastertrend(
            data_file=args.data,
            plot=args.plot
        )
        return 0
    
    # Sinon exécuter la stratégie ML
    print(f"Exécution de MasterTrendML sur {args.data}...")
    print(f"Machine Learning: Activé")
    if args.train:
        print(f"Entraînement du modèle avant exécution (lookback: {args.lookback} barres)")
    
    run_mastertrend_ml(
        data_file=args.data,
        train_model=args.train,
        optimize=args.optimize,
        plot=args.plot
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 