#!/usr/bin/env python3
"""
Script pour observer et vérifier la traduction de ISI_MAXI_WEEP
"""
import backtrader as bt
import pandas as pd
import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Mode sans interface graphique pour la compatibilité

# Importer les indicateurs et la stratégie depuis isi_maxi_weep_optimized.py
from isi_maxi_weep_optimized import PivotHigh, PivotLow, PriceLevel, ISI_MAXI_WEEP_Optimized

# Créer une version modifiée de la stratégie pour la validation
class ISI_MAXI_WEEP_Validator(ISI_MAXI_WEEP_Optimized):
    """Version modifiée pour observer plus de détails"""
    
    def __init__(self):
        # Appel du constructeur parent
        ISI_MAXI_WEEP_Optimized.__init__(self)
        
        # Variables pour le suivi des pivots et sweeps
        self.pivot_high_values = []
        self.pivot_low_values = []
        self.sweep_high_events = []
        self.sweep_low_events = []
        self.trade_events = []
        
    def next(self):
        # Enregistrer les pivots détectés
        if not pd.isna(self.pivot_high[0]):
            dt = self.datas[0].datetime.datetime(0)
            self.pivot_high_values.append((len(self), dt, self.pivot_high[0]))
            self.log(f'PIVOT HIGH DETECTED: {self.pivot_high[0]:.5f}', force=True)
            
        if not pd.isna(self.pivot_low[0]):
            dt = self.datas[0].datetime.datetime(0)
            self.pivot_low_values.append((len(self), dt, self.pivot_low[0]))
            self.log(f'PIVOT LOW DETECTED: {self.pivot_low[0]:.5f}', force=True)
        
        # Exécuter la logique normale de la stratégie
        super().next()
        
        # Enregistrer les événements de sweep
        if len(self) > 1:  # Éviter l'initialisation
            current_bar = len(self)
            if self.recent_high is not None and self.data.close[0] > self.recent_high and current_bar > self.last_sweep_bar + self.p.cooldown_bars:
                dt = self.datas[0].datetime.datetime(0)
                if self.sweep_type == 'high':  # Si mis à jour par la classe parent
                    self.sweep_high_events.append((current_bar, dt, self.data.close[0], self.recent_high))
            
            if self.recent_low is not None and self.data.close[0] < self.recent_low and current_bar > self.last_sweep_bar + self.p.cooldown_bars:
                dt = self.datas[0].datetime.datetime(0)
                if self.sweep_type == 'low':  # Si mis à jour par la classe parent
                    self.sweep_low_events.append((current_bar, dt, self.data.close[0], self.recent_low))
    
    def notify_trade(self, trade):
        super().notify_trade(trade)
        
        if trade.isclosed:
            # Enregistrer les détails du trade terminé
            entry_dt = self.data.datetime.datetime(-trade.barlen)
            exit_dt = self.data.datetime.datetime(0)
            
            trade_info = {
                'entry_bar': len(self) - trade.barlen,
                'exit_bar': len(self),
                'entry_dt': entry_dt,
                'exit_dt': exit_dt,
                'entry_price': trade.price, 
                'exit_price': trade.priceopenout if hasattr(trade, 'priceopenout') else self.data.close[0],
                'pnl': trade.pnl,
                'pnl_pct': (trade.pnl / trade.price) * 100 if trade.price != 0 else 0,
                'is_win': trade.pnl > 0
            }
            
            self.trade_events.append(trade_info)
            self.log(f'TRADE RECORDED: Entry: {entry_dt}, Exit: {exit_dt}, PnL: {trade.pnl:.2f}', force=True)
    
    def summary_report(self):
        """Générer un rapport résumé de la stratégie et ses comportements clés"""
        print("\n===== RAPPORT DE VALIDATION ISI_MAXI_WEEP =====")
        
        print("\n1. Détection des pivots")
        print(f"   Pivots Hauts détectés: {len(self.pivot_high_values)}")
        print(f"   Pivots Bas détectés: {len(self.pivot_low_values)}")
        
        if self.pivot_high_values:
            print("\n   Exemples de Pivots Hauts (5 premiers):")
            for i, (bar, dt, value) in enumerate(self.pivot_high_values[:5]):
                print(f"   - Bar {bar}, Date: {dt}, Valeur: {value:.5f}")
        
        if self.pivot_low_values:
            print("\n   Exemples de Pivots Bas (5 premiers):")
            for i, (bar, dt, value) in enumerate(self.pivot_low_values[:5]):
                print(f"   - Bar {bar}, Date: {dt}, Valeur: {value:.5f}")
        
        print("\n2. Événements de Sweep")
        print(f"   Sweeps de Hauts: {len(self.sweep_high_events)}")
        print(f"   Sweeps de Bas: {len(self.sweep_low_events)}")
        
        if self.sweep_high_events:
            print("\n   Exemples de Sweeps de Hauts (5 premiers):")
            for i, (bar, dt, close, level) in enumerate(self.sweep_high_events[:5]):
                print(f"   - Bar {bar}, Date: {dt}, Close: {close:.5f}, Niveau: {level:.5f}")
        
        if self.sweep_low_events:
            print("\n   Exemples de Sweeps de Bas (5 premiers):")
            for i, (bar, dt, close, level) in enumerate(self.sweep_low_events[:5]):
                print(f"   - Bar {bar}, Date: {dt}, Close: {close:.5f}, Niveau: {level:.5f}")
        
        print("\n3. Transactions")
        print(f"   Total: {len(self.trade_events)}")
        
        wins = [t for t in self.trade_events if t['is_win']]
        losses = [t for t in self.trade_events if not t['is_win']]
        
        print(f"   Gagnantes: {len(wins)}")
        print(f"   Perdantes: {len(losses)}")
        
        if self.trade_events:
            win_rate = (len(wins) / len(self.trade_events)) * 100 if self.trade_events else 0
            print(f"   Win Rate: {win_rate:.2f}%")
            
            avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
            
            print(f"   Gain moyen: {avg_win:.4f}")
            print(f"   Perte moyenne: {avg_loss:.4f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if losses else float('inf')
                print(f"   Facteur de profit: {profit_factor:.2f}")
        
        print("\n===== FIN DU RAPPORT =====")

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    # Charger un petit échantillon de données pour l'analyse détaillée
    sample_size = 20000  # Nombre de barres à analyser
    
    print(f"Chargement de {sample_size} barres pour l'analyse...")
    try:
        full_data = pd.read_csv("EURUSD_data_1M.csv")
        sample_data = full_data.iloc[:sample_size]
        sample_file = "EURUSD_sample.csv"
        sample_data.to_csv(sample_file, index=False)
        
        data = bt.feeds.GenericCSVData(
            dataname=sample_file,
            dtformat=('%Y-%m-%d %H:%M:%S'),
            datetime=0, open=1, high=2, low=3, close=4, volume=5,
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
            openinterest=-1
        )
        
        cerebro.adddata(data)
        
        # Paramètres pour un maximum de détails et de visualisation
        validation_params = {
            'pivot_length': 10,        # Comme dans PineScript
            'cooldown_bars': 5,        # Comme dans PineScript
            'rr_ratio': 2.0,           # Comme dans PineScript
            'show_logs': True,         # Activer tous les logs
            'log_interval': 1000,      # Logs fréquents
            'sl_buffer': 0.0002,       # Marge pour SL
        }
        
        # Ajouter la stratégie avec logs détaillés
        cerebro.addstrategy(ISI_MAXI_WEEP_Validator, **validation_params)
        
        # Paramètres du broker (comme dans PineScript)
        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.0001)
        
        # Ajouter tous les observateurs disponibles pour la visualisation
        cerebro.addobserver(bt.observers.Broker)
        cerebro.addobserver(bt.observers.BuySell)
        cerebro.addobserver(bt.observers.DrawDown)
        cerebro.addobserver(bt.observers.Trades)
        
        # Analyseurs pour statistiques
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        print('Démarrage de la vérification de la stratégie...')
        print('Capital initial: %.2f' % cerebro.broker.getvalue())
        
        results = cerebro.run()
        strat = results[0]
        
        # Appeler le rapport de validation
        strat.summary_report()
        
        print('Capital final: %.2f' % cerebro.broker.getvalue())
        print('Retour: %.2f%%' % ((cerebro.broker.getvalue() / 10000.0 - 1.0) * 100))
        
        # Afficher les statistiques des transactions
        trade_analysis = strat.analyzers.trades.get_analysis()
        
        print('\nValidation de la traduction - Statistiques des transactions:')
        print('Total des transactions:', trade_analysis.total.closed if hasattr(trade_analysis, 'total') else 0)
        print('Transactions gagnantes:', trade_analysis.won.total if hasattr(trade_analysis, 'won') else 0)
        print('Transactions perdantes:', trade_analysis.lost.total if hasattr(trade_analysis, 'lost') else 0)
        
        if hasattr(trade_analysis, 'won') and trade_analysis.won.total > 0:
            win_pct = (trade_analysis.won.total / trade_analysis.total.closed) * 100
            print(f'Win Rate: {win_pct:.2f}%')
            print(f'Gain moyen des transactions gagnantes: {trade_analysis.won.pnl.average:.6f}')
        if hasattr(trade_analysis, 'lost') and trade_analysis.lost.total > 0:
            print(f'Perte moyenne des transactions perdantes: {trade_analysis.lost.pnl.average:.6f}')
        
        # Tracer le graphique avec toutes les informations
        print('\nGénération du graphique détaillé pour validation visuelle...')
        try:
            figs = cerebro.plot(style='candlestick', barup='green', bardown='red',
                               volup='green', voldown='red',
                               plotdist=0.1, width=16, height=9)
            
            # Sauvegarder le graphique
            from matplotlib import pyplot as plt
            plt.savefig('validation_plot.png', dpi=150)
            print('Graphique sauvegardé sous validation_plot.png')
        except Exception as e:
            print(f'Erreur lors de la génération du graphique: {e}')
        
        print('\nComment vérifier la traduction:')
        print('1. Examiner les logs ci-dessus pour voir les pivots et sweeps détectés')
        print('2. Vérifier les règles d\'entrée et sortie dans le rapport')
        print('3. Observer le graphique dans validation_plot.png')
        print('4. Comparer avec le comportement attendu dans PineScript/TradingView')
        
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
    
    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(sample_file):
            try:
                os.remove(sample_file)
            except:
                pass 