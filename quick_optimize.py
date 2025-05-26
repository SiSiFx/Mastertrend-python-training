#!/usr/bin/env python3
"""
QUICK OPTIMIZER - Lancement rapide de l'optimisation automatique
Optimise automatiquement la stratégie MasterTrend pour maximiser les profits
"""
import sys
import os
from mastertrend_auto_optimizer import OptunaBayesianOptimizer, GeneticOptimizer, test_strategy

def quick_bayesian_optimization():
    """Optimisation Bayésienne rapide (recommandée)"""
    print("🎯 OPTIMISATION BAYÉSIENNE RAPIDE")
    print("=" * 50)
    
    data_file = "EURUSD_data_15M.csv"
    
    # Vérifier que le fichier de données existe
    if not os.path.exists(data_file):
        print(f"❌ ERREUR: Fichier {data_file} non trouvé!")
        print("Assurez-vous que le fichier de données EURUSD est présent.")
        return None
    
    # Optimisation avec 30 essais (rapide mais efficace)
    optimizer = OptunaBayesianOptimizer(data_file, n_trials=30)
    
    print("🚀 Démarrage de l'optimisation...")
    print("⏱️  Temps estimé: 5-10 minutes")
    
    try:
        best_params = optimizer.optimize()
        
        print("\n🏆 OPTIMISATION TERMINÉE!")
        print("📊 Test des meilleurs paramètres...")
        
        # Tester les meilleurs paramètres
        test_strategy(data_file, best_params)
        
        # Sauvegarder les paramètres
        save_best_params(best_params, "bayesian")
        
        return best_params
        
    except Exception as e:
        print(f"❌ ERREUR lors de l'optimisation: {e}")
        return None

def quick_genetic_optimization():
    """Optimisation Génétique rapide"""
    print("🧬 OPTIMISATION GÉNÉTIQUE RAPIDE")
    print("=" * 50)
    
    data_file = "EURUSD_data_15M.csv"
    
    # Vérifier que le fichier de données existe
    if not os.path.exists(data_file):
        print(f"❌ ERREUR: Fichier {data_file} non trouvé!")
        return None
    
    # Optimisation avec population réduite (rapide)
    optimizer = GeneticOptimizer(data_file, population_size=20, generations=10)
    
    print("🚀 Démarrage de l'optimisation génétique...")
    print("⏱️  Temps estimé: 10-15 minutes")
    
    try:
        best_params = optimizer.optimize()
        
        print("\n🏆 OPTIMISATION TERMINÉE!")
        print("📊 Test des meilleurs paramètres...")
        
        # Tester les meilleurs paramètres
        test_strategy(data_file, best_params)
        
        # Sauvegarder les paramètres
        save_best_params(best_params, "genetic")
        
        return best_params
        
    except Exception as e:
        print(f"❌ ERREUR lors de l'optimisation: {e}")
        return None

def save_best_params(params, method):
    """Sauvegarde les meilleurs paramètres dans un fichier"""
    filename = f"best_params_{method}.txt"
    
    try:
        with open(filename, 'w') as f:
            f.write(f"# Meilleurs paramètres - Méthode: {method.upper()}\n")
            f.write(f"# Généré automatiquement par MasterTrend Auto-Optimizer\n\n")
            
            for param, value in params.items():
                f.write(f"{param} = {value}\n")
        
        print(f"💾 Paramètres sauvegardés dans: {filename}")
        
    except Exception as e:
        print(f"⚠️  Erreur lors de la sauvegarde: {e}")

def create_optimized_strategy_file(params, method):
    """Crée un fichier de stratégie avec les paramètres optimisés"""
    filename = f"mastertrend_optimized_{method}.py"
    
    # Code de base de la stratégie optimisée
    strategy_code = f'''#!/usr/bin/env python3
"""
MASTERTREND STRATEGY - OPTIMISÉE AUTOMATIQUEMENT
Paramètres optimisés par {method.upper()}
Généré automatiquement par MasterTrend Auto-Optimizer
"""
import backtrader as bt
from mastertrend_propfirm_optimized import OptimizedPropFirmMasterTrend

# Paramètres optimisés
OPTIMIZED_PARAMS = {{
'''
    
    # Ajouter les paramètres
    for param, value in params.items():
        if isinstance(value, str):
            strategy_code += f"    '{param}': '{value}',\n"
        else:
            strategy_code += f"    '{param}': {value},\n"
    
    strategy_code += '''}

def run_optimized_strategy():
    """Lance la stratégie avec les paramètres optimisés"""
    cerebro = bt.Cerebro()
    
    # Données
    data = bt.feeds.GenericCSVData(
        dataname="EURUSD_data_15M.csv",
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes, compression=15,
        openinterest=-1, headers=True, separator=','
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(OptimizedPropFirmMasterTrend, **OPTIMIZED_PARAMS)
    
    # Configuration PropFirm
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print('🚀 STRATÉGIE OPTIMISÉE - DÉMARRAGE')
    print(f'Méthode d\\'optimisation: {method.upper()}')
    print('Capital Initial: $10,000')
    
    try:
        results = cerebro.run()
        strat = results[0]
        
        # Résultats
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - 10000) / 10000 * 100
        
        print(f'\\n📊 RÉSULTATS OPTIMISÉS:')
        print(f'Capital Final: ${final_value:.2f}')
        print(f'Rendement Total: {total_return:.2f}%')
        
        # Analyse des trades
        trades = strat.analyzers.trades.get_analysis()
        if hasattr(trades, 'total') and trades.total.closed > 0:
            win_rate = trades.won.total / trades.total.closed * 100 if hasattr(trades, 'won') else 0
            print(f'Trades Total: {trades.total.closed}')
            print(f'Win Rate: {win_rate:.1f}%')
            
            if hasattr(trades.won, 'pnl') and hasattr(trades.lost, 'pnl'):
                avg_win = trades.won.pnl.average if hasattr(trades.won.pnl, 'average') else 0
                avg_loss = trades.lost.pnl.average if hasattr(trades.lost.pnl, 'average') else 0
                if avg_loss != 0:
                    profit_factor = abs(avg_win / avg_loss)
                    print(f'Profit Factor: {profit_factor:.2f}')
        
        # Drawdown
        drawdown = strat.analyzers.drawdown.get_analysis()
        if hasattr(drawdown, 'max'):
            print(f'Drawdown Max: {drawdown.max.drawdown:.2f}%')
            
            # Vérification PropFirm
            if drawdown.max.drawdown <= 5.0 and total_return > 0:
                print('✅ RÈGLES PROPFIRM RESPECTÉES!')
            elif drawdown.max.drawdown <= 5.0:
                print('✅ Drawdown OK - Besoin de plus de profit')
            else:
                print('❌ Drawdown trop élevé pour PropFirm')
        
    except Exception as e:
        print(f'❌ ERREUR: {e}')

if __name__ == '__main__':
    run_optimized_strategy()
'''
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(strategy_code)
        
        print(f"📄 Stratégie optimisée créée: {filename}")
        print(f"   Vous pouvez l'exécuter avec: python {filename}")
        
    except Exception as e:
        print(f"⚠️  Erreur lors de la création du fichier: {e}")

def main():
    """Menu principal"""
    print("🚀 MASTERTREND QUICK OPTIMIZER")
    print("=" * 50)
    print("Optimisation automatique pour maximiser les profits!")
    print()
    
    while True:
        print("Choisissez votre méthode d'optimisation:")
        print("1. 🎯 Optimisation Bayésienne (RECOMMANDÉE - Rapide et efficace)")
        print("2. 🧬 Optimisation Génétique (Plus long mais exhaustif)")
        print("3. 🔥 Les deux méthodes (Comparaison complète)")
        print("4. ❌ Quitter")
        print()
        
        choice = input("Votre choix (1-4): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            params = quick_bayesian_optimization()
            if params:
                create_optimized_strategy_file(params, "bayesian")
            break
            
        elif choice == '2':
            print("\n" + "="*50)
            params = quick_genetic_optimization()
            if params:
                create_optimized_strategy_file(params, "genetic")
            break
            
        elif choice == '3':
            print("\n" + "="*50)
            print("🔥 OPTIMISATION COMPLÈTE - LES DEUX MÉTHODES")
            
            # Bayésienne d'abord (plus rapide)
            print("\n1️⃣ Phase 1: Optimisation Bayésienne")
            bayesian_params = quick_bayesian_optimization()
            
            print("\n2️⃣ Phase 2: Optimisation Génétique")
            genetic_params = quick_genetic_optimization()
            
            # Comparaison
            if bayesian_params and genetic_params:
                print("\n🏆 COMPARAISON DES MÉTHODES")
                print("-" * 30)
                
                print("📊 Test Bayésien:")
                test_strategy("EURUSD_data_15M.csv", bayesian_params)
                
                print("\n📊 Test Génétique:")
                test_strategy("EURUSD_data_15M.csv", genetic_params)
                
                # Créer les deux fichiers
                create_optimized_strategy_file(bayesian_params, "bayesian")
                create_optimized_strategy_file(genetic_params, "genetic")
            
            break
            
        elif choice == '4':
            print("👋 Au revoir!")
            break
            
        else:
            print("❌ Choix invalide. Veuillez choisir 1, 2, 3 ou 4.")
            print()

if __name__ == '__main__':
    main() 