#!/usr/bin/env python3
"""
MASTERTREND AUTO-OPTIMIZER
Optimiseur automatique utilisant des algorithmes génétiques et ML
pour trouver les meilleurs paramètres et maximiser le profit
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import math
from collections import deque
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')


class OptimizedMasterTrend(bt.Strategy):
    """
    Stratégie MasterTrend avec paramètres optimisables
    """
    
    params = (
        # SuperTrend
        ('st_period', 10),
        ('st_multiplier', 3.0),
        
        # MACD
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        
        # Williams Fractals
        ('williams_left', 2),
        ('williams_right', 2),
        ('williams_buffer', 0.05),
        
        # P1 Indicator
        ('p1_e31', 5),
        ('p1_m', 9),
        ('p1_l31', 14),
        
        # StopGap Filter
        ('sg_ema_short', 10),
        ('sg_ema_long', 20),
        ('sg_lookback', 4),
        ('sg_median_period', 10),
        ('sg_multiplier', 1.0),
        
        # Filtres additionnels
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('volume_ma_period', 20),
        ('min_volume_ratio', 1.0),
        
        # Risk Management
        ('max_daily_loss', 0.02),
        ('max_total_loss', 0.05),
        ('profit_target', 0.10),
        ('position_size', 0.01),
        ('max_consecutive_losses', 3),
        
        # Sessions
        ('trading_start_hour', 9),
        ('trading_start_minute', 30),
        ('trading_end_hour', 16),
        ('trading_end_minute', 0),
        
        # Filtres de marché
        ('min_atr_ratio', 0.5),
        ('max_atr_ratio', 2.0),
        ('trend_strength_period', 20),
        ('min_trend_strength', 0.5),
        
        # Nouveaux paramètres optimisables
        ('p1_threshold', 1.05),  # Seuil P1
        ('macd_threshold', 0.0001),  # Seuil MACD
        ('rsi_trend_filter', True),  # Utiliser RSI comme filtre
        ('volume_filter', True),  # Utiliser volume comme filtre
        ('atr_filter', True),  # Utiliser ATR comme filtre
        ('stop_multiplier', 1.0),  # Multiplicateur pour stops
        ('take_profit_ratio', 2.0),  # Ratio take profit / stop loss
    )
    
    def __init__(self):
        # Indicateurs principaux
        self.hl2 = (self.data.high + self.data.low) / 2.0
        self.atr = bt.indicators.ATR(self.data, period=self.p.st_period)
        self.atr_ma = bt.indicators.SMA(self.atr, period=20)
        
        # SuperTrend
        self.basic_up = self.hl2 - self.p.st_multiplier * self.atr
        self.basic_dn = self.hl2 + self.p.st_multiplier * self.atr
        
        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        
        # Volume
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)
        
        # Williams Fractals
        self.fractal_high = bt.indicators.Highest(self.data.high, period=self.p.williams_left + self.p.williams_right + 1)
        self.fractal_low = bt.indicators.Lowest(self.data.low, period=self.p.williams_left + self.p.williams_right + 1)
        
        # P1 Indicator
        self.hilow = (self.data.high - self.data.low) * 100
        self.openclose = (self.data.close - self.data.open) * 100
        self.spreadv = self.openclose * self.data.close
        self.pt_approx = bt.indicators.SMA(self.spreadv, period=75)
        
        self.ema_e31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_e31)
        self.ema_m = bt.indicators.EMA(self.pt_approx, period=self.p.p1_m)
        self.ema_l31 = bt.indicators.EMA(self.pt_approx, period=self.p.p1_l31)
        
        self.a1 = self.ema_l31 - self.ema_m
        self.b1 = self.ema_e31 - self.ema_m
        self.p1 = self.a1 + self.b1
        
        # StopGap Filter
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.sg_ema_long)
        self.highest_sg = bt.indicators.Highest(self.data.high, period=self.p.sg_lookback)
        self.lowest_sg = bt.indicators.Lowest(self.data.low, period=self.p.sg_lookback)
        
        # Trend Strength
        self.close_ma = bt.indicators.SMA(self.data.close, period=self.p.trend_strength_period)
        
        # Variables d'état
        self.trend = 1
        self.final_up = 0.0
        self.final_dn = 0.0
        
        # Williams Stops
        self.williams_long_stop = None
        self.williams_short_stop = None
        self.williams_long_active = False
        self.williams_short_active = False
        
        # Prop Firm Tracking
        self.initial_cash = None
        self.daily_start_cash = None
        self.peak_value = None
        self.current_date = None
        
        # Trade Management
        self.order = None
        self.position_entry_price = None
        self.consecutive_losses = 0
        self.last_trade_profit = 0
        
        # StopGap median calculation
        self.candle_sizes = deque(maxlen=self.p.sg_median_period)
        
        # Métriques pour l'optimisation
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        
    def start(self):
        self.initial_cash = self.broker.getvalue()
        self.daily_start_cash = self.initial_cash
        self.peak_value = self.initial_cash
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.total_trades += 1
            self.total_profit += trade.pnl
            
            if trade.pnl > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            # Calculer drawdown
            current_value = self.broker.getvalue()
            self.peak_value = max(self.peak_value, current_value)
            drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def update_supertrend(self):
        if len(self.data) < 2:
            return
            
        if self.data.close[-1] > self.final_up:
            self.final_up = max(self.basic_up[0], self.final_up)
        else:
            self.final_up = self.basic_up[0]
            
        if self.data.close[-1] < self.final_dn:
            self.final_dn = min(self.basic_dn[0], self.final_dn)
        else:
            self.final_dn = self.basic_dn[0]
            
        if self.trend == -1 and self.data.close[0] > self.final_dn:
            self.trend = 1
        elif self.trend == 1 and self.data.close[0] < self.final_up:
            self.trend = -1
    
    def update_williams_stops(self):
        if len(self.data) > self.p.williams_left + self.p.williams_right:
            center_idx = self.p.williams_right
            
            # High fractal
            is_high_fractal = True
            center_high = self.data.high[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.high[-i] >= center_high:
                    is_high_fractal = False
                    break
            
            if is_high_fractal:
                self.williams_short_stop = center_high * (1 + self.p.williams_buffer / 100)
                self.williams_short_active = True
            
            # Low fractal
            is_low_fractal = True
            center_low = self.data.low[-center_idx]
            
            for i in range(self.p.williams_left + self.p.williams_right + 1):
                if i != center_idx and self.data.low[-i] <= center_low:
                    is_low_fractal = False
                    break
            
            if is_low_fractal:
                self.williams_long_stop = center_low * (1 - self.p.williams_buffer / 100)
                self.williams_long_active = True
    
    def calculate_stopgap_filter(self):
        trend_up = self.ema_short[0] > self.ema_long[0]
        trend_down = self.ema_short[0] < self.ema_long[0]
        
        candle_size = abs(self.data.high[0] - self.data.low[0])
        self.candle_sizes.append(candle_size)
        
        if len(self.candle_sizes) == self.p.sg_median_period:
            median = np.median(list(self.candle_sizes))
        else:
            median = candle_size
        
        stop_gap = 0.0
        if trend_down and self.data.high[0] < self.highest_sg[-1]:
            stop_gap = abs(self.highest_sg[-1] - self.data.low[0])
        elif trend_up and self.data.low[0] > self.lowest_sg[-1]:
            stop_gap = abs(self.data.high[0] - self.lowest_sg[-1])
        
        return stop_gap > (median * self.p.sg_multiplier)
    
    def check_filters(self):
        """Vérification des filtres optimisables"""
        # Filtre RSI
        if self.p.rsi_trend_filter:
            if self.trend == 1 and self.rsi[0] > self.p.rsi_overbought:
                return False
            if self.trend == -1 and self.rsi[0] < self.p.rsi_oversold:
                return False
        
        # Filtre Volume
        if self.p.volume_filter and self.data.volume[0] > 0 and self.volume_ma[0] > 0:
            volume_ratio = self.data.volume[0] / self.volume_ma[0]
            if volume_ratio < self.p.min_volume_ratio:
                return False
        
        # Filtre ATR
        if self.p.atr_filter:
            atr_ratio = self.atr[0] / self.atr_ma[0] if self.atr_ma[0] > 0 else 1.0
            if atr_ratio < self.p.min_atr_ratio or atr_ratio > self.p.max_atr_ratio:
                return False
        
        return True
    
    def check_prop_firm_rules(self):
        if self.initial_cash is None:
            return True
            
        current_value = self.broker.getvalue()
        dt = self.datas[0].datetime.datetime(0)
        
        # Reset daily tracking
        if self.current_date != dt.date():
            self.current_date = dt.date()
            self.daily_start_cash = current_value
        
        # Check consecutive losses
        if self.consecutive_losses >= self.p.max_consecutive_losses:
            return False
        
        # Check daily loss limit
        daily_pnl = (current_value - self.daily_start_cash) / self.initial_cash
        if daily_pnl < -self.p.max_daily_loss:
            return False
        
        # Check total drawdown
        total_dd = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        if total_dd > self.p.max_total_loss:
            return False
        
        # Check profit target
        total_profit = (current_value - self.initial_cash) / self.initial_cash
        if total_profit >= self.p.profit_target:
            return False
        
        return True
    
    def is_trading_session(self):
        dt = self.datas[0].datetime.datetime(0)
        current_time = dt.time()
        start_time = datetime.time(self.p.trading_start_hour, self.p.trading_start_minute)
        end_time = datetime.time(self.p.trading_end_hour, self.p.trading_end_minute)
        return start_time <= current_time <= end_time
    
    def calculate_position_size(self):
        if self.initial_cash is None:
            return 1
            
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.position_size
        
        # Utiliser Williams Stop pour calculer la taille
        if self.williams_long_stop:
            stop_distance = abs(self.data.close[0] - self.williams_long_stop) * self.p.stop_multiplier
        elif self.williams_short_stop:
            stop_distance = abs(self.data.close[0] - self.williams_short_stop) * self.p.stop_multiplier
        else:
            stop_distance = self.data.close[0] * 0.01
        
        if stop_distance > 0:
            size = risk_amount / stop_distance
            max_size = account_value * 0.1 / self.data.close[0]
            size = min(size, max_size)
            return max(1, int(size))
        return 1
    
    def next(self):
        if not self.check_prop_firm_rules() or not self.is_trading_session():
            return
        
        if not self.check_filters():
            return
        
        self.update_supertrend()
        self.update_williams_stops()
        
        # MACD Crossovers avec seuil
        crossmacdbear = (self.macd.macd[0] > self.p.macd_threshold and 
                        self.macd.macd[-1] <= self.p.macd_threshold)
        crossmacd = (self.macd.macd[0] < -self.p.macd_threshold and 
                    self.macd.macd[-1] >= -self.p.macd_threshold)
        
        # P1 Conditions avec seuil optimisable
        b1_ge_p1 = self.b1[0] >= self.p1[0] * self.p.p1_threshold
        b1_le_p1 = self.b1[0] <= self.p1[0] / self.p.p1_threshold
        
        # StopGap Filter
        stopgap_ok = self.calculate_stopgap_filter()
        
        # Conditions de trading
        long_condition = (
            self.williams_long_active and 
            crossmacdbear and 
            b1_ge_p1 and 
            self.trend == 1 and 
            stopgap_ok
        )
        
        short_condition = (
            self.williams_short_active and 
            crossmacd and 
            b1_le_p1 and 
            self.trend == -1 and 
            stopgap_ok
        )
        
        if self.order:
            return
        
        if not self.position:
            if long_condition:
                size = self.calculate_position_size()
                self.order = self.buy(size=size)
                
            elif short_condition:
                size = self.calculate_position_size()
                self.order = self.sell(size=size)
        
        else:  # Gestion des positions ouvertes
            if self.position.size > 0:  # Position longue
                # Stop loss
                if (self.williams_long_stop and 
                    self.data.low[0] <= self.williams_long_stop * self.p.stop_multiplier):
                    self.order = self.close()
                # Take profit
                elif (self.position_entry_price and 
                      self.data.close[0] >= self.position_entry_price * (1 + 0.01 * self.p.take_profit_ratio)):
                    self.order = self.close()
                # Signal de sortie
                elif crossmacd:
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Position courte
                # Stop loss
                if (self.williams_short_stop and 
                    self.data.high[0] >= self.williams_short_stop * self.p.stop_multiplier):
                    self.order = self.close()
                # Take profit
                elif (self.position_entry_price and 
                      self.data.close[0] <= self.position_entry_price * (1 - 0.01 * self.p.take_profit_ratio)):
                    self.order = self.close()
                # Signal de sortie
                elif crossmacdbear:
                    self.order = self.close()
    
    def get_metrics(self):
        """Retourne les métriques pour l'optimisation"""
        final_value = self.broker.getvalue()
        if self.initial_cash is None or self.initial_cash == 0:
            return 0, 0, 0, 0
            
        total_return = (final_value - self.initial_cash) / self.initial_cash
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        profit_factor = abs(self.total_profit) if self.total_profit != 0 else 0
        
        return total_return, win_rate, profit_factor, self.max_drawdown


class GeneticOptimizer:
    """
    Optimiseur génétique pour trouver les meilleurs paramètres
    """
    
    def __init__(self, data_file, population_size=50, generations=20):
        self.data_file = data_file
        self.population_size = population_size
        self.generations = generations
        self.best_params = None
        self.best_fitness = -float('inf')
        self.results_history = []
        
        # Définir les plages de paramètres à optimiser
        self.param_ranges = {
            'st_period': (8, 20),
            'st_multiplier': (1.5, 4.0),
            'macd_fast': (8, 16),
            'macd_slow': (20, 35),
            'macd_signal': (5, 15),
            'williams_left': (1, 5),
            'williams_right': (1, 5),
            'williams_buffer': (0.01, 0.2),
            'p1_e31': (3, 12),
            'p1_m': (6, 18),
            'p1_l31': (10, 25),
            'sg_ema_short': (5, 15),
            'sg_ema_long': (15, 30),
            'sg_lookback': (2, 8),
            'sg_multiplier': (0.5, 2.0),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80),
            'position_size': (0.005, 0.02),
            'p1_threshold': (1.01, 1.15),
            'macd_threshold': (0.00001, 0.001),
            'stop_multiplier': (0.8, 1.5),
            'take_profit_ratio': (1.5, 3.0),
        }
    
    def generate_random_params(self):
        """Génère des paramètres aléatoires"""
        params = {}
        for param, (min_val, max_val) in self.param_ranges.items():
            if param in ['st_period', 'macd_fast', 'macd_slow', 'macd_signal', 
                        'williams_left', 'williams_right', 'p1_e31', 'p1_m', 
                        'p1_l31', 'sg_ema_short', 'sg_ema_long', 'sg_lookback',
                        'rsi_oversold', 'rsi_overbought']:
                params[param] = random.randint(int(min_val), int(max_val))
            else:
                params[param] = random.uniform(min_val, max_val)
        return params
    
    def evaluate_params(self, params):
        """Évalue un ensemble de paramètres"""
        try:
            cerebro = bt.Cerebro()
            
            # Charger les données
            data = bt.feeds.GenericCSVData(
                dataname=self.data_file,
                dtformat=('%Y-%m-%d %H:%M:%S'),
                datetime=0, open=1, high=2, low=3, close=4, volume=5,
                timeframe=bt.TimeFrame.Minutes, compression=15,
                openinterest=-1, headers=True, separator=','
            )
            
            cerebro.adddata(data)
            
            # Ajouter la stratégie avec les paramètres
            cerebro.addstrategy(OptimizedMasterTrend, **params)
            
            # Configuration
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0001)
            
            # Exécuter
            results = cerebro.run()
            strat = results[0]
            
            # Obtenir les métriques
            total_return, win_rate, profit_factor, max_drawdown = strat.get_metrics()
            
            # Fonction de fitness combinée
            # Priorité au profit, puis au win rate, pénalité pour le drawdown
            fitness = (total_return * 100) + (win_rate * 50) - (max_drawdown * 100)
            
            # Bonus si respecte les règles prop firm
            if max_drawdown < 0.05 and total_return > 0:
                fitness += 25
            
            return fitness, total_return, win_rate, max_drawdown, strat.total_trades
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation: {e}")
            return -1000, 0, 0, 1, 0
    
    def crossover(self, parent1, parent2):
        """Croisement entre deux parents"""
        child = {}
        for param in self.param_ranges:
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def mutate(self, params, mutation_rate=0.1):
        """Mutation des paramètres"""
        mutated = params.copy()
        for param, (min_val, max_val) in self.param_ranges.items():
            if random.random() < mutation_rate:
                if param in ['st_period', 'macd_fast', 'macd_slow', 'macd_signal', 
                            'williams_left', 'williams_right', 'p1_e31', 'p1_m', 
                            'p1_l31', 'sg_ema_short', 'sg_ema_long', 'sg_lookback',
                            'rsi_oversold', 'rsi_overbought']:
                    mutated[param] = random.randint(int(min_val), int(max_val))
                else:
                    mutated[param] = random.uniform(min_val, max_val)
        return mutated
    
    def optimize(self):
        """Processus d'optimisation génétique"""
        print("🧬 DÉMARRAGE DE L'OPTIMISATION GÉNÉTIQUE")
        print(f"Population: {self.population_size}, Générations: {self.generations}")
        
        # Génération initiale
        population = []
        for i in range(self.population_size):
            params = self.generate_random_params()
            fitness, ret, wr, dd, trades = self.evaluate_params(params)
            population.append((fitness, params, ret, wr, dd, trades))
            print(f"Individu {i+1}/{self.population_size}: Fitness={fitness:.2f}, Return={ret*100:.2f}%, WinRate={wr*100:.1f}%")
        
        # Trier par fitness
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Évolution
        for generation in range(self.generations):
            print(f"\n🔄 GÉNÉRATION {generation + 1}/{self.generations}")
            
            # Sélection des meilleurs (élitisme)
            elite_size = self.population_size // 4
            new_population = population[:elite_size]
            
            # Génération de nouveaux individus
            while len(new_population) < self.population_size:
                # Sélection des parents (tournoi)
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Croisement
                child_params = self.crossover(parent1[1], parent2[1])
                
                # Mutation
                child_params = self.mutate(child_params)
                
                # Évaluation
                fitness, ret, wr, dd, trades = self.evaluate_params(child_params)
                new_population.append((fitness, child_params, ret, wr, dd, trades))
            
            # Trier la nouvelle population
            population = sorted(new_population, key=lambda x: x[0], reverse=True)
            
            # Afficher le meilleur de cette génération
            best = population[0]
            print(f"Meilleur: Fitness={best[0]:.2f}, Return={best[2]*100:.2f}%, WinRate={best[3]*100:.1f}%, DD={best[4]*100:.2f}%, Trades={best[5]}")
            
            # Sauvegarder l'historique
            self.results_history.append({
                'generation': generation + 1,
                'best_fitness': best[0],
                'best_return': best[2],
                'best_winrate': best[3],
                'best_drawdown': best[4],
                'best_trades': best[5],
                'best_params': best[1].copy()
            })
        
        # Meilleur résultat final
        self.best_fitness, self.best_params, best_return, best_winrate, best_dd, best_trades = population[0]
        
        print(f"\n🏆 OPTIMISATION TERMINÉE!")
        print(f"Meilleur Fitness: {self.best_fitness:.2f}")
        print(f"Meilleur Return: {best_return*100:.2f}%")
        print(f"Meilleur Win Rate: {best_winrate*100:.1f}%")
        print(f"Meilleur Drawdown: {best_dd*100:.2f}%")
        print(f"Nombre de Trades: {best_trades}")
        
        return self.best_params
    
    def tournament_selection(self, population, tournament_size=3):
        """Sélection par tournoi"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x[0])


class OptunaBayesianOptimizer:
    """
    Optimiseur Bayésien utilisant Optuna pour une optimisation plus intelligente
    """
    
    def __init__(self, data_file, n_trials=100):
        self.data_file = data_file
        self.n_trials = n_trials
        self.best_params = None
        self.study = None
    
    def objective(self, trial):
        """Fonction objectif pour Optuna"""
        # Suggérer des paramètres
        params = {
            'st_period': trial.suggest_int('st_period', 8, 20),
            'st_multiplier': trial.suggest_float('st_multiplier', 1.5, 4.0),
            'macd_fast': trial.suggest_int('macd_fast', 8, 16),
            'macd_slow': trial.suggest_int('macd_slow', 20, 35),
            'macd_signal': trial.suggest_int('macd_signal', 5, 15),
            'williams_left': trial.suggest_int('williams_left', 1, 5),
            'williams_right': trial.suggest_int('williams_right', 1, 5),
            'williams_buffer': trial.suggest_float('williams_buffer', 0.01, 0.2),
            'p1_e31': trial.suggest_int('p1_e31', 3, 12),
            'p1_m': trial.suggest_int('p1_m', 6, 18),
            'p1_l31': trial.suggest_int('p1_l31', 10, 25),
            'sg_ema_short': trial.suggest_int('sg_ema_short', 5, 15),
            'sg_ema_long': trial.suggest_int('sg_ema_long', 15, 30),
            'sg_lookback': trial.suggest_int('sg_lookback', 2, 8),
            'sg_multiplier': trial.suggest_float('sg_multiplier', 0.5, 2.0),
            'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 35),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 65, 80),
            'position_size': trial.suggest_float('position_size', 0.005, 0.02),
            'p1_threshold': trial.suggest_float('p1_threshold', 1.01, 1.15),
            'macd_threshold': trial.suggest_float('macd_threshold', 0.00001, 0.001),
            'stop_multiplier': trial.suggest_float('stop_multiplier', 0.8, 1.5),
            'take_profit_ratio': trial.suggest_float('take_profit_ratio', 1.5, 3.0),
        }
        
        try:
            cerebro = bt.Cerebro()
            
            # Charger les données
            data = bt.feeds.GenericCSVData(
                dataname=self.data_file,
                dtformat=('%Y-%m-%d %H:%M:%S'),
                datetime=0, open=1, high=2, low=3, close=4, volume=5,
                timeframe=bt.TimeFrame.Minutes, compression=15,
                openinterest=-1, headers=True, separator=','
            )
            
            cerebro.adddata(data)
            cerebro.addstrategy(OptimizedMasterTrend, **params)
            cerebro.broker.setcash(10000.0)
            cerebro.broker.setcommission(commission=0.0001)
            
            results = cerebro.run()
            strat = results[0]
            
            total_return, win_rate, profit_factor, max_drawdown = strat.get_metrics()
            
            # Fonction objectif multi-critères
            # Maximiser le profit et le win rate, minimiser le drawdown
            objective_value = (total_return * 100) + (win_rate * 50) - (max_drawdown * 100)
            
            # Bonus pour respecter les règles prop firm
            if max_drawdown < 0.05 and total_return > 0:
                objective_value += 25
            
            # Pénalité si pas assez de trades
            if strat.total_trades < 5:
                objective_value -= 20
            
            return objective_value
            
        except Exception as e:
            return -1000
    
    def optimize(self):
        """Optimisation Bayésienne avec Optuna"""
        print("🎯 DÉMARRAGE DE L'OPTIMISATION BAYÉSIENNE (OPTUNA)")
        print(f"Nombre d'essais: {self.n_trials}")
        
        # Créer l'étude
        self.study = optuna.create_study(direction='maximize')
        
        # Optimiser
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Meilleurs paramètres
        self.best_params = self.study.best_params
        
        print(f"\n🏆 OPTIMISATION BAYÉSIENNE TERMINÉE!")
        print(f"Meilleure valeur objective: {self.study.best_value:.2f}")
        print(f"Meilleurs paramètres: {self.best_params}")
        
        return self.best_params


def run_optimization():
    """Fonction principale d'optimisation"""
    data_file = "EURUSD_data_15M.csv"
    
    print("🚀 MASTERTREND AUTO-OPTIMIZER")
    print("=" * 50)
    
    # Choix de la méthode d'optimisation
    method = input("Choisissez la méthode d'optimisation:\n1. Algorithme Génétique\n2. Optimisation Bayésienne (Optuna)\n3. Les deux\nChoix (1/2/3): ")
    
    results = {}
    
    if method in ['1', '3']:
        print("\n🧬 OPTIMISATION GÉNÉTIQUE")
        genetic_optimizer = GeneticOptimizer(data_file, population_size=30, generations=15)
        genetic_params = genetic_optimizer.optimize()
        results['genetic'] = genetic_params
    
    if method in ['2', '3']:
        print("\n🎯 OPTIMISATION BAYÉSIENNE")
        bayesian_optimizer = OptunaBayesianOptimizer(data_file, n_trials=50)
        bayesian_params = bayesian_optimizer.optimize()
        results['bayesian'] = bayesian_params
    
    # Tester les meilleurs paramètres
    print("\n📊 TEST DES MEILLEURS PARAMÈTRES")
    for method_name, params in results.items():
        print(f"\n--- Test {method_name.upper()} ---")
        test_strategy(data_file, params)
    
    return results


def test_strategy(data_file, params):
    """Teste une stratégie avec des paramètres donnés"""
    cerebro = bt.Cerebro()
    
    data = bt.feeds.GenericCSVData(
        dataname=data_file,
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        timeframe=bt.TimeFrame.Minutes, compression=15,
        openinterest=-1, headers=True, separator=','
    )
    
    cerebro.adddata(data)
    cerebro.addstrategy(OptimizedMasterTrend, **params)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Analyseurs
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    results = cerebro.run()
    strat = results[0]
    
    # Afficher les résultats
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - 10000) / 10000 * 100
    
    print(f"Capital Final: ${final_value:.2f}")
    print(f"Rendement Total: {total_return:.2f}%")
    
    trades = strat.analyzers.trades.get_analysis()
    if hasattr(trades, 'total') and trades.total.closed > 0:
        win_rate = trades.won.total / trades.total.closed * 100 if hasattr(trades, 'won') else 0
        print(f"Trades Total: {trades.total.closed}")
        print(f"Win Rate: {win_rate:.1f}%")
    
    drawdown = strat.analyzers.drawdown.get_analysis()
    if hasattr(drawdown, 'max'):
        print(f"Drawdown Max: {drawdown.max.drawdown:.2f}%")


if __name__ == '__main__':
    run_optimization() 