# MasterTrend Strategy

Cette implémentation est une conversion de la stratégie MasterTrend depuis PineScript (TradingView) vers Python/Backtrader.

## Description

MasterTrend est une stratégie de trading qui combine plusieurs indicateurs techniques pour prendre des décisions d'achat et de vente. La stratégie utilise:

- MACD avec Jurik Moving Average (JMA)
- SuperTrend
- Fractales de Williams
- Sessions de trading (Londres, New York, Tokyo, Sydney)
- Filtres basés sur RSI et Volume

## Installation

### Prérequis
- Python 3.6+
- Les bibliothèques suivantes:
  - backtrader
  - pandas
  - numpy
  - matplotlib
  - pytz

### Installation des dépendances

```bash
pip install backtrader pandas numpy matplotlib pytz
```

## Structure des Fichiers

- `mastertrend_strategy.py` - Implémentation principale de la stratégie
- `test_mastertrend.py` - Script pour tester la stratégie
- `README_MASTERTREND.md` - Ce fichier

## Données

La stratégie nécessite des données au format CSV avec les colonnes suivantes:
- datetime (format: 'YYYY-MM-DD HH:MM:SS')
- open
- high
- low
- close
- volume

## Utilisation

### Exécution du Test Simple

```bash
python test_mastertrend.py
```

Ce script teste d'abord les indicateurs personnalisés sur des données aléatoires, puis exécute la stratégie complète sur un fichier de données EURUSD si disponible.

### Personnalisation de la Stratégie

Pour modifier les paramètres de la stratégie, vous pouvez éditer le dictionnaire `test_params` dans `test_mastertrend.py` ou créer votre propre script:

```python
import backtrader as bt
from mastertrend_strategy import MasterTrendStrategy

# Créer un Cerebro
cerebro = bt.Cerebro()

# Ajouter les données
data = bt.feeds.GenericCSVData(
    dataname="votre_fichier_data.csv",
    dtformat=('%Y-%m-%d %H:%M:%S'),
    datetime=0, open=1, high=2, low=3, close=4, volume=5,
    timeframe=bt.TimeFrame.Minutes,
    compression=1,
    openinterest=-1
)
cerebro.adddata(data)

# Ajouter la stratégie avec vos paramètres
cerebro.addstrategy(MasterTrendStrategy,
    line1_macd=13,
    line2_macd=26,
    supertrend_period=10,
    supertrend_multiplier=3,
    jma_length=7,
    jma_phase=50,
    london_session=True,
    ny_session=True,
    tokyo_session=True
)

# Définir le capital initial et les commissions
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.0001)

# Exécuter la stratégie
results = cerebro.run()

# Optionnel: afficher un graphique
cerebro.plot()
```

## Paramètres Configurables

Les principaux paramètres de la stratégie sont:

### Paramètres MACD
- `line1_macd` (défaut: 13) - Période courte du MACD
- `line2_macd` (défaut: 26) - Période longue du MACD

### Paramètres SuperTrend
- `supertrend_period` (défaut: 10) - Période de l'ATR pour SuperTrend
- `supertrend_multiplier` (défaut: 3) - Multiplicateur d'ATR

### Paramètres JMA
- `jma_length` (défaut: 7) - Période du Jurik Moving Average
- `jma_phase` (défaut: 50) - Phase du JMA
- `jma_power` (défaut: 2) - Puissance du JMA

### Paramètres des pivots
- `pivot_length` (défaut: 10) - Longueur des pivots
- `left_range` (défaut: 2) - Nombre de barres à gauche pour les fractales de Williams
- `right_range` (défaut: 2) - Nombre de barres à droite pour les fractales de Williams

### Sessions de trading
- `london_session` (défaut: True) - Activer/désactiver la session de Londres
- `london_hours` (défaut: '0300-1200') - Heures de la session (format: 'HHMM-HHMM')
- `ny_session` (défaut: True) - Activer/désactiver la session de New York
- `ny_hours` (défaut: '0800-1700')
- `tokyo_session` (défaut: True) - Activer/désactiver la session de Tokyo
- `tokyo_hours` (défaut: '2000-0400')
- `sydney_session` (défaut: False) - Activer/désactiver la session de Sydney
- `sydney_hours` (défaut: '1700-0200')

### Autres paramètres
- `rsi_length` (défaut: 14) - Période du RSI
- `volume_mult` (défaut: 1.5) - Multiplicateur pour le volume
- `atr_length` (défaut: 14) - Période de l'ATR
- `williams_stop_buffer` (défaut: 0.0) - Buffer pour les stops des fractales de Williams

## Comparaison avec PineScript

Cette implémentation Backtrader cherche à rester fidèle à la version PineScript originale, mais il existe quelques différences:

1. **Affichage graphique**: La version PineScript inclut plus d'éléments visuels que cette conversion qui se concentre sur la logique de trading.

2. **Sessions de trading**: L'implémentation des sessions utilise pytz pour gérer les fuseaux horaires.

3. **Indicateurs personnalisés**: JMA, JMACD et WilliamsFractal ont été recréés à partir de zéro pour Backtrader.

## Avertissement

Le trading comporte des risques. Cette stratégie est fournie à titre éducatif uniquement. Effectuez toujours vos propres tests et analyses avant d'utiliser une stratégie avec de l'argent réel. 