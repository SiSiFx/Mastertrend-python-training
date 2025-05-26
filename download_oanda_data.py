#!/usr/bin/env python3
"""
Téléchargement des données forex depuis OANDA avec intervalle 1-minute
sur une période d'un mois, compatible avec le backtest ISI_MAXI_WEEP.
"""
import datetime as dt
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time
import argparse

# Configuration OANDA
ACCESS_TOKEN = "3240e71f449497ac70a3abce8989ef0a-631d288e0d87be637ba3a3a41ff998fd"
client = API(access_token=ACCESS_TOKEN)

# Parse CLI args
parser = argparse.ArgumentParser(description='Download 1M data from OANDA')
parser.add_argument('--instrument', default='EUR_USD', help='OANDA instrument code, e.g. EUR_USD')
parser.add_argument('--granularity', default='M1', help='Granularity, e.g. M1, H1')
parser.add_argument('--days', type=int, default=30, help='Days back to download')
parser.add_argument('--account-id', default=None, help='OANDA account ID (auto-detected if not provided)')
args = parser.parse_args()

INSTRUMENT = args.instrument
GRANULARITY = args.granularity
OUTPUT_FILE = f"{INSTRUMENT.replace('_', '')}_data_1M.csv"
ACCOUNT_ID = args.account_id or ""

# Essayer de récupérer l'ID de compte automatiquement
if not ACCOUNT_ID:
    try:
        from oandapyV20.endpoints.accounts import AccountList
        r = AccountList()
        client.request(r)
        accounts = r.response.get('accounts', [])
        if accounts:
            ACCOUNT_ID = accounts[0].get('id')
            print(f"ID de compte détecté: {ACCOUNT_ID}")
        else:
            print("Aucun compte trouvé. Continuer sans ACCOUNT_ID.")
    except Exception as e:
        print(f"Impossible de récupérer l'ID de compte: {e}")
        print("Continuer sans ACCOUNT_ID.")

# Paramètres de téléchargement
END_DATE = dt.datetime.utcnow()
START_DATE = END_DATE - dt.timedelta(days=args.days)

print(f"Téléchargement des données {INSTRUMENT} en {GRANULARITY} du {START_DATE} au {END_DATE}")
print(f"Les données seront sauvegardées dans: {OUTPUT_FILE}")

# Fonction pour convertir les timestamps
def parse_time(time_str):
    return dt.datetime.strptime(time_str.replace('.000000000Z', ''), '%Y-%m-%dT%H:%M:%S')

# Fonction pour télécharger par segments (OANDA a des limites de 5000 bougies par requête)
def download_data(instrument, granularity, start, end):
    all_candles = []
    current_start = start
    
    while current_start < end:
        # Calculer une période maximale (pour rester sous la limite de 5000 bougies)
        # Pour M1 (1-minute), 5000 minutes = environ 3.5 jours
        batch_end = min(current_start + dt.timedelta(days=3), end)
        
        params = {
            "granularity": granularity,
            "from": current_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "to": batch_end.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
        
        print(f"Téléchargement du segment: {params['from']} à {params['to']}")
        
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            client.request(r)
            candles = r.response.get('candles', [])
            if not candles:
                print("Aucune donnée disponible pour cette période.")
                break
                
            all_candles.extend(candles)
            print(f"Téléchargé {len(candles)} bougies")
            
            # Attendre un peu pour ne pas dépasser les limites de l'API
            time.sleep(1)
            
            # Préparer la prochaine itération
            current_start = batch_end
        except Exception as e:
            print(f"Erreur lors du téléchargement: {e}")
            # Attendre plus longtemps en cas d'erreur
            time.sleep(5)
    
    return all_candles

# Télécharger les données
all_data = download_data(INSTRUMENT, GRANULARITY, START_DATE, END_DATE)

if not all_data:
    print("Aucune donnée n'a été téléchargée. Vérifiez vos paramètres et votre connexion.")
    exit(1)

# Transformer les données au format attendu par ISI_MAXI_WEEP
rows = []
for candle in all_data:
    if candle['complete']:  # Ne prendre que les bougies complètes
        time_str = parse_time(candle['time'])
        rows.append({
            'datetime': time_str,
            'open': float(candle['mid']['o']),
            'high': float(candle['mid']['h']),
            'low': float(candle['mid']['l']),
            'close': float(candle['mid']['c']),
            'volume': 0  # OANDA ne fournit pas de volume, mais backtrader en a besoin
        })

# Créer un DataFrame et sauvegarder en CSV
if rows:
    df = pd.DataFrame(rows)
    df = df.sort_values('datetime')  # S'assurer que les données sont dans l'ordre chronologique
    
    # Formatage de la colonne datetime comme attendu par ISI_MAXI_WEEP
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Sauvegarder en CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Données sauvegardées avec succès dans {OUTPUT_FILE}")
    print(f"Période couverte: {df['datetime'].iloc[0]} à {df['datetime'].iloc[-1]}")
    print(f"Nombre total de bougies: {len(df)}")
else:
    print("Aucune donnée valide n'a été trouvée.") 