from pathlib import Path
import sqlite3
import pandas as pd

base = Path(__file__).parent.parent
db = base / 'data' / 'processed' / 'feature_store.db'
csv = base / 'data' / 'processed' / 'london_features.csv'
print('DB path:', db)
print('CSV path:', csv)
if db.exists():
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM aqi_features")
        n = cur.fetchone()[0]
        print('DB rows:', n)
    except Exception as e:
        print('DB error:', e)
    conn.close()
else:
    print('DB not found')

if csv.exists():
    df = pd.read_csv(csv, parse_dates=['timestamp'])
    print('CSV rows:', len(df))
else:
    print('CSV not found')
