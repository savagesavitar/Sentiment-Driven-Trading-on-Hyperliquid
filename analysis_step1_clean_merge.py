import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load raw data ──
hist = pd.read_csv(r'B:\Internshala\historical_data.csv')
fg = pd.read_csv(r'B:\Internshala\fear_greed.csv')

print("Historical raw shape:", hist.shape)
print("Fear/Greed raw shape:", fg.shape)

# ── 2. Clean historical data ──
# Parse trade date from "Timestamp IST" (format: DD-MM-YYYY HH:MM)
hist['trade_date'] = pd.to_datetime(hist['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
hist = hist.dropna(subset=['trade_date'])
hist['date'] = hist['trade_date'].dt.strftime('%Y-%m-%d')

# Clean numeric columns
hist['Execution Price'] = pd.to_numeric(hist['Execution Price'], errors='coerce')
hist['Size Tokens'] = pd.to_numeric(hist['Size Tokens'], errors='coerce')
hist['Size USD'] = pd.to_numeric(hist['Size USD'], errors='coerce')
hist['Start Position'] = pd.to_numeric(hist['Start Position'], errors='coerce')
hist['Closed PnL'] = pd.to_numeric(hist['Closed PnL'], errors='coerce')
hist['Fee'] = pd.to_numeric(hist['Fee'], errors='coerce')

# Drop rows with critical NaNs
hist = hist.dropna(subset=['Execution Price', 'Size USD', 'Closed PnL'])

print(f"\nAfter cleaning historical: {hist.shape[0]} rows, date range: {hist['date'].min()} to {hist['date'].max()}")

# ── 3. Clean fear/greed data ──
fg['date'] = pd.to_datetime(fg['date']).dt.strftime('%Y-%m-%d')
fg = fg[['date', 'value', 'classification']].drop_duplicates(subset='date')

print(f"Fear/Greed cleaned: {fg.shape[0]} unique dates")
print(f"Classifications: {fg['classification'].value_counts().to_dict()}")

# ── 4. Merge: left join trades onto daily sentiment ──
merged = hist.merge(fg[['date', 'value', 'classification']], on='date', how='left')

# Drop trades with no sentiment match
matched = merged.dropna(subset=['classification'])
unmatched = merged[merged['classification'].isna()]

print(f"\nMerged: {matched.shape[0]} trades matched ({matched.shape[0]/hist.shape[0]*100:.1f}%)")
print(f"Unmatched: {unmatched.shape[0]} trades (no sentiment data for those dates)")

# ── 5. Save merged dataset ──
matched.to_csv(r'B:\Internshala\merged_data.csv', index=False)
print(f"\nSaved merged_data.csv: {matched.shape}")

# Quick summary
print("\n=== MERGED DATA SUMMARY ===")
print(matched[['Execution Price', 'Size USD', 'Closed PnL', 'Start Position']].describe().round(2))
print(f"\nClassification distribution in matched trades:")
print(matched['classification'].value_counts())
