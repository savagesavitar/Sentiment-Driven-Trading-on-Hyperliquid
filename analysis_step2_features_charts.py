import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

OUT = r'B:\Internshala'

# ── Load merged data ──
df = pd.read_csv(f'{OUT}\\merged_data.csv')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df['date'] = pd.to_datetime(df['date'])

print(f"Loaded: {df.shape[0]} trades across {df['date'].nunique()} days")

# ════════════════════════════════════════════════════════════════
# STEP 1: ENGINEER FEATURES
# ════════════════════════════════════════════════════════════════
print("\n=== FEATURE ENGINEERING ===")

# 1a. Win/Loss flag
df['is_win'] = (df['Closed PnL'] > 0).astype(int)
df['is_loss'] = (df['Closed PnL'] < 0).astype(int)
df['is_breakeven'] = (df['Closed PnL'] == 0).astype(int)

# 1b. PnL as % of position size (use absolute start position as proxy for position size)
df['abs_position'] = df['Start Position'].abs().replace(0, np.nan)
df['pnl_pct'] = (df['Closed PnL'] / df['abs_position'].clip(lower=1)) * 100
df['pnl_pct'] = df['pnl_pct'].clip(-500, 500)  # clip extreme outliers for stability

# 1c. Leverage proxy: size_usd / abs(start_position)
# When Start Position is 0 (new position), leverage is undefined
df['leverage_proxy'] = df['Size USD'] / df['abs_position']
df['leverage_proxy'] = df['leverage_proxy'].replace([np.inf, -np.inf], np.nan)
df['leverage_proxy'] = df['leverage_proxy'].clip(0, 200)

# 1d. Leverage buckets
def leverage_bucket(lev):
    if pd.isna(lev) or lev <= 0:
        return 'Unknown'
    elif lev <= 2:
        return '1-2x'
    elif lev <= 5:
        return '3-5x'
    elif lev <= 10:
        return '6-10x'
    elif lev <= 25:
        return '11-25x'
    else:
        return '25x+'

df['leverage_bucket'] = df['leverage_proxy'].apply(leverage_bucket)

# 1e. Trade frequency per account per day
freq = df.groupby(['Account', 'date']).size().reset_index(name='trade_freq')
df = df.merge(freq, on=['Account', 'date'], how='left')

# 1f. Sentiment numeric mapping for analysis
sentiment_order = {'Extreme Fear': 1, 'Fear': 2, 'Neutral': 3, 'Greed': 4, 'Extreme Greed': 5}
df['sentiment_score'] = df['classification'].map(sentiment_order)

# 1g. Direction flag
df['is_long'] = (df['Side'] == 'BUY').astype(int)
df['is_short'] = (df['Side'] == 'SELL').astype(int)

# 1h. Net PnL per account per day per sentiment
daily_account = df.groupby(['Account', 'date', 'classification', 'value']).agg(
    total_pnl=('Closed PnL', 'sum'),
    total_trades=('Closed PnL', 'count'),
    wins=('is_win', 'sum'),
    losses=('is_loss', 'sum'),
    avg_trade_size=('Size USD', 'mean'),
    avg_leverage=('leverage_proxy', 'mean'),
    long_count=('is_long', 'sum'),
    short_count=('is_short', 'sum'),
    total_fee=('Fee', 'sum')
).reset_index()

daily_account['win_rate'] = daily_account['wins'] / daily_account['total_trades']
daily_account['long_bias'] = daily_account['long_count'] / (daily_account['long_count'] + daily_account['short_count'])
daily_account['net_pnl_after_fees'] = daily_account['total_pnl'] - daily_account['total_fee']

# Save engineered data
df.to_csv(f'{OUT}\\engineered_data.csv', index=False)
daily_account.to_csv(f'{OUT}\\daily_account_summary.csv', index=False)

print(f"Engineered data saved: {df.shape}")
print(f"Daily account summary saved: {daily_account.shape}")
print(f"\nLeverage bucket distribution:")
print(df['leverage_bucket'].value_counts().to_string())
print(f"\nWin rate overall: {df['is_win'].mean():.3f}")
print(f"Avg PnL per trade: ${df['Closed PnL'].mean():.2f}")
print(f"Avg trade size: ${df['Size USD'].mean():.2f}")

# ════════════════════════════════════════════════════════════════
# STEP 2: EXPLORATORY ANALYSIS WITH VISUALIZATIONS
# ════════════════════════════════════════════════════════════════
print("\n=== EXPLORATORY ANALYSIS ===")

# Define color palette for sentiment categories
colors = {
    'Extreme Fear': '#d32f2f',
    'Fear': '#ff9800',
    'Neutral': '#9e9e9e',
    'Greed': '#4caf50',
    'Extreme Greed': '#1b5e20'
}
cat_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

# ── CHART 1: PnL Distribution by Sentiment ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of PnL
data_for_box = [df[df['classification'] == c]['Closed PnL'].values for c in cat_order]
bp = axes[0].boxplot(data_for_box, labels=cat_order, patch_artist=True, showfliers=False,
                      medianprops=dict(color='black', linewidth=1.5))
for patch, cat in zip(bp['boxes'], cat_order):
    patch.set_facecolor(colors[cat])
    patch.set_alpha(0.7)
axes[0].set_title('Closed PnL Distribution by Market Sentiment', fontweight='bold')
axes[0].set_ylabel('Closed PnL ($)')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
axes[0].tick_params(axis='x', rotation=15)

# Mean PnL bar chart
mean_pnl = df.groupby('classification')['Closed PnL'].mean().reindex(cat_order)
axes[1].bar(cat_order, mean_pnl.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_title('Average Closed PnL by Market Sentiment', fontweight='bold')
axes[1].set_ylabel('Mean Closed PnL ($)')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(mean_pnl.values):
    axes[1].text(i, v + (2 if v >= 0 else -8), f'${v:.1f}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}\\chart1_pnl_by_sentiment.png', bbox_inches='tight')
plt.close()
print("Chart 1 saved: PnL by sentiment")

# ── CHART 2: Win Rate by Sentiment ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Win rate per trade
win_rate_trade = df.groupby('classification')['is_win'].mean().reindex(cat_order) * 100
axes[0].bar(cat_order, win_rate_trade.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_title('Win Rate (Trade-Level) by Market Sentiment', fontweight='bold')
axes[0].set_ylabel('Win Rate (%)')
axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(win_rate_trade.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Win rate per account-day
win_rate_daily = daily_account.groupby('classification')['win_rate'].mean().reindex(cat_order) * 100
axes[1].bar(cat_order, win_rate_daily.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_title('Avg Daily Win Rate per Account by Sentiment', fontweight='bold')
axes[1].set_ylabel('Avg Daily Win Rate (%)')
axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(win_rate_daily.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}\\chart2_winrate_by_sentiment.png', bbox_inches='tight')
plt.close()
print("Chart 2 saved: Win rate by sentiment")

# ── CHART 3: Leverage Usage by Sentiment ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Avg leverage by sentiment
avg_lev = df[df['leverage_proxy'] > 0].groupby('classification')['leverage_proxy'].mean().reindex(cat_order)
axes[0].bar(cat_order, avg_lev.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_title('Average Leverage by Market Sentiment', fontweight='bold')
axes[0].set_ylabel('Avg Leverage (x)')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(avg_lev.values):
    axes[0].text(i, v + 0.2, f'{v:.1f}x', ha='center', fontsize=9, fontweight='bold')

# Leverage bucket distribution as stacked bar
lev_cross = pd.crosstab(df['classification'], df['leverage_bucket'], normalize='index') * 100
bucket_order = ['1-2x', '3-5x', '6-10x', '11-25x', '25x+', 'Unknown']
bucket_order = [b for b in bucket_order if b in lev_cross.columns]
lev_cross = lev_cross.reindex(cat_order)[bucket_order]
bucket_colors = ['#1976d2', '#42a5f5', '#ffca28', '#ff9800', '#f44336', '#bdbdbd']
lev_cross.plot(kind='bar', stacked=True, ax=axes[1], color=bucket_colors[:len(bucket_order)], alpha=0.85, edgecolor='black', linewidth=0.3)
axes[1].set_title('Leverage Distribution by Market Sentiment', fontweight='bold')
axes[1].set_ylabel('Percentage of Trades (%)')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(title='Leverage', bbox_to_anchor=(1.0, 1), fontsize=7)

plt.tight_layout()
plt.savefig(f'{OUT}\\chart3_leverage_by_sentiment.png', bbox_inches='tight')
plt.close()
print("Chart 3 saved: Leverage by sentiment")

# ── CHART 4: Long/Short Bias by Sentiment ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Long vs Short proportion
ls_count = df.groupby(['classification', 'Side']).size().unstack(fill_value=0)
ls_pct = ls_count.div(ls_count.sum(axis=1), axis=0) * 100
ls_pct = ls_pct.reindex(cat_order)
if 'BUY' in ls_pct.columns and 'SELL' in ls_pct.columns:
    x = np.arange(len(cat_order))
    width = 0.35
    axes[0].bar(x - width/2, ls_pct['BUY'], width, label='Long (BUY)', color='#4caf50', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].bar(x + width/2, ls_pct['SELL'], width, label='Short (SELL)', color='#f44336', alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cat_order, rotation=15)
    axes[0].set_title('Long vs Short Ratio by Sentiment', fontweight='bold')
    axes[0].set_ylabel('Percentage of Trades (%)')
    axes[0].legend()
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5)

# Long bias per account-day
long_bias_daily = daily_account.groupby('classification')['long_bias'].mean().reindex(cat_order) * 100
axes[1].bar(cat_order, long_bias_daily.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_title('Avg Long Bias per Account-Day by Sentiment', fontweight='bold')
axes[1].set_ylabel('Long Bias (%)')
axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% = neutral')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(long_bias_daily.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}\\chart4_longshort_by_sentiment.png', bbox_inches='tight')
plt.close()
print("Chart 4 saved: Long/Short bias by sentiment")

# ── CHART 5: Trade Volume & Frequency ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Total trade count
trade_count = df.groupby('classification').size().reindex(cat_order)
axes[0].bar(cat_order, trade_count.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_title('Total Trade Count by Sentiment', fontweight='bold')
axes[0].set_ylabel('Number of Trades')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(trade_count.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=8, fontweight='bold')

# Avg trade frequency per account
avg_freq = df.groupby('classification')['trade_freq'].mean().reindex(cat_order)
axes[1].bar(cat_order, avg_freq.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_title('Avg Trade Frequency per Account per Day', fontweight='bold')
axes[1].set_ylabel('Avg Trades/Account/Day')
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(avg_freq.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}\\chart5_volume_frequency.png', bbox_inches='tight')
plt.close()
print("Chart 5 saved: Volume and frequency")

# ── CHART 6: PnL % by Sentiment ──
fig, ax = plt.subplots(figsize=(10, 5))
pnl_pct_data = [df[(df['classification'] == c) & (df['pnl_pct'].between(-50, 50))]['pnl_pct'].values for c in cat_order]
bp = ax.boxplot(pnl_pct_data, labels=cat_order, patch_artist=True, showfliers=False,
                medianprops=dict(color='black', linewidth=1.5))
for patch, cat in zip(bp['boxes'], cat_order):
    patch.set_facecolor(colors[cat])
    patch.set_alpha(0.7)
ax.set_title('PnL as % of Position Size by Market Sentiment (clipped -50% to +50%)', fontweight='bold')
ax.set_ylabel('PnL (%)')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}\\chart6_pnl_pct_by_sentiment.png', bbox_inches='tight')
plt.close()
print("Chart 6 saved: PnL% by sentiment")

# ── CHART 7: Sentiment Value Distribution Over Time ──
fig, ax = plt.subplots(figsize=(14, 4))
daily_sentiment = df.groupby('date').agg(
    avg_value=('value', 'mean'),
    avg_pnl=('Closed PnL', 'mean'),
    trade_count=('Closed PnL', 'count')
).reset_index()
ax.plot(daily_sentiment['date'], daily_sentiment['avg_value'], color='#1976d2', alpha=0.6, linewidth=0.8, label='Sentiment Value')
ax2 = ax.twinx()
ax2.plot(daily_sentiment['date'], daily_sentiment['avg_pnl'].rolling(7).mean(), color='#f44336', alpha=0.7, linewidth=1.2, label='Avg PnL (7d MA)')
ax.set_title('Market Sentiment vs Avg Trader PnL Over Time', fontweight='bold')
ax.set_ylabel('Fear/Greed Index Value')
ax2.set_ylabel('Avg Closed PnL ($)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}\\chart7_sentiment_vs_pnl_timeline.png', bbox_inches='tight')
plt.close()
print("Chart 7 saved: Sentiment vs PnL timeline")

print("\n=== ALL CHARTS GENERATED ===")
