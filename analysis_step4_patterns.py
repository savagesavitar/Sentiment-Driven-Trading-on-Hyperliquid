import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

OUT = r'B:\Internshala'

df = pd.read_csv(f'{OUT}\\engineered_data.csv')
daily = pd.read_csv(f'{OUT}\\daily_account_summary.csv')
daily['date'] = pd.to_datetime(daily['date'])

cat_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
colors = {
    'Extreme Fear': '#d32f2f', 'Fear': '#ff9800',
    'Neutral': '#9e9e9e', 'Greed': '#4caf50', 'Extreme Greed': '#1b5e20'
}

print("=" * 70)
print("PATTERN IDENTIFICATION & HIDDEN INSIGHTS")
print("=" * 70)

# ════════════════════════════════════════════════════════════════
# PATTERN 1: Contrarian traders who outperform during Fear
# ════════════════════════════════════════════════════════════════
print("\n--- PATTERN 1: Contrarian Traders During Fear ---")

# For each account, classify their dominant bias during Greed vs Fear
account_behavior = df.groupby(['Account', 'classification']).agg(
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('is_win', 'mean'),
    long_bias=('is_long', 'mean'),
    avg_leverage=('leverage_proxy', lambda x: x[x > 0].mean() if (x > 0).any() else np.nan),
    trade_count=('Closed PnL', 'count')
).reset_index()

# Find accounts active in both Fear and Greed periods
fear_accounts = set(account_behavior[account_behavior['classification'].isin(['Fear', 'Extreme Fear'])]['Account'])
greed_accounts = set(account_behavior[account_behavior['classification'].isin(['Greed', 'Extreme Greed'])]['Account'])
both_accounts = fear_accounts & greed_accounts

print(f"  Accounts active during Fear periods: {len(fear_accounts)}")
print(f"  Accounts active during Greed periods: {len(greed_accounts)}")
print(f"  Accounts active in BOTH: {len(both_accounts)}")

# Identify contrarians: short-biased during greed, long-biased during fear
contrarians = []
for acc in both_accounts:
    acc_data = account_behavior[account_behavior['Account'] == acc]
    
    fear_data = acc_data[acc_data['classification'].isin(['Fear', 'Extreme Fear'])]
    greed_data = acc_data[acc_data['classification'].isin(['Greed', 'Extreme Greed'])]
    
    if len(fear_data) > 0 and len(greed_data) > 0:
        fear_long_bias = fear_data['long_bias'].mean()
        greed_long_bias = greed_data['long_bias'].mean()
        fear_pnl = fear_data['avg_pnl'].mean()
        greed_pnl = greed_data['avg_pnl'].mean()
        fear_winrate = fear_data['win_rate'].mean()
        
        # Contrarian: more short during greed (low long_bias), more long during fear (high long_bias)
        is_contrarian = (greed_long_bias < 0.5) and (fear_long_bias > 0.5)
        
        if is_contrarian and fear_pnl > 0:
            contrarians.append({
                'Account': acc[:12] + '...',
                'Fear_LongBias': fear_long_bias,
                'Greed_LongBias': greed_long_bias,
                'Fear_AvgPnL': fear_pnl,
                'Greed_AvgPnL': greed_pnl,
                'Fear_WinRate': fear_winrate
            })

contrarians_df = pd.DataFrame(contrarians)
print(f"\n  Contrarian traders (short during greed, long during fear, profitable in fear): {len(contrarians_df)}")
if len(contrarians_df) > 0:
    print(contrarians_df.to_string(index=False))

# Non-contrarian comparison
non_contrarians_pnl = []
contrarians_pnl = []
for acc in both_accounts:
    acc_data = account_behavior[account_behavior['Account'] == acc]
    fear_data = acc_data[acc_data['classification'].isin(['Fear', 'Extreme Fear'])]
    greed_data = acc_data[acc_data['classification'].isin(['Greed', 'Extreme Greed'])]
    if len(fear_data) > 0:
        acc_is_contrarian = any(c['Account'].startswith(acc[:12]) for c in contrarians)
        fear_pnl = fear_data['avg_pnl'].mean()
        if acc_is_contrarian:
            contrarians_pnl.append(fear_pnl)
        else:
            non_contrarians_pnl.append(fear_pnl)

if contrarians_pnl and non_contrarians_pnl:
    print(f"\n  Avg Fear PnL - Contrarians:     ${np.mean(contrarians_pnl):.2f}")
    print(f"  Avg Fear PnL - Non-Contrarians: ${np.mean(non_contrarians_pnl):.2f}")

# ════════════════════════════════════════════════════════════════
# PATTERN 2: Lag effects between sentiment and next-day performance
# ════════════════════════════════════════════════════════════════
print("\n--- PATTERN 2: Lag Effects (Sentiment Today -> Performance Tomorrow) ---")

# Create daily-level data with lagged sentiment
daily_sent = df.groupby('date').agg(
    sentiment_value=('value', 'mean'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('is_win', 'mean'),
    avg_leverage=('leverage_proxy', lambda x: x[x > 0].mean() if (x > 0).any() else np.nan),
    total_pnl=('Closed PnL', 'sum'),
    trade_count=('Closed PnL', 'count')
).reset_index().sort_values('date')

daily_sent['next_day_pnl'] = daily_sent['avg_pnl'].shift(-1)
daily_sent['next_day_winrate'] = daily_sent['win_rate'].shift(-1)
daily_sent['next_day_leverage'] = daily_sent['avg_leverage'].shift(-1)
daily_sent['next_day_trade_count'] = daily_sent['trade_count'].shift(-1)

daily_sent_lagged = daily_sent.dropna()

# Correlations with lag
rho_pnl, p_pnl = spearmanr(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_pnl'])
rho_wr, p_wr = spearmanr(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_winrate'])
rho_lev, p_lev = spearmanr(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_leverage'])
rho_tc, p_tc = spearmanr(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_trade_count'])

print(f"  Sentiment(t) vs Next-Day PnL:        rho={rho_pnl:.4f}, p={p_pnl:.2e} {'SIG' if p_pnl < 0.05 else 'n.s.'}")
print(f"  Sentiment(t) vs Next-Day Win Rate:    rho={rho_wr:.4f}, p={p_wr:.2e} {'SIG' if p_wr < 0.05 else 'n.s.'}")
print(f"  Sentiment(t) vs Next-Day Leverage:    rho={rho_lev:.4f}, p={p_lev:.2e} {'SIG' if p_lev < 0.05 else 'n.s.'}")
print(f"  Sentiment(t) vs Next-Day Trade Count: rho={rho_tc:.4f}, p={p_tc:.2e} {'SIG' if p_tc < 0.05 else 'n.s.'}")

# Compare same-day vs next-day correlation
rho_same_pnl, _ = spearmanr(daily_sent['sentiment_value'], daily_sent['avg_pnl'])
print(f"\n  Same-day correlation (Sentiment vs PnL):    rho={rho_same_pnl:.4f}")
print(f"  Next-day correlation (Sentiment vs PnL):    rho={rho_pnl:.4f}")
if abs(rho_pnl) > abs(rho_same_pnl):
    print("  => NEXT-DAY effect is STRONGER than same-day!")
elif abs(rho_pnl) < abs(rho_same_pnl) * 0.5:
    print("  => Lag effect is much weaker; same-day dominates")
else:
    print("  => Lag effect is comparable to same-day effect")

# Lag effect chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_pnl'],
                    alpha=0.3, s=15, c='#1976d2')
z = np.polyfit(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_pnl'], 1)
p = np.poly1d(z)
x_line = np.linspace(daily_sent_lagged['sentiment_value'].min(), daily_sent_lagged['sentiment_value'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'rho={rho_pnl:.3f}')
axes[0, 0].set_title('Sentiment(t) vs Next-Day Avg PnL', fontweight='bold')
axes[0, 0].set_xlabel('Fear/Greed Index Value')
axes[0, 0].set_ylabel('Next-Day Avg PnL ($)')
axes[0, 0].legend()

axes[0, 1].scatter(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_winrate'] * 100,
                    alpha=0.3, s=15, c='#4caf50')
z = np.polyfit(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_winrate'] * 100, 1)
p = np.poly1d(z)
axes[0, 1].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'rho={rho_wr:.3f}')
axes[0, 1].set_title('Sentiment(t) vs Next-Day Win Rate', fontweight='bold')
axes[0, 1].set_xlabel('Fear/Greed Index Value')
axes[0, 1].set_ylabel('Next-Day Win Rate (%)')
axes[0, 1].legend()

axes[1, 0].scatter(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_leverage'],
                    alpha=0.3, s=15, c='#ff9800')
z = np.polyfit(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_leverage'], 1)
p = np.poly1d(z)
axes[1, 0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'rho={rho_lev:.3f}')
axes[1, 0].set_title('Sentiment(t) vs Next-Day Avg Leverage', fontweight='bold')
axes[1, 0].set_xlabel('Fear/Greed Index Value')
axes[1, 0].set_ylabel('Next-Day Avg Leverage')
axes[1, 0].legend()

axes[1, 1].scatter(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_trade_count'],
                    alpha=0.3, s=15, c='#9c27b0')
z = np.polyfit(daily_sent_lagged['sentiment_value'], daily_sent_lagged['next_day_trade_count'], 1)
p = np.poly1d(z)
axes[1, 1].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'rho={rho_tc:.3f}')
axes[1, 1].set_title('Sentiment(t) vs Next-Day Trade Count', fontweight='bold')
axes[1, 1].set_xlabel('Fear/Greed Index Value')
axes[1, 1].set_ylabel('Next-Day Trade Count')
axes[1, 1].legend()

plt.suptitle('Lag Effect Analysis: Does Today\'s Sentiment Predict Tomorrow\'s Performance?',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}\\chart8_lag_effects.png', bbox_inches='tight')
plt.close()
print("\n  Chart 8 saved: Lag effects")

# ════════════════════════════════════════════════════════════════
# PATTERN 3: Risk-taking behavior (leverage) during Greed
# ════════════════════════════════════════════════════════════════
print("\n--- PATTERN 3: Risk-Taking Behavior (Leverage) During Greed ---")

df_lev = df[df['leverage_proxy'].between(0.1, 200)].copy()

# Leverage distribution stats by sentiment
for c in cat_order:
    sub = df_lev[df_lev['classification'] == c]['leverage_proxy']
    print(f"  {c:15s}: median={sub.median():.1f}x, mean={sub.mean():.1f}x, "
          f"p75={sub.quantile(0.75):.1f}x, p90={sub.quantile(0.90):.1f}x, max={sub.max():.1f}x")

# High leverage (>10x) proportion by sentiment
df_lev['high_lev'] = df_lev['leverage_proxy'] > 10
high_lev_pct = df_lev.groupby('classification')['high_lev'].mean() * 100
high_lev_pct = high_lev_pct.reindex(cat_order)
print(f"\n  High leverage (>10x) proportion:")
for c in cat_order:
    print(f"    {c:15s}: {high_lev_pct[c]:.1f}%")

# Risk-adjusted performance: PnL per unit of leverage
df_risk = df[(df['leverage_proxy'] > 0.5) & (df['leverage_proxy'] < 100)].copy()
df_risk['pnl_per_lev'] = df_risk['Closed PnL'] / df_risk['leverage_proxy']
risk_adj = df_risk.groupby('classification')['pnl_per_lev'].mean().reindex(cat_order)
print(f"\n  PnL per unit leverage (risk-adjusted return):")
for c in cat_order:
    print(f"    {c:15s}: ${risk_adj[c]:.2f}")

# Chart for leverage behavior
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# High leverage proportion
axes[0].bar(cat_order, high_lev_pct.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_title('% of Trades with High Leverage (>10x) by Sentiment', fontweight='bold')
axes[0].set_ylabel('Percentage of Trades (%)')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(high_lev_pct.values):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Risk-adjusted PnL
axes[1].bar(cat_order, risk_adj.values, color=[colors[c] for c in cat_order], alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_title('Risk-Adjusted Return: PnL per Unit Leverage', fontweight='bold')
axes[1].set_ylabel('Avg PnL per Leverage Unit ($)')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(risk_adj.values):
    axes[1].text(i, v + (1 if v >= 0 else -5), f'${v:.1f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUT}\\chart9_leverage_risk.png', bbox_inches='tight')
plt.close()
print("\n  Chart 9 saved: Leverage & risk behavior")

# ════════════════════════════════════════════════════════════════
# PATTERN 4: Account-level sentiment-following vs contrarian
# ════════════════════════════════════════════════════════════════
print("\n--- PATTERN 4: Sentiment-Following vs Contrarian Accounts ---")

# For each account: does their long bias correlate with sentiment?
account_long_bias = df.groupby(['Account', 'date', 'value']).agg(
    long_bias=('is_long', 'mean'),
    total_pnl=('Closed PnL', 'sum'),
    trades=('Closed PnL', 'count')
).reset_index()

# Correlation of each account's long_bias with sentiment
account_styles = []
for acc in df['Account'].unique():
    acc_data = account_long_bias[account_long_bias['Account'] == acc]
    if len(acc_data) > 10:
        try:
            rho, p = spearmanr(acc_data['value'], acc_data['long_bias'])
            avg_pnl = acc_data['total_pnl'].mean()
            account_styles.append({
                'Account': acc[:12] + '...',
                'Sentiment_Corr': rho,
                'PValue': p,
                'Avg_Daily_PnL': avg_pnl,
                'Style': 'Follower' if rho > 0.2 else ('Contrarian' if rho < -0.2 else 'Neutral')
            })
        except:
            pass

styles_df = pd.DataFrame(account_styles)
print(f"  Account styles (based on long-bias vs sentiment correlation):")
print(styles_df['Style'].value_counts().to_string())
print()

for style in ['Follower', 'Contrarian', 'Neutral']:
    sub = styles_df[styles_df['Style'] == style]
    if len(sub) > 0:
        print(f"  {style:12s}: {len(sub)} accounts, avg daily PnL = ${sub['Avg_Daily_PnL'].mean():.2f}")

# Chart
fig, ax = plt.subplots(figsize=(10, 6))
scatter_colors = {'Follower': '#4caf50', 'Contrarian': '#f44336', 'Neutral': '#9e9e9e'}
for style in ['Follower', 'Contrarian', 'Neutral']:
    sub = styles_df[styles_df['Style'] == style]
    ax.scatter(sub['Sentiment_Corr'], sub['Avg_Daily_PnL'], label=f'{style} ({len(sub)})',
               c=scatter_colors[style], alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.4, label='Follower threshold')
ax.axvline(x=-0.2, color='red', linestyle='--', alpha=0.4, label='Contrarian threshold')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_title('Account Trading Style: Sentiment Correlation vs Performance', fontweight='bold')
ax.set_xlabel('Correlation (Sentiment Value vs Long Bias)')
ax.set_ylabel('Avg Daily PnL ($)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}\\chart10_account_styles.png', bbox_inches='tight')
plt.close()
print("\n  Chart 10 saved: Account trading styles")

print("\n" + "=" * 70)
print("ALL PATTERNS IDENTIFIED")
print("=" * 70)
