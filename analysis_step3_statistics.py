import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, spearmanr, pearsonr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

OUT = r'B:\Internshala'

# ── Load data ──
df = pd.read_csv(f'{OUT}\\engineered_data.csv')
daily = pd.read_csv(f'{OUT}\\daily_account_summary.csv')
daily['date'] = pd.to_datetime(daily['date'])

cat_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
colors = {
    'Extreme Fear': '#d32f2f', 'Fear': '#ff9800',
    'Neutral': '#9e9e9e', 'Greed': '#4caf50', 'Extreme Greed': '#1b5e20'
}

print("=" * 70)
print("STATISTICAL TESTS: SENTIMENT vs TRADING BEHAVIOR")
print("=" * 70)

# ════════════════════════════════════════════════════════════════
# TEST 1: Kruskal-Wallis (non-parametric ANOVA) for PnL
# ════════════════════════════════════════════════════════════════
print("\n--- Test 1: Kruskal-Wallis on Closed PnL ---")
groups_pnl = [df[df['classification'] == c]['Closed PnL'].values for c in cat_order]
stat, pval = kruskal(*groups_pnl)
print(f"  H-statistic = {stat:.4f}, p-value = {pval:.2e}")
print(f"  => {'SIGNIFICANT' if pval < 0.05 else 'NOT significant'} at alpha=0.05")

# ════════════════════════════════════════════════════════════════
# TEST 2: Kruskal-Wallis on PnL %
# ════════════════════════════════════════════════════════════════
print("\n--- Test 2: Kruskal-Wallis on PnL% ---")
df_pnl = df[df['pnl_pct'].between(-100, 100)].copy()
groups_pnl_pct = [df_pnl[df_pnl['classification'] == c]['pnl_pct'].values for c in cat_order]
stat, pval = kruskal(*groups_pnl_pct)
print(f"  H-statistic = {stat:.4f}, p-value = {pval:.2e}")
print(f"  => {'SIGNIFICANT' if pval < 0.05 else 'NOT significant'} at alpha=0.05")

# ════════════════════════════════════════════════════════════════
# TEST 3: Kruskal-Wallis on Leverage
# ════════════════════════════════════════════════════════════════
print("\n--- Test 3: Kruskal-Wallis on Leverage ---")
df_lev = df[df['leverage_proxy'].between(0.1, 200)].copy()
groups_lev = [df_lev[df_lev['classification'] == c]['leverage_proxy'].values for c in cat_order]
stat, pval = kruskal(*groups_lev)
print(f"  H-statistic = {stat:.4f}, p-value = {pval:.2e}")
print(f"  => {'SIGNIFICANT' if pval < 0.05 else 'NOT significant'} at alpha=0.05")

# ════════════════════════════════════════════════════════════════
# TEST 4: Chi-Square test for win rate across sentiments
# ════════════════════════════════════════════════════════════════
print("\n--- Test 4: Chi-Square Test on Win/Loss vs Sentiment ---")
contingency = pd.crosstab(df['classification'], df['is_win'])
chi2, pval, dof, expected = stats.chi2_contingency(contingency)
print(f"  Chi2 = {chi2:.4f}, dof = {dof}, p-value = {pval:.2e}")
print(f"  => {'SIGNIFICANT' if pval < 0.05 else 'NOT significant'} at alpha=0.05")
print(f"\n  Observed win rates:")
for c in cat_order:
    subset = df[df['classification'] == c]
    wr = subset['is_win'].mean() * 100
    print(f"    {c:15s}: {wr:.2f}%")

# ════════════════════════════════════════════════════════════════
# TEST 5: Correlation - Sentiment Value vs PnL
# ════════════════════════════════════════════════════════════════
print("\n--- Test 5: Spearman Correlation ---")
# At trade level
rho, pval = spearmanr(df['value'], df['Closed PnL'])
print(f"  Sentiment Value vs PnL (trade-level):  rho = {rho:.4f}, p = {pval:.2e}")

# At daily level
daily_stats = df.groupby('date').agg(
    avg_sentiment=('value', 'mean'),
    avg_pnl=('Closed PnL', 'mean'),
    win_rate=('is_win', 'mean'),
    avg_leverage=('leverage_proxy', lambda x: x[x > 0].mean() if (x > 0).any() else np.nan),
    avg_trade_size=('Size USD', 'mean'),
    trade_count=('Closed PnL', 'count')
).dropna()

rho, pval = spearmanr(daily_stats['avg_sentiment'], daily_stats['avg_pnl'])
print(f"  Sentiment vs Avg PnL (daily):          rho = {rho:.4f}, p = {pval:.2e}")

rho, pval = spearmanr(daily_stats['avg_sentiment'], daily_stats['win_rate'])
print(f"  Sentiment vs Win Rate (daily):         rho = {rho:.4f}, p = {pval:.2e}")

rho, pval = spearmanr(daily_stats['avg_sentiment'], daily_stats['avg_leverage'])
print(f"  Sentiment vs Avg Leverage (daily):     rho = {rho:.4f}, p = {pval:.2e}")

rho, pval = spearmanr(daily_stats['avg_sentiment'], daily_stats['trade_count'])
print(f"  Sentiment vs Trade Count (daily):      rho = {rho:.4f}, p = {pval:.2e}")

# ════════════════════════════════════════════════════════════════
# TEST 6: Pairwise Mann-Whitney U tests (Fear vs Greed)
# ════════════════════════════════════════════════════════════════
print("\n--- Test 6: Pairwise Mann-Whitney U Tests (PnL) ---")
pairs = [('Extreme Fear', 'Extreme Greed'), ('Fear', 'Greed'), ('Extreme Fear', 'Fear'), ('Greed', 'Extreme Greed')]
for a, b in pairs:
    g1 = df[df['classification'] == a]['Closed PnL'].values
    g2 = df[df['classification'] == b]['Closed PnL'].values
    stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
    print(f"  {a:15s} vs {b:15s}: U={stat:.0f}, p={pval:.2e} => {'SIG' if pval < 0.05 else 'n.s.'}")

print("\n" + "=" * 70)
print("STATISTICAL TESTS COMPLETE")
print("=" * 70)
