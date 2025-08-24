import pandas as pd

df = pd.read_csv("historical_plus_forecast.csv")

# Optional: use historical-only correlations 
USE_HISTORICAL_ONLY = True
if USE_HISTORICAL_ONLY and 'type' in df.columns:
    df = df[df['type'].str.lower() == "historical"]

# 2) Build a tidy YearMonth index (monthly grain)-
# Using a proper datetime helps keep ordering clean
df['YearMonth'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))


# 3) Pivot so each KPI is a column and each month is a row
#    - If your KPIs are additive per month, 'sum' is fine.
#    - If each KPI appears once per month, 'mean' is fine too.

pivot_df = df.pivot_table(
    index='YearMonth',
    columns='english_name',
    values='monthly_value',
    aggfunc='sum'   # or 'mean' if more appropriate
)

# NOTE: We do NOT fill NaNs with zero here; pandas' .corr() handles NaNs pairwise.
# If you truly need to fill (e.g., to avoid all-NaN columns), do it explicitly:
# pivot_df = pivot_df.fillna(0)

# 4) Correlation matrix (wide)
corr = pivot_df.corr(method='pearson')

# CRITICAL: rename axes to avoid the "cannot insert english_name, already exists" error
corr = corr.rename_axis(index='KPI1', columns='KPI2')

# Save wide format
corr.to_csv("correlation_matrix.csv")

# 5) Long format for Power BI heatmap
#    (No axis-name collisions because we renamed them)

correlation_long = corr.stack(dropna=False).reset_index(name='Correlation')

# Remove self-correlations if you don't want the diagonal
correlation_long = correlation_long[correlation_long['KPI1'] != correlation_long['KPI2']]

# Save long format
correlation_long.to_csv("correlation_long.csv", index=False)

print("\nAxis names ->", corr.index.name, corr.columns.name)
print("\n📊 Sample (long):")
print(correlation_long.head(10))

