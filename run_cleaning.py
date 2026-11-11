import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- PART 1: LOAD RAW DATA ---")
df_raw = pd.read_csv('data/Nassau Candy Distributor.csv')
df = df_raw.copy()
print("Raw data loaded successfully, working on a copy 'df'.")

print("\\n--- PART 2: DATA TYPE CLEANING ---")
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

invalid_order_dates = df['Order Date'].isna().sum()
invalid_ship_dates = df['Ship Date'].isna().sum()

print(f"Number of invalid Order Dates: {invalid_order_dates}")
print(f"Number of invalid Ship Dates: {invalid_ship_dates}")

print("\\n--- PART 3: DUPLICATE CHECK ---")
complete_duplicates = df.duplicated().sum()
duplicate_order_ids = df['Order ID'].duplicated().sum() if 'Order ID' in df.columns else 'N/A'
duplicate_row_ids = df['Row ID'].duplicated().sum() if 'Row ID' in df.columns else 'N/A'

print(f"Complete duplicate rows: {complete_duplicates}")
print(f"Duplicate Order IDs: {duplicate_order_ids}")
print(f"Duplicate Row IDs: {duplicate_row_ids}")

print("\\n--- PART 4: MISSING VALUES ---")
missing_counts = df.isnull().sum()
missing_percentages = (missing_counts / len(df)) * 100

missing_report = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing Count': missing_counts.values,
    'Missing Percentage': missing_percentages.values
}).reset_index(drop=True)

missing_report.to_csv('outputs/data_cleaning/missing_values.csv', index=False)

print("\\n--- PART 5: NUMERICAL DATA VALIDATION ---")
num_cols_to_check = ['Sales', 'Units', 'Gross Profit', 'Cost']
num_validation_data = []

for col in num_cols_to_check:
    if col in df.columns:
        s = df[col]
        num_validation_data.append({
            'Column': col,
            'minimum': s.min(),
            'maximum': s.max(),
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'zero_values': (s == 0).sum(),
            'negative_values': (s < 0).sum()
        })

num_validation_df = pd.DataFrame(num_validation_data)
num_validation_df.to_csv('outputs/data_cleaning/numerical_validation.csv', index=False)

print(num_validation_df)

print("Special Attention Values:")
print("Sales <= 0:", (df['Sales'] <= 0).sum() if 'Sales' in df.columns else 0)
print("Units <= 0:", (df['Units'] <= 0).sum() if 'Units' in df.columns else 0)
print("Cost < 0:", (df['Cost'] < 0).sum() if 'Cost' in df.columns else 0)
print("Gross Profit < 0:", (df['Gross Profit'] < 0).sum() if 'Gross Profit' in df.columns else 0)

print("\\n--- PART 6: LEAD TIME INVESTIGATION ---")
df['Raw Lead Time Days'] = (df['Ship Date'] - df['Order Date']).dt.days

lt = df['Raw Lead Time Days'].dropna()
lt_min = lt.min()
lt_max = lt.max()
lt_mean = lt.mean()
lt_median = lt.median()
lt_std = lt.std()
q1 = lt.quantile(0.25)
q3 = lt.quantile(0.75)
iqr = q3 - q1

lt_metrics = {
    'Metric': [
        'minimum', 'maximum', 'mean', 'median', 'std', 'Q1', 'Q3', 'IQR', 
        'negative_values', 'zero_values', '> 30 days', '> 90 days', '> 365 days'
    ],
    'Value': [
        lt_min, lt_max, lt_mean, lt_median, lt_std, q1, q3, iqr,
        (lt < 0).sum(), (lt == 0).sum(), (lt > 30).sum(), (lt > 90).sum(), (lt > 365).sum()
    ]
}

lt_report = pd.DataFrame(lt_metrics)
lt_report.to_csv('outputs/data_cleaning/lead_time_investigation.csv', index=False)
print("Lead time investigation saved.")

print("\\n--- PART 7: DO NOT FIX --- (Acknowledged)")

print("\\n--- PART 8: DATE PATTERN INVESTIGATION ---")
df['order_year'] = df['Order Date'].dt.year
df['order_month'] = df['Order Date'].dt.month
df['ship_year'] = df['Ship Date'].dt.year
df['ship_month'] = df['Ship Date'].dt.month

print("\\n1. Distribution of Order Date by Year")
print(df['order_year'].value_counts().sort_index())

print("\\n2. Distribution of Ship Date by Year")
print(df['ship_year'].value_counts().sort_index())

print("\\n3. Ship Date by Order Date relationship (Pivot Year)")
if df['order_year'].notna().sum() > 0:
    year_pivot = pd.crosstab(df['order_year'], df['ship_year'])
    print(year_pivot)

print("\\n4. Raw Lead Time by Ship Mode")
if 'Ship Mode' in df.columns:
    print(df.groupby('Ship Mode')['Raw Lead Time Days'].describe())

print("\\n5. Raw Lead Time by Region")
if 'Region' in df.columns:
    print(df.groupby('Region')['Raw Lead Time Days'].describe())

print("\\n6. Raw Lead Time by Division")
if 'Division' in df.columns:
    print(df.groupby('Division')['Raw Lead Time Days'].describe())

print("\\n--- PART 9: IDENTIFY POSSIBLE SYSTEMATIC DATE SHIFT ---")
df['Ship Year - Order Year'] = df['ship_year'] - df['order_year']
print("\\nShip Year - Order Year Frequency:")
print(df['Ship Year - Order Year'].value_counts().sort_index())

print("\\nOrder Month vs Ship Month (Frequency of matched vs mismatched):")
print((df['order_month'] == df['ship_month']).value_counts(normalize=True))

df['order_day'] = df['Order Date'].dt.day
df['ship_day'] = df['Ship Date'].dt.day
print("\\nOrder Day vs Ship Day (Frequency of matched vs mismatched):")
print((df['order_day'] == df['ship_day']).value_counts(normalize=True))

print("\\n--- PART 10: OUTLIER REPORT ---")
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df['is_lead_time_outlier'] = (df['Raw Lead Time Days'] < lower_bound) | (df['Raw Lead Time Days'] > upper_bound)
outliers = df[df['is_lead_time_outlier']]
outliers.to_csv('outputs/data_cleaning/lead_time_outliers.csv', index=False)
print(f"Number of statistical lead-time outliers identified: {len(outliers)}")

print("\\n--- PART 11: CREATE CLEAN DATASET ---")
df['order_dayofweek'] = df['Order Date'].dt.dayofweek
df['ship_dayofweek'] = df['Ship Date'].dt.dayofweek
df['raw_lead_time_days'] = df['Raw Lead Time Days']

final_clean_df = df.drop(columns=['Raw Lead Time Days', 'Ship Year - Order Year'])
final_clean_df.to_csv('data/processed/nassau_cleaned_stage2.csv', index=False)
print("Clean dataset saved to data/processed/nassau_cleaned_stage2.csv")

print("\\n--- PART 12: VISUALIZATIONS ---")
sns.set_theme(style="whitegrid")
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'outputs/data_cleaning/plots/{filename}')
    plt.close()

plt.figure(figsize=(10, 5))
df['Order Date'].hist(bins=50)
plt.title('Order Date Distribution')
save_plot('order_date_distribution.png')

plt.figure(figsize=(10, 5))
df['Ship Date'].hist(bins=50)
plt.title('Ship Date Distribution')
save_plot('ship_date_distribution.png')

plt.figure(figsize=(10, 5))
sns.histplot(df['raw_lead_time_days'].dropna(), bins=50)
plt.title('Raw Lead Time Distribution')
save_plot('raw_lead_time_distribution.png')

if 'Ship Mode' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Ship Mode', y='raw_lead_time_days')
    plt.title('Raw Lead Time by Ship Mode')
    plt.xticks(rotation=45)
    save_plot('raw_lead_time_by_ship_mode.png')

if 'Region' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Region', y='raw_lead_time_days')
    plt.title('Raw Lead Time by Region')
    plt.xticks(rotation=45)
    save_plot('raw_lead_time_by_region.png')

if 'Division' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Division', y='raw_lead_time_days')
    plt.title('Raw Lead Time by Division')
    plt.xticks(rotation=45)
    save_plot('raw_lead_time_by_division.png')

if df['order_year'].notna().sum() > 0:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='order_year', y='ship_year', alpha=0.3)
    plt.title('Order Year vs Ship Year')
    save_plot('order_year_vs_ship_year.png')

print("Investigation plots generated and saved.")

print("\\n--- PART 13: FINAL REPORT ---")
perc_gt_30 = (lt > 30).sum() / len(lt) * 100 if len(lt) > 0 else 0
perc_gt_90 = (lt > 90).sum() / len(lt) * 100 if len(lt) > 0 else 0
perc_gt_365 = (lt > 365).sum() / len(lt) * 100 if len(lt) > 0 else 0

print(f"Median raw lead time: {lt_median} days")
print(f"Maximum raw lead time: {lt_max} days")

print("\\nSTEP 2 DATA CLEANING AND LEAD-TIME INVESTIGATION COMPLETED")
