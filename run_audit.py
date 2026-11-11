import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("STEP 1: Dataset Audit & Understanding")

# Create directories
os.makedirs('outputs/data_audit/plots', exist_ok=True)

print("\\n--- 1. Load the CSV ---")
df = pd.read_csv('data/Nassau Candy Distributor.csv')

print("\\n--- 2. Basic Dataset Information ---")
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
print("Column Names:", df.columns.tolist())
print("\\nData Types:\\n", df.dtypes)
print("\\nDataset Information:")
df.info()

print("\\n--- 3. Data Quality Checks ---")
missing_cols = df.isnull().sum()
total_missing = missing_cols.sum()
duplicates = df.duplicated().sum()
print("Total missing values:", total_missing)
print("Total duplicate rows:", duplicates)

print("\\n--- 4. Column Grouping ---")
date_columns_raw = ['Order Date', 'Ship Date']
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in date_columns_raw]
date_cols = date_columns_raw

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)
print("Date columns:", date_cols)

print("\\n--- 5. Numerical Statistics ---")
num_summary = df[numerical_cols].describe()

print("\\n--- 6. Categorical Value Counts ---")
# Value counts logic is in notebook, doing briefly here.
important_cats = ['Ship Mode', 'Division', 'Region', 'Product Name', 'State/Province']
# (Skipping printout to terminal to reduce spam, unless needed)

print("\\n--- 7. Convert Dates ---")
df_clean = df.copy()
df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'], errors='coerce')
df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'], errors='coerce')

print("\\n--- 8. Date Ranges ---")
min_order_date = df_clean['Order Date'].min()
max_order_date = df_clean['Order Date'].max()
min_ship_date = df_clean['Ship Date'].min()
max_ship_date = df_clean['Ship Date'].max()

print("\\n--- 9. Raw Lead Time ---")
df_clean['Raw Lead Time'] = (df_clean['Ship Date'] - df_clean['Order Date']).dt.days

print("\\n--- 10. Analyze Raw Lead Time ---")
lead_times = df_clean['Raw Lead Time'].dropna()
lt_min = lead_times.min()
lt_max = lead_times.max()
lt_mean = lead_times.mean()
lt_median = lead_times.median()
lt_std = lead_times.std()
lt_quartiles = lead_times.quantile([0.25, 0.5, 0.75])

neg_lt = (lead_times < 0).sum()
zero_lt = (lead_times == 0).sum()
large_lt = (lead_times > 30).sum()

print("\\n--- 11. Export Outputs ---")
num_summary.to_csv('outputs/data_audit/numerical_summary.csv')
pd.DataFrame(missing_cols, columns=['Missing_Values']).to_csv('outputs/data_audit/missing_value_summary.csv')
df[categorical_cols].nunique().to_frame(name='Unique_Values').to_csv('outputs/data_audit/categorical_summary.csv')

date_summary_data = {
    'Metric': [
        'Min Order Date', 'Max Order Date', 
        'Min Ship Date', 'Max Ship Date', 
        'Min Lead Time', 'Max Lead Time',
        'Mean Lead Time'
    ],
    'Value': [
        min_order_date, max_order_date, 
        min_ship_date, max_ship_date, 
        lt_min, lt_max, 
        lt_mean
    ]
}
pd.DataFrame(date_summary_data).to_csv('outputs/data_audit/date_lead_time_summary.csv', index=False)

print("\\n--- 12. Visualizations ---")
sns.set_theme(style="whitegrid")
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'outputs/data_audit/plots/{filename}')
    plt.close()

if 'Ship Mode' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Ship Mode', order=df['Ship Mode'].value_counts().index)
    plt.title('Ship Mode Distribution')
    save_plot('ship_mode_distribution.png')

if 'Region' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Region', order=df['Region'].value_counts().index)
    plt.title('Region Distribution')
    save_plot('region_distribution.png')

if 'Division' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Division', order=df['Division'].value_counts().index)
    plt.title('Division Distribution')
    save_plot('division_distribution.png')

if 'Sales' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='Sales', bins=50, kde=True)
    plt.title('Sales Distribution')
    save_plot('sales_distribution.png')

if 'Gross Profit' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='Gross Profit', bins=50, kde=True)
    plt.title('Gross Profit Distribution')
    save_plot('gross_profit_distribution.png')

plt.figure(figsize=(10, 5))
sns.histplot(data=df_clean, x='Raw Lead Time', bins=50, kde=True)
plt.title('Raw Lead Time Distribution')
save_plot('raw_lead_time_distribution.png')

print("Visualizations generated.")

print("\\nSTEP 1 DATA AUDIT COMPLETED")
print("Summary: Processed data, visualized distributions, identified quality warnings, and exported results.")
