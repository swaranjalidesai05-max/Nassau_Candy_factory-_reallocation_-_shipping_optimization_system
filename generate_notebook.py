import os
import json

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    def add_md(source_text):
        nb['cells'].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + '\\n' for line in source_text.split('\\n')]
        })
        
    def add_code(source_text):
        # ensure last line does not have trailing newline if not needed, 
        # but basic split and join is fine.
        nb['cells'].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\\n' for line in source_text.split('\\n')]
        })
    
    # 1. Introduction & Setup
    add_md("# STEP 1: Dataset Audit & Understanding")
    
    code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create standard directories if they don't exist
os.makedirs('../outputs/data_audit/plots', exist_ok=True)"""
    add_code(code_setup)
    
    # 2. Requirement 1: Load CSV
    add_md("## 1. Load the CSV")
    add_code("""# Load the dataset
# Adjusted path since this notebook runs from the 'notebooks' folder
df = pd.read_csv('../data/Nassau Candy Distributor.csv')""")
    
    # 3. Requirement 2: Display shapes, head, tail, info
    add_md("## 2. Basic Dataset Information")
    add_code("""print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])""")
    add_code("""display("First 5 rows:", df.head())""")
    add_code("""display("Last 5 rows:", df.tail())""")
    add_code("""print("Column Names:")
print(df.columns.tolist())""")
    add_code("""print("Data Types:")
print(df.dtypes)""")
    add_code("""print("Dataset Information:")
df.info()""")
    
    # 4. Requirement 3: Data Quality (Missing, Dups, Uniques)
    add_md("## 3. Data Quality Checks")
    add_code("""missing_cols = df.isnull().sum()
print("Missing values for every column:")
print(missing_cols)""")
    add_code("""total_missing = df.isnull().sum().sum()
print(f"\\nTotal missing values in the dataset: {total_missing}")""")
    add_code("""duplicates = df.duplicated().sum()
print(f"\\nTotal duplicate rows: {duplicates}")""")
    add_code("""print("\\nNumber of unique values in every column:")
print(df.nunique())""")
    
    # 5. Requirement 4: Separate columns
    add_md("## 4. Column Grouping")
    add_code("""# Separate numerical and categorical columns before converting dates
# Note: 'Order Date' and 'Ship Date' are currently strings/objects
date_columns_raw = ['Order Date', 'Ship Date']
# Let's temporarily exclude dates from categorical mapping
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Exclude date cols from categorical if they are in there
categorical_cols = [c for c in categorical_cols if c not in date_columns_raw]
date_cols = date_columns_raw

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)
print("Date columns:", date_cols)""")

    # 6. Requirement 5: Descriptive stats for numerical
    add_md("## 5. Numerical Statistics")
    add_code("""num_summary = df[numerical_cols].describe()
display(num_summary)""")

    # 7. Requirement 6: Value counts for Important Categorical
    add_md("## 6. Categorical Value Counts")
    add_code("""important_cats = ['Ship Mode', 'Division', 'Region', 'Product Name', 'State/Province']

for col in important_cats:
    if col in df.columns:
        print(f"--- Value Counts for {col} ---")
        print(df[col].value_counts())
        print("\\n")""")

    # 8. Requirement 7: Convert Dates
    add_md("## 7. Convert Dates to Datetime")
    add_code("""# Using safe conversion (coerce errors to NaT if any strange format happens)
df_clean = df.copy() # Operate on a copy
# Setting date formats to ensure correctness and speed where possible, but skipping if errors
df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'], errors='coerce')
df_clean['Ship Date'] = pd.to_datetime(df_clean['Ship Date'], errors='coerce')

print("Data types after conversion:")
print(df_clean[['Order Date', 'Ship Date']].dtypes)""")

    # 9. Requirement 8: Min/Max Dates
    add_md("## 8. Date Ranges")
    add_code("""min_order_date = df_clean['Order Date'].min()
max_order_date = df_clean['Order Date'].max()
min_ship_date = df_clean['Ship Date'].min()
max_ship_date = df_clean['Ship Date'].max()

print(f"Minimum Order Date: {min_order_date}")
print(f"Maximum Order Date: {max_order_date}")
print(f"Minimum Ship Date: {min_ship_date}")
print(f"Maximum Ship Date: {max_ship_date}")""")

    # 10. Requirement 9: Raw Lead Time
    add_md("## 9. Creating Raw Lead Time Column")
    add_code("""# Create Raw Lead Time in days
df_clean['Raw Lead Time'] = (df_clean['Ship Date'] - df_clean['Order Date']).dt.days
display(df_clean[['Order Date', 'Ship Date', 'Raw Lead Time']].head())""")

    # 11. Requirement 10: Analyze Raw Lead Time
    add_md("## 10. Raw Lead Time Analysis")
    add_code("""lead_times = df_clean['Raw Lead Time'].dropna()

lt_min = lead_times.min()
lt_max = lead_times.max()
lt_mean = lead_times.mean()
lt_median = lead_times.median()
lt_std = lead_times.std()
lt_quartiles = lead_times.quantile([0.25, 0.5, 0.75])

neg_lt = (lead_times < 0).sum()
zero_lt = (lead_times == 0).sum()
# define 'extremely large' arbitrarily for the audit. Let's say > 30 days is large.
large_lt = (lead_times > 30).sum()

print(f"Min Raw Lead Time: {lt_min}")
print(f"Max Raw Lead Time: {lt_max}")
print(f"Mean Raw Lead Time: {lt_mean:.2f}")
print(f"Median Raw Lead Time: {lt_median}")
print(f"Std Dev: {lt_std:.2f}")
print(f"Quartiles:\\n{lt_quartiles}\\n")

print(f"Negative Lead Times: {neg_lt}")
print(f"Zero Lead Times: {zero_lt}")
print(f"Extremely Large Lead Times (> 30 days): {large_lt}")""")

    # 12. Requirement 11: Export Audit Outputs
    add_md("## 11. Useful Audit Outputs (CSV exports)")
    add_code("""# 1. Numerical summary
num_summary.to_csv('../outputs/data_audit/numerical_summary.csv')

# 2. Missing-value summary
missing_df = pd.DataFrame(missing_cols, columns=['Missing_Values'])
missing_df.to_csv('../outputs/data_audit/missing_value_summary.csv')

# 3. Categorical summary
cat_summary = df[categorical_cols].nunique().to_frame(name='Unique_Values')
cat_summary.to_csv('../outputs/data_audit/categorical_summary.csv')

# 4. Date / lead-time summary
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
date_summary = pd.DataFrame(date_summary_data)
date_summary.to_csv('../outputs/data_audit/date_lead_time_summary.csv', index=False)

print("Exported audit CSVs to outputs/data_audit/")""")

    # 13. Requirement 12: Visualizations
    add_md("## 12. Basic Visualizations")
    add_code("""sns.set_theme(style="whitegrid")

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'../outputs/data_audit/plots/{filename}')
    plt.show()
    plt.close()

# 1. Ship Mode Distribution
if 'Ship Mode' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Ship Mode', order=df['Ship Mode'].value_counts().index)
    plt.title('Ship Mode Distribution')
    save_plot('ship_mode_distribution.png')

# 2. Region Distribution
if 'Region' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='Region', order=df['Region'].value_counts().index)
    plt.title('Region Distribution')
    save_plot('region_distribution.png')

# 3. Division Distribution
if 'Division' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Division', order=df['Division'].value_counts().index)
    plt.title('Division Distribution')
    save_plot('division_distribution.png')""")
    
    add_code("""# 4. Sales Distribution
if 'Sales' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='Sales', bins=50, kde=True)
    plt.title('Sales Distribution')
    save_plot('sales_distribution.png')

# 5. Gross Profit Distribution
if 'Gross Profit' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='Gross Profit', bins=50, kde=True)
    plt.title('Gross Profit Distribution')
    save_plot('gross_profit_distribution.png')

# 6. Raw Lead Time Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=df_clean, x='Raw Lead Time', bins=50, kde=True)
plt.title('Raw Lead Time Distribution')
save_plot('raw_lead_time_distribution.png')""")

    # 14. Requirement 14 & 15: Initial Findings and Data Quality Warnings
    add_md("## 13. Analysis Reporting Sections")
    add_md("### Initial Findings")
    add_code("""from IPython.display import Markdown, display

findings = f\"\"\"
**Initial Findings**
- **number of rows**: {df.shape[0]:,}
- **number of columns**: {df.shape[1]:,}
- **missing values**: {total_missing:,}
- **duplicate rows**: {duplicates:,}
- **number of products**: {df['Product Name'].nunique() if 'Product Name' in df.columns else 'N/A':,}
- **number of customers**: {df['Customer ID'].nunique() if 'Customer ID' in df.columns else 'N/A':,}
- **number of orders**: {df['Order ID'].nunique() if 'Order ID' in df.columns else 'N/A':,}
- **number of regions**: {df['Region'].nunique() if 'Region' in df.columns else 'N/A'}
- **number of ship modes**: {df['Ship Mode'].nunique() if 'Ship Mode' in df.columns else 'N/A'}
- **Order Date range**: {min_order_date.date() if pd.notnull(min_order_date) else 'N/A'} to {max_order_date.date() if pd.notnull(max_order_date) else 'N/A'}
- **Ship Date range**: {min_ship_date.date() if pd.notnull(min_ship_date) else 'N/A'} to {max_ship_date.date() if pd.notnull(max_ship_date) else 'N/A'}
- **Raw Lead Time range**: {lt_min} to {lt_max} days
\"\"\"

display(Markdown(findings))""")

    add_md("### Data Quality Warnings")
    add_code("""warnings = []
if total_missing > 0:
    warnings.append(f"- **missing values**: {total_missing} missing values found.")
if duplicates > 0:
    warnings.append(f"- **duplicate rows**: {duplicates} duplicate rows found.")
if neg_lt > 0:
    warnings.append(f"- **negative lead times**: {neg_lt} orders shipped before they were ordered.")
if zero_lt > 0:
    warnings.append(f"- **zero lead times**: {zero_lt} orders shipped on the exact same day.")
if large_lt > 0:
    warnings.append(f"- **unusually large lead times**: {large_lt} orders took over 30 days to ship.")

if (pd.notnull(min_order_date) and min_order_date.year < 2000) or (pd.notnull(max_order_date) and max_order_date.year > 2030):
    warnings.append("- **suspicious date ranges**: Some dates appear to be in the far past or far future.")

if not warnings:
    warnings.append("- None detected based on initial broad criteria.")

warnings_text = "**Data Quality Warnings**\\n" + "\\n".join(warnings)
display(Markdown(warnings_text))""")

    add_code("""print("\\nSTEP 1 DATA AUDIT COMPLETED")
print("Summary: Processed data, visualized distributions, identified quality warnings, and exported results.")""")
    
    # Save the notebook
    with open('notebooks/01_data_audit.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    create_notebook()
    print("Notebook 'notebooks/01_data_audit.ipynb' created successfully.")
