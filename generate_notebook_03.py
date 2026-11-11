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
        nb['cells'].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\\n' for line in source_text.split('\\n')]
        })

    add_md("# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)\\n\\nDo NOT build ML models yet. Do NOT build factory optimization yet. Do NOT build Streamlit yet.")
    
    # 1. LOAD DATA
    add_md("## PART 1 — LOAD DATA")
    add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../outputs/eda/plots', exist_ok=True)

print("--- PART 1: LOAD FACTORY/SALES DATA ---")
df = pd.read_csv('../data/processed/nassau_cleaned_stage2.csv')

if 'Order Date' in df.columns: df['Order Date'] = pd.to_datetime(df['Order Date'])
if 'Ship Date' in df.columns: df['Ship Date'] = pd.to_datetime(df['Ship Date'])

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\\nData types:\\n{df.dtypes}")
print(f"\\nBasic statistics:")
display(df.describe())""")

    # 2. BUSINESS OVERVIEW
    add_md("## PART 2 — BUSINESS OVERVIEW")
    add_code("""total_sales = df['Sales'].sum() if 'Sales' in df.columns else 0
total_cost = df['Cost'].sum() if 'Cost' in df.columns else 0
total_profit = df['Gross Profit'].sum() if 'Gross Profit' in df.columns else 0
total_units = df['Units'].sum() if 'Units' in df.columns else 0
num_orders = df['Order ID'].nunique() if 'Order ID' in df.columns else 0
num_customers = df['Customer ID'].nunique() if 'Customer ID' in df.columns else 0
num_products = df['Product Name'].nunique() if 'Product Name' in df.columns else 0
num_regions = df['Region'].nunique() if 'Region' in df.columns else 0
num_cities = df['City'].nunique() if 'City' in df.columns else 0
num_states = df['State/Province'].nunique() if 'State/Province' in df.columns else 0

kpis = {
    'Metric': [
        'Total Sales', 'Total Cost', 'Total Gross Profit', 'Total Units Sold',
        'Number of Orders', 'Number of Customers', 'Number of Products',
        'Number of Regions', 'Number of Cities', 'Number of States/Provinces'
    ],
    'Value': [
        total_sales, total_cost, total_profit, total_units,
        num_orders, num_customers, num_products,
        num_regions, num_cities, num_states
    ]
}
kpi_df = pd.DataFrame(kpis)
kpi_df.to_csv('../outputs/eda/business_kpis.csv', index=False)
display(kpi_df)""")

    # 3. SALES ANALYSIS
    add_md("## PART 3 — SALES ANALYSIS")
    add_code("""def agg_sales(groupby_col):
    if groupby_col not in df.columns: return pd.DataFrame()
    s = df.groupby(groupby_col).agg({
        'Sales': 'sum',
        'Order ID': 'nunique',
        'Units': 'sum'
    }).rename(columns={'Order ID': 'Num Orders'})
    s['Sales Pct'] = (s['Sales'] / total_sales) * 100
    return s.sort_values(by='Sales', ascending=False)

sales_reg = agg_sales('Region')
sales_div = agg_sales('Division')
sales_prod = agg_sales('Product Name')
sales_ship = agg_sales('Ship Mode')
sales_state = agg_sales('State/Province')

print("Top 10 Products by Sales:")
top_10_products_sales = sales_prod.head(10) if not sales_prod.empty else pd.DataFrame()
display(top_10_products_sales)

print("Top 10 Regions/States by Sales:")
top_10_states_sales = sales_state.head(10) if not sales_state.empty else pd.DataFrame()
display(top_10_states_sales)""")

    # 4. PROFIT ANALYSIS
    add_md("## PART 4 — PROFIT ANALYSIS")
    add_code("""def agg_profit(groupby_col):
    if groupby_col not in df.columns: return pd.DataFrame()
    p = df.groupby(groupby_col).agg({'Gross Profit': 'sum', 'Sales': 'sum'})
    p['Gross Profit Margin'] = (p['Gross Profit'] / p['Sales']) * 100
    return p.sort_values(by='Gross Profit', ascending=False)

profit_reg = agg_profit('Region')
profit_div = agg_profit('Division')
profit_prod = agg_profit('Product Name')
profit_ship = agg_profit('Ship Mode')

print("Total Profit by Region:")
display(profit_reg)
print("Profit Margin by Division:")
display(profit_div)

top_10_profitable_products = profit_prod.head(10) if not profit_prod.empty else pd.DataFrame()
bottom_10_profitable_products = profit_prod.tail(10) if not profit_prod.empty else pd.DataFrame()

print("Top 10 Profitable Products:")
display(top_10_profitable_products)
print("Bottom 10 Profitable Products:")
display(bottom_10_profitable_products)""")

    # 5. COST ANALYSIS
    add_md("## PART 5 — COST ANALYSIS")
    add_code("""cost_sales_corr = df['Cost'].corr(df['Sales']) if ('Cost' in df.columns and 'Sales' in df.columns) else None
cost_profit_corr = df['Cost'].corr(df['Gross Profit']) if ('Cost' in df.columns and 'Gross Profit' in df.columns) else None
print(f"Correlation (Cost vs Sales): {cost_sales_corr}")
print(f"Correlation (Cost vs Gross Profit): {cost_profit_corr}")

def agg_cost(groupby_col):
    if groupby_col not in df.columns: return pd.DataFrame()
    c = df.groupby(groupby_col).agg({'Cost': 'sum', 'Sales': 'sum'})
    c['Cost Pct'] = (c['Cost'] / c['Sales']) * 100
    return c.sort_values(by='Cost Pct', ascending=False)

print("Cost by Region:")
cost_reg = agg_cost('Region')
display(cost_reg)

print("Cost by Ship Mode:")
cost_ship = agg_cost('Ship Mode')
display(cost_ship)

print("High Cost Pct Products:")
cost_prod = agg_cost('Product Name')
display(cost_prod.head(10))""")

    # 6. SHIPPING ANALYSIS
    add_md("## PART 6 — SHIPPING ANALYSIS")
    add_code("""if 'Ship Mode' in df.columns:
    shipping = df.groupby('Ship Mode').agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    })
    shipping.to_csv('../outputs/eda/ship_mode_analysis.csv')
    print("Shipping Analysis overall:")
    display(shipping)
    
if 'raw_lead_time_days' in df.columns:
    print("Raw Lead Time by Ship Mode:")
    display(df.groupby('Ship Mode')['raw_lead_time_days'].describe() if 'Ship Mode' in df.columns else None)
    print("Raw Lead Time by Region:")
    display(df.groupby('Region')['raw_lead_time_days'].describe() if 'Region' in df.columns else None)
    print("Raw Lead Time by Division:")
    display(df.groupby('Division')['raw_lead_time_days'].describe() if 'Division' in df.columns else None)
    
print("IMPORTANT Note: Raw Lead Time is still an investigation variable.")""")

    # 7. PRODUCT ANALYSIS
    add_md("## PART 7 — PRODUCT ANALYSIS")
    add_code("""if 'Product Name' in df.columns:
    prod_full = df.groupby('Product Name').agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    })
    prod_full['Profit Margin'] = (prod_full['Gross Profit'] / prod_full['Sales']) * 100
    prod_full.to_csv('../outputs/eda/product_analysis.csv')
    
    high_revenue_prod = prod_full.sort_values('Sales', ascending=False).index[0] if len(prod_full)>0 else "N/A"
    high_volume_prod = prod_full.sort_values('Units', ascending=False).index[0] if len(prod_full)>0 else "N/A"
    high_profit_prod = prod_full.sort_values('Gross Profit', ascending=False).index[0] if len(prod_full)>0 else "N/A"
    low_margin_prod = prod_full.sort_values('Profit Margin', ascending=True).index[0] if len(prod_full)>0 else "N/A"
    
    print(f"High-volume Product: {high_volume_prod}")
    print(f"High-revenue Product: {high_revenue_prod}")
    print(f"High-profit Product: {high_profit_prod}")
    print(f"Low-margin Product: {low_margin_prod}")""")

    # 8. REGIONAL ANALYSIS
    add_md("## PART 8 — REGIONAL ANALYSIS")
    add_code("""if 'Region' in df.columns:
    reg_full = df.groupby('Region').agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    })
    reg_full['Profit Margin'] = (reg_full['Gross Profit'] / reg_full['Sales']) * 100
    reg_full['AOV'] = reg_full['Sales'] / reg_full['Order ID']
    reg_full.to_csv('../outputs/eda/regional_analysis.csv')
    
    high_rev_region = reg_full.sort_values('Sales', ascending=False).index[0] if len(reg_full)>0 else "N/A"
    high_prof_region = reg_full.sort_values('Gross Profit', ascending=False).index[0] if len(reg_full)>0 else "N/A"
    high_vol_region = reg_full.sort_values('Order ID', ascending=False).index[0] if len(reg_full)>0 else "N/A"
    low_margin_region = reg_full.sort_values('Profit Margin', ascending=True).index[0] if len(reg_full)>0 else "N/A"
    high_cost_region = reg_full.sort_values('Cost', ascending=False).index[0] if len(reg_full)>0 else "N/A"

    print("Regional Overview:")
    display(reg_full)
    
if 'Division' in df.columns:
    div_full = df.groupby('Division').agg({'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'})
    div_full.to_csv('../outputs/eda/division_analysis.csv')""")

    # 9. TIME ANALYSIS
    add_md("## PART 9 — TIME ANALYSIS")
    add_code("""if 'order_year' in df.columns and 'order_month' in df.columns:
    monthly = df.groupby(['order_year', 'order_month']).agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    }).reset_index()
    monthly['Month_Label'] = monthly['order_year'].astype(str) + "-" + monthly['order_month'].astype(str).str.zfill(2)
    monthly.to_csv('../outputs/eda/monthly_analysis.csv', index=False)
    
    display(monthly.head(10))""")

    # 10. CORRELATION ANALYSIS
    add_md("## PART 10 — CORRELATION ANALYSIS")
    add_code("""cols_for_corr = [c for c in ['Sales', 'Units', 'Cost', 'Gross Profit', 'raw_lead_time_days'] if c in df.columns]
corr_matrix = df[cols_for_corr].corr()
corr_matrix.to_csv('../outputs/eda/correlation_matrix.csv')
display(corr_matrix)""")

    # 11. OUTLIER ANALYSIS
    add_md("## PART 11 — OUTLIER ANALYSIS")
    add_code("""outlier_data = []
for col in ['Sales', 'Units', 'Cost', 'Gross Profit']:
    if col in df.columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        n_outliers = ((s < lb) | (s > ub)).sum()
        pct_outliers = (n_outliers / len(s)) * 100
        outlier_data.append({
            'Column': col, 'Q1': q1, 'Q3': q3, 'IQR': iqr,
            'Lower Bound': lb, 'Upper Bound': ub,
            'Number of Outliers': n_outliers,
            'Outlier Percentage': pct_outliers
        })
outliers_df = pd.DataFrame(outlier_data)
outliers_df.to_csv('../outputs/eda/outlier_analysis.csv', index=False)
display(outliers_df)""")

    # 12. REQUIRED VISUALIZATIONS
    add_md("## PART 12 — REQUIRED VISUALIZATIONS")
    add_code("""sns.set_theme(style="whitegrid")
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'../outputs/eda/plots/{filename}')
    plt.close()

# 1. Sales by Region
if 'Region' in df.columns:
    plt.figure(figsize=(10, 5))
    sales_reg['Sales'].plot(kind='bar')
    plt.title('Sales by Region')
    save_plot('sales_by_region.png')

    plt.figure(figsize=(10, 5))
    profit_reg['Gross Profit'].plot(kind='bar')
    plt.title('Profit by Region')
    save_plot('profit_by_region.png')
    
    plt.figure(figsize=(10, 5))
    reg_full['Profit Margin'].plot(kind='bar')
    plt.title('Regional Profit Margin (%)')
    save_plot('regional_profit_margin.png')

# 3, 4. Sales/Profit by Ship Mode
if 'Ship Mode' in df.columns:
    plt.figure(figsize=(10, 5))
    sales_ship['Sales'].plot(kind='bar')
    plt.title('Sales by Ship Mode')
    save_plot('sales_by_ship_mode.png')

    plt.figure(figsize=(10, 5))
    profit_ship['Gross Profit'].plot(kind='bar')
    plt.title('Profit by Ship Mode')
    save_plot('profit_by_ship_mode.png')

# 5, 6, 14. Products
if 'Product Name' in df.columns:
    if not top_10_products_sales.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_products_sales.index, y=top_10_products_sales['Sales'])
        plt.title('Top 10 Products by Sales')
        plt.xticks(rotation=45, ha='right')
        save_plot('top_10_products_sales.png')

    if not top_10_profitable_products.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_profitable_products.index, y=top_10_profitable_products['Gross Profit'])
        plt.title('Top 10 Products by Gross Profit')
        plt.xticks(rotation=45, ha='right')
        save_plot('top_10_products_profit.png')
        
    plt.figure(figsize=(12, 6))
    if not top_10_profitable_products.empty:
        sns.barplot(x=top_10_profitable_products.index, y=prod_full.loc[top_10_profitable_products.index, 'Profit Margin'])
        plt.title('Product Profit Margin (Top 10 Profitable Products)')
        plt.xticks(rotation=45, ha='right')
        save_plot('product_profit_margin.png') 

# 7, 8. Monthly Trends
if 'order_year' in df.columns and 'order_month' in df.columns:
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x='Month_Label', y='Sales')
    plt.title('Monthly Sales Trend')
    plt.xticks(rotation=45)
    save_plot('monthly_sales_trend.png')

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x='Month_Label', y='Gross Profit')
    plt.title('Monthly Profit Trend')
    plt.xticks(rotation=45)
    save_plot('monthly_profit_trend.png')

# 9, 10, 11
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df.sample(min(len(df), 2000)), x='Units', y='Sales', alpha=0.5)
plt.title('Sales vs Units')
save_plot('sales_vs_units.png')

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df.sample(min(len(df), 2000)), x='Gross Profit', y='Sales', alpha=0.5)
plt.title('Sales vs Gross Profit')
save_plot('sales_vs_gross_profit.png')

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df.sample(min(len(df), 2000)), x='Gross Profit', y='Cost', alpha=0.5)
plt.title('Cost vs Gross Profit')
save_plot('cost_vs_gross_profit.png')

# 12. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
save_plot('correlation_heatmap.png')
""")

    # 13. BUSINESS INSIGHTS
    add_md("## PART 13 — EDA BUSINESS INSIGHTS")
    add_code("""from IPython.display import Markdown, display
most_freq_ship_mode = df['Ship Mode'].mode()[0] if 'Ship Mode' in df.columns else "N/A"
high_profit_ship_mode = profit_ship.index[0] if not profit_ship.empty else "N/A"

business_insights = f\"\"\"
**EDA BUSINESS INSIGHTS**
- Highest revenue region: {high_rev_region}
- Highest profit region: {high_prof_region}
- Highest-volume product: {high_volume_prod}
- Highest-revenue product: {high_revenue_prod}
- Most profitable product: {high_profit_prod}
- Lowest-margin product: {low_margin_prod}
- Most frequently used ship mode: {most_freq_ship_mode}
- Highest-profit ship mode: {high_profit_ship_mode}
- Highest-cost region: {high_cost_region}
- Monthly sales trend check plot `monthly_sales_trend.png`
- Monthly profit trend check plot `monthly_profit_trend.png`
\"\"\"
display(Markdown(business_insights))""")

    # 14. OPTIMIZATION RELEVANCE
    add_md("## PART 14 — INSIGHTS RELEVANT TO FACTORY REALLOCATION")
    add_code("""optimization_insights = f\"\"\"
**INSIGHTS RELEVANT TO FACTORY REALLOCATION**
- **Regions with high demand**: {high_rev_region} and {high_vol_region} will likely require prioritizing inventory allocation and minimized distribution overhead.
- **Products with high sales**: {high_revenue_prod} is a critical item. Factory positioning should reflect logistics for this high-moving item.
- **Products with high profit**: {high_profit_prod}. Shipping inefficiencies for this product disproportionately erode company profits.
- **Regions with high shipping activity**: {high_vol_region} processes the highest volume of order lines.
- **Ship modes associated with high shipping volume**: {most_freq_ship_mode} drives bulk logistics.
- **Regions/Products where cost is relatively high**: Region {high_cost_region} features high aggregate costs indicating potential fulfillment optimization prospects.
\"\"\"
display(Markdown(optimization_insights))""")

    # 15. EXPORT RESULTS (Already done during sections, but we acknowledge it)
    add_md("## PART 15 — EXPORT RESULTS")
    add_code("""print("Exports (e.g., regional_analysis.csv, monthly_analysis.csv, etc.) have successfully been deposited to `outputs/eda/`.")""")

    add_code("""print("\\nSTEP 3 EDA COMPLETED")""")

    # Save the notebook
    with open('notebooks/03_eda.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    create_notebook()
    print("Notebook 'notebooks/03_eda.ipynb' created successfully.")
