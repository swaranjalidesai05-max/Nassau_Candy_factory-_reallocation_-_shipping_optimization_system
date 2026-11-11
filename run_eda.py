import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("--- PART 1: LOAD DATA ---")
df = pd.read_csv('data/processed/nassau_cleaned_stage2.csv')

# Parse dates since they load as strings
if 'Order Date' in df.columns: df['Order Date'] = pd.to_datetime(df['Order Date'])
if 'Ship Date' in df.columns: df['Ship Date'] = pd.to_datetime(df['Ship Date'])

print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.describe())

print("\\n--- PART 2: BUSINESS OVERVIEW ---")
total_sales = df['Sales'].sum() if 'Sales' in df.columns else 0
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
pd.DataFrame(kpis).to_csv('outputs/eda/business_kpis.csv', index=False)
print("Business KPIs export complete")

print("\\n--- PART 3: SALES ANALYSIS ---")
def agg_sales(groupby_col):
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

top_10_products_sales = sales_prod.head(10) if not sales_prod.empty else None
top_10_states_sales = sales_state.head(10) if not sales_state.empty else None

print("\\n--- PART 4: PROFIT ANALYSIS ---")
def agg_profit(groupby_col):
    if groupby_col not in df.columns: return pd.DataFrame()
    p = df.groupby(groupby_col).agg({'Gross Profit': 'sum', 'Sales': 'sum'})
    p['Gross Profit Margin'] = (p['Gross Profit'] / p['Sales']) * 100
    return p.sort_values(by='Gross Profit', ascending=False)

profit_reg = agg_profit('Region')
profit_div = agg_profit('Division')
profit_prod = agg_profit('Product Name')
profit_ship = agg_profit('Ship Mode')

top_10_profitable_products = profit_prod.head(10) if not profit_prod.empty else None
bottom_10_profitable_products = profit_prod.tail(10) if not profit_prod.empty else None

print("\\n--- PART 5: COST ANALYSIS ---")
cost_sales_corr = df['Cost'].corr(df['Sales']) if ('Cost' in df.columns and 'Sales' in df.columns) else None
cost_profit_corr = df['Cost'].corr(df['Gross Profit']) if ('Cost' in df.columns and 'Gross Profit' in df.columns) else None

def agg_cost(groupby_col):
    if groupby_col not in df.columns: return pd.DataFrame()
    c = df.groupby(groupby_col).agg({'Cost': 'sum', 'Sales': 'sum'})
    c['Cost Pct'] = (c['Cost'] / c['Sales']) * 100
    return c.sort_values(by='Cost Pct', ascending=False)

cost_reg = agg_cost('Region')
cost_prod = agg_cost('Product Name')
cost_ship = agg_cost('Ship Mode')

print("\\n--- PART 6: SHIPPING ANALYSIS ---")
if 'Ship Mode' in df.columns:
    shipping = df.groupby('Ship Mode').agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    })
    shipping.to_csv('outputs/eda/ship_mode_analysis.csv')

print("\\n--- PART 7: PRODUCT ANALYSIS ---")
if 'Product Name' in df.columns:
    prod_full = df.groupby('Product Name').agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    })
    prod_full['Profit Margin'] = (prod_full['Gross Profit'] / prod_full['Sales']) * 100
    prod_full.to_csv('outputs/eda/product_analysis.csv')
    high_revenue_prod = prod_full.sort_values('Sales', ascending=False).index[0] if len(prod_full)>0 else "N/A"
    high_volume_prod = prod_full.sort_values('Units', ascending=False).index[0] if len(prod_full)>0 else "N/A"
    high_profit_prod = prod_full.sort_values('Gross Profit', ascending=False).index[0] if len(prod_full)>0 else "N/A"
    low_margin_prod = prod_full.sort_values('Profit Margin', ascending=True).index[0] if len(prod_full)>0 else "N/A"

print("\\n--- PART 8: REGIONAL ANALYSIS ---")
if 'Region' in df.columns:
    reg_full = df.groupby('Region').agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    })
    reg_full['Profit Margin'] = (reg_full['Gross Profit'] / reg_full['Sales']) * 100
    reg_full['AOV'] = reg_full['Sales'] / reg_full['Order ID']
    reg_full.to_csv('outputs/eda/regional_analysis.csv')
    
    high_rev_region = reg_full.sort_values('Sales', ascending=False).index[0] if len(reg_full)>0 else "N/A"
    high_prof_region = reg_full.sort_values('Gross Profit', ascending=False).index[0] if len(reg_full)>0 else "N/A"
    high_vol_region = reg_full.sort_values('Order ID', ascending=False).index[0] if len(reg_full)>0 else "N/A"
    low_margin_region = reg_full.sort_values('Profit Margin', ascending=True).index[0] if len(reg_full)>0 else "N/A"
    high_cost_region = reg_full.sort_values('Cost', ascending=False).index[0] if len(reg_full)>0 else "N/A"

if 'Division' in df.columns:
    div_full = df.groupby('Division').agg({'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'})
    div_full.to_csv('outputs/eda/division_analysis.csv')

print("\\n--- PART 9: TIME ANALYSIS ---")
if 'order_year' in df.columns and 'order_month' in df.columns:
    monthly = df.groupby(['order_year', 'order_month']).agg({
        'Order ID': 'nunique', 'Units': 'sum', 'Sales': 'sum', 'Cost': 'sum', 'Gross Profit': 'sum'
    }).reset_index()
    monthly['Month_Label'] = monthly['order_year'].astype(str) + "-" + monthly['order_month'].astype(str).str.zfill(2)
    monthly.to_csv('outputs/eda/monthly_analysis.csv', index=False)

print("\\n--- PART 10: CORRELATION ANALYSIS ---")
cols_for_corr = [c for c in ['Sales', 'Units', 'Cost', 'Gross Profit', 'raw_lead_time_days'] if c in df.columns]
corr_matrix = df[cols_for_corr].corr()
corr_matrix.to_csv('outputs/eda/correlation_matrix.csv')

print("\\n--- PART 11: OUTLIER ANALYSIS ---")
outlier_data = []
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
pd.DataFrame(outlier_data).to_csv('outputs/eda/outlier_analysis.csv', index=False)

print("\\n--- PART 12: REQUIRED VISUALIZATIONS ---")
sns.set_theme(style="whitegrid")
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f'outputs/eda/plots/{filename}')
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
    if top_10_products_sales is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_products_sales.index, y=top_10_products_sales['Sales'])
        plt.title('Top 10 Products by Sales')
        plt.xticks(rotation=45, ha='right')
        save_plot('top_10_products_sales.png')

    if top_10_profitable_products is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_profitable_products.index, y=top_10_profitable_products['Gross Profit'])
        plt.title('Top 10 Products by Gross Profit')
        plt.xticks(rotation=45, ha='right')
        save_plot('top_10_products_profit.png')
        
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_profitable_products.index, y=prod_full.loc[top_10_profitable_products.index, 'Profit Margin'])
    plt.title('Product Profit Margin (Top 10 Profitable Products)')
    plt.xticks(rotation=45, ha='right')
    save_plot('top_10_products_profit_margin.png') # Proxy for product profit margin

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

print("\\n--- PART 13 & 14: BUSINESS INSIGHTS RELEVANT TO FACTORY REALLOCATION ---")
most_freq_ship_mode = df['Ship Mode'].mode()[0] if 'Ship Mode' in df.columns else "N/A"
high_profit_ship_mode = profit_ship.index[0] if not profit_ship.empty else "N/A"

print(f"Highest revenue region: {high_rev_region}")
print(f"Highest profit region: {high_prof_region}")
print(f"Highest-volume product: {high_volume_prod}")
print(f"Highest-revenue product: {high_revenue_prod}")
print(f"Most profitable product: {high_profit_prod}")
print(f"Lowest-margin product: {low_margin_prod}")
print(f"Most frequently used ship mode: {most_freq_ship_mode}")
print(f"Highest-profit ship mode: {high_profit_ship_mode}")
print(f"Highest-cost region: {high_cost_region}")
if 'order_year' in df.columns:
    print(f"Monthly sales trend: Extracted dynamically into plot.")
    print(f"Monthly profit trend: Extracted dynamically into plot.")

print("\\nSTEP 3 EDA COMPLETED")
