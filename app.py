import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import FuncFormatter
import os

# Set page config
st.set_page_config(page_title="USA Regional Sales Analysis", layout="wide")

# Title
st.title("USA Regional Sales Analysis")

# Set working directory
os.chdir(r"c:\Users\MESHAK SELVA KUMAR\Desktop\Data Analytics\Sales Report Analysis")

# Load data
@st.cache_data
def load_data():
    sheets = pd.read_excel("Regional Sales Dataset.xlsx", sheet_name=None)
    df_sales = sheets['Sales Orders']
    df_customers = sheets['Customers']
    df_products = sheets['Products']
    df_regions = sheets['Regions']
    df_state_reg = sheets['State Regions']
    df_budgets = sheets['2017 Budgets']
    
    # Process state regions
    new_header = df_state_reg.iloc[0]
    df_state_reg.columns = new_header
    df_state_reg = df_state_reg[1:].reset_index(drop=True)
    
    # Merge dataframes
    df = df_sales.merge(df_customers, how='left', left_on='Customer Name Index', right_on='Customer Index')
    df = df.merge(df_products, how='left', left_on='Product Description Index', right_on='Index')
    df = df.merge(df_regions, how='left', left_on='Delivery Region Index', right_on='id')
    df = df.merge(df_state_reg[["State Code","Region"]], how='left', left_on='state_code', right_on='State Code')
    df = df.merge(df_budgets, how='left', on='Product Name')
    
    # Clean up columns
    cols_to_drop = ['Customer Index', 'Index', 'id', 'State Code']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Feature engineering
    df['total_cost'] = df['Order Quantity'] * df['Total Unit Cost']
    df['profit'] = df['Line Total'] - df['total_cost']
    df['profit_margin_pct'] = (df['profit'] / df['Line Total']) * 100
    df['order_date'] = pd.to_datetime(df['OrderDate'])
    df['order_month'] = df['order_date'].dt.to_period('M')
    df['order_month_name'] = df['order_date'].dt.month_name()
    df['order_month_num'] = df['order_date'].dt.month
    
    return df

# Load the data
df = load_data()

# Monthly Sales Trend
st.header("1. Monthly Sales Trend Over Time")
monthly_sales = df.groupby('order_month')['Line Total'].sum()

fig1, ax1 = plt.subplots(figsize=(15, 4))
monthly_sales.plot(marker='o', color='navy', ax=ax1)
formatter = FuncFormatter(lambda x, p: f'${x/1e6:.1f}M')
ax1.yaxis.set_major_formatter(formatter)
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Monthly Sales Trend (All Years Combined)
st.header("2. Monthly Sales Trend (All Years Combined)")
df_ = df[df['order_date'].dt.year != 2018]
monthly_sales = df_.groupby(['order_month_num', 'order_month_name'])['Line Total'].sum().sort_index()

fig2, ax2 = plt.subplots(figsize=(13, 4))
plt.plot(monthly_sales.index.get_level_values(1), monthly_sales.values, marker='o', color='navy')
ax2.yaxis.set_major_formatter(formatter)
plt.title('Overall Monthly Sales Trend (Excluding 2018)')
plt.xlabel('Month')
plt.ylabel('Total Revenue (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# Top 10 Products by Revenue
st.header("3. Top 10 Products by Revenue")
top_prod = df.groupby('Product Name')['Line Total'].sum() / 1_000_000
top_prod = top_prod.nlargest(10)

fig3, ax3 = plt.subplots(figsize=(9, 4))
sns.barplot(x=top_prod.values, y=top_prod.index, palette='viridis')
plt.title('Top 10 Products by Revenue (in Millions)')
plt.xlabel('Total Revenue (in Millions)')
plt.ylabel('Product Name')
plt.tight_layout()
st.pyplot(fig3)

# Top 10 Products by Avg Profit Margin
st.header("4. Top 10 Products by Avg Profit Margin")
top_margin = df.groupby('Product Name')['profit'].mean().sort_values(ascending=False).head(10)

fig4, ax4 = plt.subplots(figsize=(9, 4))
sns.barplot(x=top_margin.values, y=top_margin.index, palette='viridis')
plt.title('Top 10 Products by Avg Profit Margin')
plt.xlabel('Average Profit Margin (USD)')
plt.ylabel('Product Name')
plt.tight_layout()
st.pyplot(fig4)

# Sales by Channel
st.header("5. Sales by Channel")
chan_sales = df.groupby('Channel')['Line Total'].sum().sort_values(ascending=False)

fig5, ax5 = plt.subplots(figsize=(5, 5))
plt.pie(chan_sales.values, labels=chan_sales.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('coolwarm'))
plt.title('Total Sales by Channel')
plt.tight_layout()
st.pyplot(fig5)

# Average Order Value Distribution
st.header("6. Average Order Value (AOV) Distribution")
aov = df.groupby('OrderNumber')['Line Total'].sum()

fig6, ax6 = plt.subplots(figsize=(12, 4))
plt.hist(aov, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Average Order Value')
plt.xlabel('Order Value (USD)')
plt.ylabel('Number of Orders')
plt.tight_layout()
st.pyplot(fig6)

# Profit Margin % vs. Unit Price
st.header("7. Profit Margin % vs. Unit Price")
fig7, ax7 = plt.subplots(figsize=(6, 4))
plt.scatter(df['Unit Price'], df['profit_margin_pct'], alpha=0.6, color='green')
plt.title('Profit Margin % vs. Unit Price')
plt.xlabel('Unit Price (USD)')
plt.ylabel('Profit Margin (%)')
plt.tight_layout()
st.pyplot(fig7)

# Unit Price Distribution per Product
st.header("8. Unit Price Distribution per Product")
fig8, ax8 = plt.subplots(figsize=(12, 4))
sns.boxplot(data=df, x='Product Name', y='Unit Price', color='g')
plt.title('Unit Price Distribution per Product')
plt.xlabel('Product')
plt.ylabel('Unit Price (USD)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig8)

# Total Sales by US Region
st.header("9. Total Sales by US Region")
region_sales = df.groupby('Region')['Line Total'].sum().sort_values(ascending=False) / 1e6

fig9, ax9 = plt.subplots(figsize=(10, 4))
sns.barplot(x=region_sales.values, y=region_sales.index, palette='Greens_r')
plt.title('Total Sales by US Region', fontsize=16, pad=12)
plt.xlabel('Total Sales (in Millions USD)')
plt.ylabel('US Region')
plt.tight_layout()
st.pyplot(fig9)

# Total Sales by State
st.header("10. Total Sales by State")
state_sales = df.groupby('state_code')['Line Total'].sum().reset_index()
state_sales['revenue_m'] = state_sales['Line Total'] / 1e6

fig10 = px.choropleth(
    state_sales,
    locations='state_code',
    locationmode='USA-states',
    color='revenue_m',
    scope='usa',
    labels={'revenue_m': 'Total Sales (M USD)'},
    color_continuous_scale='Blues',
    hover_data={'revenue_m': ':.2f'}
)
fig10.update_layout(title_text='Total Sales by State')
st.plotly_chart(fig10)

# Top and Bottom 10 States
st.header("11. Top 10 States by Revenue and Order Count")
state_rev = df.groupby('state').agg(
    revenue=('Line Total', 'sum'),
    orders=('OrderNumber', 'nunique')
).sort_values('revenue', ascending=False).head(10)

fig11_1, ax11_1 = plt.subplots(figsize=(15, 4))
sns.barplot(x=state_rev.index, y=state_rev['revenue'] / 1e6, palette='coolwarm')
plt.title('Top 10 States by Revenue')
plt.xlabel('State')
plt.ylabel('Total Revenue (Million USD)')
plt.tight_layout()
st.pyplot(fig11_1)

fig11_2, ax11_2 = plt.subplots(figsize=(15, 4))
sns.barplot(x=state_rev.index, y=state_rev['orders'], palette='coolwarm')
plt.title('Top 10 States by Number of Orders')
plt.xlabel('State')
plt.ylabel('Order Count')
plt.tight_layout()
st.pyplot(fig11_2)

# Average Profit Margin by Channel
st.header("12. Average Profit Margin by Channel")
channel_margin = df.groupby('Channel')['profit_margin_pct'].mean().sort_values(ascending=False)

fig12, ax12 = plt.subplots(figsize=(6, 4))
ax = sns.barplot(x=channel_margin.index, y=channel_margin.values, palette='coolwarm')
plt.title('Average Profit Margin by Channel')
plt.xlabel('Sales Channel')
plt.ylabel('Avg Profit Margin (%)')
for i, v in enumerate(channel_margin.values):
    ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
st.pyplot(fig12)

# Top and Bottom 10 Customers
st.header("13. Top and Bottom 10 Customers by Revenue")
top_rev = df.groupby('Customer Names')['Line Total'].sum().sort_values(ascending=False).head(10)
bottom_rev = df.groupby('Customer Names')['Line Total'].sum().sort_values(ascending=True).head(10)

fig13, (ax13_1, ax13_2) = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(x=top_rev.values / 1e6, y=top_rev.index, palette='Blues_r', ax=ax13_1)
ax13_1.set_title('Top 10 Customers by Revenue')
ax13_1.set_xlabel('Revenue (Million USD)')
ax13_1.set_ylabel('Customer Name')

sns.barplot(x=bottom_rev.values / 1e6, y=bottom_rev.index, palette='Reds', ax=ax13_2)
ax13_2.set_title('Bottom 10 Customers by Revenue')
ax13_2.set_xlabel('Revenue (Million USD)')
ax13_2.set_ylabel('Customer Name')
plt.tight_layout()
st.pyplot(fig13)

# Correlation Heatmap
st.header("14. Correlation Heatmap of Numeric Features")
num_cols = ['Order Quantity', 'Unit Price', 'Line Total', 'Total Unit Cost', 'profit']
corr = df[num_cols].corr()

fig14, ax14 = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis')
plt.title('Correlation Matrix')
plt.tight_layout()
st.pyplot(fig14)

# Installation Instructions
st.header("Installation Instructions")
st.code("""
conda create -n sales-env python=3.9
conda activate sales-env
conda install streamlit pandas numpy matplotlib seaborn plotly openpyxl
""")