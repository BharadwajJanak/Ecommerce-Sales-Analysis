"""
E-commerce Sales Analysis
Author: Bharadwaj Janak

Description:
End-to-end analysis of an e-commerce sales dataset including:
- Feature engineering
- Profitability analysis
- Customer & product insights
- Static and interactive visualizations
"""


# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.express as px



# Utility Functions

def thousands(x, pos):
    """Formatter for thousands separator on plots"""
    return f'{int(x):,}'



# Data Loading

def load_data(path: str) -> pd.DataFrame:
    """Load CSV data"""
    return pd.read_csv(path)



# Feature Engineering

def preprocess_data(dt: pd.DataFrame) -> pd.DataFrame:
    """Clean data and create new features"""

    # Date handling
    dt['order_date'] = pd.to_datetime(dt['order_date'], errors='coerce')
    dt['order_year'] = dt['order_date'].dt.year
    dt['order_month'] = dt['order_date'].dt.to_period('M').astype(str)
    dt['year_month'] = dt['order_date'].dt.to_period('M')
    dt['month'] = dt['order_date'].dt.month
    dt['order_week'] = dt['order_date'].dt.isocalendar().week
    dt['order_quarter'] = dt['order_date'].dt.to_period('Q').astype(str)

    # Sales calculations
    dt['gross_sales'] = dt['unit_price'] * dt['quantity']
    dt['discount_amount'] = dt['gross_sales'] * dt['discount']
    dt['net_sales'] = dt['gross_sales'] - dt['discount_amount']

    # Cost (if missing)
    if 'cost' not in dt.columns:
        dt['cost'] = dt['gross_sales'] * np.random.uniform(0.5, 0.9, size=len(dt))

    # Profit metrics
    dt['profit'] = dt['net_sales'] - dt['cost']
    dt['profit_margin'] = dt['profit'] / dt['net_sales'].replace(0, np.nan)

    # Business flags
    dt['is_high_value'] = (dt['net_sales'] > dt['net_sales'].quantile(0.9)).astype(int)
    dt['sales_per_day'] = dt['net_sales'] / dt['shipping_days'].replace(0, 1)

    return dt



# Visualizations

def plot_monthly_sales_profit(dt: pd.DataFrame):
    monthly = dt.groupby('year_month').agg(
        net_sales=('net_sales', 'sum'),
        profit=('profit', 'sum'),
        quantity=('quantity', 'sum')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly['year_month'].astype(str), monthly['net_sales'], marker='o', label='Net Sales')
    ax.plot(monthly['year_month'].astype(str), monthly['profit'], marker='o', label='Profit')
    ax.set_title('Monthly Net Sales and Profit')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Amount')
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_profit_heatmap(dt: pd.DataFrame):
    pivot = dt.pivot_table(
        index='region',
        columns='product_category',
        values='profit',
        aggfunc='sum'
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(pivot.values, aspect='auto')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_title('Profit by Region and Product Category')
    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_revenue_sunburst(dt: pd.DataFrame):
    sun = dt.groupby(
        ['customer_segment', 'product_category', 'product_subcategory']
    ).net_sales.sum().reset_index()

    fig = px.sunburst(
        sun,
        path=['customer_segment', 'product_category', 'product_subcategory'],
        values='net_sales',
        title='Revenue Sunburst: Segment → Category → Subcategory'
    )
    fig.show()


def plot_pareto_category(dt: pd.DataFrame):
    cat = dt.groupby('product_category').net_sales.sum().sort_values(
        ascending=False
    ).reset_index()

    cat['cumperc'] = cat['net_sales'].cumsum() / cat['net_sales'].sum() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(cat['product_category'], cat['net_sales'])
    ax2 = ax.twinx()
    ax2.plot(cat['product_category'], cat['cumperc'], marker='o', linestyle='--')
    ax2.set_ylabel('Cumulative %')
    ax.set_title('Pareto: Category Contribution to Net Sales')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def plot_discount_vs_profit(dt: pd.DataFrame):
    x = dt['discount'].fillna(0)
    y = dt['profit'].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.4)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, linestyle='--')
    ax.set_xlabel('Discount')
    ax.set_ylabel('Profit')
    ax.set_title('Discount vs Profit')
    plt.tight_layout()
    plt.show()


def plot_shipping_distribution(dt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dt['shipping_days'], bins=14, edgecolor='black')
    ax.set_title('Shipping Days Distribution')
    ax.set_xlabel('Days')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_sales_per_day_by_region(dt: pd.DataFrame):
    regions = sorted(dt['region'].unique())
    data = [dt.loc[dt['region'] == r, 'sales_per_day'].dropna() for r in regions]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data, labels=regions)
    ax.set_title('Sales per Day by Region')
    plt.tight_layout()
    plt.show()


def plot_top_customers(dt: pd.DataFrame):
    if 'customer_id' in dt.columns:
        top = dt.groupby('customer_id').net_sales.sum().sort_values(
            ascending=False
        ).head(20)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(top)), top.values)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(top.index, rotation=45)
        ax.set_title('Top 20 Customers by Net Sales')
        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(dt: pd.DataFrame):
    num = dt.select_dtypes(include=[np.number])
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title('Correlation Matrix (Numeric Features)')
    plt.tight_layout()
    plt.show()


def plot_price_quantity_scatter(dt: pd.DataFrame):
    agg = dt.groupby('product_category').agg(
        unit_price=('unit_price', 'mean'),
        quantity=('quantity', 'mean'),
        net_sales=('net_sales', 'sum')
    ).reset_index()

    fig = px.scatter(
        agg,
        x='unit_price',
        y='quantity',
        size='net_sales',
        hover_name='product_category',
        title='Avg Price vs Avg Quantity by Category'
    )
    fig.show()


def plot_monthly_sales_by_region(dt: pd.DataFrame):
    ts = dt.groupby(
        [dt['year_month'].astype(str), 'region']
    ).net_sales.sum().reset_index()

    ts.columns = ['year_month', 'region', 'net_sales']

    fig = px.line(
        ts,
        x='year_month',
        y='net_sales',
        color='region',
        title='Monthly Net Sales by Region'
    )
    fig.show()



# Main Pipeline

def main():
    dt = load_data('data/advanced_ecommerce_sales_with_customers.csv')
    dt = preprocess_data(dt)

    plot_monthly_sales_profit(dt)
    plot_profit_heatmap(dt)
    plot_revenue_sunburst(dt)
    plot_pareto_category(dt)
    plot_discount_vs_profit(dt)
    plot_shipping_distribution(dt)
    plot_sales_per_day_by_region(dt)
    plot_top_customers(dt)
    plot_correlation_matrix(dt)
    plot_price_quantity_scatter(dt)
    plot_monthly_sales_by_region(dt)


if __name__ == "__main__":
    main()
