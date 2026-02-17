import pandas as pd
import numpy as np


def load_data(filepath='data/sales_data.csv'):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    print(f'  Loaded {len(df):,} records')
    return df


def clean_data(df):
    print('  Cleaning data...')
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates()
    missing = df.isnull().sum()
    if missing.any():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    df['quantity_sold'] = df['quantity_sold'].clip(lower=0)
    outliers_removed = 0
    cleaned_dfs = []
    for pid in df['product_id'].unique():
        pdf = df[df['product_id'] == pid].copy()
        mean_qty = pdf['quantity_sold'].mean()
        std_qty = pdf['quantity_sold'].std()
        before = len(pdf)
        pdf = pdf[(pdf['quantity_sold'] >= mean_qty - 3 * std_qty) &
                  (pdf['quantity_sold'] <= mean_qty + 3 * std_qty)]
        outliers_removed += before - len(pdf)
        cleaned_dfs.append(pdf)
    df = pd.concat(cleaned_dfs, ignore_index=True)
    print(f'  Removed {outliers_removed} outliers | Final: {len(df):,} records')
    return df


def get_data_summary(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()
    print(f'  Records: {len(df):,}')
    print(f'  Date Range: {min_date} to {max_date}')
    print(f'  Products: {df["product_id"].nunique()}')
    print(f'  Revenue: ${df["revenue"].sum():,.2f}')
    return {}


def prepare_product_data(df, product_id):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    pdf = df[df['product_id'] == product_id].copy()
    daily = pdf.groupby('date').agg({
        'quantity_sold': 'sum', 'revenue': 'sum',
        'promotion': 'max', 'stock_level': 'mean', 'unit_price': 'mean'
    }).reset_index()
    full_range = pd.date_range(start=daily['date'].min(), end=daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(full_range).reset_index()
    daily.rename(columns={'index': 'date'}, inplace=True)
    daily['quantity_sold'] = daily['quantity_sold'].fillna(0).astype(int)
    daily['revenue'] = daily['revenue'].fillna(0)
    daily['promotion'] = daily['promotion'].fillna(0).astype(int)
    daily['stock_level'] = daily['stock_level'].ffill().fillna(0)
    daily['unit_price'] = daily['unit_price'].ffill().fillna(0)
    daily = daily.sort_values('date').reset_index(drop=True)
    return daily