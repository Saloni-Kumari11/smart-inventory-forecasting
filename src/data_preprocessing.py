import pandas as pd
import numpy as np


def load_data(filepath='data/sales_data.csv'):
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        print(f'  Loaded {len(df):,} records')
        required = ['date', 'product_id', 'quantity_sold', 'revenue']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Missing columns: {missing}')
        return df
    except FileNotFoundError:
        print(f'  ERROR: File not found: {filepath}')
        raise
    except Exception as e:
        print(f'  ERROR loading data: {e}')
        raise


def clean_data(df):
    print('  Cleaning data...')
    initial = len(df)
    df = df.drop_duplicates()
    dupes = initial - len(df)
    if dupes > 0:
        print(f'  Removed {dupes} duplicates')
    missing = df.isnull().sum()
    if missing.any():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    df['quantity_sold'] = df['quantity_sold'].clip(lower=0)
    df['revenue'] = df['revenue'].clip(lower=0)
    outliers_removed = 0
    cleaned_dfs = []
    for pid in df['product_id'].unique():
        pdf = df[df['product_id'] == pid].copy()
        mean_qty = pdf['quantity_sold'].mean()
        std_qty = pdf['quantity_sold'].std()
        if std_qty > 0:
            before = len(pdf)
            pdf = pdf[(pdf['quantity_sold'] >= mean_qty - 3 * std_qty) &
                      (pdf['quantity_sold'] <= mean_qty + 3 * std_qty)]
            outliers_removed += before - len(pdf)
        cleaned_dfs.append(pdf)
    df = pd.concat(cleaned_dfs, ignore_index=True)
    print(f'  Removed {outliers_removed} outliers | Final: {len(df):,} records')
    return df


def get_data_summary(df):
    total_units = df['quantity_sold'].sum()
    avg_daily = df.groupby('date')['quantity_sold'].sum().mean()
    print(f'  Records: {len(df):,}')
    print(f'  Date Range: {df["date"].min().date()} to {df["date"].max().date()}')
    print(f'  Products: {df["product_id"].nunique()}')
    print(f'  Categories: {df["category"].nunique()}')
    print(f'  Total Units Sold: {total_units:,}')
    print(f'  Revenue: ${df["revenue"].sum():,.2f}')
    print(f'  Avg Daily Sales: {avg_daily:.0f}')
    return {
        'total_records': len(df),
        'total_units': total_units,
        'total_revenue': df['revenue'].sum(),
        'avg_daily': avg_daily,
    }


def prepare_product_data(df, product_id):
    pdf = df[df['product_id'] == product_id].copy()
    if len(pdf) == 0:
        print(f'  WARNING: No data for product {product_id}')
        return pd.DataFrame()
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


def detect_anomalies(df, column='quantity_sold', window=30, threshold=2.5):
    df = df.copy()
    rolling_mean = df[column].rolling(window=window, center=True).mean()
    rolling_std = df[column].rolling(window=window, center=True).std()
    z_scores = np.abs((df[column] - rolling_mean) / rolling_std)
    df['z_score'] = z_scores
    df['is_anomaly'] = z_scores > threshold
    anomaly_count = df['is_anomaly'].sum()
    print(f'  Anomalies found: {anomaly_count} (threshold={threshold})')
    return df
