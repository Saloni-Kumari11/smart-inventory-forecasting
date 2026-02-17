import pandas as pd
import numpy as np


def create_all_features(df, target_col='quantity_sold'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['is_holiday'] = (
        ((df['month'] == 11) & (df['day_of_month'] >= 24)) |
        ((df['month'] == 12) & (df['day_of_month'] >= 15)) |
        ((df['month'] == 1) & (df['day_of_month'] <= 3))
    ).astype(int)
    for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    for w in [7, 14, 30, 60, 90]:
        df[f'rmean_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        df[f'rstd_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).std()
        df[f'rmin_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).min()
        df[f'rmax_{w}'] = df[target_col].shift(1).rolling(window=w, min_periods=1).max()
    df['exp_mean'] = df[target_col].shift(1).expanding().mean()
    for span in [7, 14, 30]:
        df[f'ewm_{span}'] = df[target_col].shift(1).ewm(span=span, min_periods=1).mean()
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ['date', target_col, 'product_id', 'product_name', 'category', 'day_of_week_name']]
    return df, feature_cols
