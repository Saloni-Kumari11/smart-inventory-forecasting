import pandas as pd
import numpy as np


def create_all_features(df, target_col='quantity_sold'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Calendar features
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

    # Cyclical features
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Holiday flag
    df['is_holiday'] = (
        ((df['month'] == 11) & (df['day_of_month'] >= 24)) |
        ((df['month'] == 12) & (df['day_of_month'] >= 15)) |
        ((df['month'] == 1) & (df['day_of_month'] <= 3))
    ).astype(int)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 21, 28, 30]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling features
    shifted = df[target_col].shift(1)
    for w in [7, 14, 30, 60, 90]:
        df[f'rmean_{w}'] = shifted.rolling(window=w, min_periods=1).mean()
        df[f'rstd_{w}'] = shifted.rolling(window=w, min_periods=1).std()
        df[f'rmin_{w}'] = shifted.rolling(window=w, min_periods=1).min()
        df[f'rmax_{w}'] = shifted.rolling(window=w, min_periods=1).max()
        df[f'rmedian_{w}'] = shifted.rolling(window=w, min_periods=1).median()

    # Expanding and EWM
    df['exp_mean'] = shifted.expanding().mean()
    for span in [7, 14, 30]:
        df[f'ewm_{span}'] = shifted.ewm(span=span, min_periods=1).mean()

    # NEW: Interaction features
    if 'rmean_7' in df.columns and 'rmean_30' in df.columns:
        df['short_long_ratio'] = (df['rmean_7'] / df['rmean_30'].replace(0, np.nan)).fillna(1)

    if 'is_weekend' in df.columns and 'promotion' in df.columns:
        df['weekend_promo'] = df['is_weekend'] * df['promotion']

    if 'is_holiday' in df.columns and 'promotion' in df.columns:
        df['holiday_promo'] = df['is_holiday'] * df['promotion']

    if 'unit_price' in df.columns:
        df['price_change'] = df['unit_price'].pct_change().fillna(0)

    if 'rmean_7' in df.columns:
        df['demand_accel'] = df['rmean_7'].diff().diff().fillna(0)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ['date', target_col, 'product_id', 'product_name', 'category', 'day_of_week_name']]
    return df, feature_cols
