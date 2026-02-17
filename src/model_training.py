import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
import warnings
import pickle
import os
warnings.filterwarnings('ignore')


def train_test_split_time(df, feature_cols, target_col='quantity_sold', test_days=90):
    split = len(df) - test_days
    train, test = df.iloc[:split], df.iloc[split:]
    print(f'  Train: {len(train)} | Test: {len(test)}')
    return train[feature_cols], test[feature_cols], train[target_col], test[target_col], train, test


def train_models(X_train, X_test, y_train, y_test):
    print('  Training models...')
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42, verbosity=0),
    }
    results, trained, preds = [], {}, {}
    for name, model in models.items():
        if 'Linear' in name or 'Ridge' in name:
            model.fit(X_tr_s, y_train)
            yp = model.predict(X_te_s)
        else:
            model.fit(X_train, y_train)
            yp = model.predict(X_test)
        yp = np.maximum(yp, 0)
        mae = mean_absolute_error(y_test, yp)
        rmse = np.sqrt(mean_squared_error(y_test, yp))
        mask = y_test != 0
        mape = mean_absolute_percentage_error(y_test[mask], yp[mask]) * 100 if mask.sum() > 0 else 0
        r2 = r2_score(y_test, yp)
        results.append({'model': name, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2), 'R2': round(r2, 4)})
        trained[name] = model
        preds[name] = yp
        print(f'    {name}: MAE={mae:.2f} R2={r2:.4f}')
    rdf = pd.DataFrame(results).sort_values('MAE')
    best = rdf.iloc[0]['model']
    print(f'  Best Model: {best}')
    return trained, scaler, rdf, preds, best


def get_feature_importance(model, feature_cols, model_name):
    if hasattr(model, 'feature_importances_'):
        return pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    return None


def cross_validate_model(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for ti, vi in tscv.split(X):
        model.fit(X.iloc[ti], y.iloc[ti])
        yp = np.maximum(model.predict(X.iloc[vi]), 0)
        scores.append(mean_absolute_error(y.iloc[vi], yp))
    print(f'  CV MAE: {np.mean(scores):.2f} +/- {np.std(scores):.2f}')
    return scores


def save_model(model, scaler, feature_cols, product_id, path='models'):
    os.makedirs(path, exist_ok=True)
    fp = os.path.join(path, f'model_{product_id}.pkl')
    with open(fp, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, f)
    print(f'  Model saved: {fp}')
