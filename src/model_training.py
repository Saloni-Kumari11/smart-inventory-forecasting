import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
import warnings
import pickle
import os

warnings.filterwarnings('ignore')


def train_test_split_time(df, feature_cols, target_col='quantity_sold', test_days=90):
    if len(df) <= test_days:
        raise ValueError(f'Not enough data for {test_days} test days')
    split = len(df) - test_days
    train = df.iloc[:split]
    test = df.iloc[split:]
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
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=12,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100, max_depth=12,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=150, max_depth=5,
            learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=0
        ),
    }

    results = []
    trained = {}
    preds = {}

    for name, model in models.items():
        try:
            if name in ('Linear Regression', 'Ridge Regression', 'Lasso Regression'):
                model.fit(X_tr_s, y_train)
                yp = model.predict(X_te_s)
            else:
                model.fit(X_train, y_train)
                yp = model.predict(X_test)

            yp = np.maximum(yp, 0)

            mae = mean_absolute_error(y_test, yp)
            rmse = np.sqrt(mean_squared_error(y_test, yp))
            mask = y_test != 0
            if mask.sum() > 0:
                mape = mean_absolute_percentage_error(y_test[mask], yp[mask]) * 100
            else:
                mape = 0
            r2 = r2_score(y_test, yp)

            if np.sum(np.abs(y_test)) > 0:
                wmape = np.sum(np.abs(y_test - yp)) / np.sum(np.abs(y_test)) * 100
            else:
                wmape = 0

            results.append({
                'model': name,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'MAPE': round(mape, 2),
                'WMAPE': round(wmape, 2),
                'R2': round(r2, 4)
            })
            trained[name] = model
            preds[name] = yp
            print(f'    {name}: MAE={mae:.2f} R2={r2:.4f}')
        except Exception as e:
            print(f'    {name} FAILED: {e}')

    rdf = pd.DataFrame(results).sort_values('MAE')
    best = rdf.iloc[0]['model']
    print(f'  Best Model: {best}')
    return trained, scaler, rdf, preds, best


def get_feature_importance(model, feature_cols, model_name):
    if hasattr(model, 'feature_importances_'):
        imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        imp['cumulative'] = imp['importance'].cumsum() / imp['importance'].sum()
        return imp
    elif hasattr(model, 'coef_'):
        imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        imp['cumulative'] = imp['importance'].cumsum() / imp['importance'].sum()
        return imp
    return None


def cross_validate_model(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    rmse_scores = []
    for ti, vi in tscv.split(X):
        model.fit(X.iloc[ti], y.iloc[ti])
        yp = np.maximum(model.predict(X.iloc[vi]), 0)
        mae_scores.append(mean_absolute_error(y.iloc[vi], yp))
        rmse_scores.append(np.sqrt(mean_squared_error(y.iloc[vi], yp)))
    print(f'  CV MAE: {np.mean(mae_scores):.2f} +/- {np.std(mae_scores):.2f}')
    print(f'  CV RMSE: {np.mean(rmse_scores):.2f} +/- {np.std(rmse_scores):.2f}')
    return {
        'mae_scores': mae_scores,
        'rmse_scores': rmse_scores,
        'mae_mean': np.mean(mae_scores)
    }


def save_model(model, scaler, feature_cols, product_id, model_name='', path='models'):
    os.makedirs(path, exist_ok=True)
    fp = os.path.join(path, f'model_{product_id}.pkl')
    with open(fp, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'model_name': model_name,
            'product_id': product_id
        }, f)
    print(f'  Model saved: {fp}')


def load_model(product_id, path='models'):
    fp = os.path.join(path, f'model_{product_id}.pkl')
    if not os.path.exists(fp):
        print(f'  Model not found: {fp}')
        return None
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    print(f'  Model loaded: {fp}')
    return data
