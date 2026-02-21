import pandas as pd
import numpy as np
from src.feature_engineering import create_all_features


def forecast_future(model, last_known_data, feature_cols, days_ahead=30,
                    is_linear=False, scaler=None):
    forecast_data = last_known_data.copy()
    predictions = []
    for i in range(days_ahead):
        recent_data = forecast_data.tail(120).copy().reset_index(drop=True)
        fdf, _ = create_all_features(recent_data, 'quantity_sold')
        if len(fdf) == 0:
            break
        last_row = fdf.iloc[[-1]]
        X = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in last_row.columns:
                X[col] = last_row[col].values
            else:
                X[col] = [0]
        if is_linear and scaler is not None:
            pred = model.predict(scaler.transform(X))[0]
        else:
            pred = model.predict(X)[0]
        pred = max(0, round(pred))
        next_date = forecast_data['date'].max() + pd.Timedelta(days=1)
        new_row = pd.DataFrame([{
            'date': next_date,
            'quantity_sold': pred,
            'revenue': 0,
            'promotion': 0,
            'stock_level': 0,
            'unit_price': forecast_data['unit_price'].iloc[-1]
        }])
        forecast_data = pd.concat([forecast_data, new_row], ignore_index=True)
        predictions.append({
            'date': next_date,
            'predicted_demand': pred,
            'day_of_week': next_date.strftime('%A')
        })
    fdf = pd.DataFrame(predictions)
    if len(fdf) > 0:
        print(f'  Forecast: {len(fdf)} days | Avg: {fdf["predicted_demand"].mean():.1f}/day | Total: {fdf["predicted_demand"].sum()}')
    return fdf


def calculate_inventory_recommendations(forecast_df, product_name,
                                         lead_time_days=7, safety_stock_factor=1.5,
                                         current_stock=100):
    total = forecast_df['predicted_demand'].sum()
    avg = forecast_df['predicted_demand'].mean()
    std = forecast_df['predicted_demand'].std()
    max_demand = int(forecast_df['predicted_demand'].max())
    safety = int(np.ceil(safety_stock_factor * std * np.sqrt(lead_time_days)))
    reorder = int(np.ceil(avg * lead_time_days + safety))
    days_left = current_stock / avg if avg > 0 else float('inf')
    order = max(0, int(np.ceil(total - current_stock + safety)))
    ordering_cost = 50
    holding_cost_per_unit = 0.50
    if avg > 0 and holding_cost_per_unit > 0:
        eoq = int(np.ceil(np.sqrt((2 * total * ordering_cost) / holding_cost_per_unit)))
    else:
        eoq = order
    if current_stock < safety:
        risk = 'CRITICAL'
    elif current_stock < reorder:
        risk = 'HIGH'
    elif current_stock < reorder * 1.5:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'
    cumulative_demand = forecast_df['predicted_demand'].cumsum()
    stockout_mask = cumulative_demand >= current_stock
    stockout_date = None
    stockout_day = None
    if stockout_mask.any():
        stockout_idx = stockout_mask.idxmax()
        stockout_date = forecast_df.loc[stockout_idx, 'date']
        stockout_day = int(stockout_idx) + 1
    excess_stock = max(0, current_stock - total)
    holding_cost = round(excess_stock * holding_cost_per_unit * len(forecast_df), 2)
    potential_lost = max(0, total - current_stock)
    stockout_cost = round(potential_lost * 5.00, 2)
    rec = {
        'product': product_name,
        'forecast_period_days': len(forecast_df),
        'total_forecasted_demand': int(total),
        'avg_daily_demand': round(avg, 1),
        'std_daily_demand': round(std, 1),
        'max_daily_demand': max_demand,
        'current_stock': current_stock,
        'days_of_stock_remaining': round(days_left, 1),
        'safety_stock': safety,
        'reorder_point': reorder,
        'recommended_order_qty': order,
        'economic_order_qty': eoq,
        'stockout_risk': risk,
        'stockout_date': stockout_date,
        'stockout_day': stockout_day,
        'lead_time_days': lead_time_days,
        'estimated_holding_cost': holding_cost,
        'estimated_stockout_cost': stockout_cost,
    }
    print(f'  Inventory: {product_name}')
    print(f'    Demand: {rec["total_forecasted_demand"]} | Stock: {rec["current_stock"]} | Order: {rec["recommended_order_qty"]} | Risk: {risk}')
    return rec


def scenario_analysis(model, last_known_data, feature_cols,
                      days_ahead=30, is_linear=False, scaler=None,
                      current_stock=100):
    print('  Running scenario analysis...')
    base = forecast_future(model, last_known_data, feature_cols, days_ahead, is_linear, scaler)
    scenarios = {}
    scenarios['normal'] = {
        'forecast': base,
        'recommendation': calculate_inventory_recommendations(base, 'Product', current_stock=current_stock),
    }
    optimistic = base.copy()
    optimistic['predicted_demand'] = (optimistic['predicted_demand'] * 0.8).astype(int)
    scenarios['optimistic'] = {
        'forecast': optimistic,
        'recommendation': calculate_inventory_recommendations(optimistic, 'Product', current_stock=current_stock),
    }
    pessimistic = base.copy()
    pessimistic['predicted_demand'] = (pessimistic['predicted_demand'] * 1.3).astype(int)
    scenarios['pessimistic'] = {
        'forecast': pessimistic,
        'recommendation': calculate_inventory_recommendations(pessimistic, 'Product', current_stock=current_stock),
    }
    print('  Scenario analysis complete')
    return scenarios
