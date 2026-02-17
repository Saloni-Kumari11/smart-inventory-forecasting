import pandas as pd
import numpy as np
from src.feature_engineering import create_all_features


def forecast_future(model, last_known_data, feature_cols, days_ahead=30,
                    is_linear=False, scaler=None):
    forecast_data = last_known_data.copy()
    predictions = []
    for i in range(days_ahead):
        fdf, _ = create_all_features(forecast_data, 'quantity_sold')
        if len(fdf) == 0:
            break
        last_row = fdf.iloc[[-1]]
        X = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            X[col] = last_row[col].values if col in last_row.columns else [0]
        if is_linear and scaler is not None:
            pred = model.predict(scaler.transform(X))[0]
        else:
            pred = model.predict(X)[0]
        pred = max(0, round(pred))
        next_date = forecast_data['date'].max() + pd.Timedelta(days=1)
        forecast_data = pd.concat([forecast_data, pd.DataFrame([{
            'date': next_date, 'quantity_sold': pred, 'revenue': 0,
            'promotion': 0, 'stock_level': 0,
            'unit_price': forecast_data['unit_price'].iloc[-1]
        }])], ignore_index=True)
        predictions.append({'date': next_date, 'predicted_demand': pred,
                           'day_of_week': next_date.strftime('%A')})
    fdf = pd.DataFrame(predictions)
    print(f'  Forecast: {len(fdf)} days | Avg: {fdf["predicted_demand"].mean():.1f}/day | Total: {fdf["predicted_demand"].sum()}')
    return fdf


def calculate_inventory_recommendations(forecast_df, product_name,
                                         lead_time_days=7, safety_stock_factor=1.5,
                                         current_stock=100):
    total = forecast_df['predicted_demand'].sum()
    avg = forecast_df['predicted_demand'].mean()
    std = forecast_df['predicted_demand'].std()
    safety = int(np.ceil(safety_stock_factor * std * np.sqrt(lead_time_days)))
    reorder = int(np.ceil(avg * lead_time_days + safety))
    days_left = current_stock / avg if avg > 0 else float('inf')
    order = max(0, int(np.ceil(total - current_stock + safety)))
    if current_stock < reorder:
        risk = 'HIGH'
    elif current_stock < reorder * 1.5:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'
    rec = {'product': product_name, 'forecast_period_days': len(forecast_df),
           'total_forecasted_demand': int(total), 'avg_daily_demand': round(avg, 1),
           'max_daily_demand': int(forecast_df['predicted_demand'].max()),
           'current_stock': current_stock, 'days_of_stock_remaining': round(days_left, 1),
           'safety_stock': safety, 'reorder_point': reorder,
           'recommended_order_qty': order, 'stockout_risk': risk,
           'lead_time_days': lead_time_days}
    print(f'  Inventory: {product_name}')
    print(f'    Demand: {rec["total_forecasted_demand"]} | Stock: {rec["current_stock"]} | Order: {rec["recommended_order_qty"]} | Risk: {risk}')
    return rec
