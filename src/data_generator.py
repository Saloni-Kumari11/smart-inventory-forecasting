import pandas as pd
import numpy as np
import os


def generate_sales_data(start_date='2021-01-01', end_date='2024-12-31',
                        output_path='data/sales_data.csv'):
    np.random.seed(42)
    products = {
        'P001': {'name': 'Wireless Mouse', 'base_demand': 15, 'price': 29.99, 'category': 'Electronics', 'seasonal': 'neutral'},
        'P002': {'name': 'USB-C Cable', 'base_demand': 25, 'price': 12.99, 'category': 'Electronics', 'seasonal': 'neutral'},
        'P003': {'name': 'Notebook A5', 'base_demand': 30, 'price': 5.99, 'category': 'Stationery', 'seasonal': 'back_to_school'},
        'P004': {'name': 'Ballpoint Pen Pack', 'base_demand': 40, 'price': 3.99, 'category': 'Stationery', 'seasonal': 'back_to_school'},
        'P005': {'name': 'Desk Lamp', 'base_demand': 8, 'price': 45.99, 'category': 'Furniture', 'seasonal': 'winter'},
        'P006': {'name': 'Hand Sanitizer', 'base_demand': 50, 'price': 4.99, 'category': 'Health', 'seasonal': 'winter'},
        'P007': {'name': 'Water Bottle', 'base_demand': 20, 'price': 15.99, 'category': 'Lifestyle', 'seasonal': 'summer'},
        'P008': {'name': 'Phone Case', 'base_demand': 18, 'price': 19.99, 'category': 'Electronics', 'seasonal': 'neutral'},
        'P009': {'name': 'Sticky Notes', 'base_demand': 35, 'price': 2.99, 'category': 'Stationery', 'seasonal': 'back_to_school'},
        'P010': {'name': 'Coffee Mug', 'base_demand': 12, 'price': 9.99, 'category': 'Lifestyle', 'seasonal': 'winter'},
    }
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    records = []
    for date in date_range:
        day_of_week = date.dayofweek
        month = date.month
        day_of_year = date.timetuple().tm_yday
        for product_id, info in products.items():
            base = info['base_demand']
            years_passed = (date - pd.Timestamp(start_date)).days / 365.25
            trend = 1 + 0.05 * years_passed
            seasonal_factor = 1.0
            if info['seasonal'] == 'summer':
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            elif info['seasonal'] == 'winter':
                seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * (day_of_year - 260) / 365)
            elif info['seasonal'] == 'back_to_school':
                seasonal_factor = 1 + 0.6 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
                if month == 1:
                    seasonal_factor += 0.3
            dow_factor = {5: 1.3, 6: 1.2, 0: 0.85}.get(day_of_week, 1.0)
            holiday_factor = 1.0
            if month == 11 and 24 <= date.day <= 30:
                holiday_factor = 2.5
            elif month == 12 and date.day >= 15:
                holiday_factor = 2.0
            elif month == 1 and date.day <= 3:
                holiday_factor = 1.5
            promotion = 0
            promo_factor = 1.0
            if np.random.random() < 0.10:
                promotion = 1
                promo_factor = 1.4 + np.random.uniform(0, 0.3)
            demand = base * trend * seasonal_factor * dow_factor * holiday_factor * promo_factor
            noise = np.random.normal(0, base * 0.15)
            demand = max(0, int(round(demand + noise)))
            actual_price = info['price'] * (0.8 if promotion else 1.0)
            revenue = round(demand * actual_price, 2)
            max_stock = base * 10
            stock_level = max(0, int(max_stock - demand + np.random.randint(-20, 20)))
            records.append({
                'date': date.strftime('%Y-%m-%d'), 'product_id': product_id,
                'product_name': info['name'], 'category': info['category'],
                'quantity_sold': demand, 'unit_price': round(actual_price, 2),
                'revenue': revenue, 'stock_level': stock_level,
                'promotion': promotion, 'day_of_week': date.strftime('%A'),
                'month': month, 'year': date.year,
            })
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'  Generated {len(df):,} sales records')
    print(f'  Saved to: {output_path}')
    return df
