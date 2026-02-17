import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

OUTPUT_DIR = 'output/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    pass


def plot_sales_overview(df, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sales Overview', fontsize=16, fontweight='bold')
    ds = df.groupby('date')['quantity_sold'].sum()
    axes[0,0].plot(ds.index, ds.values, alpha=0.5, linewidth=0.5)
    axes[0,0].plot(ds.rolling(30).mean(), color='red', linewidth=2)
    axes[0,0].set_title('Daily Sales')
    mr = df.groupby([pd.Grouper(key='date', freq='M'), 'category'])['revenue'].sum().reset_index()
    for cat in mr['category'].unique():
        cd = mr[mr['category']==cat]
        axes[0,1].plot(cd['date'], cd['revenue'], label=cat)
    axes[0,1].set_title('Revenue by Category')
    axes[0,1].legend(fontsize=7)
    dwo = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dws = df.groupby('day_of_week')['quantity_sold'].mean().reindex(dwo)
    axes[1,0].bar(range(7), dws.values)
    axes[1,0].set_xticks(range(7))
    axes[1,0].set_xticklabels(['M','T','W','T','F','S','S'])
    axes[1,0].set_title('Avg by Day')
    tp = df.groupby('product_name')['quantity_sold'].sum().sort_values(ascending=True)
    axes[1,1].barh(range(len(tp)), tp.values)
    axes[1,1].set_yticks(range(len(tp)))
    axes[1,1].set_yticklabels(tp.index, fontsize=7)
    axes[1,1].set_title('Sales by Product')
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'sales_overview.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_product_analysis(product_daily, product_name, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Analysis: {product_name}', fontsize=16, fontweight='bold')
    axes[0,0].plot(product_daily['date'], product_daily['quantity_sold'], alpha=0.4, linewidth=0.5)
    axes[0,0].plot(product_daily['date'], product_daily['quantity_sold'].rolling(30).mean(), color='red', linewidth=2)
    axes[0,0].set_title('Daily Sales')
    pdf = product_daily.copy()
    pdf['month'] = pdf['date'].dt.month
    md = [pdf[pdf['month']==m]['quantity_sold'].values for m in range(1,13)]
    axes[0,1].boxplot(md, labels=['J','F','M','A','M','J','J','A','S','O','N','D'])
    axes[0,1].set_title('Monthly')
    axes[1,0].hist(product_daily['quantity_sold'], bins=50, color='steelblue', edgecolor='white')
    axes[1,0].set_title('Distribution')
    pdf['year'] = pdf['date'].dt.year
    for y in sorted(pdf['year'].unique()):
        yd = pdf[pdf['year']==y]
        axes[1,1].plot(yd.groupby('month')['quantity_sold'].mean(), marker='o', label=str(y))
    axes[1,1].set_title('YoY')
    axes[1,1].legend(fontsize=7)
    plt.tight_layout()
    if save:
        sn = product_name.replace(' ','_').lower()
        fp = os.path.join(OUTPUT_DIR, f'product_{sn}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_model_comparison(results_df, save=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    x = range(len(results_df))
    m = results_df['model'].values
    axes[0].bar(x, results_df['MAE'].values, color='salmon')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(m, rotation=45, ha='right', fontsize=7)
    axes[0].set_title('MAE')
    axes[1].bar(x, results_df['RMSE'].values, color='lightskyblue')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(m, rotation=45, ha='right', fontsize=7)
    axes[1].set_title('RMSE')
    axes[2].bar(x, results_df['R2'].values, color='lightgreen')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(m, rotation=45, ha='right', fontsize=7)
    axes[2].set_title('R2')
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'model_comparison.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_predictions_vs_actual(test_df, predictions_dict, best_model_name, save=True):
    fig, ax = plt.subplots(figsize=(16, 6))
    dates = test_df['date'].values
    actual = test_df['quantity_sold'].values
    ax.plot(dates, actual, 'k-', linewidth=2, label='Actual')
    bp = predictions_dict[best_model_name]
    ax.plot(dates, bp, 'r-', linewidth=2, label=f'Best: {best_model_name}')
    ax.fill_between(dates, actual, bp, alpha=0.2, color='red')
    ax.set_title('Predictions vs Actual')
    ax.legend()
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'predictions_vs_actual.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_forecast(historical_df, forecast_df, product_name, save=True):
    fig, ax = plt.subplots(figsize=(16, 6))
    recent = historical_df.tail(90)
    ax.plot(recent['date'], recent['quantity_sold'], 'b-', linewidth=1.5, label='Historical')
    ax.plot(forecast_df['date'], forecast_df['predicted_demand'], 'r--', linewidth=2, label='Forecast')
    ax.fill_between(forecast_df['date'], forecast_df['predicted_demand']*0.8, forecast_df['predicted_demand']*1.2, alpha=0.15, color='red')
    ax.axvline(x=forecast_df['date'].iloc[0], color='gray', linestyle=':')
    ax.set_title(f'Forecast: {product_name}')
    ax.legend()
    plt.tight_layout()
    if save:
        sn = product_name.replace(' ','_').lower()
        fp = os.path.join(OUTPUT_DIR, f'forecast_{sn}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_feature_importance(importance_df, model_name, top_n=15, save=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance_df.head(top_n)
    ax.barh(range(top_n), top['importance'].values[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=8)
    ax.set_title(f'Top Features ({model_name})')
    plt.tight_layout()
    if save:
        fp = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()


def plot_inventory_dashboard(forecast_df, recommendations, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pn = recommendations['product']
    fig.suptitle(f'Inventory: {pn}', fontsize=16, fontweight='bold')
    axes[0].bar(forecast_df['date'], forecast_df['predicted_demand'], color='steelblue', alpha=0.7)
    axes[0].axhline(y=recommendations['avg_daily_demand'], color='red', linestyle='--')
    axes[0].set_title('Forecast Demand')
    axes[0].tick_params(axis='x', rotation=45)
    stk = recommendations['current_stock']
    sl = [stk]
    for d in forecast_df['predicted_demand'].values:
        stk = max(0, stk-d)
        sl.append(stk)
    axes[1].plot(sl, 'b-', linewidth=2)
    axes[1].axhline(y=recommendations['reorder_point'], color='orange', linestyle='--', label='Reorder')
    axes[1].axhline(y=recommendations['safety_stock'], color='red', linestyle='--', label='Safety')
    axes[1].set_title('Stock Projection')
    axes[1].legend()
    plt.tight_layout()
    if save:
        sn = pn.replace(' ','_').lower()
        fp = os.path.join(OUTPUT_DIR, f'dashboard_{sn}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f'  Plot: {fp}')
    plt.close()
