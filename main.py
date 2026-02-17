import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_sales_data
from src.data_preprocessing import load_data, clean_data, get_data_summary, prepare_product_data
from src.feature_engineering import create_all_features
from src.model_training import train_test_split_time, train_models, get_feature_importance, cross_validate_model, save_model
from src.forecasting import forecast_future, calculate_inventory_recommendations
from src.visualization import (plot_sales_overview, plot_product_analysis, plot_model_comparison,
                                plot_predictions_vs_actual, plot_forecast, plot_feature_importance,
                                plot_inventory_dashboard)


def main():
    print('=' * 65)
    print('  SMART INVENTORY FORECASTING FOR SMALL BUSINESS')
    print('=' * 65)

    print('\n[STEP 1] Data Generation')
    data_path = 'data/sales_data.csv'
    if not os.path.exists(data_path):
        raw_df = generate_sales_data(output_path=data_path)
    else:
        raw_df = load_data(data_path)

    print('\n[STEP 2] Preprocessing')
    df = clean_data(raw_df)
    get_data_summary(df)

    print('\n[STEP 3] Visualization')
    plot_sales_overview(df)

    products = {'P001': 'Wireless Mouse', 'P003': 'Notebook A5', 'P006': 'Hand Sanitizer'}
    all_results = {}

    for pid, pname in products.items():
        print(f'\n{"="*65}')
        print(f'  Processing: {pname} ({pid})')
        print(f'{"="*65}')

        pd_data = prepare_product_data(df, pid)
        print(f'  Data points: {len(pd_data)}')
        plot_product_analysis(pd_data, pname)

        fdf, fcols = create_all_features(pd_data)
        print(f'  Features: {len(fcols)} | Rows: {len(fdf)}')

        Xtr, Xte, ytr, yte, tr_df, te_df = train_test_split_time(fdf, fcols, test_days=90)

        trained, scaler, rdf, preds, best = train_models(Xtr, Xte, ytr, yte)

        plot_model_comparison(rdf)
        plot_predictions_vs_actual(te_df, preds, best)

        bm = trained[best]
        if hasattr(bm, 'feature_importances_'):
            imp = get_feature_importance(bm, fcols, best)
            if imp is not None:
                plot_feature_importance(imp, best)

        if best in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            cross_validate_model(fdf[fcols], fdf['quantity_sold'], trained[best])

        is_lin = best in ['Linear Regression', 'Ridge Regression']
        fcast = forecast_future(bm, pd_data, fcols, 30, is_lin, scaler if is_lin else None)
        plot_forecast(pd_data, fcast, pname)

        cs = int(pd_data['stock_level'].iloc[-1])
        if cs == 0:
            cs = 100
        rec = calculate_inventory_recommendations(fcast, pname, 7, 1.5, cs)
        plot_inventory_dashboard(fcast, rec)
        save_model(bm, scaler, fcols, pid)

        all_results[pid] = {'name': pname, 'best': best, 'results': rdf, 'rec': rec}

    print(f'\n{"="*65}')
    print('  FINAL SUMMARY')
    print(f'{"="*65}')
    for pid, r in all_results.items():
        b = r['results'].iloc[0]
        rc = r['rec']
        print(f'  {r["name"]} ({pid}): {r["best"]} | MAE={b["MAE"]} | R2={b["R2"]}')
        print(f'    Demand: {rc["total_forecasted_demand"]} | Order: {rc["recommended_order_qty"]} | Risk: {rc["stockout_risk"]}')

    print(f'\n  Plots: output/plots/')
    print(f'  Models: models/')
    print(f'\n{"="*65}')
    print('  DONE! Now run: streamlit run app.py')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
