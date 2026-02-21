import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_sales_data
from src.data_preprocessing import load_data, clean_data, prepare_product_data, detect_anomalies
from src.feature_engineering import create_all_features
from src.model_training import train_test_split_time, train_models, get_feature_importance
from src.forecasting import forecast_future, calculate_inventory_recommendations, scenario_analysis

st.set_page_config(page_title='Smart Inventory Forecast', page_icon='📦', layout='wide')


@st.cache_data
def get_data():
    p = 'data/sales_data.csv'
    if not os.path.exists(p):
        generate_sales_data(output_path=p)
    return clean_data(load_data(p))


@st.cache_data
def run_pipeline(_df, pid, fdays, tdays):
    pd_data = prepare_product_data(_df, pid)
    fdf, fcols = create_all_features(pd_data)
    Xtr, Xte, ytr, yte, _, tdf = train_test_split_time(fdf, fcols, test_days=tdays)
    trained, scaler, rdf, preds, best = train_models(Xtr, Xte, ytr, yte)
    bm = trained[best]
    il = best in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
    fc = forecast_future(bm, pd_data, fcols, fdays, il, scaler if il else None)
    imp = None
    if hasattr(bm, 'feature_importances_'):
        imp = pd.DataFrame({'feature': fcols, 'importance': bm.feature_importances_}).sort_values('importance', ascending=False)
    elif hasattr(bm, 'coef_'):
        imp = pd.DataFrame({'feature': fcols, 'importance': np.abs(bm.coef_)}).sort_values('importance', ascending=False)
    scenarios = scenario_analysis(bm, pd_data, fcols, fdays, il, scaler if il else None, current_stock=150)
    return {'pd': pd_data, 'rdf': rdf, 'preds': preds, 'best': best, 'tdf': tdf, 'fc': fc, 'imp': imp, 'scenarios': scenarios}


def main():
    st.title('📦 Smart Inventory Forecasting')
    st.caption('AI-Powered Demand Prediction & Inventory Optimization')
    df = get_data()
    prods = df.groupby(['product_id','product_name','category']).size().reset_index()[['product_id','product_name','category']]

    st.sidebar.title('⚙️ Settings')
    sel = st.sidebar.selectbox('📦 Product', prods['product_name'].values)
    pr = prods[prods['product_name']==sel].iloc[0]
    pid = pr['product_id']
    st.sidebar.markdown(f'**ID:** `{pid}` | **Category:** {pr["category"]}')
    st.sidebar.markdown('---')
    fdays = st.sidebar.slider('Forecast Days', 7, 90, 30)
    tdays = st.sidebar.slider('Test Days', 30, 180, 90)
    st.sidebar.markdown('---')
    cstock = st.sidebar.number_input('Current Stock', 0, 10000, 150)
    lt = st.sidebar.number_input('Lead Time (days)', 1, 30, 7)
    sf = st.sidebar.slider('Safety Factor', 1.0, 3.0, 1.5, 0.1)
    run = st.sidebar.button('🚀 Run Forecast', type='primary', use_container_width=True)

    t1, t2, t3, t4, t5, t6 = st.tabs(['📊 Overview', '🔍 Product', '🤖 Models', '📈 Forecast', '📦 Inventory', '🎯 Scenarios'])

    with t1:
        st.header('Sales Overview')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Records', f'{len(df):,}')
        c2.metric('Products', df['product_id'].nunique())
        c3.metric('Revenue', f'${df["revenue"].sum():,.0f}')
        c4.metric('Avg Daily', f'{df.groupby("date")["quantity_sold"].sum().mean():.0f}')
        dt = df.groupby('date')['quantity_sold'].sum().reset_index()
        dt['MA30'] = dt['quantity_sold'].rolling(30).mean()
        dt['MA90'] = dt['quantity_sold'].rolling(90).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dt['date'], y=dt['quantity_sold'], mode='lines', name='Daily', opacity=0.3))
        fig.add_trace(go.Scatter(x=dt['date'], y=dt['MA30'], mode='lines', name='30d MA', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=dt['date'], y=dt['MA90'], mode='lines', name='90d MA', line=dict(color='green', width=2, dash='dash')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            cr = df.groupby('category')['revenue'].sum().reset_index()
            st.plotly_chart(px.pie(cr, values='revenue', names='category', hole=0.4, title='Revenue by Category'), use_container_width=True)
        with c2:
            ps = df.groupby('product_name')['quantity_sold'].sum().sort_values(ascending=True).reset_index()
            st.plotly_chart(px.bar(ps, x='quantity_sold', y='product_name', orientation='h', title='Sales by Product', color='quantity_sold', color_continuous_scale='Blues'), use_container_width=True)

    with t2:
        st.header(f'🔍 {sel}')
        pdd = prepare_product_data(df, pid)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total Sold', f'{pdd["quantity_sold"].sum():,}')
        c2.metric('Avg/Day', f'{pdd["quantity_sold"].mean():.1f}')
        c3.metric('Revenue', f'${pdd["revenue"].sum():,.0f}')
        c4.metric('Max Daily', f'{pdd["quantity_sold"].max()}')
        pdd_plot = pdd.copy()
        pdd_plot['MA30'] = pdd_plot['quantity_sold'].rolling(30).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pdd_plot['date'], y=pdd_plot['quantity_sold'], mode='lines', name='Daily', opacity=0.4))
        fig.add_trace(go.Scatter(x=pdd_plot['date'], y=pdd_plot['MA30'], mode='lines', name='30d MA', line=dict(color='red', width=2)))
        promo_data = pdd_plot[pdd_plot['promotion']==1]
        if len(promo_data) > 0:
            fig.add_trace(go.Scatter(x=promo_data['date'], y=promo_data['quantity_sold'], mode='markers', name='Promotions', marker=dict(color='green', size=5, symbol='triangle-up')))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            pdd_plot['month_name'] = pdd_plot['date'].dt.month_name()
            month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
            st.plotly_chart(px.box(pdd_plot, x='month_name', y='quantity_sold', title='Monthly Distribution', category_orders={'month_name': month_order}), use_container_width=True)
        with c2:
            anom = detect_anomalies(pdd)
            anomalies = anom[anom['is_anomaly']]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=anom['date'], y=anom['quantity_sold'], mode='lines', name='Sales', opacity=0.6))
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['quantity_sold'], mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
            fig.update_layout(title=f'Anomalies ({len(anomalies)} found)', height=400)
            st.plotly_chart(fig, use_container_width=True)

    if run or 'res' in st.session_state:
        if run:
            with st.spinner('🔄 Training models and generating forecast...'):
                st.session_state['res'] = run_pipeline(df, pid, fdays, tdays)
        if 'res' in st.session_state:
            r = st.session_state['res']
            with t3:
                st.header('🤖 Model Performance')
                st.success(f'🏆 Best: **{r["best"]}**')
                st.dataframe(r['rdf'], use_container_width=True)
                c1,c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.bar(r['rdf'], x='model', y='MAE', color='MAE', title='MAE (lower=better)', color_continuous_scale='RdYlGn_r'), use_container_width=True)
                with c2:
                    st.plotly_chart(px.bar(r['rdf'], x='model', y='R2', color='R2', title='R² (higher=better)', color_continuous_scale='RdYlGn'), use_container_width=True)
                td = r['tdf']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=td['date'], y=td['quantity_sold'], mode='lines', name='Actual', line=dict(color='black', width=2)))
                cols = px.colors.qualitative.Set2
                for i,(n,p) in enumerate(r['preds'].items()):
                    opacity = 1.0 if n == r['best'] else 0.4
                    width = 2.5 if n == r['best'] else 1.0
                    fig.add_trace(go.Scatter(x=td['date'], y=p, mode='lines', name=n, opacity=opacity, line=dict(color=cols[i%len(cols)], width=width)))
                fig.update_layout(height=500, title='Predictions vs Actual')
                st.plotly_chart(fig, use_container_width=True)
                if r['imp'] is not None:
                    st.subheader('🔑 Top Features')
                    st.plotly_chart(px.bar(r['imp'].head(15), x='importance', y='feature', orientation='h', title='Feature Importance', color='importance', color_continuous_scale='Viridis').update_layout(yaxis=dict(autorange='reversed'), height=500), use_container_width=True)
            with t4:
                st.header(f'📈 {fdays}-Day Forecast')
                fc = r['fc']
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Total Demand', f'{fc["predicted_demand"].sum():,}')
                c2.metric('Avg/Day', f'{fc["predicted_demand"].mean():.1f}')
                c3.metric('Peak', f'{fc["predicted_demand"].max()}')
                c4.metric('Min', f'{fc["predicted_demand"].min()}')
                hist = r['pd'].tail(90)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['quantity_sold'], mode='lines', name='Historical', line=dict(color='steelblue', width=2)))
                fig.add_trace(go.Scatter(x=fc['date'], y=fc['predicted_demand'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2, dash='dash'), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=fc['date'], y=fc['predicted_demand']*1.2, mode='lines', showlegend=False, line=dict(width=0)))
                fig.add_trace(go.Scatter(x=fc['date'], y=fc['predicted_demand']*0.8, fill='tonexty', fillcolor='rgba(255,0,0,0.1)', mode='lines', name='±20% Band', line=dict(width=0)))
                fig.add_vline(x=fc['date'].iloc[0], line_dash='dot', line_color='gray')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                dd = fc.copy()
                dd['date'] = dd['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(dd, use_container_width=True)
                csv = dd.to_csv(index=False)
                st.download_button('📥 Download Forecast CSV', csv, 'forecast.csv', 'text/csv', use_container_width=True)
            with t5:
                st.header('📦 Inventory Plan')
                fc = r['fc']
                rec = calculate_inventory_recommendations(fc, sel, lt, sf, cstock)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Stock', f'{rec["current_stock"]} units')
                c2.metric('Days Left', f'{rec["days_of_stock_remaining"]:.0f}')
                c3.metric('Reorder Point', f'{rec["reorder_point"]}')
                risk_emoji = {'LOW':'🟢','MEDIUM':'🟡','HIGH':'🟠','CRITICAL':'🔴'}
                c4.metric('Risk', f'{risk_emoji.get(rec["stockout_risk"],"")} {rec["stockout_risk"]}')
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Order Qty', f'{rec["recommended_order_qty"]} units')
                c2.metric('Safety Stock', f'{rec["safety_stock"]} units')
                c3.metric('Holding Cost', f'${rec.get("estimated_holding_cost",0):.2f}')
                c4.metric('Stockout Cost', f'${rec.get("estimated_stockout_cost",0):.2f}')
                stk = cstock
                sd = [{'day':0,'stock':stk}]
                for d in fc['predicted_demand'].values:
                    stk = max(0, stk-d)
                    sd.append({'day':len(sd),'stock':stk})
                spd = pd.DataFrame(sd)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spd['day'], y=spd['stock'], mode='lines+markers', name='Stock', marker=dict(size=4)))
                fig.add_hline(y=rec['reorder_point'], line_dash='dash', line_color='orange', annotation_text='Reorder')
                fig.add_hline(y=rec['safety_stock'], line_dash='dash', line_color='red', annotation_text='Safety')
                fig.add_hrect(y0=0, y1=rec['safety_stock'], fillcolor='red', opacity=0.1, line_width=0)
                fig.update_layout(height=400, xaxis_title='Days', yaxis_title='Stock')
                st.plotly_chart(fig, use_container_width=True)
                risk = rec['stockout_risk']
                if risk == 'CRITICAL':
                    st.error(f'🚨 CRITICAL: Order {rec["recommended_order_qty"]} units IMMEDIATELY!')
                elif risk == 'HIGH':
                    st.warning(f'⚠️ Order {rec["recommended_order_qty"]} units within {lt} days!')
                elif risk == 'MEDIUM':
                    st.info(f'📋 Plan order: {rec["recommended_order_qty"]} units soon.')
                else:
                    st.success(f'✅ Stock OK. Next order: {rec["recommended_order_qty"]} units.')
                if rec.get('stockout_date'):
                    st.error(f'📅 Stockout projected: Day {rec["stockout_day"]}')
            with t6:
                st.header('🎯 Scenario Analysis')
                st.markdown('Compare optimistic, normal, and pessimistic demand scenarios.')
                scenarios = r['scenarios']
                fig = go.Figure()
                sc_colors = {'optimistic':'green','normal':'steelblue','pessimistic':'red'}
                for name, data in scenarios.items():
                    sfc = data['forecast']
                    fig.add_trace(go.Scatter(x=sfc['date'], y=sfc['predicted_demand'], mode='lines', name=name.title(), line=dict(color=sc_colors.get(name,'gray'), width=2)))
                fig.update_layout(height=400, title='Demand Scenarios')
                st.plotly_chart(fig, use_container_width=True)
                cols_sc = st.columns(3)
                for i, (name, data) in enumerate(scenarios.items()):
                    srec = data['recommendation']
                    with cols_sc[i]:
                        st.subheader(f'{name.title()}')
                        st.metric('Total Demand', srec['total_forecasted_demand'])
                        st.metric('Order Qty', srec['recommended_order_qty'])
                        st.metric('Risk', srec['stockout_risk'])
                        st.metric('Days of Stock', f'{srec["days_of_stock_remaining"]:.0f}')
    else:
        for t in [t3,t4,t5,t6]:
            with t:
                st.info('👈 Click **Run Forecast** in sidebar')

    st.markdown('---')
    st.markdown('<div style="text-align:center;color:gray;">📦 Smart Inventory Forecasting v2.0 | ML Powered</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
