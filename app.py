import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_sales_data
from src.data_preprocessing import load_data, clean_data, prepare_product_data
from src.feature_engineering import create_all_features
from src.model_training import train_test_split_time, train_models, get_feature_importance
from src.forecasting import forecast_future, calculate_inventory_recommendations

st.set_page_config(page_title='Inventory Forecast', page_icon='📦', layout='wide')


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
    il = best in ['Linear Regression', 'Ridge Regression']
    fc = forecast_future(bm, pd_data, fcols, fdays, il, scaler if il else None)
    imp = None
    if hasattr(bm, 'feature_importances_'):
        imp = pd.DataFrame({'feature': fcols, 'importance': bm.feature_importances_}).sort_values('importance', ascending=False)
    return {'pd': pd_data, 'rdf': rdf, 'preds': preds, 'best': best, 'tdf': tdf, 'fc': fc, 'imp': imp}


def main():
    st.title('📦 Smart Inventory Forecasting')
    st.caption('AI-Powered Demand Prediction')
    df = get_data()
    prods = df.groupby(['product_id','product_name','category']).size().reset_index()[['product_id','product_name','category']]

    st.sidebar.title('Settings')
    sel = st.sidebar.selectbox('Product', prods['product_name'].values)
    pr = prods[prods['product_name']==sel].iloc[0]
    pid = pr['product_id']
    st.sidebar.markdown(f'**{pid}** | {pr["category"]}')
    st.sidebar.markdown('---')
    fdays = st.sidebar.slider('Forecast Days', 7, 90, 30)
    tdays = st.sidebar.slider('Test Days', 30, 180, 90)
    st.sidebar.markdown('---')
    cstock = st.sidebar.number_input('Current Stock', 0, 10000, 150)
    lt = st.sidebar.number_input('Lead Time', 1, 30, 7)
    sf = st.sidebar.slider('Safety Factor', 1.0, 3.0, 1.5, 0.1)
    run = st.sidebar.button('Run Forecast', type='primary', use_container_width=True)

    t1, t2, t3, t4, t5 = st.tabs(['Overview','Product','Models','Forecast','Inventory'])

    with t1:
        st.header('Sales Overview')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Records', f'{len(df):,}')
        c2.metric('Products', df['product_id'].nunique())
        c3.metric('Revenue', f'${df["revenue"].sum():,.0f}')
        c4.metric('Avg Daily', f'{df.groupby("date")["quantity_sold"].sum().mean():.0f}')
        dt = df.groupby('date')['quantity_sold'].sum().reset_index()
        dt['MA30'] = dt['quantity_sold'].rolling(30).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dt['date'], y=dt['quantity_sold'], mode='lines', name='Daily', opacity=0.3))
        fig.add_trace(go.Scatter(x=dt['date'], y=dt['MA30'], mode='lines', name='30d MA', line=dict(color='red', width=2)))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            cr = df.groupby('category')['revenue'].sum().reset_index()
            st.plotly_chart(px.pie(cr, values='revenue', names='category', hole=0.4, title='Revenue'), use_container_width=True)
        with c2:
            ps = df.groupby('product_name')['quantity_sold'].sum().sort_values(ascending=True).reset_index()
            st.plotly_chart(px.bar(ps, x='quantity_sold', y='product_name', orientation='h', title='Sales'), use_container_width=True)

    with t2:
        st.header(f'{sel}')
        pdd = prepare_product_data(df, pid)
        c1,c2,c3 = st.columns(3)
        c1.metric('Total', f'{pdd["quantity_sold"].sum():,}')
        c2.metric('Avg/Day', f'{pdd["quantity_sold"].mean():.1f}')
        c3.metric('Revenue', f'${pdd["revenue"].sum():,.0f}')
        pdd['MA30'] = pdd['quantity_sold'].rolling(30).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pdd['date'], y=pdd['quantity_sold'], mode='lines', name='Daily', opacity=0.4))
        fig.add_trace(go.Scatter(x=pdd['date'], y=pdd['MA30'], mode='lines', name='30d MA', line=dict(color='red', width=2)))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if run or 'res' in st.session_state:
        if run:
            with st.spinner('Running...'):
                st.session_state['res'] = run_pipeline(df, pid, fdays, tdays)
        if 'res' in st.session_state:
            r = st.session_state['res']
            with t3:
                st.header('Models')
                st.success(f'Best: **{r["best"]}**')
                st.dataframe(r['rdf'], use_container_width=True)
                c1,c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.bar(r['rdf'], x='model', y='MAE', color='MAE', title='MAE'), use_container_width=True)
                with c2:
                    st.plotly_chart(px.bar(r['rdf'], x='model', y='R2', color='R2', title='R2'), use_container_width=True)
                td = r['tdf']
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=td['date'], y=td['quantity_sold'], mode='lines', name='Actual', line=dict(color='black', width=2)))
                cols = px.colors.qualitative.Set2
                for i,(n,p) in enumerate(r['preds'].items()):
                    fig.add_trace(go.Scatter(x=td['date'], y=p, mode='lines', name=n, opacity=0.7, line=dict(color=cols[i%len(cols)])))
                fig.update_layout(height=500, title='Predictions vs Actual')
                st.plotly_chart(fig, use_container_width=True)
                if r['imp'] is not None:
                    st.subheader('Top Features')
                    st.plotly_chart(px.bar(r['imp'].head(15), x='importance', y='feature', orientation='h').update_layout(yaxis=dict(autorange='reversed'), height=500), use_container_width=True)
            with t4:
                st.header(f'{fdays}-Day Forecast')
                fc = r['fc']
                c1,c2,c3 = st.columns(3)
                c1.metric('Total', f'{fc["predicted_demand"].sum():,}')
                c2.metric('Avg/Day', f'{fc["predicted_demand"].mean():.1f}')
                c3.metric('Peak', f'{fc["predicted_demand"].max()}')
                hist = r['pd'].tail(90)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['quantity_sold'], mode='lines', name='Historical', line=dict(color='steelblue', width=2)))
                fig.add_trace(go.Scatter(x=fc['date'], y=fc['predicted_demand'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=fc['date'], y=fc['predicted_demand']*1.2, mode='lines', showlegend=False, line=dict(width=0)))
                fig.add_trace(go.Scatter(x=fc['date'], y=fc['predicted_demand']*0.8, fill='tonexty', fillcolor='rgba(255,0,0,0.1)', mode='lines', name='Band', line=dict(width=0)))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                dd = fc.copy()
                dd['date'] = dd['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(dd, use_container_width=True)
            with t5:
                st.header('Inventory Plan')
                fc = r['fc']
                rec = calculate_inventory_recommendations(fc, sel, lt, sf, cstock)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Stock', rec['current_stock'])
                c2.metric('Days Left', f'{rec["days_of_stock_remaining"]:.0f}')
                c3.metric('Reorder', rec['reorder_point'])
                c4.metric('Risk', rec['stockout_risk'])
                c1,c2 = st.columns(2)
                c1.metric('Order Qty', f'{rec["recommended_order_qty"]} units')
                c2.metric('Safety Stock', f'{rec["safety_stock"]} units')
                stk = cstock
                sd = [{'day':0,'stock':stk}]
                for d in fc['predicted_demand'].values:
                    stk = max(0, stk-d)
                    sd.append({'day':len(sd),'stock':stk})
                spd = pd.DataFrame(sd)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spd['day'], y=spd['stock'], mode='lines+markers', name='Stock'))
                fig.add_hline(y=rec['reorder_point'], line_dash='dash', line_color='orange', annotation_text='Reorder')
                fig.add_hline(y=rec['safety_stock'], line_dash='dash', line_color='red', annotation_text='Safety')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                risk = rec['stockout_risk']
                if 'HIGH' in risk:
                    st.error(f'Order {rec["recommended_order_qty"]} units NOW!')
                elif 'MEDIUM' in risk:
                    st.warning(f'Plan order: {rec["recommended_order_qty"]} units soon.')
                else:
                    st.success(f'Stock OK. Next: {rec["recommended_order_qty"]} units.')
    else:
        for t in [t3,t4,t5]:
            with t:
                st.info('Click Run Forecast in sidebar')

    st.markdown('---')
    st.caption('Smart Inventory Forecasting | ML Powered')


if __name__ == '__main__':
    main()
