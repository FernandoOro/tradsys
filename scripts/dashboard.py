import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os
from sqlalchemy import create_engine
from datetime import datetime

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import config
from src.database.models import Trade, Signal

# Page Config
st.set_page_config(
    page_title="Smart Spot Cockpit",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database Connection
@st.cache_resource
def get_engine():
    db_path = config.get_db_path()
    return create_engine(f'sqlite:///{db_path}')

engine = get_engine()

# --- Sidebar ---
st.sidebar.title("🦅 Smart Spot System")
st.sidebar.markdown("---")
refresh_rate = st.sidebar.slider("Refresh Rate (s)", 5, 60, 10)
if st.sidebar.button("Refresh Now"):
    st.cache_data.clear()

st.sidebar.markdown("### Modes")
st.sidebar.info(f"DB Path: {config.BASE_DIR / 'tradsys.db'}")

# --- Data Fetching ---
def load_trades():
    try:
        query = "SELECT * FROM trades ORDER BY timestamp DESC"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()

def load_signals():
    try:
        query = "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 200"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error loading signals: {e}")
        return pd.DataFrame()

# --- Main Layout ---
col1, col2, col3 = st.columns(3)

# Load Data
df_trades = load_trades()
df_signals = load_signals()

# Metrics
total_trades = len(df_trades)
if not df_trades.empty:
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / total_trades * 100 if 'pnl' in df_trades.columns else 0
    # Approx PnL if not populated yet
    pnl_sum = df_trades['pnl'].sum() if 'pnl' in df_trades.columns else 0.0
else:
    win_rate = 0
    pnl_sum = 0.0

col1.metric("Total Trades", total_trades)
col2.metric("Win Rate", f"{win_rate:.1f}%")
col3.metric("Total PnL (Est)", f"${pnl_sum:.2f}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🚀 Execution", "🧠 Signals (Brain)", "📉 Performance"])

with tab1:
    st.subheader("Recent Executions")
    if not df_trades.empty:
        st.dataframe(df_trades.head(20), use_container_width=True)
    else:
        st.info("No trades recorded yet.")

with tab2:
    st.subheader("AI Alpha Signals")
    if not df_signals.empty:
        # Charts for Confidence
        fig_conf = px.line(df_signals, x='timestamp', y='confidence', title="Model Confidence over Time")
        st.plotly_chart(fig_conf, use_container_width=True)
        
        st.dataframe(df_signals, use_container_width=True)
    else:
        st.info("No signals recorded yet.")

with tab3:
    st.subheader("Equity Curve")
    if not df_trades.empty and 'pnl' in df_trades.columns:
        # Cumulative Sum ignoring NaNs
        df_trades = df_trades.sort_values('timestamp')
        df_trades['equity'] = df_trades['pnl'].fillna(0).cumsum()
        
        fig = px.line(df_trades, x='timestamp', y='equity', title="Cumulative PnL")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough PnL data to plot curve.")

# --- Footer ---
st.markdown("---")
st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
time.sleep(refresh_rate)
st.rerun()
