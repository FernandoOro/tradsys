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
# --- Classification Logic ---
def classify_bot(status):
    status = str(status).upper()
    # Hybrid Architecture Map
    if '1H' in status and ('SHORT' in status or 'FUT' in status or 'OPEN_1H_SHORT' in status): return 'Agent 1H (Futures)'
    if '1H' in status: return 'Agent 1H (Spot)'
    
    if 'AUDITED' in status and 'FUT' in status: return 'Agent 1 (Audited) [FUTURES]'
    if 'AUDITED' in status: return 'Agent 1 (Audited)'
    
    if 'SNIPER' in status and 'FUT' in status: return 'Agent 1 (Sniper) [FUTURES]'
    if 'SNIPER' in status: return 'Agent 1 (Sniper)'
    
    if 'RECKLESS' in status and 'FUT' in status: return 'Agent 1 (Reckless) [FUTURES]'
    if 'RECKLESS' in status: return 'Agent 1 (Reckless)'
    
    if 'TESTNET' in status or '1M' in status: return 'Agent 1 (Legacy)'
    return 'Unknown'

col1, col2 = st.columns([1, 2])

# Load Data
df_trades = load_trades()
df_signals = load_signals()

# Global Metrics
total_trades = len(df_trades)
if not df_trades.empty:
    # Ensure PnL exists
    if 'pnl' not in df_trades.columns: df_trades['pnl'] = 0.0
    
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / total_trades * 100
    pnl_sum = df_trades['pnl'].sum()
    
    # --- BOT COMPARISON ---
    df_trades['Bot'] = df_trades['status'].apply(classify_bot)
    
    summary = df_trades.groupby('Bot').agg(
        Trades=('id', 'count'),
        Win_Rate=('pnl', lambda x: (x > 0).mean() * 100),
        Total_PnL=('pnl', 'sum'),
        Avg_PnL=('pnl', 'mean')
    ).reset_index()
    
    # Ensure expected bots exist
    # Ensure expected bots exist (Hybrid Fleet)
    expected_bots = [
        'Agent 1H (Spot)', 'Agent 1H (Futures)',
        'Agent 1 (Audited)', 'Agent 1 (Audited) [FUTURES]',
        'Agent 1 (Sniper)', 'Agent 1 (Sniper) [FUTURES]',
        'Agent 1 (Reckless)', 'Agent 1 (Reckless) [FUTURES]'
    ]
    for bot in expected_bots:
        if bot not in summary['Bot'].values:
            new_row = {'Bot': bot, 'Trades': 0, 'Win_Rate': 0.0, 'Total_PnL': 0.0, 'Avg_PnL': 0.0}
            summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)

    # Format for display
    summary['Win_Rate'] = summary['Win_Rate'].map('{:.1f}%'.format)
    summary['Total_PnL'] = summary['Total_PnL'].map('${:.2f}'.format)
    summary['Avg_PnL'] = summary['Avg_PnL'].map('${:.2f}'.format)
    
    # Sort: 1H first, then Audited, Sniper, Reckless
    def sort_bots(name):
        if '1H' in name and 'Futures' in name: return 1
        if '1H' in name: return 0
        
        if 'Audited' in name and 'FUTURES' in name: return 3
        if 'Audited' in name: return 2
        
        if 'Sniper' in name and 'FUTURES' in name: return 5
        if 'Sniper' in name: return 4
        
        if 'Reckless' in name and 'FUTURES' in name: return 7
        if 'Reckless' in name: return 6
        return 99
        return 99
        
    summary['sort_key'] = summary['Bot'].apply(sort_bots)
    summary = summary.sort_values('sort_key').drop(columns='sort_key')

else:
    win_rate = 0
    pnl_sum = 0.0
    summary = pd.DataFrame()

with col1:
    st.metric("Total Trades", total_trades)
    st.metric("Global Win Rate", f"{win_rate:.1f}%")
    st.metric("Global PnL (Est)", f"${pnl_sum:.2f}")

with col2:
    st.subheader("🤖 Bot Performance Comparison")
    if not summary.empty:
        st.dataframe(summary, width="stretch", hide_index=True)
    else:
        st.info("Waiting for trade data...")



# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🚀 Execution", "🧠 Signals (Brain)", "📉 Performance"])

with tab1:
    st.subheader("Recent Executions")
    if not df_trades.empty:
        st.dataframe(df_trades.head(20), width="stretch")
    else:
        st.info("No trades recorded yet.")

with tab2:
    st.subheader("AI Alpha Signals")
    if not df_signals.empty:
        # Charts for Confidence
        fig_conf = px.line(df_signals, x='timestamp', y='confidence', title="Model Confidence over Time")
        st.plotly_chart(fig_conf)
        
        st.dataframe(df_signals, width="stretch")
    else:
        st.info("No signals recorded yet.")

with tab3:
    st.subheader("Equity Curve")
    if not df_trades.empty and 'pnl' in df_trades.columns:
        # Cumulative Sum ignoring NaNs
        df_trades = df_trades.sort_values('timestamp')
        df_trades['equity'] = df_trades['pnl'].fillna(0).cumsum()
        
        fig = px.line(df_trades, x='timestamp', y='equity', title="Cumulative PnL")
        st.plotly_chart(fig)
    else:
        st.warning("Not enough PnL data to plot curve.")

# --- Footer ---
st.markdown("---")
st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
time.sleep(refresh_rate)
st.rerun()
