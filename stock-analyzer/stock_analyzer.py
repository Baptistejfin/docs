"""
Stock Analyzer Pro v8.0 - Version Complète avec Analyse Cygne Noir + Assistant IA
Analyse ESG + Fondamentale Avancée + Historique 10 ans + DCF + Indicateurs Techniques + Black Swan + ALADDIN-Like AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import search
import yfinance as yf
from datetime import datetime, timedelta

# Gestion du module ESG (Fallback si absent)
try:
    from esg_module_pro import (
        get_simple_esg_display,
        analyze_greenwashing_pro,
    )
except ImportError:
    def get_simple_esg_display(ticker): return {}
    def analyze_greenwashing_pro(ticker, name, info):
        return {
            'risk_score': 50, 'risk_level': 'MEDIUM', 'risk_color': '#f1c40f',
            'detailed_scores': {},
            'esg_data': {'summary': {'scandals_found': 0, 'total_fines_usd': 0, 'sanctions_matches': 0, 'governance_issues': 0, 'eu_facilities': 0, 'jurisdictions': 0}},
            'flags': [], 'recommendations': []
        }

# Gestion du module Black Swan (Cygne Noir)
try:
    from black_swan_analyzer import (
        BlackSwanAnalyzer,
        get_risk_category_emoji,
        get_risk_category_name_fr,
        get_trend_display
    )
    BLACK_SWAN_AVAILABLE = True
except ImportError:
    BLACK_SWAN_AVAILABLE = False

    # Fallback classes si le module n'est pas disponible
    class BlackSwanAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze(self, ticker):
            return None
        def plot_risk_matrix(self, analysis):
            return None
        def plot_risk_bars(self, analysis, top_n=10):
            return None

    def get_risk_category_emoji(cat): return "⚠️"
    def get_risk_category_name_fr(cat): return cat
    def get_trend_display(trend, change): return ("→", "#95a5a6")

# Gestion du module AI Assistant (ALADDIN-Like)
try:
    from ai_assistant import (
        MistralAssistant,
        AIProvider,
        QUICK_ACTIONS,
        REPORT_TYPES,
        get_assistant_css,
        format_financial_context,
        StreamlitChatUI
    )
    AI_ASSISTANT_AVAILABLE = True
except ImportError:
    AI_ASSISTANT_AVAILABLE = False

    class AIProvider:
        OLLAMA = "ollama"
        MISTRAL_API = "mistral_api"

    class MistralAssistant:
        def __init__(self, *args, **kwargs):
            self.is_available = False
        def check_availability(self):
            return False
        def set_context(self, data):
            pass
        def chat(self, msg, stream=True):
            yield "Module AI Assistant non disponible."
        def get_welcome_message(self):
            return "Module AI Assistant non disponible."
        def clear_history(self):
            pass

    QUICK_ACTIONS = []
    REPORT_TYPES = []
    def get_assistant_css(): return ""
    def format_financial_context(info, data=None): return {}
    class StreamlitChatUI:
        @staticmethod
        def init_session_state(): pass

st.set_page_config(page_title="Stock Analyzer Pro", page_icon="📈", layout="wide")
st.title("📊 Analyseur d'Actions - Niveau Institutionnel")
st.markdown("---")


# ============ FONCTIONS UTILITAIRES ============

@st.cache_data(ttl=300)
def get_ticker_from_name(name: str) -> str | None:
    try:
        results = search(name.strip()).get('quotes')
        return results[0]['symbol'] if results else None
    except:
        return None

@st.cache_data(ttl=60)
def get_stock_data_full(ticker: str, period: str = '10y'):
    """Récupère toutes les données incluant l'historique."""
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        hist = tkr.history(period=period)

        # Données financières historiques
        financials = tkr.financials
        quarterly_financials = tkr.quarterly_financials
        balance_sheet = tkr.balance_sheet
        quarterly_balance = tkr.quarterly_balance_sheet
        cashflow = tkr.cashflow
        quarterly_cashflow = tkr.quarterly_cashflow
        income_stmt = tkr.income_stmt

        return {
            'info': info,
            'hist': hist,
            'financials': financials,
            'quarterly_financials': quarterly_financials,
            'balance_sheet': balance_sheet,
            'quarterly_balance': quarterly_balance,
            'cashflow': cashflow,
            'quarterly_cashflow': quarterly_cashflow,
            'income_stmt': income_stmt,
        }
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

@st.cache_data(ttl=60)
def get_stock_history(ticker: str, period: str = '1y', interval: str = '1d'):
    """Récupère l'historique des prix avec intervalle personnalisé."""
    try:
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period=period, interval=interval)
        return hist
    except:
        return None

def format_value(val, decimal=2):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        val = float(val)
        if abs(val) >= 1e12: return f'{val/1e12:,.{decimal}f} T'
        if abs(val) >= 1e9: return f'{val/1e9:,.{decimal}f} B'
        if abs(val) >= 1e6: return f'{val/1e6:,.{decimal}f} M'
        return f'{val:,.{decimal}f}'
    except:
        return 'N/A'

def format_percent(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f"{float(val) * 100:.2f}%"
    except:
        return 'N/A'

def format_ratio(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f"{float(val):.2f}"
    except:
        return 'N/A'

def safe_get(df, key, default=None):
    """Récupère une ligne d'un DataFrame de manière sécurisée."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return default
    try:
        if key in df.index:
            return df.loc[key]
        return default
    except:
        return default

def is_series_valid(series):
    """Vérifie si une Series pandas contient des données valides non-nulles."""
    if series is None:
        return False
    if isinstance(series, pd.Series):
        if series.empty:
            return False
        valid_values = series.apply(lambda x: pd.notna(x) and x != 0 and x > 0)
        return valid_values.any()
    elif isinstance(series, list):
        return bool(series) and any(x is not None and x != 0 and x > 0 for x in series)
    return False

def get_valid_values(series):
    """Extrait les valeurs valides d'une Series pandas."""
    if series is None:
        return []
    if isinstance(series, pd.Series):
        return [x for x in series.values if pd.notna(x) and x > 0]
    elif isinstance(series, list):
        return [x for x in series if x is not None and x > 0]
    return []

def calculate_growth(series):
    """Calcule le taux de croissance annuel."""
    if series is None:
        return None
    if isinstance(series, pd.Series) and series.empty:
        return None
    if len(series) < 2:
        return None
    try:
        first = series.iloc[-1]
        last = series.iloc[0]
        years = len(series) - 1
        if pd.notna(first) and pd.notna(last) and first > 0 and last > 0 and years > 0:
            return (last / first) ** (1/years) - 1
        return None
    except:
        return None


# ============ INDICATEURS TECHNIQUES ============

def calculate_rsi(prices, period=14):
    """Calcule le RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calcule les bandes de Bollinger."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calcule l'oscillateur stochastique."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d


# ============ GRAPHIQUE AVEC INDICATEURS TECHNIQUES ============

def display_price_chart_with_indicators(ticker, info, hist_full):
    """Affiche le graphique des prix avec sélecteur de période et indicateurs techniques."""

    st.subheader("📈 Graphique des Cours")

    # === SÉLECTEUR DE PÉRIODE ===
    col_period, col_indicators = st.columns([2, 3])

    with col_period:
        st.markdown("**📅 Période**")
        period_options = {
            "1J": ("1d", "1m"),
            "1S": ("5d", "5m"),
            "1M": ("1mo", "1h"),
            "YTD": ("ytd", "1d"),
            "1A": ("1y", "1d"),
            "5A": ("5y", "1wk"),
            "10A": ("10y", "1mo"),
            "Max": ("max", "1mo")
        }

        selected_period = st.radio(
            "Période",
            options=list(period_options.keys()),
            horizontal=True,
            index=4,  # Default: 1A
            label_visibility="collapsed"
        )

        period, interval = period_options[selected_period]

    with col_indicators:
        st.markdown("**📊 Indicateurs Techniques**")

        col_ind1, col_ind2, col_ind3, col_ind4 = st.columns(4)

        with col_ind1:
            show_ma50 = st.checkbox("MA50", value=True)
        with col_ind2:
            show_ma200 = st.checkbox("MA200", value=True)
        with col_ind3:
            show_rsi = st.checkbox("RSI", value=False)
        with col_ind4:
            show_bollinger = st.checkbox("Bollinger", value=False)

    # Options supplémentaires
    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

    with col_opt1:
        show_volume = st.checkbox("Volume", value=True)
    with col_opt2:
        show_macd = st.checkbox("MACD", value=False)
    with col_opt3:
        show_stochastic = st.checkbox("Stochastique", value=False)
    with col_opt4:
        fullscreen = st.checkbox("🔍 Agrandir", value=False)

    # === RÉCUPÉRATION DES DONNÉES ===
    hist = get_stock_history(ticker, period, interval)

    if hist is None or hist.empty:
        st.warning("Données non disponibles pour cette période")
        return

    # === CALCUL DE LA PERFORMANCE ===
    if len(hist) > 1:
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        perf_percent = ((end_price - start_price) / start_price) * 100
        perf_color = "green" if perf_percent >= 0 else "red"
    else:
        perf_percent = 0
        perf_color = "gray"

    # === CRÉATION DU GRAPHIQUE ===
    # Déterminer le nombre de sous-graphiques
    num_rows = 1
    row_heights = [0.6]

    if show_volume:
        num_rows += 1
        row_heights.append(0.15)
    if show_rsi:
        num_rows += 1
        row_heights.append(0.15)
    if show_macd:
        num_rows += 1
        row_heights.append(0.15)
    if show_stochastic:
        num_rows += 1
        row_heights.append(0.15)

    # Normaliser les hauteurs
    total = sum(row_heights)
    row_heights = [h/total for h in row_heights]

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )

    current_row = 1

    # === GRAPHIQUE PRINCIPAL (Candlestick ou Line) ===
    if interval in ['1m', '5m', '15m', '1h']:
        # Chandelier pour les intervalles courts
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Prix',
                increasing_line_color='#2ecc71',
                decreasing_line_color='#e74c3c'
            ),
            row=1, col=1
        )
    else:
        # Ligne pour les intervalles longs
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name='Prix',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.1)'
            ),
            row=1, col=1
        )

    # === MOYENNES MOBILES ===
    if show_ma50 and len(hist) >= 50:
        ma50 = hist['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(x=hist.index, y=ma50, mode='lines', name='MA50',
                       line=dict(color='#f39c12', width=1.5)),
            row=1, col=1
        )

    if show_ma200 and len(hist) >= 200:
        ma200 = hist['Close'].rolling(window=200).mean()
        fig.add_trace(
            go.Scatter(x=hist.index, y=ma200, mode='lines', name='MA200',
                       line=dict(color='#e74c3c', width=1.5)),
            row=1, col=1
        )

    # === BANDES DE BOLLINGER ===
    if show_bollinger and len(hist) >= 20:
        upper, middle, lower = calculate_bollinger_bands(hist['Close'])
        fig.add_trace(
            go.Scatter(x=hist.index, y=upper, mode='lines', name='BB Upper',
                       line=dict(color='rgba(128, 128, 128, 0.5)', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=lower, mode='lines', name='BB Lower',
                       line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                       fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)'),
            row=1, col=1
        )

    current_row = 2

    # === VOLUME ===
    if show_volume:
        colors = ['#2ecc71' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] else '#e74c3c'
                  for i in range(len(hist))]
        fig.add_trace(
            go.Bar(x=hist.index, y=hist['Volume'], name='Volume',
                   marker_color=colors, opacity=0.7),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1

    # === RSI ===
    if show_rsi and len(hist) >= 14:
        rsi = calculate_rsi(hist['Close'])
        fig.add_trace(
            go.Scatter(x=hist.index, y=rsi, mode='lines', name='RSI',
                       line=dict(color='#9b59b6', width=1.5)),
            row=current_row, col=1
        )
        # Zones de surachat/survente
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1

    # === MACD ===
    if show_macd and len(hist) >= 26:
        macd, signal, histogram = calculate_macd(hist['Close'])
        fig.add_trace(
            go.Scatter(x=hist.index, y=macd, mode='lines', name='MACD',
                       line=dict(color='#3498db', width=1.5)),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=signal, mode='lines', name='Signal',
                       line=dict(color='#e74c3c', width=1.5)),
            row=current_row, col=1
        )
        colors_macd = ['#2ecc71' if h >= 0 else '#e74c3c' for h in histogram]
        fig.add_trace(
            go.Bar(x=hist.index, y=histogram, name='Histogram',
                   marker_color=colors_macd, opacity=0.5),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1

    # === STOCHASTIQUE ===
    if show_stochastic and len(hist) >= 14:
        k, d = calculate_stochastic(hist['High'], hist['Low'], hist['Close'])
        fig.add_trace(
            go.Scatter(x=hist.index, y=k, mode='lines', name='%K',
                       line=dict(color='#3498db', width=1.5)),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=d, mode='lines', name='%D',
                       line=dict(color='#e74c3c', width=1.5)),
            row=current_row, col=1
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.update_yaxes(title_text="Stoch", range=[0, 100], row=current_row, col=1)

    # === MISE EN FORME ===
    chart_height = 700 if fullscreen else 500
    if num_rows > 2:
        chart_height += (num_rows - 2) * 100

    fig.update_layout(
        template='plotly_dark',
        height=chart_height,
        title=f"Cours de {info.get('shortName', ticker)} - {selected_period}",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    fig.update_yaxes(title_text="Prix ($)", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # === MÉTRIQUES DE PERFORMANCE ===
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            f"Performance {selected_period}",
            f"{perf_percent:+.2f}%",
            delta=f"${end_price - start_price:+.2f}" if len(hist) > 1 else None,
            delta_color="normal" if perf_percent >= 0 else "inverse"
        )

    with col2:
        st.metric("Prix actuel", f"${end_price:.2f}")

    with col3:
        high_period = hist['High'].max()
        st.metric(f"Plus haut {selected_period}", f"${high_period:.2f}")

    with col4:
        low_period = hist['Low'].min()
        st.metric(f"Plus bas {selected_period}", f"${low_period:.2f}")

    # === SIGNAUX TECHNIQUES ===
    if show_rsi or show_macd or show_stochastic:
        st.markdown("### 📊 Signaux Techniques")

        signal_cols = st.columns(4)

        with signal_cols[0]:
            if show_rsi and len(hist) >= 14:
                rsi_current = calculate_rsi(hist['Close']).iloc[-1]
                if pd.notna(rsi_current):
                    if rsi_current > 70:
                        st.error(f"RSI: {rsi_current:.1f} - Surachat 🔴")
                    elif rsi_current < 30:
                        st.success(f"RSI: {rsi_current:.1f} - Survente 🟢")
                    else:
                        st.info(f"RSI: {rsi_current:.1f} - Neutre ⚪")

        with signal_cols[1]:
            if show_ma50 and show_ma200 and len(hist) >= 200:
                ma50_val = hist['Close'].rolling(50).mean().iloc[-1]
                ma200_val = hist['Close'].rolling(200).mean().iloc[-1]
                if pd.notna(ma50_val) and pd.notna(ma200_val):
                    if ma50_val > ma200_val:
                        st.success("MA50 > MA200 - Tendance haussière 🟢")
                    else:
                        st.error("MA50 < MA200 - Tendance baissière 🔴")

        with signal_cols[2]:
            if show_macd and len(hist) >= 26:
                macd_val, signal_val, _ = calculate_macd(hist['Close'])
                macd_current = macd_val.iloc[-1]
                signal_current = signal_val.iloc[-1]
                if pd.notna(macd_current) and pd.notna(signal_current):
                    if macd_current > signal_current:
                        st.success("MACD > Signal - Achat 🟢")
                    else:
                        st.error("MACD < Signal - Vente 🔴")

        with signal_cols[3]:
            if show_stochastic and len(hist) >= 14:
                k_val, d_val = calculate_stochastic(hist['High'], hist['Low'], hist['Close'])
                k_current = k_val.iloc[-1]
                if pd.notna(k_current):
                    if k_current > 80:
                        st.error(f"Stoch: {k_current:.1f} - Surachat 🔴")
                    elif k_current < 20:
                        st.success(f"Stoch: {k_current:.1f} - Survente 🟢")
                    else:
                        st.info(f"Stoch: {k_current:.1f} - Neutre ⚪")


# ============ ANALYSE HISTORIQUE 10 ANS ============

def get_historical_metrics(data):
    """Extrait les métriques historiques sur 10 ans."""
    metrics = {
        'years': [],
        'revenue': None,
        'net_income': None,
        'gross_profit': None,
        'operating_income': None,
        'ebitda': None,
        'free_cash_flow': None,
        'total_debt': None,
        'total_cash': None,
        'total_equity': None,
        'total_assets': None,
        'dividends_paid': None,
        'shares_outstanding': None,
    }

    income = data.get('income_stmt')
    balance = data.get('balance_sheet')
    cashflow = data.get('cashflow')

    if income is not None and not income.empty:
        years = [col.year if hasattr(col, 'year') else str(col)[:4] for col in income.columns]
        metrics['years'] = years

        metrics['revenue'] = safe_get(income, 'Total Revenue')
        if metrics['revenue'] is None:
            metrics['revenue'] = safe_get(income, 'Operating Revenue')

        metrics['net_income'] = safe_get(income, 'Net Income')
        metrics['gross_profit'] = safe_get(income, 'Gross Profit')
        metrics['operating_income'] = safe_get(income, 'Operating Income')
        metrics['ebitda'] = safe_get(income, 'EBITDA')
        if metrics['ebitda'] is None:
            metrics['ebitda'] = safe_get(income, 'Normalized EBITDA')

    if balance is not None and not balance.empty:
        metrics['total_debt'] = safe_get(balance, 'Total Debt')
        metrics['total_cash'] = safe_get(balance, 'Cash And Cash Equivalents')
        metrics['total_equity'] = safe_get(balance, 'Total Equity Gross Minority Interest')
        if metrics['total_equity'] is None:
            metrics['total_equity'] = safe_get(balance, 'Stockholders Equity')
        metrics['total_assets'] = safe_get(balance, 'Total Assets')
        metrics['shares_outstanding'] = safe_get(balance, 'Ordinary Shares Number')

    if cashflow is not None and not cashflow.empty:
        operating_cf = safe_get(cashflow, 'Operating Cash Flow')
        capex = safe_get(cashflow, 'Capital Expenditure')

        if operating_cf is not None and capex is not None:
            try:
                metrics['free_cash_flow'] = operating_cf + capex
            except:
                metrics['free_cash_flow'] = None

        metrics['dividends_paid'] = safe_get(cashflow, 'Common Stock Dividend Paid')

    return metrics


# ============ OUTIL DCF AMÉLIORÉ ============

def calculate_dcf(fcf_values, wacc, terminal_growth_rate, shares_outstanding):
    """Calcule la valeur DCF."""
    if not fcf_values or shares_outstanding <= 0:
        return None

    pv_fcfs = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcf_values)])
    terminal_fcf = fcf_values[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
    pv_terminal = terminal_value / ((1 + wacc) ** len(fcf_values))
    enterprise_value = pv_fcfs + pv_terminal
    equity_value = enterprise_value
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
    return {
        'pv_fcfs': pv_fcfs,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'value_per_share': value_per_share
    }

def display_dcf_tool(data, info):
    """Affiche l'outil DCF interactif amélioré."""
    st.subheader("🧮 Outil DCF - Valorisation par Flux de Trésorerie")

    metrics = get_historical_metrics(data)
    fcf = metrics.get('free_cash_flow')

    if not is_series_valid(fcf):
        st.warning("Données de Free Cash Flow insuffisantes pour le DCF")
        return

    # === PARAMÈTRES DCF ===
    st.markdown("### ⚙️ Paramètres DCF")

    param_col1, param_col2, param_col3, param_col4 = st.columns(4)

    with param_col1:
        default_wacc = info.get('returnOnEquity', 0.10) if info.get('returnOnEquity') else 0.10
        if pd.isna(default_wacc) or default_wacc <= 0:
            default_wacc = 0.10
        wacc = st.slider(
            "WACC (Coût du Capital)",
            min_value=0.02,
            max_value=0.20,
            value=min(max(float(default_wacc), 0.02), 0.20),
            step=0.01,
            help="Coût moyen pondéré du capital - généralement entre 5% et 12%"
        )

    with param_col2:
        terminal_growth = st.slider(
            "Croissance Terminale",
            min_value=0.01,
            max_value=0.05,
            value=0.025,
            step=0.005,
            help="Croissance perpétuelle estimée - généralement 2-3%"
        )

    with param_col3:
        projection_years = st.slider(
            "Années de projection",
            min_value=3,
            max_value=10,
            value=5,
            help="Nombre d'années à projeter"
        )

    with param_col4:
        fcf_growth_rate = st.slider(
            "Croissance FCF estimée",
            min_value=-0.10,
            max_value=0.30,
            value=0.05,
            step=0.01,
            help="Taux de croissance annuel du FCF"
        )

    # === CALCULS ===
    fcf_hist = get_valid_values(fcf)

    if not fcf_hist:
        st.warning("Pas de Free Cash Flow positif disponible")
        return

    last_fcf = fcf_hist[-1]
    shares_out = info.get('sharesOutstanding', 1) or 1
    current_price = info.get('regularMarketPrice', 0) or 0

    # Projection avec le taux choisi par l'utilisateur
    projected_fcfs = []
    yearly_prices = []

    for year in range(1, projection_years + 1):
        projected_fcf = last_fcf * ((1 + fcf_growth_rate) ** year)
        projected_fcfs.append(projected_fcf)

        # Calculer le prix cible pour chaque année
        partial_dcf = calculate_dcf(projected_fcfs, wacc, terminal_growth, shares_out)
        if partial_dcf:
            yearly_prices.append(partial_dcf['value_per_share'])
        else:
            yearly_prices.append(0)

    dcf_result = calculate_dcf(projected_fcfs, wacc, terminal_growth, shares_out)

    if dcf_result is None:
        st.warning("Impossible de calculer le DCF")
        return

    st.markdown("---")

    # === GRAPHIQUE PROJECTION 5 ANS ===
    st.markdown("### 📊 Projection sur 5 ans")

    # Légende des couleurs
    st.markdown("""
    <div style="display: flex; gap: 20px; margin-bottom: 15px;">
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background-color: #3498db; border-radius: 3px;"></div>
            <span>FCF Historique (données réelles)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background-color: #2ecc71; border-radius: 3px;"></div>
            <span>FCF Projeté (estimation)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 20px; height: 20px; background-color: #f39c12; border-radius: 3px;"></div>
            <span>Prix Cible par Action</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    years = metrics['years']
    all_fcf_values = list(fcf.values) if isinstance(fcf, pd.Series) else list(fcf)
    all_fcf = all_fcf_values + projected_fcfs

    # Générer les années futures
    if years:
        try:
            last_year = int(years[-1]) if isinstance(years[-1], (int, str)) else years[-1]
            future_years = [str(last_year + i) for i in range(1, projection_years + 1)]
        except:
            future_years = [f"A+{i}" for i in range(1, projection_years + 1)]
    else:
        future_years = [f"A+{i}" for i in range(1, projection_years + 1)]

    all_years = [str(y) for y in years] + future_years
    colors = ['#3498db'] * len(years) + ['#2ecc71'] * projection_years

    # Graphique avec deux axes Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Barres FCF
    fig.add_trace(
        go.Bar(
            x=all_years,
            y=[x/1e9 if pd.notna(x) else 0 for x in all_fcf],
            marker=dict(color=colors),
            text=[f"${x/1e9:.1f}B" if pd.notna(x) else "" for x in all_fcf],
            textposition='outside',
            name='Free Cash Flow'
        ),
        secondary_y=False
    )

    # Ligne prix cible (seulement pour les années projetées)
    fig.add_trace(
        go.Scatter(
            x=future_years,
            y=yearly_prices,
            mode='lines+markers+text',
            name='Prix Cible',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=12, symbol='diamond'),
            text=[f"${p:.0f}" for p in yearly_prices],
            textposition='top center',
            textfont=dict(size=12, color='#f39c12')
        ),
        secondary_y=True
    )

    # Ligne du prix actuel
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Prix actuel: ${current_price:.2f}",
        annotation_position="right",
        secondary_y=True
    )

    fig.add_vline(
        x=len(years)-0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Début Projection",
        annotation_position="top"
    )

    fig.update_layout(
        template='plotly_dark',
        height=500,
        title=f"FCF Historique + Projection ({projection_years} ans) avec Prix Cible",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode='relative'
    )

    fig.update_yaxes(title_text="FCF (Milliards $)", secondary_y=False)
    fig.update_yaxes(title_text="Prix Cible ($)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # === RÉSULTATS DCF ===
    st.markdown("### 💰 Résultats DCF")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        upside = ((dcf_result['value_per_share'] - current_price) / current_price * 100) if current_price > 0 else 0
        st.metric(
            "Valeur Intrinsèque",
            f"${dcf_result['value_per_share']:.2f}",
            delta=f"{upside:+.1f}% vs prix actuel",
            delta_color="normal" if upside >= 0 else "inverse"
        )

    with res_col2:
        st.metric("Prix Actuel", f"${current_price:.2f}")

    with res_col3:
        st.metric("Valeur Entreprise", format_value(dcf_result['enterprise_value']))

    with res_col4:
        margin_of_safety = max(0, (dcf_result['value_per_share'] - current_price) / dcf_result['value_per_share'] * 100)
        st.metric("Marge de Sécurité", f"{margin_of_safety:.1f}%")

    # === TABLEAU PRIX CIBLE PAR ANNÉE ===
    st.markdown("### 📅 Prix Cible par Année")

    price_data = {
        'Année': future_years,
        'FCF Projeté': [format_value(p) for p in projected_fcfs],
        'Prix Cible': [f"${p:.2f}" for p in yearly_prices],
        'Potentiel vs Actuel': [f"{((p - current_price) / current_price * 100):+.1f}%" if current_price > 0 else "N/A" for p in yearly_prices]
    }

    st.dataframe(pd.DataFrame(price_data), hide_index=True, use_container_width=True)

    # === ANALYSE DE SENSIBILITÉ ===
    with st.expander("📊 Analyse de Sensibilité"):
        st.markdown("**Impact des paramètres sur la valorisation:**")

        sens_col1, sens_col2 = st.columns(2)

        with sens_col1:
            st.markdown("**Variation du WACC:**")
            wacc_variations = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
            wacc_prices = []
            for w in wacc_variations:
                if w > terminal_growth:
                    dcf_temp = calculate_dcf(projected_fcfs, w, terminal_growth, shares_out)
                    wacc_prices.append(dcf_temp['value_per_share'] if dcf_temp else 0)
                else:
                    wacc_prices.append(0)

            wacc_df = pd.DataFrame({
                'WACC': [f"{w*100:.1f}%" for w in wacc_variations],
                'Prix Cible': [f"${p:.2f}" for p in wacc_prices]
            })
            st.dataframe(wacc_df, hide_index=True, use_container_width=True)

        with sens_col2:
            st.markdown("**Variation de la Croissance:**")
            growth_variations = [fcf_growth_rate - 0.02, fcf_growth_rate - 0.01, fcf_growth_rate, fcf_growth_rate + 0.01, fcf_growth_rate + 0.02]
            growth_prices = []
            for g in growth_variations:
                projected_temp = [last_fcf * ((1 + g) ** y) for y in range(1, projection_years + 1)]
                dcf_temp = calculate_dcf(projected_temp, wacc, terminal_growth, shares_out)
                growth_prices.append(dcf_temp['value_per_share'] if dcf_temp else 0)

            growth_df = pd.DataFrame({
                'Croissance FCF': [f"{g*100:.1f}%" for g in growth_variations],
                'Prix Cible': [f"${p:.2f}" for p in growth_prices]
            })
            st.dataframe(growth_df, hide_index=True, use_container_width=True)

    # === MODE ÉDUCATION ===
    with st.expander("📚 Comprendre le DCF"):
        edu_col1, edu_col2, edu_col3, edu_col4 = st.columns(4)

        with edu_col1:
            st.info("""
            **WACC**

            Coût moyen pondéré du capital.
            Plus il est bas, plus l'entreprise
            est finançable à bon marché.

            💡 Typiquement 5-12%
            """)

        with edu_col2:
            st.info("""
            **Free Cash Flow**

            Flux de trésorerie disponible
            après dépenses d'exploitation.

            💡 Doit être positif et croissant
            """)

        with edu_col3:
            st.info("""
            **Croissance Terminale**

            Croissance perpétuelle supposée.
            Ne dépasse pas la croissance
            économique mondiale.

            💡 Généralement 2-3% max
            """)

        with edu_col4:
            st.info("""
            **Marge de Sécurité**

            Différence entre valeur intrinsèque
            et prix actuel. Plus elle est haute,
            plus l'investissement est sûr.

            💡 Viser > 20%
            """)


# ============ FREE FORM TOOL ============

def display_free_form_tool(data, info):
    """Outil de création de graphiques personnalisés."""
    st.subheader("📊 Free Form Tool - Créer Vos Graphiques")

    metrics = get_historical_metrics(data)
    years = metrics['years']

    if not years:
        st.warning("Données insuffisantes")
        return

    available_metrics = {
        'Revenus': metrics['revenue'],
        'Bénéfice Net': metrics['net_income'],
        'Profit Brut': metrics['gross_profit'],
        'Résultat Opérationnel': metrics['operating_income'],
        'EBITDA': metrics['ebitda'],
        'Free Cash Flow': metrics['free_cash_flow'],
        'Dette Totale': metrics['total_debt'],
        'Trésorerie': metrics['total_cash'],
        'Capitaux Propres': metrics['total_equity'],
        'Actifs Totaux': metrics['total_assets'],
    }

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Sélection des métriques")

        selected_metrics = st.multiselect(
            "Choisir jusqu'à 5 métriques",
            options=list(available_metrics.keys()),
            max_selections=5,
            default=["Revenus", "Bénéfice Net"],
            help="Sélectionnez les métriques à afficher"
        )

        st.markdown("---")

        normalize = st.checkbox("Normaliser (index 100)", value=False,
                                help="Affiche tous les graphiques à partir de 100")

        chart_type = st.radio("Type de graphique", ["Ligne", "Barre", "Ligne+Points"])

    with col2:
        if selected_metrics:
            fig = go.Figure()

            colors_palette = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
            y_label = "Milliards $"

            for idx, metric_name in enumerate(selected_metrics):
                metric_data = available_metrics[metric_name]

                if not is_series_valid(metric_data):
                    continue

                if isinstance(metric_data, pd.Series):
                    cleaned_data = [x if pd.notna(x) and x > 0 else None for x in metric_data.values]
                else:
                    cleaned_data = [x if x is not None and x > 0 else None for x in metric_data]

                if normalize and any(x for x in cleaned_data if x):
                    first_val = next((x for x in cleaned_data if x), 1)
                    normalized_data = [(x / first_val * 100) if x else None for x in cleaned_data]
                    y_data = normalized_data
                    y_label = "Index (100 = année 1)"
                else:
                    y_data = [x/1e9 if x else None for x in cleaned_data]
                    y_label = "Milliards $"

                if chart_type == "Ligne":
                    fig.add_trace(go.Scatter(
                        x=[str(y) for y in years], y=y_data, name=metric_name,
                        mode='lines',
                        line=dict(color=colors_palette[idx], width=2)
                    ))
                elif chart_type == "Barre":
                    fig.add_trace(go.Bar(
                        x=[str(y) for y in years], y=y_data, name=metric_name,
                        marker_color=colors_palette[idx]
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[str(y) for y in years], y=y_data, name=metric_name,
                        mode='lines+markers',
                        line=dict(color=colors_palette[idx], width=2),
                        marker=dict(size=8)
                    ))

            fig.update_layout(
                template='plotly_dark',
                height=450,
                title="Graphique Personnalisé",
                yaxis_title=y_label,
                hovermode='x unified',
                legend=dict(orientation="v", yanchor="top", xanchor="right")
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Sélectionnez au moins une métrique")


# ============ SÉLECTEUR D'ANNÉES INTERACTIF ============

def display_year_selector_view(data, info):
    """Affiche les données pour une année spécifique sélectionnée."""
    st.subheader("📅 Analyse par Année - Vue Détaillée")

    metrics = get_historical_metrics(data)
    years = metrics['years']

    if not years:
        st.warning("Données insuffisantes")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        years_options = [str(y) for y in years]
        years_options_sorted = sorted(years_options)

        selected_year_str = st.select_slider(
            "Sélectionner l'année",
            options=years_options_sorted,
            value=years_options_sorted[-1] if years_options_sorted else None
        )

        try:
            selected_year_idx = years_options.index(selected_year_str)
        except (ValueError, IndexError):
            selected_year_idx = 0

    if not years:
        return

    selected_year = years[selected_year_idx]

    with col2:
        st.markdown(f"### 📊 Métriques pour l'année **{selected_year}**")

    def get_metric_value(metric_series, idx):
        if metric_series is None:
            return None
        if isinstance(metric_series, pd.Series):
            if metric_series.empty or idx >= len(metric_series):
                return None
            val = metric_series.iloc[idx]
            return val if pd.notna(val) else None
        elif isinstance(metric_series, list):
            if idx >= len(metric_series):
                return None
            return metric_series[idx]
        return None

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    data_points = [
        (col1, "Revenus", metrics['revenue']),
        (col2, "Bénéfice Net", metrics['net_income']),
        (col3, "EBITDA", metrics['ebitda']),
        (col4, "FCF", metrics['free_cash_flow']),
        (col5, "Dette", metrics['total_debt']),
        (col6, "Trésorerie", metrics['total_cash']),
    ]

    for col, label, metric_series in data_points:
        with col:
            val = get_metric_value(metric_series, selected_year_idx)
            st.metric(label, format_value(val))


# ============ DONNÉES ESG DANS ANALYSE FONDAMENTALE ============

def display_esg_section(ticker, info):
    """Affiche les données ESG dans l'analyse fondamentale."""
    st.subheader("🌱 Données ESG")

    esg = get_simple_esg_display(ticker)

    if esg.get('available'):
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            env_score = esg.get('environment', 0)
            color = "#2ecc71" if env_score <= 20 else "#f1c40f" if env_score <= 35 else "#e74c3c"
            st.metric("🌍 Environnement", f"{env_score:.0f}/100")
            st.progress(env_score / 100)

        with col2:
            soc_score = esg.get('social', 0)
            st.metric("👥 Social", f"{soc_score:.0f}/100")
            st.progress(soc_score / 100)

        with col3:
            gov_score = esg.get('governance', 0)
            st.metric("🏛️ Gouvernance", f"{gov_score:.0f}/100")
            st.progress(gov_score / 100)

        with col4:
            total_esg = esg.get('total_esg', 0)
            st.metric("📊 Score Total", f"{total_esg:.0f}/100")
            st.progress(total_esg / 100)

        with col5:
            controversy = esg.get('controversy_level', 0)
            controversy_text = {0: "Aucune", 1: "Faible", 2: "Modérée", 3: "Significative", 4: "Élevée", 5: "Sévère"}
            st.metric("⚠️ Controverse", f"Niveau {controversy}/5")
            st.caption(controversy_text.get(controversy, "N/A"))

        # Analyse rapide
        st.markdown("### 📋 Analyse Rapide")

        avg_score = (env_score + soc_score + gov_score) / 3

        if avg_score <= 20:
            st.success("✅ **Excellent profil ESG** - L'entreprise démontre de solides pratiques environnementales, sociales et de gouvernance.")
        elif avg_score <= 30:
            st.info("ℹ️ **Bon profil ESG** - L'entreprise a des pratiques ESG supérieures à la moyenne du secteur.")
        elif avg_score <= 40:
            st.warning("⚠️ **Profil ESG moyen** - Des améliorations sont possibles dans certains domaines.")
        else:
            st.error("🔴 **Profil ESG à risque** - L'entreprise présente des risques ESG significatifs à surveiller.")

        # Détails par catégorie
        with st.expander("📊 Détails ESG"):
            st.markdown("""
            **Comprendre les scores ESG:**

            - **Score Environnement**: Évalue l'impact environnemental (émissions CO2, gestion des déchets, utilisation des ressources)
            - **Score Social**: Évalue les relations avec les employés, la diversité, la sécurité au travail
            - **Score Gouvernance**: Évalue la structure de direction, l'éthique des affaires, la transparence

            *Note: Plus le score est bas, meilleure est la performance ESG (échelle Sustainalytics)*
            """)
    else:
        st.info("📊 Données ESG non disponibles pour cette entreprise via Yahoo Finance.")
        st.caption("Conseil: Utilisez le mode 'Analyse ESG Pro' pour une analyse plus complète avec 10 sources de données.")


# ============ MODULE CYGNE NOIR (BLACK SWAN) ============

@st.cache_data(ttl=3600)  # Cache de 1 heure
def get_black_swan_analysis(ticker: str) -> dict:
    """Récupère l'analyse Cygne Noir avec cache."""
    if not BLACK_SWAN_AVAILABLE:
        return None
    try:
        analyzer = BlackSwanAnalyzer()
        return analyzer.analyze(ticker)
    except Exception as e:
        st.error(f"Erreur analyse Cygne Noir: {e}")
        return None


def display_black_swan_section(ticker: str, info: dict):
    """Affiche la section d'analyse des risques Cygne Noir."""

    st.subheader("🦢 Analyse Cygne Noir - Risques Extrêmes")

    if not BLACK_SWAN_AVAILABLE:
        st.warning("⚠️ Module Black Swan non disponible. Vérifiez que `black_swan_analyzer.py` est présent.")
        return

    # Explication du concept
    with st.expander("ℹ️ Qu'est-ce qu'un Cygne Noir ?", expanded=False):
        st.markdown("""
        **La théorie du Cygne Noir** (Nassim Nicholas Taleb) décrit des événements :
        - **Rares** : Hors du domaine des attentes normales
        - **Impact extrême** : Conséquences majeures sur les marchés
        - **Rationalisés a posteriori** : Expliqués après coup comme prévisibles

        **Exemples historiques :**
        - Crise financière de 2008
        - Pandémie COVID-19 (2020)
        - Guerre en Ukraine (2022)
        - Effondrement de Lehman Brothers

        Cette analyse identifie les risques potentiels spécifiques à l'entreprise et son secteur.
        """)

    # Récupérer l'analyse
    with st.spinner("🔍 Analyse des risques en cours..."):
        analysis = get_black_swan_analysis(ticker)

    if not analysis:
        st.error("❌ Impossible de récupérer l'analyse des risques")
        return

    # === SCORE GLOBAL ===
    st.markdown("### 🎯 Score de Risque Global")

    score = analysis['global_risk_score']
    risk_level = analysis['risk_level']
    risk_color = analysis['risk_color']

    # Jauge de risque
    col_score1, col_score2, col_score3 = st.columns([1, 2, 1])

    with col_score1:
        st.metric(
            "Score Global",
            f"{score}/10",
            delta=risk_level,
            delta_color="inverse" if score < 5 else "normal"
        )

    with col_score2:
        # Barre de progression stylisée
        progress_html = f"""
        <div style="background: linear-gradient(to right, #27ae60 0%, #f1c40f 40%, #e67e22 60%, #e74c3c 100%);
                    height: 30px; border-radius: 15px; position: relative; margin: 10px 0;">
            <div style="position: absolute; left: {score * 10}%; top: -5px;
                        width: 40px; height: 40px; background: white; border-radius: 50%;
                        border: 4px solid {risk_color}; transform: translateX(-50%);
                        display: flex; align-items: center; justify-content: center;
                        font-weight: bold; font-size: 14px; color: {risk_color};">{score}</div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888;">
            <span>Faible</span>
            <span>Modéré</span>
            <span>Élevé</span>
            <span>Critique</span>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)

    with col_score3:
        st.metric("Risques Analysés", analysis['total_risks_analyzed'])
        st.caption(f"Secteur: {analysis['company_info']['sector']}")

    # === MÉTRIQUES DE VOLATILITÉ ===
    st.markdown("### 📊 Indicateurs de Stress")

    vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)

    with vol_col1:
        vol_1y = analysis['company_info']['volatility_1y']
        st.metric("Volatilité 1 an", f"{vol_1y:.1f}%")

    with vol_col2:
        vol_20d = analysis['company_info']['volatility_20d']
        vol_ratio = analysis['company_info']['volatility_ratio']
        delta_color = "inverse" if vol_ratio > 1.2 else "normal"
        st.metric(
            "Volatilité 20 jours",
            f"{vol_20d:.1f}%",
            delta=f"x{vol_ratio:.2f} vs 1Y",
            delta_color=delta_color
        )

    with vol_col3:
        current_price = analysis['company_info']['current_price']
        st.metric("Prix Actuel", f"${current_price:.2f}")

    with vol_col4:
        market_cap = analysis['company_info'].get('market_cap', 0)
        if market_cap:
            if market_cap >= 1e12:
                cap_str = f"${market_cap/1e12:.1f}T"
            elif market_cap >= 1e9:
                cap_str = f"${market_cap/1e9:.1f}B"
            else:
                cap_str = f"${market_cap/1e6:.1f}M"
            st.metric("Market Cap", cap_str)

    st.markdown("---")

    # === TOP RISQUES ===
    st.markdown("### ⚠️ Principaux Risques Identifiés")

    top_risks = analysis.get('top_risks', [])[:5]

    if not top_risks:
        st.info("Aucun risque majeur identifié pour cette entreprise.")
        return

    for i, risk in enumerate(top_risks, 1):
        emoji = get_risk_category_emoji(risk['category'])
        category_name = get_risk_category_name_fr(risk['category'])
        trend_text, trend_color = get_trend_display(risk['probability_trend'], risk['probability_change'])

        # Couleur basée sur la probabilité
        if risk['probability_percent'] >= 50:
            prob_color = "#e74c3c"
            prob_bg = "rgba(231, 76, 60, 0.1)"
        elif risk['probability_percent'] >= 30:
            prob_color = "#e67e22"
            prob_bg = "rgba(230, 126, 34, 0.1)"
        elif risk['probability_percent'] >= 15:
            prob_color = "#f1c40f"
            prob_bg = "rgba(241, 196, 15, 0.1)"
        else:
            prob_color = "#27ae60"
            prob_bg = "rgba(39, 174, 96, 0.1)"

        with st.expander(f"{emoji} **{risk['name_fr']}** — Probabilité: {risk['probability_percent']:.1f}%", expanded=(i <= 2)):

            risk_col1, risk_col2, risk_col3 = st.columns([1, 1, 1])

            with risk_col1:
                st.markdown(f"**Catégorie:** {category_name}")
                st.markdown(f"**Probabilité:** <span style='color:{prob_color}; font-size: 1.2em; font-weight: bold;'>{risk['probability_percent']:.1f}%</span>", unsafe_allow_html=True)
                st.markdown(f"**Tendance:** <span style='color:{trend_color};'>{trend_text}</span>", unsafe_allow_html=True)

            with risk_col2:
                st.markdown("**Scénarios d'Impact:**")
                scenarios = risk['impact_scenarios']
                st.markdown(f"- 🟡 Modéré: **{scenarios['moderate']['percentage']:.0f}%** → ${scenarios['moderate']['price_target']:.2f}")
                st.markdown(f"- 🟠 Sévère: **{scenarios['severe']['percentage']:.0f}%** → ${scenarios['severe']['price_target']:.2f}")
                st.markdown(f"- 🔴 Extrême: **{scenarios['extreme']['percentage']:.0f}%** → ${scenarios['extreme']['price_target']:.2f}")

            with risk_col3:
                st.markdown("**Indicateurs News:**")
                sentiment = risk['news_sentiment']
                if sentiment < -0.3:
                    sent_text = "🔴 Négatif"
                elif sentiment > 0.3:
                    sent_text = "🟢 Positif"
                else:
                    sent_text = "⚪ Neutre"
                st.markdown(f"- Sentiment: {sent_text} ({sentiment:.2f})")
                st.markdown(f"- Articles pertinents: {risk['relevant_news_count']}")
                st.markdown(f"- Récupération estimée: ~{scenarios['recovery_months']} mois")

            # Actualités pertinentes
            if risk.get('relevant_headlines'):
                st.markdown("**📰 Actualités récentes:**")
                for headline in risk['relevant_headlines'][:3]:
                    sent_emoji = "🔴" if headline['sentiment'] < -0.2 else "🟢" if headline['sentiment'] > 0.2 else "⚪"
                    st.markdown(f"- {sent_emoji} {headline['title'][:80]}...")

    st.markdown("---")

    # === VISUALISATIONS ===
    st.markdown("### 📈 Visualisations")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🎯 Matrice Risque/Impact", "📊 Probabilités", "💰 Scénarios Prix"])

    with viz_tab1:
        try:
            analyzer = BlackSwanAnalyzer()
            fig_matrix = analyzer.plot_risk_matrix(analysis)
            if fig_matrix:
                st.plotly_chart(fig_matrix, use_container_width=True)

                st.caption("""
                **Lecture de la matrice:**
                - **Axe X** : Probabilité d'occurrence (%)
                - **Axe Y** : Impact potentiel sur le cours (%)
                - **Taille des bulles** : Score de risque combiné
                - Les risques dans la zone rouge (haut-droite) sont les plus critiques
                """)
        except Exception as e:
            st.error(f"Erreur lors de la génération de la matrice: {e}")

    with viz_tab2:
        try:
            analyzer = BlackSwanAnalyzer()
            fig_bars = analyzer.plot_risk_bars(analysis, top_n=10)
            if fig_bars:
                st.plotly_chart(fig_bars, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique: {e}")

    with viz_tab3:
        try:
            analyzer = BlackSwanAnalyzer()
            fig_scenarios = analyzer.plot_impact_scenarios(analysis, top_n=5)
            if fig_scenarios:
                st.plotly_chart(fig_scenarios, use_container_width=True)

                st.caption("""
                **Scénarios d'impact:**
                - 🟡 **Modéré** : Impact limité, récupération rapide
                - 🟠 **Sévère** : Impact significatif, récupération lente
                - 🔴 **Extrême** : Impact majeur type "Cygne Noir"
                """)
        except Exception as e:
            st.error(f"Erreur lors de la génération des scénarios: {e}")

    # === TABLEAU RÉCAPITULATIF ===
    with st.expander("📋 Tableau Complet des Risques"):
        risks_data = []
        for risk in analysis.get('risks', []):
            risks_data.append({
                'Risque': risk['name_fr'],
                'Catégorie': get_risk_category_name_fr(risk['category']),
                'Probabilité': f"{risk['probability_percent']:.1f}%",
                'Impact Min': f"{risk['impact_range'][0]}%",
                'Impact Max': f"{risk['impact_range'][1]}%",
                'Tendance': risk['probability_trend'].upper(),
                'Sentiment': f"{risk['news_sentiment']:.2f}"
            })

        if risks_data:
            df_risks = pd.DataFrame(risks_data)
            st.dataframe(df_risks, hide_index=True, use_container_width=True)

    # === MÉTHODOLOGIE ===
    with st.expander("🔬 Méthodologie"):
        st.markdown(f"""
        **Sources de données:**
        - Données financières: Yahoo Finance (yfinance)
        - Actualités: {analysis['methodology']['news_source']}
        - Analyse sentiment: {analysis['methodology']['sentiment_analysis']}

        **Calcul des probabilités:**
        - Score de base par type de risque (données historiques)
        - Ajustement sentiment news (-15% à +20%)
        - Ajustement fréquence mentions (0 à +15%)
        - Ajustement volatilité marché (-5% à +15%)

        **Calcul des impacts:**
        - Basé sur des événements historiques similaires
        - Ajusté selon le beta de l'action
        - 3 scénarios: modéré, sévère, extrême

        **Fréquence de mise à jour:** {analysis['methodology']['update_frequency']}

        **Dernière mise à jour:** {analysis['last_updated']}
        """)


# ============ ANALYSE HISTORIQUE COMPLÈTE ============

def display_historical_analysis(data, info, ticker=None):
    """Affiche l'analyse historique sur 10 ans avec tous les outils."""

    # Récupérer le ticker si non fourni
    if not ticker:
        ticker = info.get('symbol', '')

    metrics = get_historical_metrics(data)

    if not metrics['years']:
        st.warning("Données historiques non disponibles")
        return

    st.subheader("📅 Analyse Fondamentale Avancée")

    # === SÉLECTION DES MODULES À AFFICHER ===
    with st.expander("⚙️ Personnaliser l'affichage", expanded=False):
        mod_col1, mod_col2, mod_col3, mod_col4 = st.columns(4)
        with mod_col1:
            show_revenue = st.checkbox("Revenus & Profits", value=True)
        with mod_col2:
            show_profitability = st.checkbox("Rentabilité", value=True)
        with mod_col3:
            show_balance = st.checkbox("Bilan", value=True)
        with mod_col4:
            show_cashflow = st.checkbox("Cash Flow", value=True)

        mod_col5, mod_col6, mod_col7, mod_col8 = st.columns(4)
        with mod_col5:
            show_ratios = st.checkbox("Ratios", value=True)
        with mod_col6:
            show_dcf = st.checkbox("DCF", value=True)
        with mod_col7:
            show_freeform = st.checkbox("Free Form", value=True)
        with mod_col8:
            show_blackswan = st.checkbox("🦢 Cygne Noir", value=True)

    # === ONGLETS DYNAMIQUES ===
    tabs_list = []
    tabs_names = []

    if show_revenue:
        tabs_names.append("📈 Revenus & Profits")
    if show_profitability:
        tabs_names.append("💰 Rentabilité")
    if show_balance:
        tabs_names.append("🏦 Bilan")
    if show_cashflow:
        tabs_names.append("💵 Cash Flow")
    if show_ratios:
        tabs_names.append("📊 Ratios")
    if show_dcf:
        tabs_names.append("🧮 DCF")
    if show_freeform:
        tabs_names.append("📋 Free Form")
    if show_blackswan:
        tabs_names.append("🦢 Cygne Noir")

    if not tabs_names:
        st.info("Sélectionnez au moins un module à afficher")
        return

    tabs = st.tabs(tabs_names)

    years = metrics['years']
    tab_idx = 0

    # TAB 1: REVENUS & PROFITS
    if show_revenue:
        with tabs[tab_idx]:
            st.markdown("### 📈 Évolution des Revenus et Profits")

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                if is_series_valid(metrics['revenue']):
                    try:
                        rev_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['revenue'].values]
                        fig.add_trace(
                            go.Bar(x=[str(y) for y in years], y=rev_values, name="Revenus", marker_color='#3498db'),
                            secondary_y=False
                        )
                    except:
                        pass

                if is_series_valid(metrics['net_income']):
                    try:
                        ni_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['net_income'].values]
                        fig.add_trace(
                            go.Scatter(x=[str(y) for y in years], y=ni_values, name="Bénéfice Net",
                                       mode='lines+markers', line=dict(color='#2ecc71', width=3)),
                            secondary_y=False
                        )
                    except:
                        pass

                if is_series_valid(metrics['operating_income']):
                    try:
                        op_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['operating_income'].values]
                        fig.add_trace(
                            go.Scatter(x=[str(y) for y in years], y=op_values, name="Résultat Opérationnel",
                                       mode='lines+markers', line=dict(color='#f1c40f', width=2, dash='dash')),
                            secondary_y=False
                        )
                    except:
                        pass

                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    title="Revenus et Profits (Milliards $)",
                    yaxis_title="Milliards $",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    barmode='overlay',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Croissance annuelle moyenne (CAGR)**")

                if is_series_valid(metrics['revenue']):
                    cagr_rev = calculate_growth(metrics['revenue'])
                    st.metric("📊 Revenus", format_percent(cagr_rev) if cagr_rev else "N/A")

                if is_series_valid(metrics['net_income']):
                    cagr_ni = calculate_growth(metrics['net_income'])
                    st.metric("💰 Bénéfice Net", format_percent(cagr_ni) if cagr_ni else "N/A")

                if is_series_valid(metrics['operating_income']):
                    cagr_op = calculate_growth(metrics['operating_income'])
                    st.metric("📈 Résultat Opé.", format_percent(cagr_op) if cagr_op else "N/A")

            with st.expander("📋 Données détaillées"):
                df_data = {'Année': [str(y) for y in years]}
                if is_series_valid(metrics['revenue']):
                    df_data['Revenus'] = [format_value(v) for v in metrics['revenue'].values]
                if is_series_valid(metrics['gross_profit']):
                    df_data['Profit Brut'] = [format_value(v) for v in metrics['gross_profit'].values]
                if is_series_valid(metrics['operating_income']):
                    df_data['Résultat Opé.'] = [format_value(v) for v in metrics['operating_income'].values]
                if is_series_valid(metrics['net_income']):
                    df_data['Bénéfice Net'] = [format_value(v) for v in metrics['net_income'].values]

                st.dataframe(pd.DataFrame(df_data), hide_index=True, use_container_width=True)
        tab_idx += 1

    # TAB 2: RENTABILITÉ
    if show_profitability:
        with tabs[tab_idx]:
            st.markdown("### 💰 Évolution de la Rentabilité")

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure()

                if is_series_valid(metrics['revenue']) and is_series_valid(metrics['gross_profit']):
                    try:
                        gross_margin = [(gp/rev)*100 if pd.notna(rev) and pd.notna(gp) and rev > 0 else 0
                                        for rev, gp in zip(metrics['revenue'].values, metrics['gross_profit'].values)]
                        fig.add_trace(go.Scatter(x=[str(y) for y in years], y=gross_margin, name="Marge Brute %",
                                                 mode='lines+markers', line=dict(color='#2ecc71', width=2)))
                    except:
                        pass

                if is_series_valid(metrics['revenue']) and is_series_valid(metrics['operating_income']):
                    try:
                        op_margin = [(oi/rev)*100 if pd.notna(rev) and pd.notna(oi) and rev > 0 else 0
                                     for rev, oi in zip(metrics['revenue'].values, metrics['operating_income'].values)]
                        fig.add_trace(go.Scatter(x=[str(y) for y in years], y=op_margin, name="Marge Opérationnelle %",
                                                 mode='lines+markers', line=dict(color='#3498db', width=2)))
                    except:
                        pass

                if is_series_valid(metrics['revenue']) and is_series_valid(metrics['net_income']):
                    try:
                        net_margin = [(ni/rev)*100 if pd.notna(rev) and pd.notna(ni) and rev > 0 else 0
                                      for rev, ni in zip(metrics['revenue'].values, metrics['net_income'].values)]
                        fig.add_trace(go.Scatter(x=[str(y) for y in years], y=net_margin, name="Marge Nette %",
                                                 mode='lines+markers', line=dict(color='#9b59b6', width=2)))
                    except:
                        pass

                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    title="Évolution des Marges (%)",
                    yaxis_title="Marge (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Rentabilité actuelle**")
                st.metric("ROE", format_percent(info.get('returnOnEquity')))
                st.metric("ROA", format_percent(info.get('returnOnAssets')))

                if is_series_valid(metrics['net_income']) and is_series_valid(metrics['total_equity']):
                    try:
                        roe_values = [(ni/eq)*100 if pd.notna(eq) and pd.notna(ni) and eq > 0 else None
                                      for ni, eq in zip(metrics['net_income'].values, metrics['total_equity'].values)]
                        valid_roe = [r for r in roe_values if r is not None]
                        if valid_roe:
                            avg_roe = np.mean(valid_roe)
                            st.metric("ROE moyen 10Y", f"{avg_roe:.1f}%")
                    except:
                        pass
        tab_idx += 1

    # TAB 3: BILAN
    if show_balance:
        with tabs[tab_idx]:
            st.markdown("### 🏦 Évolution du Bilan")

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure()

                if is_series_valid(metrics['total_debt']):
                    try:
                        debt_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['total_debt'].values]
                        fig.add_trace(go.Bar(x=[str(y) for y in years], y=debt_values, name="Dette Totale", marker_color='#e74c3c'))
                    except:
                        pass

                if is_series_valid(metrics['total_cash']):
                    try:
                        cash_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['total_cash'].values]
                        fig.add_trace(go.Bar(x=[str(y) for y in years], y=cash_values, name="Trésorerie", marker_color='#2ecc71'))
                    except:
                        pass

                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    title="Dette vs Trésorerie (Milliards $)",
                    yaxis_title="Milliards $",
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Situation actuelle**")
                st.metric("Dette Totale", format_value(info.get('totalDebt')))
                st.metric("Trésorerie", format_value(info.get('totalCash')))

                debt = info.get('totalDebt', 0) or 0
                cash = info.get('totalCash', 0) or 0
                st.metric("Dette Nette", format_value(debt - cash))

                st.markdown("---")
                st.metric("Debt/Equity", format_ratio(info.get('debtToEquity')))
                st.metric("Current Ratio", format_ratio(info.get('currentRatio')))
        tab_idx += 1

    # TAB 4: CASH FLOW
    if show_cashflow:
        with tabs[tab_idx]:
            st.markdown("### 💵 Évolution des Cash Flows")

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure()

                if is_series_valid(metrics['free_cash_flow']):
                    try:
                        fcf_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['free_cash_flow'].values]
                        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in fcf_values]
                        fig.add_trace(go.Bar(x=[str(y) for y in years], y=fcf_values, name="Free Cash Flow", marker_color=colors))
                    except:
                        pass

                if is_series_valid(metrics['dividends_paid']):
                    try:
                        div_values = [abs(v)/1e9 if pd.notna(v) else 0 for v in metrics['dividends_paid'].values]
                        fig.add_trace(go.Scatter(x=[str(y) for y in years], y=div_values, name="Dividendes Versés",
                                                 mode='lines+markers', line=dict(color='#f1c40f', width=2)))
                    except:
                        pass

                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    title="Free Cash Flow & Dividendes (Milliards $)",
                    yaxis_title="Milliards $",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Cash Flow actuel**")
                st.metric("Operating CF", format_value(info.get('operatingCashflow')))
                st.metric("Free Cash Flow", format_value(info.get('freeCashflow')))

                fcf = info.get('freeCashflow')
                mcap = info.get('marketCap')
                if fcf and mcap and mcap > 0:
                    fcf_yield = (fcf / mcap) * 100
                    st.metric("FCF Yield", f"{fcf_yield:.2f}%")

                if is_series_valid(metrics['free_cash_flow']):
                    cagr_fcf = calculate_growth(metrics['free_cash_flow'])
                    if cagr_fcf:
                        st.metric("CAGR FCF", format_percent(cagr_fcf))
        tab_idx += 1

    # TAB 5: RATIOS
    if show_ratios:
        with tabs[tab_idx]:
            st.markdown("### 📊 Évolution des Ratios Clés")

            if is_series_valid(metrics['net_income']) and is_series_valid(metrics['shares_outstanding']):
                col1, col2 = st.columns(2)

                with col1:
                    try:
                        eps_hist = [ni/shares if pd.notna(shares) and pd.notna(ni) and shares > 0 else None
                                    for ni, shares in zip(metrics['net_income'].values, metrics['shares_outstanding'].values)]

                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=[str(y) for y in years], y=eps_hist, name="EPS", marker_color='#3498db',
                                             text=[f"${v:.2f}" if v else "" for v in eps_hist],
                                             textposition='outside'))
                        fig.update_layout(template='plotly_dark', height=350, title="Bénéfice par Action (EPS)",
                                          yaxis_title="$ par action", hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)

                        eps_series = pd.Series([e for e in eps_hist if e is not None])
                        if len(eps_series) >= 2:
                            cagr_eps = calculate_growth(eps_series)
                            st.metric("CAGR EPS", format_percent(cagr_eps) if cagr_eps else "N/A")
                    except:
                        st.info("Données EPS insuffisantes")

                with col2:
                    if is_series_valid(metrics['total_equity']):
                        try:
                            bvps = [eq/shares if pd.notna(shares) and pd.notna(eq) and shares > 0 else None
                                    for eq, shares in zip(metrics['total_equity'].values, metrics['shares_outstanding'].values)]

                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=[str(y) for y in years], y=bvps, name="Book Value", marker_color='#2ecc71',
                                                 text=[f"${v:.2f}" if v else "" for v in bvps],
                                                 textposition='outside'))
                            fig.update_layout(template='plotly_dark', height=350,
                                              title="Valeur Comptable par Action",
                                              yaxis_title="$ par action", hovermode='x unified')
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.info("Données BVPS insuffisantes")

            st.markdown("### 📊 Résumé des Croissances (CAGR)")

            cagr_data = []

            if is_series_valid(metrics['revenue']):
                cagr = calculate_growth(metrics['revenue'])
                cagr_data.append({'Métrique': 'Revenus', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                                  'Rating': '🟢' if cagr and cagr > 0.10 else '🟡' if cagr and cagr > 0.05 else '🔴'})

            if is_series_valid(metrics['net_income']):
                cagr = calculate_growth(metrics['net_income'])
                cagr_data.append({'Métrique': 'Bénéfice Net', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                                  'Rating': '🟢' if cagr and cagr > 0.12 else '🟡' if cagr and cagr > 0.05 else '🔴'})

            if is_series_valid(metrics['free_cash_flow']):
                cagr = calculate_growth(metrics['free_cash_flow'])
                cagr_data.append({'Métrique': 'Free Cash Flow', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                                  'Rating': '🟢' if cagr and cagr > 0.10 else '🟡' if cagr and cagr > 0.05 else '🔴'})

            if cagr_data:
                st.dataframe(pd.DataFrame(cagr_data), hide_index=True, use_container_width=True)
        tab_idx += 1

    # TAB 6: DCF
    if show_dcf:
        with tabs[tab_idx]:
            display_dcf_tool(data, info)
        tab_idx += 1

    # TAB 7: FREE FORM
    if show_freeform:
        with tabs[tab_idx]:
            display_free_form_tool(data, info)
        tab_idx += 1

    # TAB 8: CYGNE NOIR (BLACK SWAN)
    if show_blackswan:
        with tabs[tab_idx]:
            # Utiliser le ticker passé en paramètre
            display_black_swan_section(ticker, info)

    # BONUS: Sélecteur d'années
    st.markdown("---")
    display_year_selector_view(data, info)


# ============ ASSISTANT IA (ALADDIN-LIKE) - FONCTIONS ============

def init_ai_assistant():
    """Initialise l'assistant IA dans la session."""
    import os

    if "ai_assistant" not in st.session_state:
        # Vérifier si une clé API Mistral est configurée
        mistral_api_key = os.environ.get("MISTRAL_API_KEY", "")

        if mistral_api_key:
            # Utiliser l'API Mistral Cloud
            st.session_state.ai_assistant = MistralAssistant(
                provider=AIProvider.MISTRAL_API,
                api_key=mistral_api_key,
                model="mistral-small-latest"
            )
            st.session_state.ai_provider = "cloud"
        else:
            # Utiliser Ollama en local
            st.session_state.ai_assistant = MistralAssistant(
                provider=AIProvider.OLLAMA,
                model="mistral:latest"
            )
            st.session_state.ai_provider = "local"

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "current_company_context" not in st.session_state:
        st.session_state.current_company_context = None


def update_assistant_context(ticker: str, info: dict, data: dict = None):
    """Met à jour le contexte de l'assistant avec les données de l'entreprise."""
    if not AI_ASSISTANT_AVAILABLE:
        return

    context = format_financial_context(info, data)

    if data:
        if "dcf_results" in data:
            context["dcf_results"] = data["dcf_results"]

    st.session_state.ai_assistant.set_context(context)
    st.session_state.current_company_context = context


# ============ SIDEBAR ============

with st.sidebar:
    st.header("⚙️ Paramètres")
    analysis_mode = st.radio("Mode", ['📈 Analyse Fondamentale', '🌱 Analyse ESG Pro'], index=0)
    st.markdown("---")

    company_name = st.text_input("Entreprise", value="Apple" if analysis_mode == '📈 Analyse Fondamentale' else "TotalEnergies")

    st.markdown("---")
    analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)

    if analysis_mode == '📈 Analyse Fondamentale':
        st.markdown("---")
        st.markdown("### 📊 Fonctionnalités")
        st.caption("""
        **🆕 v8.0 - Nouveautés:**
        - 🦢 Analyse Cygne Noir (risques extrêmes)
        - Scoring dynamique des risques
        - Analyse sentiment actualités
        - Matrice risque/impact interactive
        - Scénarios d'impact financier

        **v7.0:**
        - Sélecteur de période (1J à Max)
        - Indicateurs techniques
        - DCF amélioré
        """)


# ============ ANALYSE FONDAMENTALE ============

if analysis_mode == '📈 Analyse Fondamentale':
    st.header("📈 Analyse Fondamentale Avancée")

    if analyze_button and company_name:
        with st.spinner("Recherche..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            with st.spinner("Chargement des données (10 ans)..."):
                data = get_stock_data_full(ticker, '10y')

            if data and data['info'] and data['hist'] is not None and not data['hist'].empty:
                info = data['info']
                hist = data['hist']

                # Mettre à jour le contexte de l'assistant IA
                if AI_ASSISTANT_AVAILABLE:
                    init_ai_assistant()
                    update_assistant_context(ticker, info, {"data": data})

                st.subheader(f"🏢 {info.get('shortName', ticker)} ({ticker})")

                if info.get('longBusinessSummary'):
                    with st.expander("📋 Description"):
                        st.write(info['longBusinessSummary'])

                # Métriques clés
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Prix", f"${info.get('regularMarketPrice', 0):.2f}")
                with col2:
                    change = info.get('regularMarketChangePercent', 0) or 0
                    st.metric("Var. Jour", f"{change:.2f}%")
                with col3:
                    st.metric("Market Cap", format_value(info.get('marketCap')))
                with col4:
                    st.metric("P/E", format_ratio(info.get('trailingPE')))
                with col5:
                    st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                with col6:
                    st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")

                st.markdown("---")

                # === GRAPHIQUE AVEC INDICATEURS TECHNIQUES ===
                display_price_chart_with_indicators(ticker, info, hist)

                st.markdown("---")

                # === DONNÉES ESG ===
                display_esg_section(ticker, info)

                st.markdown("---")

                # === ANALYSE HISTORIQUE 10 ANS ===
                display_historical_analysis(data, info, ticker)

        else:
            st.error(f"❌ Entreprise '{company_name}' non trouvée")


# ============ ANALYSE ESG PRO ============

elif analysis_mode == '🌱 Analyse ESG Pro':
    st.header("🌱 Analyse ESG - Niveau Institutionnel")

    if analyze_button and company_name:
        with st.spinner("Recherche..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            with st.spinner("Chargement..."):
                data = get_stock_data_full(ticker, '1y')

            if data:
                info = data['info']
                full_name = info.get('shortName', company_name) if info else company_name

                st.subheader(f"🏢 {full_name} ({ticker})")

                with st.spinner("🔍 Collecte ESG (10 sources)..."):
                    analysis = analyze_greenwashing_pro(ticker, full_name, info)

                st.markdown("---")

                # Score de risque
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Score Global", f"{analysis['risk_score']:.0f}/100",
                        delta=analysis['risk_level'], delta_color="inverse" if analysis['risk_score'] < 30 else "normal")
                with col2:
                    st.metric("🗞️ Médias", f"{analysis['detailed_scores'].get('media', 0):.0f}/100")
                with col3:
                    st.metric("🌍 Environnement", f"{analysis['detailed_scores'].get('environmental', 0):.0f}/100")
                with col4:
                    st.metric("⚖️ Sanctions", f"{analysis['detailed_scores'].get('sanctions', 0):.0f}/100")
                with col5:
                    st.metric("🏛️ Gouvernance", f"{analysis['detailed_scores'].get('governance', 0):.0f}/100")

                # Jauge
                st.markdown(f"""
                <div style="background: linear-gradient(to right, #27ae60, #f1c40f, #e74c3c);
                            height: 25px; border-radius: 12px; position: relative; margin: 20px 0;">
                    <div style="position: absolute; left: {analysis['risk_score']}%; top: -8px;
                                width: 40px; height: 40px; background: white; border-radius: 50%;
                                border: 4px solid {analysis['risk_color']}; transform: translateX(-50%);
                                display: flex; align-items: center; justify-content: center;
                                font-weight: bold;">{analysis['risk_score']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Résumé
                summary = analysis['esg_data']['summary']
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1: st.metric("🗞️ Scandales", summary['scandals_found'])
                with col2: st.metric("💰 Amendes", f"${summary['total_fines_usd']:,.0f}")
                with col3: st.metric("🚫 Sanctions", summary['sanctions_matches'])
                with col4: st.metric("⚠️ Gouv.", summary['governance_issues'])
                with col5: st.metric("🏭 Sites EU", summary['eu_facilities'])
                with col6: st.metric("🌐 Pays", summary['jurisdictions'])

                st.markdown("---")

                # Alertes
                if analysis['flags']:
                    st.subheader("🚩 Alertes")
                    for flag in analysis['flags']:
                        sev = flag['severity']
                        emoji = {'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟡'}.get(sev, '⚪')
                        msg = f"{emoji} **{flag['title']}** - {flag['description']}"
                        if sev == 'HIGH': st.error(msg)
                        elif sev == 'MEDIUM': st.warning(msg)
                        else: st.info(msg)

                # Recommandations
                st.subheader("💡 Recommandations")
                for rec in analysis['recommendations']:
                    st.info(f"→ {rec}")
        else:
            st.error(f"❌ Entreprise non trouvée")


# ============ ASSISTANT IA (ALADDIN-LIKE) - UI ============

def render_chat_panel():
    """Affiche le panneau de chat latéral."""
    if not AI_ASSISTANT_AVAILABLE:
        return

    # CSS pour le chat
    st.markdown(get_assistant_css(), unsafe_allow_html=True)

    # Bouton flottant pour ouvrir le chat
    chat_button_html = """
    <style>
    .floating-chat-button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 65px;
        height: 65px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        transition: all 0.3s ease;
    }
    .floating-chat-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.7);
    }
    </style>
    """
    st.markdown(chat_button_html, unsafe_allow_html=True)


def display_ai_chat_sidebar():
    """Affiche l'interface de chat dans la sidebar ou un expander."""
    if not AI_ASSISTANT_AVAILABLE:
        return

    init_ai_assistant()

    # Créer une section dans le sidebar pour le chat
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 Assistant IA")

        # Vérifier la disponibilité
        assistant = st.session_state.ai_assistant
        provider = st.session_state.get("ai_provider", "local")
        is_available = assistant.check_availability()

        if not is_available:
            if provider == "cloud":
                st.error("""
                ❌ **Erreur API Mistral**

                Vérifiez que :
                1. Votre clé API est valide
                2. Vous avez des crédits disponibles
                """)
            else:
                st.warning("""
                ⚠️ **Ollama non détecté**

                Pour utiliser l'assistant IA :
                1. Installez [Ollama](https://ollama.ai)
                2. Lancez : `ollama run mistral`
                3. Rafraîchissez la page

                **Ou** configurez `MISTRAL_API_KEY` pour utiliser le cloud.
                """)
            return

        # Afficher le provider utilisé
        if provider == "cloud":
            st.success("✅ IA Cloud (Mistral API)")
        else:
            st.success("✅ IA Locale (Ollama)")

        # Contexte actuel
        if st.session_state.current_company_context:
            ctx = st.session_state.current_company_context
            st.caption(f"📊 Analyse: **{ctx.get('name', 'N/A')}**")

        # Zone de chat dans un expander
        with st.expander("💬 Ouvrir le chat", expanded=False):
            # Afficher les messages
            chat_container = st.container()

            with chat_container:
                # Message de bienvenue si pas de messages
                if not st.session_state.chat_messages:
                    welcome = assistant.get_welcome_message()
                    st.markdown(f"""
                    <div style="background: #2d2d44; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    {welcome}
                    </div>
                    """, unsafe_allow_html=True)

                # Afficher l'historique des messages
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white; padding: 10px 15px; border-radius: 15px;
                                    margin: 5px 0; margin-left: 20%; text-align: right;">
                        {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #2d2d44; color: #e0e0e0;
                                    padding: 10px 15px; border-radius: 15px;
                                    margin: 5px 0; margin-right: 20%;">
                        {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)

            # Zone de saisie
            user_input = st.text_input(
                "Message",
                key="chat_input",
                placeholder="Posez votre question...",
                label_visibility="collapsed"
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                send_btn = st.button("📤 Envoyer", use_container_width=True)
            with col2:
                clear_btn = st.button("🗑️", help="Effacer l'historique")

            if clear_btn:
                st.session_state.chat_messages = []
                assistant.clear_history()
                st.rerun()

            if send_btn and user_input:
                # Ajouter le message utilisateur
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": user_input
                })

                # Obtenir la réponse de l'IA
                with st.spinner("🤔 Réflexion..."):
                    response = ""
                    for chunk in assistant.chat(user_input, stream=False):
                        response += chunk

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response
                    })

                st.rerun()

        # Actions rapides
        if QUICK_ACTIONS:
            st.markdown("**⚡ Actions rapides**")

            action_cols = st.columns(2)
            for i, action in enumerate(QUICK_ACTIONS[:4]):
                with action_cols[i % 2]:
                    if st.button(action["label"], key=f"quick_{action['id']}", use_container_width=True):
                        # Exécuter l'action
                        st.session_state.chat_messages.append({
                            "role": "user",
                            "content": action["prompt"]
                        })
                        with st.spinner("🤔 Analyse..."):
                            response = ""
                            for chunk in assistant.chat(action["prompt"], stream=False):
                                response += chunk
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response
                            })
                        st.rerun()

        # Génération de rapports
        with st.expander("📄 Générer un rapport"):
            report_type = st.selectbox(
                "Type de rapport",
                options=[r["id"] for r in REPORT_TYPES],
                format_func=lambda x: next((r["label"] for r in REPORT_TYPES if r["id"] == x), x),
                label_visibility="collapsed"
            )

            if st.button("📝 Générer", use_container_width=True):
                report_info = next((r for r in REPORT_TYPES if r["id"] == report_type), None)
                if report_info:
                    with st.spinner("📝 Génération du rapport..."):
                        response = ""
                        for chunk in assistant.chat(report_info["prompt"], stream=False):
                            response += chunk
                        st.session_state.chat_messages.append({
                            "role": "user",
                            "content": f"[Génération rapport: {report_info['label']}]"
                        })
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response
                        })
                    st.rerun()


# Afficher l'interface de chat dans la sidebar
if AI_ASSISTANT_AVAILABLE:
    display_ai_chat_sidebar()


# ============ FOOTER ============

st.markdown("---")
st.caption("📊 Stock Analyzer Pro v8.0 - Données: Yahoo Finance | 🦢 Cygne Noir | 🤖 Assistant IA | Indicateurs Techniques | DCF | ESG")
