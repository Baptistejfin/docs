"""
Stock Analyzer Pro v5.2 - Analyse ESG + Fondamentale + Historique 10 ans
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import search
import yfinance as yf
from datetime import datetime, timedelta

from esg_module_pro import (
    get_simple_esg_display,
    analyze_greenwashing_pro,
)

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

@st.cache_data(ttl=300)
def get_stock_data_full(ticker: str, period: str = '10y'):
    """Récupère toutes les données incluant l'historique 10 ans."""
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

def format_value(val, decimal=2):
    try:
        if val is None: return 'N/A'
        val = float(val)
        if abs(val) >= 1e12: return f'{val/1e12:,.{decimal}f} T'
        if abs(val) >= 1e9: return f'{val/1e9:,.{decimal}f} B'
        if abs(val) >= 1e6: return f'{val/1e6:,.{decimal}f} M'
        return f'{val:,.{decimal}f}'
    except:
        return 'N/A'

def format_percent(val):
    try:
        if val is None: return 'N/A'
        return f"{float(val) * 100:.2f}%"
    except:
        return 'N/A'

def format_ratio(val):
    try:
        if val is None: return 'N/A'
        return f"{float(val):.2f}"
    except:
        return 'N/A'

def safe_get(df, key, default=None):
    """Récupère une ligne d'un DataFrame de manière sécurisée."""
    if df is None or df.empty:
        return default
    try:
        if key in df.index:
            return df.loc[key]
        return default
    except:
        return default

def calculate_growth(series):
    """Calcule le taux de croissance annuel."""
    if series is None or len(series) < 2:
        return None
    try:
        first = series.iloc[-1]
        last = series.iloc[0]
        years = len(series) - 1
        if first > 0 and last > 0 and years > 0:
            return (last / first) ** (1/years) - 1
        return None
    except:
        return None


# ============ ANALYSE HISTORIQUE 10 ANS ============

def get_historical_metrics(data):
    """Extrait les métriques historiques sur 10 ans."""
    metrics = {
        'years': [],
        'revenue': [],
        'net_income': [],
        'gross_profit': [],
        'operating_income': [],
        'ebitda': [],
        'free_cash_flow': [],
        'total_debt': [],
        'total_cash': [],
        'total_equity': [],
        'total_assets': [],
        'dividends_paid': [],
        'shares_outstanding': [],
    }

    income = data.get('income_stmt')
    balance = data.get('balance_sheet')
    cashflow = data.get('cashflow')

    if income is not None and not income.empty:
        # Années disponibles
        years = [col.year if hasattr(col, 'year') else str(col)[:4] for col in income.columns]
        metrics['years'] = years

        # Revenus
        metrics['revenue'] = safe_get(income, 'Total Revenue', pd.Series([None]*len(years)))
        if metrics['revenue'] is None:
            metrics['revenue'] = safe_get(income, 'Operating Revenue', pd.Series([None]*len(years)))

        # Bénéfice net
        metrics['net_income'] = safe_get(income, 'Net Income', pd.Series([None]*len(years)))

        # Bénéfice brut
        metrics['gross_profit'] = safe_get(income, 'Gross Profit', pd.Series([None]*len(years)))

        # Résultat opérationnel
        metrics['operating_income'] = safe_get(income, 'Operating Income', pd.Series([None]*len(years)))

        # EBITDA
        metrics['ebitda'] = safe_get(income, 'EBITDA', pd.Series([None]*len(years)))
        if metrics['ebitda'] is None:
            metrics['ebitda'] = safe_get(income, 'Normalized EBITDA', pd.Series([None]*len(years)))

    if balance is not None and not balance.empty:
        # Dette totale
        metrics['total_debt'] = safe_get(balance, 'Total Debt', pd.Series([None]*len(metrics['years'])))

        # Trésorerie
        metrics['total_cash'] = safe_get(balance, 'Cash And Cash Equivalents', pd.Series([None]*len(metrics['years'])))

        # Capitaux propres
        metrics['total_equity'] = safe_get(balance, 'Total Equity Gross Minority Interest', pd.Series([None]*len(metrics['years'])))
        if metrics['total_equity'] is None:
            metrics['total_equity'] = safe_get(balance, 'Stockholders Equity', pd.Series([None]*len(metrics['years'])))

        # Actifs totaux
        metrics['total_assets'] = safe_get(balance, 'Total Assets', pd.Series([None]*len(metrics['years'])))

        # Actions en circulation
        metrics['shares_outstanding'] = safe_get(balance, 'Ordinary Shares Number', pd.Series([None]*len(metrics['years'])))

    if cashflow is not None and not cashflow.empty:
        # Free Cash Flow
        operating_cf = safe_get(cashflow, 'Operating Cash Flow', pd.Series([0]*len(metrics['years'])))
        capex = safe_get(cashflow, 'Capital Expenditure', pd.Series([0]*len(metrics['years'])))

        if operating_cf is not None and capex is not None:
            try:
                metrics['free_cash_flow'] = operating_cf + capex  # capex est négatif
            except:
                metrics['free_cash_flow'] = None

        # Dividendes versés
        metrics['dividends_paid'] = safe_get(cashflow, 'Common Stock Dividend Paid', pd.Series([None]*len(metrics['years'])))

    return metrics


def display_historical_analysis(data, info):
    """Affiche l'analyse historique sur 10 ans."""

    metrics = get_historical_metrics(data)

    if not metrics['years']:
        st.warning("Données historiques non disponibles")
        return

    st.subheader("📅 Évolution sur 10 ans")

    # === ONGLETS HISTORIQUES ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Revenus & Profits",
        "💰 Rentabilité",
        "🏦 Bilan",
        "💵 Cash Flow",
        "📊 Ratios"
    ])

    years = metrics['years']

    # TAB 1: REVENUS & PROFITS
    with tab1:
        st.markdown("### 📈 Évolution des Revenus et Profits")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Revenus
            if metrics['revenue'] is not None:
                try:
                    rev_values = [v/1e9 if v else 0 for v in metrics['revenue'].values]
                    fig.add_trace(
                        go.Bar(x=years, y=rev_values, name="Revenus", marker_color='#3498db'),
                        secondary_y=False
                    )
                except:
                    pass

            # Bénéfice net
            if metrics['net_income'] is not None:
                try:
                    ni_values = [v/1e9 if v else 0 for v in metrics['net_income'].values]
                    fig.add_trace(
                        go.Scatter(x=years, y=ni_values, name="Bénéfice Net",
                                   mode='lines+markers', line=dict(color='#2ecc71', width=3)),
                        secondary_y=False
                    )
                except:
                    pass

            # Résultat opérationnel
            if metrics['operating_income'] is not None:
                try:
                    op_values = [v/1e9 if v else 0 for v in metrics['operating_income'].values]
                    fig.add_trace(
                        go.Scatter(x=years, y=op_values, name="Résultat Opérationnel",
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
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Tableau récapitulatif
            st.markdown("**Croissance annuelle moyenne (CAGR)**")

            if metrics['revenue'] is not None:
                cagr_rev = calculate_growth(metrics['revenue'])
                st.metric("📊 Revenus", format_percent(cagr_rev) if cagr_rev else "N/A")

            if metrics['net_income'] is not None:
                cagr_ni = calculate_growth(metrics['net_income'])
                st.metric("💰 Bénéfice Net", format_percent(cagr_ni) if cagr_ni else "N/A")

            if metrics['operating_income'] is not None:
                cagr_op = calculate_growth(metrics['operating_income'])
                st.metric("📈 Résultat Opé.", format_percent(cagr_op) if cagr_op else "N/A")

            # Dernier vs Premier
            st.markdown("---")
            st.markdown("**Dernière année vs Première**")
            if metrics['revenue'] is not None:
                try:
                    first = metrics['revenue'].iloc[-1]
                    last = metrics['revenue'].iloc[0]
                    if first and last:
                        mult = last / first
                        st.metric("Multiplicateur CA", f"x{mult:.1f}")
                except:
                    pass

        # Tableau détaillé
        with st.expander("📋 Données détaillées"):
            df_data = {'Année': years}
            if metrics['revenue'] is not None:
                df_data['Revenus'] = [format_value(v) for v in metrics['revenue'].values]
            if metrics['gross_profit'] is not None:
                df_data['Profit Brut'] = [format_value(v) for v in metrics['gross_profit'].values]
            if metrics['operating_income'] is not None:
                df_data['Résultat Opé.'] = [format_value(v) for v in metrics['operating_income'].values]
            if metrics['net_income'] is not None:
                df_data['Bénéfice Net'] = [format_value(v) for v in metrics['net_income'].values]

            st.dataframe(pd.DataFrame(df_data), hide_index=True, use_container_width=True)

    # TAB 2: RENTABILITÉ
    with tab2:
        st.markdown("### 💰 Évolution de la Rentabilité")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()

            # Calcul des marges
            if metrics['revenue'] is not None and metrics['gross_profit'] is not None:
                try:
                    gross_margin = [(gp/rev)*100 if rev and gp else 0
                                    for rev, gp in zip(metrics['revenue'].values, metrics['gross_profit'].values)]
                    fig.add_trace(go.Scatter(x=years, y=gross_margin, name="Marge Brute %",
                                             mode='lines+markers', line=dict(color='#2ecc71', width=2)))
                except:
                    pass

            if metrics['revenue'] is not None and metrics['operating_income'] is not None:
                try:
                    op_margin = [(oi/rev)*100 if rev and oi else 0
                                 for rev, oi in zip(metrics['revenue'].values, metrics['operating_income'].values)]
                    fig.add_trace(go.Scatter(x=years, y=op_margin, name="Marge Opérationnelle %",
                                             mode='lines+markers', line=dict(color='#3498db', width=2)))
                except:
                    pass

            if metrics['revenue'] is not None and metrics['net_income'] is not None:
                try:
                    net_margin = [(ni/rev)*100 if rev and ni else 0
                                  for rev, ni in zip(metrics['revenue'].values, metrics['net_income'].values)]
                    fig.add_trace(go.Scatter(x=years, y=net_margin, name="Marge Nette %",
                                             mode='lines+markers', line=dict(color='#9b59b6', width=2)))
                except:
                    pass

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Évolution des Marges (%)",
                yaxis_title="Marge (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # ROE / ROA historique
            st.markdown("**Rentabilité actuelle**")
            st.metric("ROE", format_percent(info.get('returnOnEquity')))
            st.metric("ROA", format_percent(info.get('returnOnAssets')))

            # Calcul ROE historique si possible
            if metrics['net_income'] is not None and metrics['total_equity'] is not None:
                try:
                    roe_values = [(ni/eq)*100 if eq and ni else None
                                  for ni, eq in zip(metrics['net_income'].values, metrics['total_equity'].values)]
                    avg_roe = np.mean([r for r in roe_values if r is not None])
                    st.metric("ROE moyen 10Y", f"{avg_roe:.1f}%")
                except:
                    pass

        # Graphique ROE/ROA historique
        if metrics['net_income'] is not None and metrics['total_equity'] is not None and metrics['total_assets'] is not None:
            with st.expander("📈 ROE & ROA historiques"):
                fig2 = go.Figure()

                try:
                    roe_hist = [(ni/eq)*100 if eq and ni and eq > 0 else None
                                for ni, eq in zip(metrics['net_income'].values, metrics['total_equity'].values)]
                    roa_hist = [(ni/ta)*100 if ta and ni and ta > 0 else None
                                for ni, ta in zip(metrics['net_income'].values, metrics['total_assets'].values)]

                    fig2.add_trace(go.Scatter(x=years, y=roe_hist, name="ROE %",
                                              mode='lines+markers', line=dict(color='#e74c3c')))
                    fig2.add_trace(go.Scatter(x=years, y=roa_hist, name="ROA %",
                                              mode='lines+markers', line=dict(color='#3498db')))

                    fig2.update_layout(template='plotly_dark', height=300, yaxis_title="%")
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

    # TAB 3: BILAN
    with tab3:
        st.markdown("### 🏦 Évolution du Bilan")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()

            # Dette vs Cash
            if metrics['total_debt'] is not None:
                try:
                    debt_values = [v/1e9 if v else 0 for v in metrics['total_debt'].values]
                    fig.add_trace(go.Bar(x=years, y=debt_values, name="Dette Totale", marker_color='#e74c3c'))
                except:
                    pass

            if metrics['total_cash'] is not None:
                try:
                    cash_values = [v/1e9 if v else 0 for v in metrics['total_cash'].values]
                    fig.add_trace(go.Bar(x=years, y=cash_values, name="Trésorerie", marker_color='#2ecc71'))
                except:
                    pass

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Dette vs Trésorerie (Milliards $)",
                yaxis_title="Milliards $",
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
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

        # Évolution Debt/Equity
        if metrics['total_debt'] is not None and metrics['total_equity'] is not None:
            with st.expander("📈 Évolution Debt/Equity"):
                try:
                    de_ratio = [(d/e)*100 if e and d and e > 0 else None
                                for d, e in zip(metrics['total_debt'].values, metrics['total_equity'].values)]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=years, y=de_ratio, mode='lines+markers+text',
                                              line=dict(color='#e74c3c'),
                                              text=[f"{v:.0f}%" if v else "" for v in de_ratio],
                                              textposition="top center"))
                    fig2.add_hline(y=100, line_dash="dash", line_color="white",
                                   annotation_text="Seuil 100%")
                    fig2.update_layout(template='plotly_dark', height=300, yaxis_title="Debt/Equity %")
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

    # TAB 4: CASH FLOW
    with tab4:
        st.markdown("### 💵 Évolution des Cash Flows")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()

            # Free Cash Flow
            if metrics['free_cash_flow'] is not None:
                try:
                    fcf_values = [v/1e9 if v else 0 for v in metrics['free_cash_flow'].values]
                    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in fcf_values]
                    fig.add_trace(go.Bar(x=years, y=fcf_values, name="Free Cash Flow", marker_color=colors))
                except:
                    pass

            # Dividendes
            if metrics['dividends_paid'] is not None:
                try:
                    div_values = [abs(v)/1e9 if v else 0 for v in metrics['dividends_paid'].values]
                    fig.add_trace(go.Scatter(x=years, y=div_values, name="Dividendes Versés",
                                             mode='lines+markers', line=dict(color='#f1c40f', width=2)))
                except:
                    pass

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Free Cash Flow & Dividendes (Milliards $)",
                yaxis_title="Milliards $",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Cash Flow actuel**")
            st.metric("Operating CF", format_value(info.get('operatingCashflow')))
            st.metric("Free Cash Flow", format_value(info.get('freeCashflow')))

            # FCF Yield
            fcf = info.get('freeCashflow')
            mcap = info.get('marketCap')
            if fcf and mcap and mcap > 0:
                fcf_yield = (fcf / mcap) * 100
                st.metric("FCF Yield", f"{fcf_yield:.2f}%")

            # CAGR FCF
            if metrics['free_cash_flow'] is not None:
                cagr_fcf = calculate_growth(metrics['free_cash_flow'])
                if cagr_fcf:
                    st.metric("CAGR FCF", format_percent(cagr_fcf))

        # Payout ratio historique
        if metrics['dividends_paid'] is not None and metrics['net_income'] is not None:
            with st.expander("📈 Évolution Payout Ratio"):
                try:
                    payout = [(abs(d)/ni)*100 if ni and d and ni > 0 else None
                              for d, ni in zip(metrics['dividends_paid'].values, metrics['net_income'].values)]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=years, y=payout, mode='lines+markers',
                                              line=dict(color='#f1c40f')))
                    fig2.add_hline(y=60, line_dash="dash", line_color="orange",
                                   annotation_text="Seuil 60%")
                    fig2.update_layout(template='plotly_dark', height=300, yaxis_title="Payout %")
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

    # TAB 5: RATIOS
    with tab5:
        st.markdown("### 📊 Évolution des Ratios Clés")

        # Calcul EPS historique
        if metrics['net_income'] is not None and metrics['shares_outstanding'] is not None:
            col1, col2 = st.columns(2)

            with col1:
                try:
                    eps_hist = [ni/shares if shares and ni else None
                                for ni, shares in zip(metrics['net_income'].values, metrics['shares_outstanding'].values)]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=years, y=eps_hist, name="EPS", marker_color='#3498db',
                                         text=[f"${v:.2f}" if v else "" for v in eps_hist],
                                         textposition='outside'))
                    fig.update_layout(template='plotly_dark', height=350, title="Bénéfice par Action (EPS)",
                                      yaxis_title="$ par action")
                    st.plotly_chart(fig, use_container_width=True)

                    # CAGR EPS
                    eps_series = pd.Series([e for e in eps_hist if e is not None])
                    if len(eps_series) >= 2:
                        cagr_eps = calculate_growth(eps_series)
                        st.metric("CAGR EPS", format_percent(cagr_eps) if cagr_eps else "N/A")
                except:
                    st.info("Données EPS insuffisantes")

            with col2:
                # Book Value per Share
                if metrics['total_equity'] is not None:
                    try:
                        bvps = [eq/shares if shares and eq else None
                                for eq, shares in zip(metrics['total_equity'].values, metrics['shares_outstanding'].values)]

                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=years, y=bvps, name="Book Value", marker_color='#2ecc71',
                                             text=[f"${v:.2f}" if v else "" for v in bvps],
                                             textposition='outside'))
                        fig.update_layout(template='plotly_dark', height=350,
                                          title="Valeur Comptable par Action",
                                          yaxis_title="$ par action")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Données BVPS insuffisantes")

        # Revenue per Share
        if metrics['revenue'] is not None and metrics['shares_outstanding'] is not None:
            with st.expander("📈 Revenu par Action"):
                try:
                    rps = [rev/shares if shares and rev else None
                           for rev, shares in zip(metrics['revenue'].values, metrics['shares_outstanding'].values)]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=years, y=rps, marker_color='#9b59b6'))
                    fig.update_layout(template='plotly_dark', height=300, yaxis_title="$ par action")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

        # Tableau récapitulatif des CAGR
        st.markdown("### 📊 Résumé des Croissances (CAGR)")

        cagr_data = []

        if metrics['revenue'] is not None:
            cagr = calculate_growth(metrics['revenue'])
            cagr_data.append({'Métrique': 'Revenus', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                              'Rating': '🟢' if cagr and cagr > 0.10 else '🟡' if cagr and cagr > 0.05 else '🔴'})

        if metrics['net_income'] is not None:
            cagr = calculate_growth(metrics['net_income'])
            cagr_data.append({'Métrique': 'Bénéfice Net', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                              'Rating': '🟢' if cagr and cagr > 0.12 else '🟡' if cagr and cagr > 0.05 else '🔴'})

        if metrics['free_cash_flow'] is not None:
            cagr = calculate_growth(metrics['free_cash_flow'])
            cagr_data.append({'Métrique': 'Free Cash Flow', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                              'Rating': '🟢' if cagr and cagr > 0.10 else '🟡' if cagr and cagr > 0.05 else '🔴'})

        if cagr_data:
            st.dataframe(pd.DataFrame(cagr_data), hide_index=True, use_container_width=True)


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
        st.markdown("### 📊 Données incluses")
        st.caption("""
        - Valorisation (P/E, P/B, EV/EBITDA...)
        - Rentabilité (ROE, ROA, Marges)
        - Croissance (CA, Bénéfices, FCF)
        - Bilan (Dette, Cash, Equity)
        - **Historique 10 ans**
        """)


# ============ ANALYSE FONDAMENTALE ============

if analysis_mode == '📈 Analyse Fondamentale':
    st.header("📈 Analyse Fondamentale")

    if analyze_button and company_name:
        with st.spinner("Recherche..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            with st.spinner("Chargement des données (10 ans)..."):
                data = get_stock_data_full(ticker, '10y')

            if data and data['info'] and data['hist'] is not None and not data['hist'].empty:
                info = data['info']
                hist = data['hist']

                st.subheader(f"🏢 {info.get('shortName', ticker)} ({ticker})")

                if info.get('longBusinessSummary'):
                    with st.expander("📋 Description"):
                        st.write(info['longBusinessSummary'])

                # Métriques clés
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Prix", f"${info.get('regularMarketPrice', 0):.2f}")
                with col2:
                    change = info.get('regularMarketChangePercent', 0)
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

                # Graphique prix 10 ans
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Prix',
                                         line=dict(color='#3498db', width=2)))
                if len(hist) > 200:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(),
                                             mode='lines', name='MM200', line=dict(color='red', width=1)))
                fig.update_layout(template='plotly_dark', height=400, title="Évolution du cours (10 ans)")
                st.plotly_chart(fig, use_container_width=True)

                # Performance 10 ans
                perf_10y = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prix il y a 10 ans", f"${hist['Close'].iloc[0]:.2f}")
                with col2:
                    st.metric("Prix actuel", f"${hist['Close'].iloc[-1]:.2f}")
                with col3:
                    st.metric("Performance 10 ans", f"{perf_10y:+.0f}%")

                st.markdown("---")

                # === ANALYSE HISTORIQUE 10 ANS ===
                display_historical_analysis(data, info)

                st.markdown("---")

                # ESG rapide
                st.subheader("🌱 ESG Rapide")
                esg = get_simple_esg_display(ticker)
                if esg.get('available'):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Environnement", f"{esg.get('environment', 0):.0f}/100")
                    with col2:
                        st.metric("Social", f"{esg.get('social', 0):.0f}/100")
                    with col3:
                        st.metric("Gouvernance", f"{esg.get('governance', 0):.0f}/100")
                    with col4:
                        st.metric("Controverse", f"{esg.get('controversy_level', 0)}/5")
                else:
                    st.info("Données ESG non disponibles")
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


# ============ FOOTER ============

st.markdown("---")
st.caption("📊 Stock Analyzer Pro v5.2 - Données: Yahoo Finance (10 ans), GDELT, EPA, E-PRTR, OpenSanctions, SEC")
