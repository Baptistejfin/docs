"""
Stock Analyzer Pro v5.1 - Analyse ESG + Fondamentale
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
    get_comprehensive_esg_pro,
    analyze_greenwashing_pro,
    get_epa_violations_detailed,
    get_gdelt_controversies,
    get_opensanctions_data,
    get_eprtr_violations,
    get_sec_governance,
    get_wikidata_controversies,
    get_opencorporates_data,
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
def get_stock_data(ticker: str, period: str):
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        hist = tkr.history(period=period) if period != 'ytd' else tkr.history(start=datetime(datetime.now().year-1, 12, 28))
        return info, hist, tkr.financials, tkr.balance_sheet, tkr.cashflow, tkr.income_stmt
    except:
        return None, None, None, None, None, None

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

def get_rating(value, thresholds, reverse=False):
    """Retourne une note (A à F) basée sur des seuils."""
    if value is None:
        return '—', '⚪'
    try:
        val = float(value)
        if reverse:
            if val <= thresholds[0]: return 'A', '🟢'
            if val <= thresholds[1]: return 'B', '🟢'
            if val <= thresholds[2]: return 'C', '🟡'
            if val <= thresholds[3]: return 'D', '🟠'
            return 'F', '🔴'
        else:
            if val >= thresholds[0]: return 'A', '🟢'
            if val >= thresholds[1]: return 'B', '🟢'
            if val >= thresholds[2]: return 'C', '🟡'
            if val >= thresholds[3]: return 'D', '🟠'
            return 'F', '🔴'
    except:
        return '—', '⚪'


# ============ ANALYSE FONDAMENTALE ============

def analyze_fundamentals(info, financials, balance_sheet, cashflow, income_stmt):
    """Analyse fondamentale complète."""
    analysis = {
        'valorisation': {},
        'rentabilite': {},
        'croissance': {},
        'sante_financiere': {},
        'dividendes': {},
        'scores': {},
    }

    # === VALORISATION ===
    analysis['valorisation'] = {
        'P/E (TTM)': info.get('trailingPE'),
        'P/E (Forward)': info.get('forwardPE'),
        'PEG Ratio': info.get('pegRatio'),
        'P/B (Price/Book)': info.get('priceToBook'),
        'P/S (Price/Sales)': info.get('priceToSalesTrailing12Months'),
        'EV/EBITDA': info.get('enterpriseToEbitda'),
        'EV/Revenue': info.get('enterpriseToRevenue'),
        'Price/FCF': None,  # Calculé si possible
    }

    # Calcul Price/FCF
    fcf = info.get('freeCashflow')
    market_cap = info.get('marketCap')
    if fcf and market_cap and fcf > 0:
        analysis['valorisation']['Price/FCF'] = market_cap / fcf

    # === RENTABILITÉ ===
    analysis['rentabilite'] = {
        'ROE': info.get('returnOnEquity'),
        'ROA': info.get('returnOnAssets'),
        'ROIC': None,  # Calculé si données disponibles
        'Marge Brute': info.get('grossMargins'),
        'Marge Opérationnelle': info.get('operatingMargins'),
        'Marge Nette': info.get('profitMargins'),
        'Marge EBITDA': info.get('ebitdaMargins'),
    }

    # === CROISSANCE ===
    analysis['croissance'] = {
        'Croissance CA (YoY)': info.get('revenueGrowth'),
        'Croissance Bénéfices': info.get('earningsGrowth'),
        'Croissance EPS (QoQ)': info.get('earningsQuarterlyGrowth'),
        'CAGR Revenue 5Y': None,  # À calculer
        'CAGR EPS 5Y': None,
    }

    # Calcul CAGR si données historiques disponibles
    if income_stmt is not None and not income_stmt.empty:
        try:
            revenues = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
            if revenues is not None and len(revenues) >= 2:
                first_val = revenues.iloc[-1]
                last_val = revenues.iloc[0]
                years = len(revenues) - 1
                if first_val > 0 and last_val > 0 and years > 0:
                    cagr = (last_val / first_val) ** (1/years) - 1
                    analysis['croissance']['CAGR Revenue'] = cagr
        except:
            pass

    # === SANTÉ FINANCIÈRE ===
    analysis['sante_financiere'] = {
        'Current Ratio': info.get('currentRatio'),
        'Quick Ratio': info.get('quickRatio'),
        'Debt/Equity': info.get('debtToEquity'),
        'Debt/EBITDA': None,
        'Interest Coverage': None,
        'Dette Totale': info.get('totalDebt'),
        'Trésorerie': info.get('totalCash'),
        'Dette Nette': None,
    }

    # Calcul Dette Nette
    total_debt = info.get('totalDebt', 0) or 0
    total_cash = info.get('totalCash', 0) or 0
    analysis['sante_financiere']['Dette Nette'] = total_debt - total_cash

    # Calcul Debt/EBITDA
    ebitda = info.get('ebitda')
    if ebitda and ebitda > 0 and total_debt:
        analysis['sante_financiere']['Debt/EBITDA'] = total_debt / ebitda

    # === DIVIDENDES ===
    analysis['dividendes'] = {
        'Dividend Yield': info.get('dividendYield'),
        'Payout Ratio': info.get('payoutRatio'),
        'Dividende/Action': info.get('dividendRate'),
        '5Y Avg Yield': info.get('fiveYearAvgDividendYield'),
    }

    # === CALCUL DES SCORES ===
    scores = {}

    # Score Valorisation (inverse: plus bas = mieux)
    pe = info.get('trailingPE')
    if pe:
        if pe < 15: scores['valorisation'] = 90
        elif pe < 20: scores['valorisation'] = 75
        elif pe < 25: scores['valorisation'] = 60
        elif pe < 35: scores['valorisation'] = 40
        else: scores['valorisation'] = 20

    # Score Rentabilité
    roe = info.get('returnOnEquity')
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20: scores['rentabilite'] = 90
        elif roe_pct > 15: scores['rentabilite'] = 75
        elif roe_pct > 10: scores['rentabilite'] = 60
        elif roe_pct > 5: scores['rentabilite'] = 40
        else: scores['rentabilite'] = 20

    # Score Croissance
    rev_growth = info.get('revenueGrowth')
    if rev_growth:
        growth_pct = rev_growth * 100
        if growth_pct > 20: scores['croissance'] = 90
        elif growth_pct > 10: scores['croissance'] = 75
        elif growth_pct > 5: scores['croissance'] = 60
        elif growth_pct > 0: scores['croissance'] = 40
        else: scores['croissance'] = 20

    # Score Santé Financière
    de = info.get('debtToEquity')
    if de is not None:
        if de < 50: scores['sante'] = 90
        elif de < 100: scores['sante'] = 75
        elif de < 150: scores['sante'] = 60
        elif de < 200: scores['sante'] = 40
        else: scores['sante'] = 20

    # Score Global
    if scores:
        analysis['scores'] = scores
        analysis['score_global'] = sum(scores.values()) / len(scores)
    else:
        analysis['score_global'] = None

    return analysis


def display_fundamental_analysis(info, financials, balance_sheet, cashflow, income_stmt):
    """Affiche l'analyse fondamentale."""

    analysis = analyze_fundamentals(info, financials, balance_sheet, cashflow, income_stmt)

    # === SCORE GLOBAL ===
    st.subheader("🎯 Score Fondamental Global")

    if analysis['score_global']:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            score = analysis['score_global']
            color = '#27ae60' if score >= 70 else '#f1c40f' if score >= 50 else '#e74c3c'
            st.metric("Score Global", f"{score:.0f}/100")
        with col2:
            st.metric("📊 Valorisation", f"{analysis['scores'].get('valorisation', 'N/A')}/100")
        with col3:
            st.metric("💰 Rentabilité", f"{analysis['scores'].get('rentabilite', 'N/A')}/100")
        with col4:
            st.metric("📈 Croissance", f"{analysis['scores'].get('croissance', 'N/A')}/100")
        with col5:
            st.metric("🏦 Santé Fin.", f"{analysis['scores'].get('sante', 'N/A')}/100")

    st.markdown("---")

    # === ONGLETS ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Valorisation",
        "💰 Rentabilité",
        "📈 Croissance",
        "🏦 Santé Financière",
        "💵 Dividendes"
    ])

    # TAB 1: VALORISATION
    with tab1:
        st.markdown("### 📊 Ratios de Valorisation")
        st.caption("Évalue si l'action est chère ou bon marché par rapport à ses fondamentaux.")

        col1, col2 = st.columns(2)

        with col1:
            val = analysis['valorisation']

            # Tableau des ratios
            data = []
            interpretations = {
                'P/E (TTM)': {'seuils': [15, 20, 25, 35], 'reverse': True, 'desc': 'Prix / Bénéfices'},
                'P/E (Forward)': {'seuils': [15, 18, 22, 30], 'reverse': True, 'desc': 'P/E estimé'},
                'PEG Ratio': {'seuils': [1, 1.5, 2, 2.5], 'reverse': True, 'desc': 'P/E / Croissance'},
                'P/B (Price/Book)': {'seuils': [1.5, 3, 5, 8], 'reverse': True, 'desc': 'Prix / Valeur comptable'},
                'P/S (Price/Sales)': {'seuils': [2, 5, 10, 15], 'reverse': True, 'desc': 'Prix / Chiffre d\'affaires'},
                'EV/EBITDA': {'seuils': [8, 12, 15, 20], 'reverse': True, 'desc': 'Valeur d\'entreprise / EBITDA'},
            }

            for metric, value in val.items():
                if metric in interpretations:
                    interp = interpretations[metric]
                    rating, emoji = get_rating(value, interp['seuils'], interp['reverse'])
                    data.append({
                        'Métrique': metric,
                        'Valeur': format_ratio(value),
                        'Note': f"{emoji} {rating}",
                        'Description': interp['desc'],
                    })

            st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

        with col2:
            # Graphique radar
            pe = val.get('P/E (TTM)') or 0
            pb = val.get('P/B (Price/Book)') or 0
            ps = val.get('P/S (Price/Sales)') or 0
            ev_ebitda = val.get('EV/EBITDA') or 0

            # Normaliser (inverse car plus bas = mieux)
            scores = [
                max(0, 100 - (pe or 25) * 3),
                max(0, 100 - (pb or 3) * 15),
                max(0, 100 - (ps or 5) * 8),
                max(0, 100 - (ev_ebitda or 12) * 5),
            ]

            fig = go.Figure(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=['P/E', 'P/B', 'P/S', 'EV/EBITDA', 'P/E'],
                fill='toself',
                line_color='#3498db',
                fillcolor='rgba(52, 152, 219, 0.3)'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False, template='plotly_dark', height=300,
                margin=dict(l=60, r=60, t=40, b=40),
                title="Attractivité de la valorisation"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interprétation:**
        - **P/E < 15**: Action potentiellement sous-évaluée
        - **PEG < 1**: Croissance sous-évaluée par le marché
        - **EV/EBITDA < 10**: Valorisation attractive
        """)

    # TAB 2: RENTABILITÉ
    with tab2:
        st.markdown("### 💰 Ratios de Rentabilité")
        st.caption("Mesure la capacité de l'entreprise à générer des profits.")

        col1, col2 = st.columns(2)

        with col1:
            rent = analysis['rentabilite']

            data = []
            interpretations = {
                'ROE': {'seuils': [0.20, 0.15, 0.10, 0.05], 'reverse': False, 'desc': 'Return on Equity'},
                'ROA': {'seuils': [0.10, 0.07, 0.05, 0.02], 'reverse': False, 'desc': 'Return on Assets'},
                'Marge Brute': {'seuils': [0.50, 0.35, 0.25, 0.15], 'reverse': False, 'desc': 'Gross Margin'},
                'Marge Opérationnelle': {'seuils': [0.25, 0.15, 0.10, 0.05], 'reverse': False, 'desc': 'Operating Margin'},
                'Marge Nette': {'seuils': [0.20, 0.10, 0.05, 0.02], 'reverse': False, 'desc': 'Net Margin'},
            }

            for metric, value in rent.items():
                if metric in interpretations:
                    interp = interpretations[metric]
                    rating, emoji = get_rating(value, interp['seuils'], interp['reverse'])
                    data.append({
                        'Métrique': metric,
                        'Valeur': format_percent(value),
                        'Note': f"{emoji} {rating}",
                    })

            st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

        with col2:
            # Graphique des marges
            fig = go.Figure()

            marges = ['Marge Brute', 'Marge Opérationnelle', 'Marge Nette']
            values = [rent.get(m, 0) or 0 for m in marges]
            values = [v * 100 for v in values]

            fig.add_trace(go.Bar(
                x=marges,
                y=values,
                marker_color=['#2ecc71', '#3498db', '#9b59b6'],
                text=[f"{v:.1f}%" for v in values],
                textposition='outside'
            ))
            fig.update_layout(
                template='plotly_dark', height=300,
                yaxis_title='Marge (%)',
                title="Cascade des marges"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interprétation:**
        - **ROE > 15%**: Excellente rentabilité des capitaux propres
        - **Marge nette > 10%**: Forte profitabilité
        - **Marges stables ou croissantes**: Signe de pricing power
        """)

    # TAB 3: CROISSANCE
    with tab3:
        st.markdown("### 📈 Indicateurs de Croissance")
        st.caption("Mesure la dynamique de développement de l'entreprise.")

        col1, col2 = st.columns(2)

        with col1:
            croiss = analysis['croissance']

            data = []
            interpretations = {
                'Croissance CA (YoY)': {'seuils': [0.20, 0.10, 0.05, 0], 'reverse': False},
                'Croissance Bénéfices': {'seuils': [0.25, 0.15, 0.08, 0], 'reverse': False},
                'Croissance EPS (QoQ)': {'seuils': [0.20, 0.10, 0.05, 0], 'reverse': False},
            }

            for metric, value in croiss.items():
                if value is not None:
                    if metric in interpretations:
                        interp = interpretations[metric]
                        rating, emoji = get_rating(value, interp['seuils'], interp['reverse'])
                    else:
                        rating, emoji = '—', '⚪'
                    data.append({
                        'Métrique': metric,
                        'Valeur': format_percent(value),
                        'Note': f"{emoji} {rating}",
                    })

            if data:
                st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
            else:
                st.info("Données de croissance non disponibles")

        with col2:
            # Jauge de croissance
            rev_growth = (croiss.get('Croissance CA (YoY)') or 0) * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rev_growth,
                title={'text': "Croissance CA (%)"},
                delta={'reference': 10},
                gauge={
                    'axis': {'range': [-20, 50]},
                    'bar': {'color': "#3498db"},
                    'steps': [
                        {'range': [-20, 0], 'color': "#e74c3c"},
                        {'range': [0, 10], 'color': "#f1c40f"},
                        {'range': [10, 50], 'color': "#27ae60"},
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': rev_growth
                    }
                }
            ))
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interprétation:**
        - **Croissance CA > 15%**: Forte expansion
        - **Croissance EPS > CA**: Amélioration des marges
        - **CAGR 5Y > 10%**: Croissance soutenue long terme
        """)

    # TAB 4: SANTÉ FINANCIÈRE
    with tab4:
        st.markdown("### 🏦 Santé Financière")
        st.caption("Évalue la solidité du bilan et la capacité à honorer les dettes.")

        col1, col2 = st.columns(2)

        with col1:
            sante = analysis['sante_financiere']

            data = []
            interpretations = {
                'Current Ratio': {'seuils': [2, 1.5, 1.2, 1], 'reverse': False, 'desc': 'Liquidité CT'},
                'Quick Ratio': {'seuils': [1.5, 1.2, 1, 0.8], 'reverse': False, 'desc': 'Liquidité immédiate'},
                'Debt/Equity': {'seuils': [50, 100, 150, 200], 'reverse': True, 'desc': 'Endettement'},
                'Debt/EBITDA': {'seuils': [2, 3, 4, 5], 'reverse': True, 'desc': 'Levier opérationnel'},
            }

            for metric in ['Current Ratio', 'Quick Ratio', 'Debt/Equity', 'Debt/EBITDA']:
                value = sante.get(metric)
                if metric in interpretations:
                    interp = interpretations[metric]
                    rating, emoji = get_rating(value, interp['seuils'], interp['reverse'])
                    data.append({
                        'Métrique': metric,
                        'Valeur': format_ratio(value),
                        'Note': f"{emoji} {rating}",
                        'Description': interp['desc'],
                    })

            st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

            # Valeurs absolues
            st.markdown("**Montants:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Dette Totale", format_value(sante.get('Dette Totale')))
            with col_b:
                st.metric("Trésorerie", format_value(sante.get('Trésorerie')))
            with col_c:
                dette_nette = sante.get('Dette Nette', 0)
                color = "normal" if dette_nette > 0 else "inverse"
                st.metric("Dette Nette", format_value(dette_nette))

        with col2:
            # Graphique dette vs trésorerie
            dette = abs(sante.get('Dette Totale', 0) or 0) / 1e9
            cash = abs(sante.get('Trésorerie', 0) or 0) / 1e9

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Dette Totale', 'Trésorerie', 'Dette Nette'],
                y=[dette, cash, max(0, dette - cash)],
                marker_color=['#e74c3c', '#27ae60', '#f1c40f'],
                text=[f"{v:.1f}B" for v in [dette, cash, max(0, dette - cash)]],
                textposition='outside'
            ))
            fig.update_layout(
                template='plotly_dark', height=300,
                yaxis_title='Milliards $',
                title="Structure du bilan"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Interprétation:**
        - **Current Ratio > 1.5**: Bonne liquidité
        - **Debt/Equity < 100%**: Endettement raisonnable
        - **Debt/EBITDA < 3x**: Capacité de remboursement saine
        - **Dette Nette négative**: L'entreprise a plus de cash que de dette
        """)

    # TAB 5: DIVIDENDES
    with tab5:
        st.markdown("### 💵 Politique de Dividendes")
        st.caption("Analyse du retour aux actionnaires.")

        div = analysis['dividendes']

        col1, col2 = st.columns(2)

        with col1:
            yield_val = div.get('Dividend Yield')
            payout = div.get('Payout Ratio')
            div_rate = div.get('Dividende/Action')

            st.metric("Rendement du dividende", format_percent(yield_val))
            st.metric("Payout Ratio", format_percent(payout))
            st.metric("Dividende par action", f"${div_rate:.2f}" if div_rate else "N/A")

            if payout:
                payout_pct = payout * 100
                if payout_pct < 40:
                    st.success("✅ Payout ratio conservateur - Marge de sécurité élevée")
                elif payout_pct < 60:
                    st.info("ℹ️ Payout ratio équilibré")
                elif payout_pct < 80:
                    st.warning("⚠️ Payout ratio élevé - Peu de marge pour augmenter")
                else:
                    st.error("🔴 Payout ratio très élevé - Dividende potentiellement non soutenable")

        with col2:
            if yield_val:
                # Comparaison avec benchmarks
                yield_pct = yield_val * 100

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=yield_pct,
                    title={'text': "Dividend Yield (%)"},
                    gauge={
                        'axis': {'range': [0, 8]},
                        'bar': {'color': "#27ae60"},
                        'steps': [
                            {'range': [0, 2], 'color': "#ecf0f1"},
                            {'range': [2, 4], 'color': "#bdc3c7"},
                            {'range': [4, 8], 'color': "#95a5a6"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 2  # Moyenne S&P 500
                        }
                    }
                ))
                fig.add_annotation(x=0.5, y=-0.15, text="Ligne rouge = Moyenne S&P 500 (~2%)",
                    showarrow=False, font=dict(size=10))
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pas de dividende ou données non disponibles")

        st.markdown("""
        **Interprétation:**
        - **Yield > 3%**: Rendement attractif pour les revenus
        - **Payout < 60%**: Dividende soutenable et potentiel de croissance
        - **Croissance du dividende**: Signe de confiance du management
        """)


# ============ SIDEBAR ============

with st.sidebar:
    st.header("⚙️ Paramètres")
    analysis_mode = st.radio("Mode", ['📈 Analyse Fondamentale', '🌱 Analyse ESG Pro'], index=0)
    st.markdown("---")

    company_name = st.text_input("Entreprise", value="Apple" if analysis_mode == '📈 Analyse Fondamentale' else "TotalEnergies")

    if analysis_mode == '📈 Analyse Fondamentale':
        period = st.selectbox("Période graphique", ['1y', '2y', '5y', 'max'], index=0)

    st.markdown("---")
    analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)

    if analysis_mode == '🌱 Analyse ESG Pro':
        st.markdown("---")
        st.markdown("### 📊 Sources ESG")
        st.caption("GDELT, EPA, E-PRTR, OpenSanctions, SEC, Wikidata, OpenCorporates")


# ============ ANALYSE FONDAMENTALE ============

if analysis_mode == '📈 Analyse Fondamentale':
    st.header("📈 Analyse Fondamentale")

    if analyze_button and company_name:
        with st.spinner("Recherche..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            info, hist, financials, balance_sheet, cashflow, income_stmt = get_stock_data(ticker, period)

            if info and hist is not None and not hist.empty:
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

                # Graphique prix
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Prix'
                ))
                if len(hist) > 50:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(),
                        mode='lines', name='MM50', line=dict(color='orange', width=1)))
                if len(hist) > 200:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(),
                        mode='lines', name='MM200', line=dict(color='red', width=1)))
                fig.update_layout(template='plotly_dark', height=400, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # Analyse fondamentale
                display_fundamental_analysis(info, financials, balance_sheet, cashflow, income_stmt)

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
                    st.info("💡 Pour l'analyse ESG complète, utilisez le mode **Analyse ESG Pro**")
                else:
                    st.info("Données ESG non disponibles")
        else:
            st.error(f"❌ Entreprise '{company_name}' non trouvée")


# ============ ANALYSE ESG PRO ============

elif analysis_mode == '🌱 Analyse ESG Pro':
    st.header("🌱 Analyse ESG - Niveau Institutionnel")

    if analyze_button and company_name:
        with st.spinner("Recherche du ticker..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            info, hist, _, _, _, _ = get_stock_data(ticker, '1y')
            full_name = info.get('shortName', company_name) if info else company_name

            st.subheader(f"🏢 {full_name} ({ticker})")

            if info and info.get('longBusinessSummary'):
                with st.expander("📋 Description"):
                    st.write(info['longBusinessSummary'])

            with st.spinner("🔍 Collecte des données (10 sources)..."):
                analysis = analyze_greenwashing_pro(ticker, full_name, info)

            st.markdown("---")

            # Score de risque
            st.subheader("🎯 Score de Risque ESG Global")

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
                            font-weight: bold; font-size: 12px;">{analysis['risk_score']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Résumé
            summary = analysis['esg_data']['summary']
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1: st.metric("🗞️ Scandales", summary['scandals_found'])
            with col2: st.metric("💰 Amendes US", f"${summary['total_fines_usd']:,.0f}")
            with col3: st.metric("🚫 Sanctions", summary['sanctions_matches'])
            with col4: st.metric("⚠️ Gouv.", summary['governance_issues'])
            with col5: st.metric("🏭 Sites EU", summary['eu_facilities'])
            with col6: st.metric("🌐 Juridictions", summary['jurisdictions'])

            st.markdown("---")

            # Alertes
            if analysis['flags']:
                st.subheader("🚩 Alertes")
                for flag in analysis['flags']:
                    sev = flag['severity']
                    emoji = {'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟡'}.get(sev, '⚪')
                    msg = f"{emoji} **{flag['title']}** - {flag['description']} _{flag.get('source', '')}_"
                    if sev == 'HIGH': st.error(msg)
                    elif sev == 'MEDIUM': st.warning(msg)
                    else: st.info(msg)

            # Recommandations
            st.subheader("💡 Recommandations")
            for rec in analysis['recommendations']:
                st.info(f"→ {rec}")

        else:
            st.error(f"❌ Entreprise '{company_name}' non trouvée")


# ============ FOOTER ============

st.markdown("---")
st.caption("📊 Stock Analyzer Pro v5.1 - Données: Yahoo Finance, GDELT, EPA, E-PRTR, OpenSanctions, SEC")
