"""
Stock Analyzer Pro v4.0 - Avec Module ESG Amélioré
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import search
import yfinance as yf
from datetime import datetime, timedelta

from esg_module import (
    get_simple_esg_display,
    get_comprehensive_esg,
    analyze_greenwashing_risk,
    get_epa_violations,
    get_yfinance_esg,
    get_sector_esg_benchmark,
    GreenwashingDetector,
    configure_api_keys,
)

st.set_page_config(page_title="Stock Analyzer Pro", page_icon="📈", layout="wide")
st.title("📊 Analyseur d'Actions Boursières Pro")
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
def get_stock_data(ticker_symbol: str, period: str):
    try:
        tkr = yf.Ticker(ticker_symbol)
        info = tkr.info
        if period == 'ytd':
            start_date = datetime(datetime.now().year - 1, 12, 28)
            hist = tkr.history(start=start_date)
        else:
            hist = tkr.history(period=period)
        return info, hist, tkr.financials, tkr.balance_sheet, tkr.cashflow, tkr.recommendations_summary
    except:
        return None, None, None, None, None, None


def format_value(val, unit='', decimal=2):
    try:
        if val is None: return 'N/A'
        val = float(val)
        if abs(val) >= 1e12: return f'{val/1e12:,.{decimal}f} T {unit}'.strip()
        if abs(val) >= 1e9: return f'{val/1e9:,.{decimal}f} B {unit}'.strip()
        if abs(val) >= 1e6: return f'{val/1e6:,.{decimal}f} M {unit}'.strip()
        return f'{val:,.{decimal}f} {unit}'.strip()
    except:
        return 'N/A'


def format_percent(val):
    try:
        return f"{float(val) * 100:.2f}%" if val else 'N/A'
    except:
        return 'N/A'


def format_ratio(val):
    try:
        return f"{float(val):.2f}" if val else 'N/A'
    except:
        return 'N/A'


def monte_carlo_simulation(hist, days=252, sims=1000):
    returns = hist['Close'].pct_change().dropna()
    last_price = hist['Close'].iloc[-1]
    random_returns = np.random.normal(returns.mean(), returns.std(), (days, sims))
    paths = last_price * np.cumprod(1 + random_returns, axis=0)
    return paths, np.percentile(paths, 5, axis=1), np.percentile(paths, 50, axis=1), np.percentile(paths, 95, axis=1), last_price


def get_fundamental_metrics(info, balance_sheet, cashflow):
    return {
        'P/E Ratio': info.get('trailingPE'), 'Forward P/E': info.get('forwardPE'),
        'PEG Ratio': info.get('pegRatio'), 'P/B Ratio': info.get('priceToBook'),
        'P/S Ratio': info.get('priceToSalesTrailing12Months'), 'EV/EBITDA': info.get('enterpriseToEbitda'),
        'ROE': info.get('returnOnEquity'), 'ROA': info.get('returnOnAssets'),
        'Marge Brute': info.get('grossMargins'), 'Marge Opérationnelle': info.get('operatingMargins'),
        'Marge Nette': info.get('profitMargins'), 'Croissance CA': info.get('revenueGrowth'),
        'Current Ratio': info.get('currentRatio'), 'Quick Ratio': info.get('quickRatio'),
        'Debt/Equity': info.get('debtToEquity'), 'Dividend Yield': info.get('dividendYield'),
        'Payout Ratio': info.get('payoutRatio'), 'Operating Cash Flow': info.get('operatingCashflow'),
        'Free Cash Flow': info.get('freeCashflow'),
    }


# ============ SIDEBAR ============

with st.sidebar:
    st.header("⚙️ Paramètres")

    analysis_mode = st.radio("Mode d'analyse", ['Analyse simple', 'Comparaison', '🌱 Analyse ESG'], index=0)
    st.markdown("---")

    if analysis_mode == 'Analyse simple':
        company_name = st.text_input("Nom de l'entreprise", value="Apple")
        companies_to_analyze = [company_name] if company_name else []
    elif analysis_mode == 'Comparaison':
        company_1 = st.text_input("Entreprise 1", value="Apple", key="c1")
        company_2 = st.text_input("Entreprise 2", value="Microsoft", key="c2")
        company_3 = st.text_input("Entreprise 3 (optionnel)", value="", key="c3")
        companies_to_analyze = [c for c in [company_1, company_2, company_3] if c.strip()]
        reference_index = st.selectbox("Indice de référence", ['Aucun', 'S&P 500', 'NASDAQ', 'CAC 40'])
        index_symbols = {'S&P 500': '^GSPC', 'NASDAQ': '^IXIC', 'CAC 40': '^FCHI'}
    else:
        company_name_esg = st.text_input("Nom de l'entreprise", value="TotalEnergies", key="esg")
        companies_to_analyze = [company_name_esg] if company_name_esg else []
        st.info("💡 Détection du greenwashing via croisement ESG + violations EPA")

    st.markdown("---")

    if analysis_mode != '🌱 Analyse ESG':
        period = st.selectbox("Période", ['ytd', '6mo', '1y', '2y', '5y', 'max'], index=2,
            format_func=lambda x: {'ytd': 'YTD', '6mo': '6 mois', '1y': '1 an', '2y': '2 ans', '5y': '5 ans', 'max': 'Max'}.get(x, x))
        if analysis_mode == 'Analyse simple':
            model_choice = st.selectbox("Prévision", ['Monte Carlo', 'Aucun'])
            if model_choice == 'Monte Carlo':
                mc_simulations = st.slider("Simulations", 100, 2000, 500, 100)
                mc_days = st.slider("Jours", 30, 252, 126)

    st.markdown("---")
    analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)


# ============ SESSION STATE ============

if 'all_data' not in st.session_state: st.session_state.all_data = {}
if 'valid_tickers' not in st.session_state: st.session_state.valid_tickers = []
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False


# ============ ANALYSE ESG ============

if analysis_mode == '🌱 Analyse ESG':
    st.header("🌱 Analyse ESG & Détection de Greenwashing")

    if analyze_button and companies_to_analyze:
        company = companies_to_analyze[0]

        with st.spinner(f"Analyse de {company}..."):
            ticker_symbol = get_ticker_from_name(company)

        if ticker_symbol:
            info, hist, _, _, _, _ = get_stock_data(ticker_symbol, '1y')
            company_full_name = info.get('shortName', company) if info else company
            gw_analysis = analyze_greenwashing_risk(ticker_symbol, company_full_name, info)

            st.subheader(f"🏢 {company_full_name} ({ticker_symbol})")

            if info and info.get('longBusinessSummary'):
                with st.expander("📋 Description"):
                    st.write(info['longBusinessSummary'])

            st.markdown("---")

            # Score de risque
            st.subheader("🔍 Score de Risque Greenwashing")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score de Risque", f"{gw_analysis['risk_score']}/100", delta=gw_analysis['risk_level'],
                    delta_color="inverse" if gw_analysis['risk_score'] < 30 else "normal")
            with col2:
                composite = gw_analysis['esg_data'].get('composite_score')
                st.metric("Score ESG", f"{composite}/100" if composite else "N/A",
                    delta=gw_analysis['esg_data'].get('overall_rating', ''))
            with col3:
                epa = gw_analysis.get('epa_data', {})
                st.metric("Pénalités EPA", f"${epa.get('total_penalties', 0):,.0f}",
                    delta="Violations" if epa.get('has_violations') else "OK")
            with col4:
                quality = gw_analysis['esg_data'].get('data_quality', 'low')
                st.metric("Qualité Données", quality.upper())

            # Jauge visuelle
            risk_color = gw_analysis.get('risk_color', '#888')
            st.markdown(f"""
            <div style="background: linear-gradient(to right, #27ae60, #f1c40f, #e74c3c);
                        height: 20px; border-radius: 10px; position: relative; margin: 20px 0;">
                <div style="position: absolute; left: {gw_analysis['risk_score']}%; top: -5px;
                            width: 30px; height: 30px; background: white; border-radius: 50%;
                            border: 3px solid {risk_color}; transform: translateX(-50%);
                            display: flex; align-items: center; justify-content: center;
                            font-weight: bold; font-size: 10px;">{gw_analysis['risk_score']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Graphique radar ESG
            if gw_analysis.get('detailed_scores'):
                st.subheader("📊 Scores ESG Détaillés")
                scores = gw_analysis['detailed_scores']

                col_radar, col_bars = st.columns([1, 1])

                with col_radar:
                    categories = ['Environnement', 'Social', 'Gouvernance', 'Anti-Controverse']
                    values = [scores.get('environment', 0), scores.get('social', 0),
                              scores.get('governance', 0), scores.get('controversy', 0)]

                    fig = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                        fill='toself', line_color='#2ecc71', fillcolor='rgba(46, 204, 113, 0.3)'))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False, template='plotly_dark', height=350, margin=dict(l=60, r=60, t=40, b=40))
                    st.plotly_chart(fig, use_container_width=True)

                with col_bars:
                    for label, key, emoji in [('Environnement', 'environment', '🌍'), ('Social', 'social', '👥'),
                                               ('Gouvernance', 'governance', '🏛️'), ('Anti-Controverse', 'controversy', '⚖️')]:
                        score = scores.get(key, 0)
                        st.markdown(f"{emoji} **{label}**: {score:.0f}/100")
                        st.progress(int(score)/100)

            st.markdown("---")

            # Alertes
            st.subheader("🚩 Alertes Greenwashing")
            if gw_analysis['flags']:
                for flag in gw_analysis['flags']:
                    sev = flag['severity']
                    emoji = {'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟡'}.get(sev, '⚪')
                    msg = f"{emoji} **{flag['title']}**\n\n{flag['description']}"
                    if flag.get('detail'): msg += f"\n\n_{flag['detail']}_"

                    if sev == 'HIGH': st.error(msg)
                    elif sev == 'MEDIUM': st.warning(msg)
                    else: st.info(msg)
            else:
                st.success("✅ Aucune alerte de greenwashing détectée")

            # Points positifs
            if gw_analysis['positive_points']:
                st.subheader("✅ Points Positifs")
                cols = st.columns(min(len(gw_analysis['positive_points']), 3))
                for i, point in enumerate(gw_analysis['positive_points']):
                    with cols[i % 3]:
                        st.success(f"**{point['title']}**\n\n{point['description']}")

            st.markdown("---")

            # Violations EPA
            epa_data = gw_analysis.get('epa_data', {})
            if epa_data.get('available') and epa_data.get('facilities'):
                st.subheader(f"🏭 Violations EPA ({epa_data.get('facilities_found', 0)} installations)")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Pénalités totales", f"${epa_data.get('total_penalties', 0):,.0f}")
                    st.metric("Trimestres non-conformes", epa_data.get('total_violations_quarters', 0))
                    st.metric("Sévérité", epa_data.get('severity', 'N/A'))
                with col2:
                    df = pd.DataFrame(epa_data['facilities'][:10])
                    if not df.empty:
                        st.dataframe(df[['facility_name', 'state', 'quarters_noncompliance', 'penalties']],
                            use_container_width=True, height=250)

            st.markdown("---")

            # Recommandations
            st.subheader("💡 Recommandations")
            for rec in gw_analysis['recommendations']:
                st.info(f"→ {rec}")

            # Méthodologie
            with st.expander("📚 Méthodologie"):
                st.markdown("""
                **Sources:** Yahoo Finance (Sustainalytics), EPA ECHO, SEC EDGAR, News Sentiment

                **Facteurs de risque:**
                - Bon score ESG + violations EPA: +35 pts
                - Pénalités > $500k: +25 pts
                - Controverses niveau 4-5: +25 pts
                - Secteurs controversés: +10-25 pts

                **Interprétation:** 0-10 Minimal | 10-30 Faible | 30-50 Modéré | 50-70 Élevé | 70+ Critique
                """)
        else:
            st.error(f"❌ Entreprise '{company}' non trouvée")


# ============ ANALYSE SIMPLE ============

elif analysis_mode == 'Analyse simple':
    if analyze_button and companies_to_analyze:
        st.session_state.all_data = {}
        st.session_state.valid_tickers = []

        for company in companies_to_analyze:
            with st.spinner(f"Chargement de {company}..."):
                ticker_symbol = get_ticker_from_name(company)
                if ticker_symbol:
                    info, hist, fin, bs, cf, rec = get_stock_data(ticker_symbol, period)
                    if info and hist is not None and not hist.empty:
                        st.session_state.all_data[ticker_symbol] = {
                            'name': info.get('shortName', ticker_symbol), 'info': info, 'hist': hist,
                            'financials': fin, 'balance_sheet': bs, 'cashflow': cf, 'recommendations': rec}
                        st.session_state.valid_tickers.append(ticker_symbol)

        if st.session_state.valid_tickers:
            st.session_state.analysis_done = True

    if st.session_state.analysis_done and st.session_state.valid_tickers:
        ticker = st.session_state.valid_tickers[0]
        data = st.session_state.all_data[ticker]
        info, hist = data['info'], data['hist']

        st.header(f"🏢 {data['name']} ({ticker})")

        if info.get('longBusinessSummary'):
            with st.expander("📋 Description"):
                st.write(info['longBusinessSummary'])

        # Indicateurs clés
        st.subheader("📊 Indicateurs Clés")
        price = info.get('regularMarketPrice')
        eps = info.get('trailingEps')

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("💰 Prix", f"${price:.2f}" if price else "N/A")
        with col2: st.metric("📈 P/E", f"{price/eps:.2f}" if price and eps else "N/A")
        with col3: st.metric("🏦 Market Cap", format_value(info.get('marketCap')))
        with col4:
            change = info.get('regularMarketChangePercent')
            st.metric("📊 Variation", f"{change:.2f}%" if change else "N/A")

        st.markdown("---")

        # ESG simplifié
        st.subheader("🌱 Données ESG")
        esg = get_simple_esg_display(ticker)

        if esg.get('available'):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌍 Environnement", f"{esg.get('environment', 0):.0f}/100")
                st.progress(int(esg.get('environment', 0))/100)
            with col2:
                st.metric("👥 Social", f"{esg.get('social', 0):.0f}/100")
                st.progress(int(esg.get('social', 0))/100)
            with col3:
                st.metric("🏛️ Gouvernance", f"{esg.get('governance', 0):.0f}/100")
                st.progress(int(esg.get('governance', 0))/100)
            with col4:
                controversy = esg.get('controversy_level', 0)
                st.metric("⚠️ Controverse", f"{controversy}/5")

            exposures = [k for k in ['coal', 'tobacco', 'controversial_weapons', 'nuclear', 'palm_oil'] if esg.get(k)]
            if exposures:
                st.warning(f"⚠️ Expositions: {', '.join(exposures)}")
            st.info("💡 Pour l'analyse complète, utilisez le mode **🌱 Analyse ESG**")
        else:
            st.info("📊 Données ESG non disponibles")

        st.markdown("---")

        # Fondamentaux
        st.subheader("📈 Analyse Fondamentale")
        metrics = get_fundamental_metrics(info, data['balance_sheet'], data['cashflow'])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**📘 Valorisation**")
            st.dataframe(pd.DataFrame({'Indicateur': ['P/E', 'Forward P/E', 'PEG', 'P/B', 'EV/EBITDA'],
                'Valeur': [format_ratio(metrics[k]) for k in ['P/E Ratio', 'Forward P/E', 'PEG Ratio', 'P/B Ratio', 'EV/EBITDA']]}),
                hide_index=True, use_container_width=True)
        with col2:
            st.markdown("**📙 Rentabilité**")
            st.dataframe(pd.DataFrame({'Indicateur': ['ROE', 'ROA', 'Marge Brute', 'Marge Op.', 'Marge Nette'],
                'Valeur': [format_percent(metrics[k]) for k in ['ROE', 'ROA', 'Marge Brute', 'Marge Opérationnelle', 'Marge Nette']]}),
                hide_index=True, use_container_width=True)
        with col3:
            st.markdown("**📗 Santé Financière**")
            st.dataframe(pd.DataFrame({'Indicateur': ['Current Ratio', 'Quick Ratio', 'Debt/Equity'],
                'Valeur': [format_ratio(metrics[k]) for k in ['Current Ratio', 'Quick Ratio', 'Debt/Equity']]}),
                hide_index=True, use_container_width=True)
        with col4:
            st.markdown("**📕 Cash Flow**")
            st.dataframe(pd.DataFrame({'Indicateur': ['Dividend Yield', 'Payout Ratio', 'Free CF'],
                'Valeur': [format_percent(metrics['Dividend Yield']), format_percent(metrics['Payout Ratio']),
                          format_value(metrics['Free Cash Flow'])]}), hide_index=True, use_container_width=True)

        st.markdown("---")

        # Graphique prix
        st.subheader("📈 Évolution du Cours")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Prix', line=dict(color='#1f77b4', width=2)))
        if len(hist) > 50:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), mode='lines', name='MM50', line=dict(color='orange', dash='dash')))
        if len(hist) > 200:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(), mode='lines', name='MM200', line=dict(color='red', dash='dot')))
        fig.update_layout(template='plotly_dark', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        perf = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Prix début", f"${hist['Close'].iloc[0]:.2f}")
        with col2: st.metric("Prix actuel", f"${hist['Close'].iloc[-1]:.2f}")
        with col3: st.metric("Performance", f"{perf:+.2f}%")

        # Monte Carlo
        if model_choice == 'Monte Carlo' and len(hist) > 30:
            st.markdown("---")
            st.subheader("🎲 Simulation Monte Carlo")
            paths, p5, p50, p95, last = monte_carlo_simulation(hist, mc_days, mc_simulations)
            dates = pd.bdate_range(start=hist.index[-1], periods=mc_days + 1)[1:]

            fig = go.Figure()
            for i in range(min(50, mc_simulations)):
                fig.add_trace(go.Scatter(x=dates, y=paths[:, i], mode='lines', line=dict(width=0.3, color='rgba(150,150,150,0.3)'), showlegend=False))
            fig.add_trace(go.Scatter(x=dates, y=p95, mode='lines', name='95%', line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(x=dates, y=p50, mode='lines', name='Médiane', line=dict(color='gold', width=3)))
            fig.add_trace(go.Scatter(x=dates, y=p5, mode='lines', name='5%', line=dict(color='red', dash='dash')))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("📊 Médiane", f"${p50[-1]:.2f}", delta=f"{((p50[-1]/last - 1)*100):.1f}%")
            with col2: st.metric("🟢 Optimiste", f"${p95[-1]:.2f}", delta=f"{((p95[-1]/last - 1)*100):.1f}%")
            with col3: st.metric("🔴 Pessimiste", f"${p5[-1]:.2f}", delta=f"{((p5[-1]/last - 1)*100):.1f}%")


# ============ COMPARAISON ============

elif analysis_mode == 'Comparaison':
    if analyze_button and len(companies_to_analyze) >= 2:
        st.session_state.all_data = {}
        st.session_state.valid_tickers = []

        for company in companies_to_analyze:
            with st.spinner(f"Chargement de {company}..."):
                ticker_symbol = get_ticker_from_name(company)
                if ticker_symbol:
                    info, hist, fin, bs, cf, rec = get_stock_data(ticker_symbol, period)
                    if info and hist is not None and not hist.empty:
                        st.session_state.all_data[ticker_symbol] = {'name': info.get('shortName', ticker_symbol), 'info': info, 'hist': hist}
                        st.session_state.valid_tickers.append(ticker_symbol)

        if len(st.session_state.valid_tickers) >= 2:
            st.session_state.analysis_done = True

    if st.session_state.analysis_done and len(st.session_state.valid_tickers) >= 2:
        st.header("📊 Comparaison")

        # Tableau
        st.subheader("🔑 Indicateurs Clés")
        rows = []
        for t in st.session_state.valid_tickers:
            d = st.session_state.all_data[t]
            info = d['info']
            m = get_fundamental_metrics(info, None, None)
            rows.append({'Entreprise': d['name'], 'Prix': f"${info.get('regularMarketPrice', 0):.2f}",
                'Market Cap': format_value(info.get('marketCap')), 'P/E': format_ratio(m['P/E Ratio']),
                'ROE': format_percent(m['ROE']), 'Marge Nette': format_percent(m['Marge Nette'])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("---")

        # Graphique
        st.subheader("📈 Évolution Comparée (Base 100)")
        fig = go.Figure()
        for t in st.session_state.valid_tickers:
            d = st.session_state.all_data[t]
            norm = (d['hist']['Close'] / d['hist']['Close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(x=d['hist'].index, y=norm, mode='lines', name=d['name']))

        if reference_index != 'Aucun':
            try:
                idx = yf.Ticker(index_symbols[reference_index]).history(period=period)
                if not idx.empty:
                    norm = (idx['Close'] / idx['Close'].iloc[0]) * 100
                    fig.add_trace(go.Scatter(x=idx.index, y=norm, mode='lines', name=reference_index, line=dict(dash='dash', color='white')))
            except: pass

        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Performance
        st.subheader("📊 Performance")
        cols = st.columns(len(st.session_state.valid_tickers))
        for i, t in enumerate(st.session_state.valid_tickers):
            d = st.session_state.all_data[t]
            perf = ((d['hist']['Close'].iloc[-1] - d['hist']['Close'].iloc[0]) / d['hist']['Close'].iloc[0]) * 100
            with cols[i]:
                st.metric(d['name'], f"${d['hist']['Close'].iloc[-1]:.2f}", delta=f"{perf:.2f}%")


# ============ FOOTER ============

st.markdown("---")
st.caption("📊 Stock Analyzer Pro v4.0 - Données Yahoo Finance & EPA ECHO")
