"""
Stock Analyzer Pro v5.0 - Analyse ESG Niveau Institutionnel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
        return info, hist, tkr.financials, tkr.balance_sheet
    except:
        return None, None, None, None

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


# ============ SIDEBAR ============

with st.sidebar:
    st.header("⚙️ Paramètres")
    analysis_mode = st.radio("Mode", ['Analyse Financière', '🌱 Analyse ESG Pro'], index=1)
    st.markdown("---")

    company_name = st.text_input("Entreprise", value="TotalEnergies" if analysis_mode == '🌱 Analyse ESG Pro' else "Apple")

    if analysis_mode == 'Analyse Financière':
        period = st.selectbox("Période", ['1y', '2y', '5y', 'max'], index=0)

    st.markdown("---")
    analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)

    if analysis_mode == '🌱 Analyse ESG Pro':
        st.markdown("---")
        st.markdown("### 📊 Sources de données")
        st.caption("""
        - **GDELT**: Scandales mondiaux
        - **EPA ECHO**: Amendes US
        - **E-PRTR**: Pollutions EU
        - **OpenSanctions**: Sanctions
        - **SEC EDGAR**: Gouvernance
        - **Wikidata**: Historique
        - **OpenCorporates**: Structure
        """)


# ============ ANALYSE ESG PRO ============

if analysis_mode == '🌱 Analyse ESG Pro':
    st.header("🌱 Analyse ESG - Niveau Institutionnel")

    if analyze_button and company_name:
        with st.spinner("Recherche du ticker..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            info, hist, _, _ = get_stock_data(ticker, '1y')
            full_name = info.get('shortName', company_name) if info else company_name

            st.subheader(f"🏢 {full_name} ({ticker})")

            if info and info.get('longBusinessSummary'):
                with st.expander("📋 Description"):
                    st.write(info['longBusinessSummary'])

            # === ANALYSE COMPLÈTE ===
            with st.spinner("🔍 Collecte des données (10 sources)..."):
                analysis = analyze_greenwashing_pro(ticker, full_name, info)

            st.markdown("---")

            # === SCORE DE RISQUE GLOBAL ===
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

            # Jauge visuelle
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

            # === RÉSUMÉ EXÉCUTIF ===
            st.subheader("📊 Résumé Exécutif")
            summary = analysis['esg_data']['summary']

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("🗞️ Scandales", summary['scandals_found'])
            with col2:
                st.metric("💰 Amendes US", f"${summary['total_fines_usd']:,.0f}")
            with col3:
                st.metric("🚫 Sanctions", summary['sanctions_matches'])
            with col4:
                st.metric("⚠️ Gouv. Issues", summary['governance_issues'])
            with col5:
                st.metric("🏭 Sites EU", summary['eu_facilities'])
            with col6:
                st.metric("🌐 Juridictions", summary['jurisdictions'])

            st.markdown("---")

            # === ALERTES ===
            st.subheader("🚩 Alertes Détectées")

            if analysis['flags']:
                for flag in analysis['flags']:
                    sev = flag['severity']
                    emoji = {'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟡'}.get(sev, '⚪')
                    msg = f"{emoji} **{flag['title']}**\n\n{flag['description']}\n\n_Source: {flag.get('source', 'N/A')}_"

                    if sev == 'HIGH':
                        st.error(msg)
                    elif sev == 'MEDIUM':
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.success("✅ Aucune alerte majeure détectée")

            # === POINTS POSITIFS ===
            if analysis['positive_points']:
                st.subheader("✅ Points Positifs")
                cols = st.columns(min(len(analysis['positive_points']), 4))
                for i, point in enumerate(analysis['positive_points']):
                    with cols[i % 4]:
                        st.success(f"**{point['title']}**\n\n{point['description']}")

            st.markdown("---")

            # === ONGLETS DÉTAILLÉS ===
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🗞️ Scandales (GDELT)",
                "🏭 Amendes US (EPA)",
                "🌍 Pollutions EU (E-PRTR)",
                "⚖️ Sanctions",
                "🏛️ Gouvernance (SEC)"
            ])

            # TAB 1: GDELT Scandales
            with tab1:
                gdelt = analysis['esg_data']['sources'].get('gdelt', {})
                if gdelt.get('available'):
                    st.markdown(f"**{gdelt.get('article_count', 0)} articles analysés** | Sentiment moyen: {gdelt.get('sentiment_avg', 'N/A')}")

                    if gdelt.get('top_themes'):
                        st.markdown("**Catégories:**")
                        theme_df = pd.DataFrame(gdelt['top_themes'], columns=['Catégorie', 'Articles'])
                        st.bar_chart(theme_df.set_index('Catégorie'))

                    if gdelt.get('controversies'):
                        st.markdown("**Articles les plus négatifs:**")
                        for art in gdelt['controversies'][:10]:
                            with st.expander(f"📰 {art['title'][:80]}..." if len(art.get('title', '')) > 80 else f"📰 {art.get('title', 'N/A')}"):
                                st.write(f"**Source:** {art.get('source')} | **Date:** {art.get('date')} | **Tone:** {art.get('tone')}")
                                st.write(f"**Catégorie:** {art.get('category')}")
                                if art.get('url'):
                                    st.markdown(f"[Lire l'article]({art['url']})")
                else:
                    st.info("Aucune donnée GDELT disponible")

            # TAB 2: EPA Violations
            with tab2:
                epa = analysis['esg_data']['sources'].get('epa_echo', {})
                if epa.get('available'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Installations", epa.get('facilities_count', 0))
                    with col2:
                        st.metric("Pénalités totales", f"${epa.get('total_penalties', 0):,.0f}")
                    with col3:
                        st.metric("Sévérité", epa.get('severity', 'N/A'))

                    if epa.get('violations_by_law'):
                        st.markdown("**Violations par loi:**")
                        viol_df = pd.DataFrame([
                            {'Loi': k, 'Violations': v}
                            for k, v in epa['violations_by_law'].items() if v > 0
                        ])
                        if not viol_df.empty:
                            st.bar_chart(viol_df.set_index('Loi'))

                    if epa.get('violation_history'):
                        st.markdown("**Top amendes:**")
                        st.dataframe(pd.DataFrame(epa['violation_history']), use_container_width=True)

                    if epa.get('facilities'):
                        st.markdown("**Installations:**")
                        fac_df = pd.DataFrame(epa['facilities'][:15])
                        st.dataframe(fac_df[['facility_name', 'state', 'penalties', 'quarters_noncompliance']], use_container_width=True)
                else:
                    st.info("Aucune donnée EPA disponible")

            # TAB 3: E-PRTR Europe
            with tab3:
                eprtr = analysis['esg_data']['sources'].get('eprtr', {})
                if eprtr.get('available'):
                    st.markdown(f"**{len(eprtr.get('facilities', []))} installations en Europe**")
                    st.markdown(f"**Pays:** {', '.join(eprtr.get('countries', []))}")

                    if eprtr.get('total_releases'):
                        st.markdown("**Principaux polluants:**")
                        poll_df = pd.DataFrame([
                            {'Polluant': k, 'Quantité': v}
                            for k, v in list(eprtr['total_releases'].items())[:10]
                        ])
                        if not poll_df.empty:
                            st.bar_chart(poll_df.set_index('Polluant'))

                    if eprtr.get('facilities'):
                        st.dataframe(pd.DataFrame(eprtr['facilities']), use_container_width=True)
                else:
                    st.info("Aucune donnée E-PRTR disponible")

            # TAB 4: Sanctions
            with tab4:
                sanctions = analysis['esg_data']['sources'].get('opensanctions', {})
                if sanctions.get('available'):
                    st.markdown(f"**Niveau de risque:** {sanctions.get('risk_level', 'N/A')}")

                    if sanctions.get('sanctions'):
                        st.error("⚠️ ENTITÉ SOUS SANCTIONS")
                        for s in sanctions['sanctions']:
                            st.write(f"- **{s['entity']}**: {s['reason']}")
                    else:
                        st.success("✅ Aucune sanction active trouvée")

                    if sanctions.get('matches'):
                        st.markdown("**Correspondances trouvées:**")
                        st.dataframe(pd.DataFrame(sanctions['matches']), use_container_width=True)
                else:
                    st.info("Aucune donnée OpenSanctions disponible")

            # TAB 5: Gouvernance SEC
            with tab5:
                sec = analysis['esg_data']['sources'].get('sec_edgar', {})
                if sec.get('available'):
                    st.markdown(f"**Entreprise:** {sec.get('company_name')} (CIK: {sec.get('cik')})")
                    st.markdown(f"**Secteur:** {sec.get('sic_description', 'N/A')}")

                    if sec.get('governance_issues'):
                        st.warning(f"⚠️ {len(sec['governance_issues'])} événements matériels détectés")
                        for issue in sec['governance_issues']:
                            st.write(f"- **{issue.get('date')}**: {issue.get('description')}")

                    if sec.get('filings'):
                        st.markdown("**Filings récents:**")
                        st.dataframe(pd.DataFrame(sec['filings'][:15]), use_container_width=True)
                else:
                    st.info("Aucune donnée SEC disponible")

            st.markdown("---")

            # === RECOMMANDATIONS ===
            st.subheader("💡 Recommandations")
            for rec in analysis['recommendations']:
                st.info(f"→ {rec}")

            # === MÉTHODOLOGIE ===
            with st.expander("📚 Méthodologie & Sources"):
                st.markdown("""
                ### Sources de données (toutes gratuites)

                | Source | Couverture | Données |
                |--------|------------|---------|
                | **GDELT Project** | Mondial | Scandales, polémiques, sentiment médiatique |
                | **EPA ECHO** | USA | Violations environnementales, amendes |
                | **E-PRTR** | Europe | Rejets polluants, installations industrielles |
                | **OpenSanctions** | Mondial | Sanctions, PEP, entités à risque |
                | **SEC EDGAR** | USA | Filings, gouvernance, Form 8-K |
                | **UK Environment Agency** | UK | Violations environnementales UK |
                | **Wikidata** | Mondial | Controverses historiques |
                | **OpenCorporates** | Mondial | Structure juridique, filiales |
                | **LobbyFacts EU** | Europe | Lobbying européen |
                | **Yahoo Finance** | Mondial | Scores Sustainalytics |

                ### Calcul du score de risque

                Le score est pondéré:
                - **Médias (15%)**: Nombre et sentiment des articles GDELT
                - **Environnement (30%)**: Amendes EPA + E-PRTR
                - **Sanctions (20%)**: Correspondances OpenSanctions
                - **Gouvernance (20%)**: Issues SEC + controverses
                - **Social (15%)**: Articles GDELT catégorie "SOCIAL"

                ### Interprétation

                - **0-10**: Risque minimal ✅
                - **10-30**: Risque faible 🟢
                - **30-50**: Risque modéré 🟡
                - **50-70**: Risque élevé 🟠
                - **70-100**: Risque critique 🔴
                """)

        else:
            st.error(f"❌ Entreprise '{company_name}' non trouvée")


# ============ ANALYSE FINANCIÈRE ============

elif analysis_mode == 'Analyse Financière':
    if analyze_button and company_name:
        with st.spinner("Chargement..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            info, hist, fin, bs = get_stock_data(ticker, period)

            if info and hist is not None and not hist.empty:
                st.header(f"🏢 {info.get('shortName', ticker)} ({ticker})")

                # Métriques clés
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Prix", f"${info.get('regularMarketPrice', 0):.2f}")
                with col2:
                    pe = info.get('trailingPE')
                    st.metric("P/E", f"{pe:.2f}" if pe else "N/A")
                with col3:
                    st.metric("Market Cap", format_value(info.get('marketCap')))
                with col4:
                    change = info.get('regularMarketChangePercent', 0)
                    st.metric("Var. Jour", f"{change:.2f}%")

                st.markdown("---")

                # Graphique
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Prix'))
                if len(hist) > 50:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(50).mean(), mode='lines', name='MM50', line=dict(dash='dash')))
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Performance
                perf = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                st.metric("Performance", f"{perf:+.2f}%")

                # ESG rapide
                st.markdown("---")
                st.subheader("🌱 ESG Rapide")
                esg = get_simple_esg_display(ticker)
                if esg.get('available'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Environnement", f"{esg.get('environment', 0):.0f}/100")
                    with col2:
                        st.metric("Social", f"{esg.get('social', 0):.0f}/100")
                    with col3:
                        st.metric("Gouvernance", f"{esg.get('governance', 0):.0f}/100")
                    st.info("💡 Pour l'analyse complète, utilisez le mode **Analyse ESG Pro**")
                else:
                    st.info("Données ESG non disponibles")
        else:
            st.error(f"❌ Entreprise non trouvée")


# ============ FOOTER ============

st.markdown("---")
st.caption("📊 Stock Analyzer Pro v5.0 - Données: GDELT, EPA, E-PRTR, OpenSanctions, SEC, Yahoo Finance")
