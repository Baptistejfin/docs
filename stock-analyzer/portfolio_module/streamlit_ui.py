"""
Streamlit UI for Portfolio Module
=================================
Interactive interface for portfolio tracking and analysis.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import tempfile
import os
import yfinance as yf

from .trade_republic import TradeRepublicClient, Portfolio, Position, create_sample_portfolio
from .analytics import PortfolioAnalyzer, PerformanceMetrics, AllocationAnalysis
from .alerts import AlertManager, RecommendationEngine, Alert, InvestmentRecommendation
from .config import (
    PortfolioConfig,
    DEFAULT_CONFIG,
    format_currency,
    format_percent,
    get_color_for_value,
    ALERT_MESSAGES,
    RECOMMENDATION_MESSAGES
)

# ==============================================================================
# CONFIGURATION PERSISTANCE
# ==============================================================================

# Chemin de sauvegarde automatique du portefeuille
PORTFOLIO_AUTO_SAVE_PATH = os.path.join(os.path.expanduser("~"), ".stock_analyzer_portfolio.json")

def auto_save_portfolio(portfolio: Portfolio) -> bool:
    """Sauvegarde automatique du portefeuille."""
    try:
        client = TradeRepublicClient()
        return client.save_portfolio(PORTFOLIO_AUTO_SAVE_PATH, portfolio)
    except Exception as e:
        print(f"Erreur sauvegarde auto: {e}")
        return False

def auto_load_portfolio() -> Optional[Portfolio]:
    """Charge automatiquement le portefeuille sauvegardé s'il existe."""
    if os.path.exists(PORTFOLIO_AUTO_SAVE_PATH):
        try:
            client = TradeRepublicClient()
            return client.load_portfolio(PORTFOLIO_AUTO_SAVE_PATH)
        except Exception as e:
            print(f"Erreur chargement auto: {e}")
    return None

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def init_portfolio_session_state():
    """Initialize session state for portfolio module."""
    if "portfolio_initialized" not in st.session_state:
        st.session_state.portfolio_initialized = False

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = None

    # Auto-load portfolio on first initialization
    if not st.session_state.portfolio_initialized:
        saved_portfolio = auto_load_portfolio()
        if saved_portfolio:
            st.session_state.portfolio = saved_portfolio
        st.session_state.portfolio_initialized = True

    if "portfolio_client" not in st.session_state:
        st.session_state.portfolio_client = TradeRepublicClient()
    if "portfolio_alerts" not in st.session_state:
        st.session_state.portfolio_alerts = []
    if "portfolio_recommendations" not in st.session_state:
        st.session_state.portfolio_recommendations = []
    if "portfolio_last_refresh" not in st.session_state:
        st.session_state.portfolio_last_refresh = None

# ==============================================================================
# MAIN RENDER FUNCTION
# ==============================================================================

def render_portfolio_tab(current_ticker: Optional[str] = None) -> None:
    """
    Render the portfolio tracking tab in Streamlit.

    Args:
        current_ticker: Optional pre-selected ticker
    """
    init_portfolio_session_state()

    st.header("💼 Mon Portefeuille Trade Republic")
    st.markdown("*Suivi en temps réel de vos investissements*")

    # Sidebar for data input
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💼 Configuration Portefeuille")

        input_method = st.radio(
            "Source des données",
            ["📝 Saisie manuelle", "📁 Import CSV", "🔄 Charger sauvegarde", "🎮 Données démo"],
            help="Choisissez comment importer vos positions"
        )

    # Main content based on input method
    if input_method == "📝 Saisie manuelle":
        render_manual_input()
    elif input_method == "📁 Import CSV":
        render_csv_import()
    elif input_method == "🔄 Charger sauvegarde":
        render_load_portfolio()
    else:
        render_demo_portfolio()

    # Display portfolio if available
    if st.session_state.portfolio:
        st.markdown("---")
        render_portfolio_dashboard()


def render_manual_input():
    """Render manual position input form."""
    st.subheader("📝 Saisie de vos positions")

    # Number of positions
    if "num_positions" not in st.session_state:
        st.session_state.num_positions = 3

    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.num_positions = st.number_input(
            "Nombre de positions",
            min_value=1,
            max_value=50,
            value=st.session_state.num_positions
        )
    with col2:
        cash_balance = st.number_input(
            "Cash disponible (€)",
            min_value=0.0,
            value=0.0,
            step=100.0
        )

    # Position inputs
    positions_data = []
    st.markdown("### Vos positions")

    for i in range(int(st.session_state.num_positions)):
        with st.expander(f"Position {i+1}", expanded=i < 3):
            cols = st.columns(4)
            with cols[0]:
                symbol = st.text_input(
                    "Symbole",
                    key=f"symbol_{i}",
                    placeholder="AAPL"
                ).upper()
            with cols[1]:
                quantity = st.number_input(
                    "Quantité",
                    min_value=0.0,
                    step=1.0,
                    key=f"qty_{i}"
                )
            with cols[2]:
                avg_price = st.number_input(
                    "Prix moyen (€)",
                    min_value=0.0,
                    step=0.01,
                    key=f"price_{i}"
                )
            with cols[3]:
                name = st.text_input(
                    "Nom (optionnel)",
                    key=f"name_{i}",
                    placeholder="Apple Inc."
                )

            if symbol and quantity > 0 and avg_price > 0:
                positions_data.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "average_buy_price": avg_price,
                    "name": name if name else None
                })

    # Create portfolio button
    if st.button("💼 Créer mon portefeuille", type="primary", use_container_width=True):
        if positions_data:
            with st.spinner("Récupération des cours en temps réel..."):
                client = st.session_state.portfolio_client
                portfolio = client.create_manual_portfolio(positions_data, cash_balance)
                st.session_state.portfolio = portfolio
                st.session_state.portfolio_last_refresh = datetime.now()
                # Sauvegarde automatique
                auto_save_portfolio(portfolio)
                st.success(f"✅ Portefeuille créé et sauvegardé avec {len(portfolio.positions)} positions!")
                st.rerun()
        else:
            st.warning("⚠️ Veuillez saisir au moins une position valide")


def render_csv_import():
    """Render CSV import interface."""
    st.subheader("📁 Import depuis CSV")

    st.info("""
    **Format CSV attendu (séparateur ;):**
    - ISIN ou Symbol
    - Nom de l'entreprise
    - Quantité
    - Prix d'achat moyen

    Exportez votre historique depuis l'app Trade Republic.
    """)

    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"],
        help="Fichier CSV exporté de Trade Republic"
    )

    if uploaded_file:
        # Preview
        try:
            df = pd.read_csv(uploaded_file, sep=';', decimal=',', nrows=5)
            st.markdown("**Aperçu du fichier:**")
            st.dataframe(df)

            # Reset file pointer
            uploaded_file.seek(0)

            if st.button("📥 Importer le portefeuille", type="primary"):
                # Save to temporary file (cross-platform)
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    with st.spinner("Import en cours..."):
                        client = st.session_state.portfolio_client
                        portfolio = client.import_from_csv(tmp_path)

                        if portfolio:
                            st.session_state.portfolio = portfolio
                            st.session_state.portfolio_last_refresh = datetime.now()
                            # Sauvegarde automatique
                            auto_save_portfolio(portfolio)
                            st.success(f"✅ {len(portfolio.positions)} positions importées et sauvegardées!")
                            st.rerun()
                        else:
                            st.error("❌ Erreur lors de l'import. Vérifiez le format du fichier.")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        except Exception as e:
            st.error(f"❌ Erreur de lecture: {e}")


def render_load_portfolio():
    """Render portfolio load interface."""
    st.subheader("🔄 Charger une sauvegarde")

    uploaded_file = st.file_uploader(
        "Choisir un fichier JSON",
        type=["json"],
        help="Fichier de sauvegarde du portefeuille"
    )

    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            st.json(data)

            if st.button("📥 Charger le portefeuille", type="primary"):
                # Save to temporary file (cross-platform)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                    json.dump(data, tmp_file)
                    tmp_path = tmp_file.name

                try:
                    client = st.session_state.portfolio_client
                    portfolio = client.load_portfolio(tmp_path)

                    if portfolio:
                        # Refresh prices
                        with st.spinner("Actualisation des cours..."):
                            portfolio = client.refresh_prices(portfolio)
                        st.session_state.portfolio = portfolio
                        st.session_state.portfolio_last_refresh = datetime.now()
                        # Sauvegarde automatique
                        auto_save_portfolio(portfolio)
                        st.success("✅ Portefeuille chargé, mis à jour et sauvegardé!")
                        st.rerun()
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        except Exception as e:
            st.error(f"❌ Erreur: {e}")


def render_demo_portfolio():
    """Render demo portfolio."""
    st.subheader("🎮 Portefeuille de démonstration")

    st.info("""
    Cliquez sur le bouton ci-dessous pour charger un portefeuille de démonstration
    avec des actions populaires (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA).
    """)

    if st.button("🎮 Charger le portefeuille démo", type="primary", use_container_width=True):
        with st.spinner("Chargement des données..."):
            portfolio = create_sample_portfolio()
            st.session_state.portfolio = portfolio
            st.session_state.portfolio_last_refresh = datetime.now()
            # Sauvegarde automatique
            auto_save_portfolio(portfolio)
            st.success("✅ Portefeuille démo chargé et sauvegardé!")
            st.rerun()


# ==============================================================================
# PORTFOLIO DASHBOARD
# ==============================================================================

def render_portfolio_dashboard():
    """Render the main portfolio dashboard."""
    portfolio = st.session_state.portfolio
    if not portfolio:
        return

    # Refresh button
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        if portfolio.creation_date:
            days_since_creation = (datetime.now() - portfolio.creation_date).days
            st.markdown(f"**Portefeuille créé il y a {days_since_creation} jour(s)** | Dernière MAJ: {st.session_state.portfolio_last_refresh.strftime('%H:%M:%S') if st.session_state.portfolio_last_refresh else 'N/A'}")
        else:
            st.markdown(f"**Dernière mise à jour:** {st.session_state.portfolio_last_refresh.strftime('%H:%M:%S') if st.session_state.portfolio_last_refresh else 'N/A'}")
    with col2:
        if st.button("🔄 Actualiser les cours"):
            with st.spinner("Mise à jour..."):
                client = st.session_state.portfolio_client
                portfolio = client.refresh_prices(portfolio)
                # Enregistrer la valeur dans l'historique
                portfolio.record_value()
                st.session_state.portfolio = portfolio
                st.session_state.portfolio_last_refresh = datetime.now()
                # Sauvegarde automatique après actualisation
                auto_save_portfolio(portfolio)
                st.rerun()
    with col3:
        if st.button("💾 Exporter JSON"):
            client = st.session_state.portfolio_client
            client.save_portfolio("portfolio_export.json", portfolio)
            st.success("Exporté vers portfolio_export.json!")
    with col4:
        if st.button("🗑️ Réinitialiser"):
            if os.path.exists(PORTFOLIO_AUTO_SAVE_PATH):
                os.remove(PORTFOLIO_AUTO_SAVE_PATH)
            st.session_state.portfolio = None
            st.session_state.portfolio_initialized = False
            st.rerun()

    # Main metrics
    render_main_metrics(portfolio)

    # Tabs - Ajout des nouveaux onglets
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Vue d'ensemble",
        "🌍 Répartition Géographique",
        "📈 Performance Historique",
        "📋 Positions",
        "⚠️ Alertes",
        "💡 Recommandations",
        "📉 Positions en baisse"
    ])

    with tab1:
        render_overview_tab(portfolio)

    with tab2:
        render_geographic_tab(portfolio)

    with tab3:
        render_performance_history_tab(portfolio)

    with tab4:
        render_positions_tab(portfolio)

    with tab5:
        render_alerts_tab(portfolio)

    with tab6:
        render_recommendations_tab(portfolio)

    with tab7:
        render_declining_tab(portfolio)


def render_main_metrics(portfolio: Portfolio):
    """Render main portfolio metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        value_color = get_color_for_value(portfolio.total_profit_loss)
        st.metric(
            "💰 Valeur totale",
            f"{portfolio.total_value:,.2f} €",
            delta=f"{portfolio.total_profit_loss:+,.2f} €"
        )

    with col2:
        st.metric(
            "📈 Performance",
            f"{portfolio.total_profit_loss_percent:+.2f}%",
            delta=f"{portfolio.daily_change:+.2f} € aujourd'hui"
        )

    with col3:
        st.metric(
            "💵 Investi",
            f"{portfolio.total_invested:,.2f} €"
        )

    with col4:
        st.metric(
            "🏦 Cash disponible",
            f"{portfolio.cash_balance:,.2f} €"
        )


def render_overview_tab(portfolio: Portfolio):
    """Render portfolio overview."""
    analyzer = PortfolioAnalyzer()

    col1, col2 = st.columns(2)

    with col1:
        # Allocation pie chart
        st.markdown("### 📊 Allocation du portefeuille")
        fig = create_allocation_chart(portfolio)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Performance by position
        st.markdown("### 📈 Performance par position")
        fig = create_performance_chart(portfolio)
        st.plotly_chart(fig, use_container_width=True)

    # Allocation analysis
    allocation = analyzer.analyze_allocation(portfolio)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏢 Répartition sectorielle")
        if allocation.by_sector and any(v > 0 for v in allocation.by_sector.values()):
            # Graphique en barres horizontales pour les secteurs
            sectors = list(allocation.by_sector.keys())
            values = list(allocation.by_sector.values())

            # Trier par valeur décroissante
            sorted_data = sorted(zip(sectors, values), key=lambda x: x[1], reverse=True)
            sectors, values = zip(*sorted_data) if sorted_data else ([], [])

            # Couleurs par secteur
            sector_colors = {
                'Technology': '#3498db',
                'Financial Services': '#2ecc71',
                'Healthcare': '#e74c3c',
                'Consumer Cyclical': '#f39c12',
                'Communication Services': '#9b59b6',
                'Industrials': '#1abc9c',
                'Consumer Defensive': '#34495e',
                'Energy': '#e67e22',
                'Utilities': '#95a5a6',
                'Real Estate': '#d35400',
                'Basic Materials': '#16a085',
                'Non classé': '#bdc3c7'
            }

            colors = [sector_colors.get(s, '#7f8c8d') for s in sectors]

            fig_sector = go.Figure(go.Bar(
                x=values,
                y=sectors,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition='outside'
            ))
            fig_sector.update_layout(
                height=max(300, len(sectors) * 35),
                margin=dict(t=20, b=20, l=120, r=40),
                xaxis_title="Poids (%)",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_sector, use_container_width=True)

            # Tableau détaillé
            sector_df = pd.DataFrame([
                {
                    "Secteur": k,
                    "Poids (%)": f"{v:.1f}%",
                    "Valeur (€)": f"{v * portfolio.total_value / 100:,.2f}"
                }
                for k, v in sorted(allocation.by_sector.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
        else:
            st.info("Données sectorielles non disponibles")

    with col2:
        st.markdown("### 📦 Type d'actifs")
        if allocation.by_asset_type and any(v > 0 for v in allocation.by_asset_type.values()):
            # Graphique en camembert pour les types d'actifs
            fig_type = go.Figure(data=[go.Pie(
                labels=[k.upper() for k in allocation.by_asset_type.keys()],
                values=list(allocation.by_asset_type.values()),
                hole=0.4,
                textinfo='label+percent',
                marker_colors=['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
            )])
            fig_type.update_layout(
                height=250,
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_type, use_container_width=True)

            type_df = pd.DataFrame([
                {"Type": k.upper(), "Poids (%)": f"{v:.1f}%", "Valeur (€)": f"{v * portfolio.total_value / 100:,.2f}"}
                for k, v in allocation.by_asset_type.items()
            ])
            st.dataframe(type_df, use_container_width=True, hide_index=True)
        else:
            st.info("Données non disponibles")

        # Indice de concentration
        st.markdown("### 📊 Concentration")
        concentration = allocation.concentration
        if concentration > 0.25:
            conc_status = "🔴 Élevée"
            conc_msg = "Portefeuille très concentré - risque élevé"
        elif concentration > 0.15:
            conc_status = "🟡 Modérée"
            conc_msg = "Concentration acceptable"
        else:
            conc_status = "🟢 Faible"
            conc_msg = "Bonne diversification"

        st.metric("Indice HHI", f"{concentration:.4f}", help="Indice Herfindahl-Hirschman - plus il est bas, plus le portefeuille est diversifié")
        st.markdown(f"**{conc_status}** - {conc_msg}")


def render_positions_tab(portfolio: Portfolio):
    """Render positions table."""
    st.markdown("### 📋 Détail des positions")

    # Convert to DataFrame
    df = portfolio.to_dataframe()

    if df.empty:
        st.warning("Aucune position dans le portefeuille")
        return

    # Format columns
    display_df = df[[
        'symbol', 'name', 'quantity', 'average_buy_price', 'current_price',
        'market_value', 'profit_loss', 'profit_loss_percent', 'daily_change_percent'
    ]].copy()

    display_df.columns = [
        'Symbole', 'Nom', 'Qté', 'PRU (€)', 'Cours (€)',
        'Valeur (€)', 'P/L (€)', 'P/L (%)', 'Jour (%)'
    ]

    # Style function
    def color_negative_red(val):
        if isinstance(val, (int, float)):
            color = '#FF4444' if val < 0 else '#44FF44' if val > 0 else 'white'
            return f'color: {color}'
        return ''

    # Apply styling
    styled_df = display_df.style.applymap(
        color_negative_red,
        subset=['P/L (€)', 'P/L (%)', 'Jour (%)']
    ).format({
        'Qté': '{:.2f}',
        'PRU (€)': '{:.2f}',
        'Cours (€)': '{:.2f}',
        'Valeur (€)': '{:,.2f}',
        'P/L (€)': '{:+,.2f}',
        'P/L (%)': '{:+.2f}',
        'Jour (%)': '{:+.2f}'
    })

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Export button
    csv = df.to_csv(index=False, sep=';', decimal=',')
    st.download_button(
        "📥 Exporter en CSV",
        csv,
        "portfolio_positions.csv",
        "text/csv"
    )


def render_alerts_tab(portfolio: Portfolio):
    """Render alerts tab."""
    st.markdown("### ⚠️ Alertes du portefeuille")

    alert_manager = AlertManager()
    alerts = alert_manager.check_all_alerts(portfolio)

    if not alerts:
        st.success("✅ Aucune alerte pour le moment!")
        return

    st.warning(f"**{len(alerts)} alerte(s) active(s)**")

    for alert in alerts:
        alert_info = ALERT_MESSAGES.get(alert.alert_type, {})

        if alert.severity == "error":
            container = st.error
        elif alert.severity == "warning":
            container = st.warning
        elif alert.severity == "success":
            container = st.success
        else:
            container = st.info

        with st.container():
            container(f"""
            **{alert.title}**

            {alert.message}

            {f"💡 *{alert.suggested_action}*" if alert.suggested_action else ""}
            """)


def render_recommendations_tab(portfolio: Portfolio):
    """Render investment recommendations tab."""
    st.markdown("### 💡 Recommandations d'investissement")

    engine = RecommendationEngine()
    recommendations = engine.generate_recommendations(portfolio)

    if not recommendations:
        st.info("Pas de recommandations particulières pour le moment.")
        return

    for rec in recommendations:
        rec_info = RECOMMENDATION_MESSAGES.get(rec.recommendation_type, {})

        color = rec_info.get("color", "#9E9E9E")
        icon = rec_info.get("icon", "")

        with st.container():
            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {color};
                    padding: 15px;
                    margin: 10px 0;
                    background: {color}11;
                    border-radius: 5px;
                ">
                    <h4 style="margin: 0; color: {color};">
                        {icon} {rec.symbol} - {rec.recommendation_type.value}
                    </h4>
                    <p style="margin: 5px 0;"><strong>{rec.name}</strong></p>
                    <p style="margin: 5px 0;">Confiance: {rec.confidence:.0f}%</p>
                    <ul style="margin: 5px 0;">
                        {"".join(f"<li>{r}</li>" for r in rec.reasoning)}
                    </ul>
                    {f"<p><strong>Action suggérée:</strong> {rec.target_action}</p>" if rec.target_action else ""}
                </div>
                """,
                unsafe_allow_html=True
            )


def render_declining_tab(portfolio: Portfolio):
    """Render declining positions tab."""
    st.markdown("### 📉 Positions en baisse")

    analyzer = PortfolioAnalyzer()

    # Threshold selector
    threshold = st.slider(
        "Seuil de perte (%)",
        min_value=-50,
        max_value=0,
        value=-5,
        help="Afficher les positions avec une perte supérieure à ce seuil"
    )

    declining = analyzer.get_declining_positions(portfolio, threshold=threshold)

    if not declining:
        st.success(f"✅ Aucune position en baisse de plus de {threshold}%")
        return

    st.warning(f"**{len(declining)} position(s) en difficulté**")

    for pos in declining:
        severity_color = {
            "CRITIQUE": "#FF0000",
            "IMPORTANT": "#FF6600",
            "MODÉRÉ": "#FFAA00",
            "FAIBLE": "#FFCC00"
        }.get(pos["severity"], "#999999")

        st.markdown(
            f"""
            <div style="
                border: 2px solid {severity_color};
                padding: 15px;
                margin: 10px 0;
                border-radius: 10px;
                background: {severity_color}11;
            ">
                <h4 style="margin: 0;">
                    🔴 {pos["symbol"]} - {pos["name"]}
                </h4>
                <div style="display: flex; gap: 20px; margin-top: 10px;">
                    <div>
                        <small>Quantité</small><br>
                        <strong>{pos["quantity"]:.2f}</strong>
                    </div>
                    <div>
                        <small>PRU</small><br>
                        <strong>{pos["average_buy_price"]:.2f} €</strong>
                    </div>
                    <div>
                        <small>Cours actuel</small><br>
                        <strong>{pos["current_price"]:.2f} €</strong>
                    </div>
                    <div>
                        <small>P/L</small><br>
                        <strong style="color: #FF4444;">{pos["profit_loss"]:+.2f} € ({pos["profit_loss_percent"]:+.1f}%)</strong>
                    </div>
                    <div>
                        <small>Jour</small><br>
                        <strong style="color: {"#FF4444" if pos["daily_change_percent"] < 0 else "#44FF44"};">{pos["daily_change_percent"]:+.1f}%</strong>
                    </div>
                </div>
                <div style="margin-top: 10px; padding: 10px; background: {severity_color}22; border-radius: 5px;">
                    <strong>Sévérité:</strong> {pos["severity"]}<br>
                    <strong>Recommandation:</strong> {pos["recommendation"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==============================================================================
# GEOGRAPHIC TAB
# ==============================================================================

def render_geographic_tab(portfolio: Portfolio):
    """Affiche la répartition géographique du portefeuille."""
    st.markdown("### 🌍 Répartition Géographique")

    analyzer = PortfolioAnalyzer()
    allocation = analyzer.analyze_allocation(portfolio)

    col1, col2 = st.columns(2)

    with col1:
        # Répartition par région
        st.markdown("#### 🗺️ Par Région")
        if allocation.by_region and any(v > 0 for v in allocation.by_region.values()):
            # Graphique en camembert pour les régions
            fig_region = go.Figure(data=[go.Pie(
                labels=list(allocation.by_region.keys()),
                values=list(allocation.by_region.values()),
                hole=0.4,
                textinfo='label+percent',
                marker_colors=px.colors.qualitative.Set2
            )])
            fig_region.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            st.plotly_chart(fig_region, use_container_width=True)

            # Tableau détaillé
            region_df = pd.DataFrame([
                {"Région": k, "Poids (%)": f"{v:.1f}%", "Valeur (€)": f"{v * portfolio.total_value / 100:,.2f}"}
                for k, v in sorted(allocation.by_region.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(region_df, use_container_width=True, hide_index=True)
        else:
            st.info("Données régionales non disponibles")

    with col2:
        # Répartition par pays
        st.markdown("#### 🏳️ Par Pays")
        if allocation.by_country and any(v > 0 for v in allocation.by_country.values()):
            # Graphique en barres horizontales pour les pays
            countries = list(allocation.by_country.keys())
            values = list(allocation.by_country.values())

            # Trier par valeur décroissante
            sorted_data = sorted(zip(countries, values), key=lambda x: x[1], reverse=True)
            countries, values = zip(*sorted_data) if sorted_data else ([], [])

            fig_country = go.Figure(go.Bar(
                x=values,
                y=countries,
                orientation='h',
                marker_color=px.colors.qualitative.Pastel,
                text=[f"{v:.1f}%" for v in values],
                textposition='outside'
            ))
            fig_country.update_layout(
                height=max(300, len(countries) * 35),
                margin=dict(t=20, b=20, l=100, r=40),
                xaxis_title="Poids (%)",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.info("Données par pays non disponibles")

    # Carte du monde (représentation simplifiée)
    st.markdown("#### 📊 Diversification Géographique")

    # Analyse de la diversification
    num_regions = len([r for r, v in allocation.by_region.items() if v > 0 and r != "Non classé"])
    num_countries = len([c for c, v in allocation.by_country.items() if v > 0 and c != "Non classé"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Régions couvertes", f"{num_regions}")
    with col2:
        st.metric("Pays couverts", f"{num_countries}")
    with col3:
        # Score de diversification géographique
        if num_regions >= 3:
            diversification = "Excellente"
            color = "🟢"
        elif num_regions >= 2:
            diversification = "Bonne"
            color = "🟡"
        else:
            diversification = "À améliorer"
            color = "🔴"
        st.metric("Diversification", f"{color} {diversification}")

    # Recommandations géographiques
    st.markdown("#### 💡 Recommandations")

    recommendations = []
    us_weight = allocation.by_country.get("United States", 0)
    europe_weight = sum(v for k, v in allocation.by_region.items() if k == "Europe")

    if us_weight > 70:
        recommendations.append("⚠️ **Exposition US élevée** ({:.1f}%) - Considérez diversifier vers l'Europe ou l'Asie".format(us_weight))
    if europe_weight < 10 and num_countries > 2:
        recommendations.append("💡 **Faible exposition européenne** - Les marchés européens peuvent offrir une diversification intéressante")
    if allocation.by_region.get("Asie-Pacifique", 0) < 5:
        recommendations.append("🌏 **Marchés émergents** - L'Asie-Pacifique représente une opportunité de croissance")
    if num_regions < 2:
        recommendations.append("🔄 **Diversification limitée** - Votre portefeuille est concentré sur une seule région")

    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.success("✅ Votre diversification géographique semble équilibrée!")


# ==============================================================================
# PERFORMANCE HISTORY TAB
# ==============================================================================

# Indices de référence disponibles
BENCHMARK_INDICES = {
    "S&P 500": "^GSPC",
    "CAC 40": "^FCHI",
    "NASDAQ 100": "^NDX",
    "DAX": "^GDAXI",
    "EURO STOXX 50": "^STOXX50E",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "MSCI World": "URTH",
}

# Périodes disponibles
PERFORMANCE_PERIODS = {
    "1 Jour": "1d",
    "1 Semaine": "5d",
    "1 Mois": "1mo",
    "3 Mois": "3mo",
    "6 Mois": "6mo",
    "YTD": "ytd",
    "1 An": "1y",
    "2 Ans": "2y",
    "5 Ans": "5y",
    "Max": "max"
}


def get_portfolio_historical_performance(portfolio: Portfolio, period: str = "1y") -> pd.DataFrame:
    """
    Calcule la performance historique du portefeuille basée sur les positions actuelles.

    Args:
        portfolio: Portfolio à analyser
        period: Période d'historique (1d, 5d, 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y, max)

    Returns:
        DataFrame avec la valeur du portefeuille dans le temps
    """
    if not portfolio.positions:
        return pd.DataFrame()

    # Récupérer l'historique de chaque position
    all_histories = {}
    weights = {}
    total_invested = portfolio.total_invested

    for pos in portfolio.positions:
        try:
            ticker = yf.Ticker(pos.symbol)
            hist = ticker.history(period=period)

            if not hist.empty:
                all_histories[pos.symbol] = hist['Close']
                # Poids basé sur le coût total de la position
                weights[pos.symbol] = pos.total_cost / total_invested if total_invested > 0 else 0
        except Exception as e:
            print(f"Erreur récupération {pos.symbol}: {e}")
            continue

    if not all_histories:
        return pd.DataFrame()

    # Combiner les historiques
    combined_df = pd.DataFrame(all_histories)
    combined_df = combined_df.dropna()

    if combined_df.empty:
        return pd.DataFrame()

    # Calculer les rendements normalisés (base 100)
    normalized_df = combined_df.copy()
    for col in normalized_df.columns:
        first_val = normalized_df[col].iloc[0]
        if first_val > 0:
            normalized_df[col] = (normalized_df[col] / first_val) * 100

    # Calculer la valeur pondérée du portefeuille
    portfolio_value = pd.Series(index=normalized_df.index, dtype=float)
    portfolio_value[:] = 0

    for symbol in normalized_df.columns:
        if symbol in weights:
            portfolio_value += normalized_df[symbol] * weights[symbol]

    result_df = pd.DataFrame({
        'date': portfolio_value.index,
        'portfolio_value': portfolio_value.values,
        'portfolio_return': (portfolio_value.values - 100)  # Rendement en %
    })
    result_df['date'] = pd.to_datetime(result_df['date']).dt.tz_localize(None)

    return result_df


def get_benchmark_performance(benchmark_symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Récupère la performance d'un indice de référence.

    Args:
        benchmark_symbol: Symbole de l'indice (ex: ^GSPC pour S&P 500)
        period: Période d'historique

    Returns:
        DataFrame avec la performance de l'indice normalisée (base 100)
    """
    try:
        ticker = yf.Ticker(benchmark_symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return pd.DataFrame()

        # Normaliser (base 100)
        first_val = hist['Close'].iloc[0]
        normalized = (hist['Close'] / first_val) * 100

        result_df = pd.DataFrame({
            'date': normalized.index,
            'value': normalized.values,
            'return': (normalized.values - 100)  # Rendement en %
        })
        result_df['date'] = pd.to_datetime(result_df['date']).dt.tz_localize(None)

        return result_df

    except Exception as e:
        print(f"Erreur récupération benchmark {benchmark_symbol}: {e}")
        return pd.DataFrame()


def render_performance_history_tab(portfolio: Portfolio):
    """Affiche l'historique de performance du portefeuille avec comparaison aux indices."""
    st.markdown("### 📈 Performance du Portefeuille")

    # Informations sur la création
    if portfolio.creation_date:
        days_since_creation = (datetime.now() - portfolio.creation_date).days
        st.info(f"📅 Portefeuille créé le **{portfolio.creation_date.strftime('%d/%m/%Y à %H:%M')}** (il y a {days_since_creation} jour(s))")

    # === SÉLECTEURS ===
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        selected_period_label = st.selectbox(
            "📅 Période",
            options=list(PERFORMANCE_PERIODS.keys()),
            index=6,  # Default: 1 An
            help="Sélectionnez la période d'analyse"
        )
        selected_period = PERFORMANCE_PERIODS[selected_period_label]

    with col2:
        selected_benchmarks = st.multiselect(
            "📊 Indices de comparaison",
            options=list(BENCHMARK_INDICES.keys()),
            default=["S&P 500", "CAC 40"],
            help="Sélectionnez les indices pour comparer votre performance"
        )

    with col3:
        chart_mode = st.radio(
            "Type de graphique",
            options=["Plotly (interactif)", "TradingView"],
            horizontal=True
        )

    # === CALCUL DE LA PERFORMANCE ===
    with st.spinner("Calcul de la performance..."):
        portfolio_perf = get_portfolio_historical_performance(portfolio, selected_period)

        # Récupérer les performances des benchmarks
        benchmark_data = {}
        for bench_name in selected_benchmarks:
            bench_symbol = BENCHMARK_INDICES[bench_name]
            bench_perf = get_benchmark_performance(bench_symbol, selected_period)
            if not bench_perf.empty:
                benchmark_data[bench_name] = bench_perf

    # === MÉTRIQUES DE PERFORMANCE ===
    if not portfolio_perf.empty:
        st.markdown("---")
        st.markdown("#### 📊 Performance sur la période")

        # Calcul des métriques
        portfolio_return = portfolio_perf['portfolio_return'].iloc[-1]

        cols = st.columns(len(selected_benchmarks) + 1)

        with cols[0]:
            color = "normal" if portfolio_return >= 0 else "inverse"
            st.metric(
                "🎯 Mon Portefeuille",
                f"{portfolio_return:+.2f}%",
                delta=f"vs période",
                delta_color=color
            )

        for i, bench_name in enumerate(selected_benchmarks):
            if bench_name in benchmark_data:
                bench_return = benchmark_data[bench_name]['return'].iloc[-1]
                diff = portfolio_return - bench_return
                with cols[i + 1]:
                    st.metric(
                        f"📈 {bench_name}",
                        f"{bench_return:+.2f}%",
                        delta=f"{diff:+.2f}% vs Portef.",
                        delta_color="normal" if diff >= 0 else "inverse"
                    )

        # === GRAPHIQUE ===
        st.markdown("---")

        if chart_mode == "Plotly (interactif)":
            st.markdown("#### 📈 Performance comparée (Base 100)")

            fig = go.Figure()

            # Ligne du portefeuille
            fig.add_trace(go.Scatter(
                x=portfolio_perf['date'],
                y=portfolio_perf['portfolio_value'],
                mode='lines',
                name='Mon Portefeuille',
                line=dict(color='#3498db', width=3),
                hovertemplate='%{x|%d/%m/%Y}<br>Valeur: %{y:.2f}<br>Rendement: %{customdata:+.2f}%<extra>Portefeuille</extra>',
                customdata=portfolio_perf['portfolio_return']
            ))

            # Lignes des benchmarks
            benchmark_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
            for i, (bench_name, bench_df) in enumerate(benchmark_data.items()):
                # Aligner les dates
                merged = portfolio_perf[['date']].merge(
                    bench_df[['date', 'value', 'return']],
                    on='date',
                    how='inner'
                )
                if not merged.empty:
                    fig.add_trace(go.Scatter(
                        x=merged['date'],
                        y=merged['value'],
                        mode='lines',
                        name=bench_name,
                        line=dict(color=benchmark_colors[i % len(benchmark_colors)], width=2, dash='dash'),
                        hovertemplate='%{x|%d/%m/%Y}<br>Valeur: %{y:.2f}<br>Rendement: %{customdata:+.2f}%<extra>' + bench_name + '</extra>',
                        customdata=merged['return']
                    ))

            # Ligne de base à 100
            fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5)

            fig.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Valeur (Base 100)",
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                template='plotly_dark'
            )

            st.plotly_chart(fig, use_container_width=True)

            # === GRAPHIQUE DES RENDEMENTS EN % ===
            st.markdown("#### 📊 Rendement cumulé (%)")

            fig_return = go.Figure()

            # Barre du portefeuille
            fig_return.add_trace(go.Scatter(
                x=portfolio_perf['date'],
                y=portfolio_perf['portfolio_return'],
                mode='lines',
                name='Mon Portefeuille',
                line=dict(color='#3498db', width=2),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))

            # Lignes des benchmarks
            for i, (bench_name, bench_df) in enumerate(benchmark_data.items()):
                merged = portfolio_perf[['date']].merge(
                    bench_df[['date', 'return']],
                    on='date',
                    how='inner'
                )
                if not merged.empty:
                    fig_return.add_trace(go.Scatter(
                        x=merged['date'],
                        y=merged['return'],
                        mode='lines',
                        name=bench_name,
                        line=dict(color=benchmark_colors[i % len(benchmark_colors)], width=2, dash='dash')
                    ))

            fig_return.add_hline(y=0, line_dash="solid", line_color="gray")

            fig_return.update_layout(
                height=350,
                xaxis_title="Date",
                yaxis_title="Rendement (%)",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                template='plotly_dark'
            )

            st.plotly_chart(fig_return, use_container_width=True)

        else:
            # Mode TradingView
            st.markdown("#### 📊 Comparaison avec TradingView")

            # Créer les symboles pour TradingView
            tv_symbols = []
            for bench_name in selected_benchmarks[:4]:  # Max 4 pour TradingView
                symbol = BENCHMARK_INDICES[bench_name]
                # Convertir les symboles Yahoo en TradingView
                tv_mapping = {
                    "^GSPC": "SP:SPX",
                    "^FCHI": "EURONEXT:PX1",
                    "^NDX": "NASDAQ:NDX",
                    "^GDAXI": "XETR:DAX",
                    "^STOXX50E": "EUREX:FESX1!",
                    "^FTSE": "LSE:UKX",
                    "^N225": "TVC:NI225",
                    "URTH": "AMEX:URTH"
                }
                if symbol in tv_mapping:
                    tv_symbols.append(tv_mapping[symbol])

            # Widget TradingView de comparaison
            symbols_json = ", ".join([f'"{s}"' for s in tv_symbols[:4]])

            tradingview_html = f"""
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
              <div id="tradingview_comparison"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget({{
                "width": "100%",
                "height": 500,
                "symbol": "{tv_symbols[0] if tv_symbols else 'SP:SPX'}",
                "interval": "D",
                "timezone": "Europe/Paris",
                "theme": "dark",
                "style": "2",
                "locale": "fr",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "compareSymbols": [{symbols_json}],
                "container_id": "tradingview_comparison"
              }});
              </script>
            </div>
            <!-- TradingView Widget END -->
            """

            components.html(tradingview_html, height=520)

            st.info("💡 **Astuce:** Utilisez les outils TradingView pour analyser en détail la performance des indices. Vous pouvez ajouter vos propres symboles directement dans le widget.")

        # === STATISTIQUES DÉTAILLÉES ===
        st.markdown("---")
        st.markdown("#### 📈 Statistiques détaillées")

        # Calculer les statistiques
        stats_data = []

        # Stats portefeuille
        port_returns = portfolio_perf['portfolio_return'].pct_change().dropna() * 100
        stats_data.append({
            "Actif": "🎯 Mon Portefeuille",
            "Rendement Total": f"{portfolio_return:+.2f}%",
            "Volatilité": f"{port_returns.std():.2f}%" if len(port_returns) > 1 else "N/A",
            "Max": f"{portfolio_perf['portfolio_return'].max():+.2f}%",
            "Min": f"{portfolio_perf['portfolio_return'].min():+.2f}%",
            "Sharpe (approx)": f"{(portfolio_return / (port_returns.std() * np.sqrt(252))):.2f}" if len(port_returns) > 1 and port_returns.std() > 0 else "N/A"
        })

        # Stats benchmarks
        for bench_name, bench_df in benchmark_data.items():
            bench_return = bench_df['return'].iloc[-1]
            bench_returns = bench_df['return'].pct_change().dropna() * 100
            stats_data.append({
                "Actif": f"📈 {bench_name}",
                "Rendement Total": f"{bench_return:+.2f}%",
                "Volatilité": f"{bench_returns.std():.2f}%" if len(bench_returns) > 1 else "N/A",
                "Max": f"{bench_df['return'].max():+.2f}%",
                "Min": f"{bench_df['return'].min():+.2f}%",
                "Sharpe (approx)": f"{(bench_return / (bench_returns.std() * np.sqrt(252))):.2f}" if len(bench_returns) > 1 and bench_returns.std() > 0 else "N/A"
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    else:
        st.warning("""
        ⚠️ **Impossible de calculer la performance historique**

        Cela peut arriver si :
        - Les données de marché ne sont pas disponibles
        - La période sélectionnée est trop courte
        - Les symboles de vos positions ne sont pas reconnus

        Essayez de sélectionner une période plus longue ou vérifiez vos positions.
        """)

        # Afficher les données actuelles
        st.markdown("#### 📍 Situation Actuelle")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valeur Totale", f"{portfolio.total_value:,.2f} €")
        with col2:
            st.metric("Capital Investi", f"{portfolio.total_invested:,.2f} €")
        with col3:
            color = "normal" if portfolio.total_profit_loss >= 0 else "inverse"
            st.metric(
                "P/L Actuel",
                f"{portfolio.total_profit_loss:+,.2f} €",
                delta=f"{portfolio.total_profit_loss_percent:+.2f}%",
                delta_color=color
            )


# ==============================================================================
# CHART FUNCTIONS
# ==============================================================================

def create_allocation_chart(portfolio: Portfolio) -> go.Figure:
    """Create portfolio allocation pie chart."""
    labels = [p.symbol for p in portfolio.positions]
    values = [p.market_value for p in portfolio.positions]

    # Add cash if present
    if portfolio.cash_balance > 0:
        labels.append("Cash")
        values.append(portfolio.cash_balance)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        showlegend=False,
        height=350,
        margin=dict(t=20, b=20, l=20, r=20)
    )

    return fig


def create_performance_chart(portfolio: Portfolio) -> go.Figure:
    """Create performance bar chart by position."""
    positions = sorted(portfolio.positions, key=lambda p: p.profit_loss_percent)

    fig = go.Figure(go.Bar(
        x=[p.profit_loss_percent for p in positions],
        y=[p.symbol for p in positions],
        orientation='h',
        marker_color=[
            '#00C853' if p.profit_loss_percent >= 0 else '#FF1744'
            for p in positions
        ],
        text=[f"{p.profit_loss_percent:+.1f}%" for p in positions],
        textposition='outside'
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        xaxis_title="Performance (%)",
        yaxis_title="",
        height=max(300, len(positions) * 40),
        margin=dict(t=20, b=40, l=80, r=40)
    )

    return fig
