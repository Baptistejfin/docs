"""
Streamlit UI for Portfolio Module
=================================
Interactive interface for portfolio tracking and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import tempfile
import os

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

def render_performance_history_tab(portfolio: Portfolio):
    """Affiche l'historique de performance du portefeuille."""
    st.markdown("### 📈 Performance Historique")

    # Informations sur la création
    if portfolio.creation_date:
        days_since_creation = (datetime.now() - portfolio.creation_date).days
        st.info(f"📅 Portefeuille créé le **{portfolio.creation_date.strftime('%d/%m/%Y à %H:%M')}** (il y a {days_since_creation} jour(s))")

    # Vérifier s'il y a un historique
    if portfolio.historical_values and len(portfolio.historical_values) > 0:
        # Créer un DataFrame à partir de l'historique
        hist_df = pd.DataFrame(portfolio.historical_values)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df = hist_df.sort_values('date')

        # Graphique de l'évolution de la valeur
        st.markdown("#### 💰 Évolution de la Valeur du Portefeuille")

        fig_value = go.Figure()

        # Ligne de la valeur totale
        fig_value.add_trace(go.Scatter(
            x=hist_df['date'],
            y=hist_df['total_value'],
            mode='lines+markers',
            name='Valeur totale',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ))

        # Ligne du capital investi
        fig_value.add_trace(go.Scatter(
            x=hist_df['date'],
            y=hist_df['total_invested'],
            mode='lines',
            name='Capital investi',
            line=dict(color='#95a5a6', width=2, dash='dash')
        ))

        fig_value.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Valeur (€)",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig_value, use_container_width=True)

        # Graphique du P/L
        st.markdown("#### 📊 Évolution du Profit/Perte")

        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in hist_df['profit_loss']]

        fig_pl = go.Figure()
        fig_pl.add_trace(go.Bar(
            x=hist_df['date'],
            y=hist_df['profit_loss'],
            marker_color=colors,
            name='P/L (€)'
        ))

        fig_pl.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_pl.update_layout(
            height=300,
            xaxis_title="Date",
            yaxis_title="Profit/Perte (€)"
        )

        st.plotly_chart(fig_pl, use_container_width=True)

        # Statistiques
        st.markdown("#### 📈 Statistiques de Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            initial_value = hist_df['total_value'].iloc[0]
            current_value = hist_df['total_value'].iloc[-1]
            total_return = ((current_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
            st.metric("Rendement Total", f"{total_return:+.2f}%")

        with col2:
            max_value = hist_df['total_value'].max()
            st.metric("Valeur Max", f"{max_value:,.2f} €")

        with col3:
            min_value = hist_df['total_value'].min()
            st.metric("Valeur Min", f"{min_value:,.2f} €")

        with col4:
            if len(hist_df) > 1:
                volatility = hist_df['profit_loss_percent'].std()
                st.metric("Volatilité P/L", f"{volatility:.2f}%")
            else:
                st.metric("Volatilité P/L", "N/A")

    else:
        st.warning("""
        📊 **Pas encore d'historique de performance**

        L'historique sera enregistré automatiquement à chaque actualisation des cours.
        Cliquez sur **🔄 Actualiser les cours** pour commencer à suivre la performance.
        """)

        # Afficher les données actuelles comme point de départ
        st.markdown("#### 📍 Situation Actuelle")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valeur Totale", f"{portfolio.total_value:,.2f} €")
        with col2:
            st.metric("Capital Investi", f"{portfolio.total_invested:,.2f} €")
        with col3:
            st.metric("P/L Actuel", f"{portfolio.total_profit_loss:+,.2f} € ({portfolio.total_profit_loss_percent:+.2f}%)")


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
