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
# SESSION STATE INITIALIZATION
# ==============================================================================

def init_portfolio_session_state():
    """Initialize session state for portfolio module."""
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = None
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
                st.success(f"✅ Portefeuille créé avec {len(portfolio.positions)} positions!")
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
                            st.success(f"✅ {len(portfolio.positions)} positions importées!")
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
                        st.success("✅ Portefeuille chargé et mis à jour!")
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
            st.success("✅ Portefeuille démo chargé!")
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
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Dernière mise à jour:** {st.session_state.portfolio_last_refresh.strftime('%H:%M:%S') if st.session_state.portfolio_last_refresh else 'N/A'}")
    with col2:
        if st.button("🔄 Actualiser les cours"):
            with st.spinner("Mise à jour..."):
                client = st.session_state.portfolio_client
                portfolio = client.refresh_prices(portfolio)
                st.session_state.portfolio = portfolio
                st.session_state.portfolio_last_refresh = datetime.now()
                st.rerun()
    with col3:
        if st.button("💾 Sauvegarder"):
            client = st.session_state.portfolio_client
            client.save_portfolio("portfolio_backup.json", portfolio)
            st.success("Portefeuille sauvegardé!")

    # Main metrics
    render_main_metrics(portfolio)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'ensemble",
        "📈 Positions",
        "⚠️ Alertes",
        "💡 Recommandations",
        "📉 Positions en baisse"
    ])

    with tab1:
        render_overview_tab(portfolio)

    with tab2:
        render_positions_tab(portfolio)

    with tab3:
        render_alerts_tab(portfolio)

    with tab4:
        render_recommendations_tab(portfolio)

    with tab5:
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
        if allocation.by_sector:
            sector_df = pd.DataFrame([
                {"Secteur": k, "Poids (%)": v}
                for k, v in sorted(allocation.by_sector.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
        else:
            st.info("Données sectorielles non disponibles")

    with col2:
        st.markdown("### 📦 Type d'actifs")
        if allocation.by_asset_type:
            type_df = pd.DataFrame([
                {"Type": k.upper(), "Poids (%)": f"{v:.1f}"}
                for k, v in allocation.by_asset_type.items()
            ])
            st.dataframe(type_df, use_container_width=True, hide_index=True)


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
