"""
Streamlit UI for Prediction Module
==================================
Interactive interface for stock price predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import warnings

# Suppress warnings in UI
warnings.filterwarnings("ignore")

from . import StockPredictor, PredictionConfig
from .config import DISCLAIMERS, DEFAULT_CONFIG

def render_prediction_tab(current_ticker: Optional[str] = None) -> None:
    """
    Render the stock prediction tab in Streamlit.

    Args:
        current_ticker: Optional pre-selected ticker
    """
    st.header("🎯 Prévision d'Actions")
    st.markdown("*Analyse prédictive avec Monte Carlo, ARIMA, GARCH et Sentiment*")

    # Initialize session state
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
    if "prediction_ticker" not in st.session_state:
        st.session_state.prediction_ticker = current_ticker or ""
    if "prediction_figures" not in st.session_state:
        st.session_state.prediction_figures = None

    # Configuration sidebar
    col1, col2 = st.columns([2, 1])

    with col1:
        ticker = st.text_input(
            "📊 Ticker (symbole boursier)",
            value=st.session_state.prediction_ticker,
            placeholder="Ex: AAPL, MSFT, TSLA",
            help="Entrez le symbole boursier de l'action à analyser"
        )

    with col2:
        days_ahead = st.selectbox(
            "⏰ Horizon de prévision",
            options=[7, 14, 30, 60, 90],
            index=1,
            help="Nombre de jours à prévoir"
        )

    # Advanced settings in expander
    with st.expander("⚙️ Paramètres avancés", expanded=False):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            monte_carlo_sims = st.number_input(
                "Simulations Monte Carlo",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                help="Plus de simulations = plus de précision mais plus lent"
            )

        with col_b:
            news_lookback = st.number_input(
                "Jours d'actualités",
                min_value=7,
                max_value=90,
                value=30,
                help="Période d'analyse des actualités"
            )

        with col_c:
            include_garch = st.checkbox(
                "Inclure GARCH",
                value=True,
                help="Modélisation de la volatilité conditionnelle"
            )

        # Model weights
        st.markdown("**Pondération des modèles:**")
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)

        with wcol1:
            w_mc = st.slider("Monte Carlo", 0.0, 1.0, 0.25, 0.05)
        with wcol2:
            w_arima = st.slider("ARIMA", 0.0, 1.0, 0.20, 0.05)
        with wcol3:
            w_garch = st.slider("GARCH", 0.0, 1.0, 0.15, 0.05)
        with wcol4:
            w_sent = st.slider("Sentiment", 0.0, 1.0, 0.40, 0.05)

        # Normalize weights
        total_weight = w_mc + w_arima + w_garch + w_sent
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Les poids doivent sommer à 1.0 (actuellement: {total_weight:.2f})")

    # Run analysis button
    analyze_button = st.button(
        "🚀 Lancer l'analyse prédictive",
        type="primary",
        use_container_width=True,
        disabled=not ticker
    )

    if analyze_button and ticker:
        st.session_state.prediction_ticker = ticker.upper()

        with st.spinner(f"🔄 Analyse en cours pour {ticker.upper()}..."):
            try:
                # Create predictor
                predictor = StockPredictor(ticker.upper())

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run Monte Carlo
                status_text.text("📊 Simulation Monte Carlo...")
                progress_bar.progress(10)

                try:
                    predictor.run_monte_carlo(
                        days_ahead=days_ahead,
                        num_simulations=monte_carlo_sims,
                        return_paths=True
                    )
                    progress_bar.progress(35)
                except Exception as e:
                    st.warning(f"⚠️ Monte Carlo: {e}")

                # Run Time Series
                status_text.text("📈 Analyse ARIMA/GARCH...")
                try:
                    predictor.run_time_series(
                        forecast_days=days_ahead,
                        include_garch=include_garch
                    )
                    progress_bar.progress(60)
                except Exception as e:
                    st.warning(f"⚠️ ARIMA/GARCH: {e}")

                # Run Sentiment
                status_text.text("📰 Analyse du sentiment...")
                try:
                    predictor.run_sentiment_analysis(lookback_days=news_lookback)
                    progress_bar.progress(85)
                except Exception as e:
                    st.warning(f"⚠️ Sentiment: {e}")

                # Aggregate results
                status_text.text("🎯 Agrégation des résultats...")

                # Get the prediction result
                result = predictor._probability_engine.aggregate(
                    ticker=predictor.ticker,
                    current_price=predictor.current_price,
                    horizon_days=days_ahead,
                    monte_carlo=predictor._monte_carlo_results,
                    time_series=predictor._time_series_results,
                    sentiment=predictor._sentiment_results
                )

                predictor._prediction_results = result

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                # Store results in session state
                st.session_state.prediction_results = {
                    "predictor": predictor,
                    "result": result,
                    "monte_carlo": predictor._monte_carlo_results,
                    "time_series": predictor._time_series_results,
                    "sentiment": predictor._sentiment_results,
                    "historical": predictor.historical_prices
                }

                # Generate figures
                st.session_state.prediction_figures = predictor.get_dashboard_figures()

                st.success(f"✅ Analyse terminée pour {predictor.company_name}")

            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                return

    # Display results if available
    if st.session_state.prediction_results:
        display_prediction_results()


def display_prediction_results() -> None:
    """Display prediction results from session state."""
    data = st.session_state.prediction_results
    figures = st.session_state.prediction_figures

    if not data:
        return

    predictor = data["predictor"]
    result = data["result"]
    mc = data["monte_carlo"]
    ts = data["time_series"]
    sent = data["sentiment"]

    st.markdown("---")

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Prix actuel",
            f"${result.current_price:.2f}"
        )

    with col2:
        prob_color = "normal" if result.probability_hausse > 0.5 else "inverse"
        st.metric(
            "Probabilité hausse",
            f"{result.probability_hausse:.1%}",
            delta=f"{(result.probability_hausse - 0.5) * 100:.1f}%"
        )

    with col3:
        st.metric(
            "Signal",
            result.signal
        )

    with col4:
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "EXTREME": "⚫"}
        st.metric(
            "Risque",
            f"{risk_emoji.get(result.risk_assessment.risk_level, '⚪')} {result.risk_assessment.risk_level}"
        )

    # Recommendation banner
    rec_color = "#4CAF50" if "ACHAT" in result.recommendation else "#f44336" if "VENTE" in result.recommendation else "#2196F3"
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, {rec_color}22, {rec_color}11);
            border-left: 4px solid {rec_color};
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        ">
            <h3 style="margin: 0; color: {rec_color};">{result.recommendation}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.8;">
                Horizon: {result.horizon_days} jours |
                Confiance: [{result.confidence_interval[0]:.1%} - {result.confidence_interval[1]:.1%}]
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tabs for detailed results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Synthèse",
        "🎲 Monte Carlo",
        "📈 ARIMA/GARCH",
        "📰 Sentiment",
        "⚠️ Risques"
    ])

    with tab1:
        render_synthesis_tab(result, figures)

    with tab2:
        render_monte_carlo_tab(mc, figures, data.get("historical"))

    with tab3:
        render_time_series_tab(ts, figures, data.get("historical"))

    with tab4:
        render_sentiment_tab(sent, figures)

    with tab5:
        render_risks_tab(result)

    # Disclaimers
    with st.expander("⚠️ Avertissements légaux", expanded=False):
        for d in DISCLAIMERS:
            st.markdown(f"- {d}")


def render_synthesis_tab(result, figures) -> None:
    """Render synthesis tab."""
    col1, col2 = st.columns(2)

    with col1:
        if "probability_gauge" in figures:
            st.plotly_chart(figures["probability_gauge"], use_container_width=True)

    with col2:
        if "contributions" in figures:
            st.plotly_chart(figures["contributions"], use_container_width=True)

    # Contributions table
    st.markdown("### 📊 Contributions des modèles")
    contrib_df = pd.DataFrame([
        {
            "Modèle": m.model_name,
            "Probabilité": f"{m.probability:.1%}",
            "Poids": f"{m.weight:.0%}",
            "Contribution": f"{m.contribution:+.2f}",
            "Statut": "✅" if m.available else "❌"
        }
        for m in result.model_details
    ])
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # Price targets
    if result.target_price_mean:
        st.markdown("### 🎯 Objectifs de prix")
        col1, col2, col3 = st.columns(3)
        with col1:
            change = ((result.target_price_mean / result.current_price) - 1) * 100
            st.metric("Prix cible moyen", f"${result.target_price_mean:.2f}", f"{change:+.1f}%")
        if result.target_price_range:
            with col2:
                st.metric("Borne basse (95%)", f"${result.target_price_range[0]:.2f}")
            with col3:
                st.metric("Borne haute (95%)", f"${result.target_price_range[1]:.2f}")


def render_monte_carlo_tab(mc, figures, historical) -> None:
    """Render Monte Carlo tab."""
    if not mc:
        st.warning("Données Monte Carlo non disponibles")
        return

    st.markdown("### 🎲 Simulation Monte Carlo")
    st.markdown(f"*{mc.num_simulations:,} trajectoires simulées sur {mc.days_simulated} jours*")

    # Charts
    if "monte_carlo_paths" in figures:
        st.plotly_chart(figures["monte_carlo_paths"], use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if "monte_carlo_distribution" in figures:
            st.plotly_chart(figures["monte_carlo_distribution"], use_container_width=True)

    with col2:
        st.markdown("### 📈 Statistiques")
        stats_df = pd.DataFrame({
            "Métrique": [
                "Prix actuel", "Prix moyen attendu", "Prix médian",
                "Écart-type", "Min", "Max",
                "VaR 5%", "VaR 1%", "CVaR 5%"
            ],
            "Valeur": [
                f"${mc.current_price:.2f}",
                f"${mc.mean_price:.2f}",
                f"${mc.median_price:.2f}",
                f"${mc.std_price:.2f}",
                f"${mc.min_price:.2f}",
                f"${mc.max_price:.2f}",
                f"{mc.var_5:.1f}%",
                f"{mc.var_1:.1f}%",
                f"{mc.cvar_5:.1f}%"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Confidence intervals
    st.markdown("### 📊 Intervalles de confiance")
    ci_data = []
    for level, (lower, upper) in mc.confidence_intervals.items():
        ci_data.append({
            "Niveau": level,
            "Borne basse": f"${lower:.2f}",
            "Borne haute": f"${upper:.2f}",
            "Amplitude": f"${upper - lower:.2f}"
        })
    st.dataframe(pd.DataFrame(ci_data), use_container_width=True, hide_index=True)


def render_time_series_tab(ts, figures, historical) -> None:
    """Render time series tab."""
    if not ts:
        st.warning("Données ARIMA/GARCH non disponibles")
        return

    st.markdown("### 📈 Prévision ARIMA")

    # ARIMA chart
    if "arima_forecast" in figures:
        st.plotly_chart(figures["arima_forecast"], use_container_width=True)

    # ARIMA metrics
    arima = ts.arima
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ordre du modèle", f"ARIMA{arima.model_order}")
    with col2:
        st.metric("AIC", f"{arima.aic:.1f}")
    with col3:
        st.metric("BIC", f"{arima.bic:.1f}")

    # Forecast table
    st.markdown("### 📅 Prévisions")
    forecast_df = pd.DataFrame({
        "Date": arima.forecast_dates,
        "Prix prévu": [f"${p:.2f}" for p in arima.forecast],
        "IC Bas": [f"${p:.2f}" for p in arima.confidence_interval_lower],
        "IC Haut": [f"${p:.2f}" for p in arima.confidence_interval_upper]
    })
    st.dataframe(forecast_df.head(10), use_container_width=True, hide_index=True)

    # GARCH if available
    if ts.garch:
        st.markdown("---")
        st.markdown("### 📉 Analyse de volatilité (GARCH)")

        if "garch_volatility" in figures:
            st.plotly_chart(figures["garch_volatility"], use_container_width=True)

        garch = ts.garch
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modèle", f"{garch.model_type}({garch.order[0]},{garch.order[1]})")
        with col2:
            st.metric("Volatilité annualisée", f"{garch.annualized_volatility:.1%}")
        with col3:
            regime_color = {"BASSE": "🟢", "MOYENNE": "🟡", "HAUTE": "🔴", "EXTREME": "⚫"}
            st.metric("Régime", f"{regime_color.get(garch.volatility_regime, '')} {garch.volatility_regime}")


def render_sentiment_tab(sent, figures) -> None:
    """Render sentiment tab."""
    if not sent or sent.news_analyzed == 0:
        st.warning("Données de sentiment non disponibles ou aucune actualité trouvée")
        return

    st.markdown(f"### 📰 Analyse du Sentiment ({sent.news_analyzed} articles)")

    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        color = "🟢" if sent.sentiment_global > 0.1 else "🔴" if sent.sentiment_global < -0.1 else "🟡"
        st.metric("Sentiment global", f"{color} {sent.sentiment_global:.2f}")
    with col2:
        st.metric("Probabilité hausse", f"{sent.probability_hausse:.1%}")
    with col3:
        st.metric("Modèle utilisé", sent.model_used.upper())

    # Charts
    if "sentiment_timeline" in figures:
        st.plotly_chart(figures["sentiment_timeline"], use_container_width=True)

    # Top headlines
    st.markdown("### 📰 Actualités principales")
    for article in sent.top_headlines[:5]:
        color = "#4CAF50" if article.get("sentiment", 0) > 0 else "#f44336" if article.get("sentiment", 0) < 0 else "#9E9E9E"
        st.markdown(
            f"""
            <div style="border-left: 3px solid {color}; padding-left: 10px; margin: 10px 0;">
                <strong>{article.get('title', 'N/A')}</strong><br>
                <small>{article.get('source', 'Unknown')} | {article.get('date', 'N/A')} |
                Sentiment: {article.get('sentiment', 0):.2f}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Word cloud
    if "word_frequencies" in figures:
        st.markdown("### 🔤 Mots-clés fréquents")
        st.plotly_chart(figures["word_frequencies"], use_container_width=True)


def render_risks_tab(result) -> None:
    """Render risks tab."""
    risk = result.risk_assessment

    st.markdown("### ⚠️ Évaluation des risques")

    # Risk gauge
    col1, col2 = st.columns(2)

    with col1:
        # Risk level display
        colors = {"LOW": "#4CAF50", "MEDIUM": "#FF9800", "HIGH": "#f44336", "EXTREME": "#9C27B0"}
        color = colors.get(risk.risk_level, "#9E9E9E")

        st.markdown(
            f"""
            <div style="
                background: {color}22;
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            ">
                <h2 style="color: {color}; margin: 0;">Niveau de risque: {risk.risk_level}</h2>
                <h1 style="color: {color}; margin: 10px 0;">{risk.risk_score:.0f}/100</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Risk indicators
        st.markdown("**Indicateurs:**")
        indicators = [
            ("Volatilité élevée", risk.volatility_warning),
            ("Faible confiance", risk.confidence_warning),
            ("Signaux contradictoires", risk.contradiction_warning)
        ]
        for name, active in indicators:
            icon = "🔴" if active else "🟢"
            st.markdown(f"{icon} {name}")

    # Identified risks
    st.markdown("### 📋 Risques identifiés")
    for r in risk.identified_risks:
        st.markdown(f"- {r}")

    # Model details
    st.markdown("### 🔍 Détails par modèle")
    for detail in result.model_details:
        status = "✅ Disponible" if detail.available else f"❌ {detail.warning or 'Indisponible'}"
        st.markdown(f"**{detail.model_name}**: {status}")
