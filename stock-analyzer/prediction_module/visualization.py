"""
Visualization Module
====================
Interactive charts for prediction results using Plotly.

Features:
- Monte Carlo simulation paths and distribution
- ARIMA forecast with confidence intervals
- GARCH volatility visualization
- Sentiment timeline and word cloud
- Probability gauge and contribution charts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import (
    VisualizationConfig,
    DEFAULT_CONFIG,
    setup_logger
)
from .monte_carlo import MonteCarloResults
from .time_series import TimeSeriesResults
from .news_sentiment import SentimentResults
from .probability_engine import PredictionResult

logger = setup_logger(__name__)

# ==============================================================================
# CHART CONFIGURATION
# ==============================================================================

class ChartConfig:
    """Default chart configuration."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or DEFAULT_CONFIG.visualization

    @property
    def layout_defaults(self) -> Dict[str, Any]:
        return {
            "template": self.config.theme,
            "font": {"family": self.config.font_family},
            "hovermode": "x unified",
            "showlegend": True,
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02}
        }

    @property
    def colors(self) -> Dict[str, str]:
        return {
            "positive": self.config.color_positive,
            "negative": self.config.color_negative,
            "neutral": self.config.color_neutral,
            "bands": self.config.color_confidence_bands
        }

# ==============================================================================
# MONTE CARLO VISUALIZATIONS
# ==============================================================================

def plot_monte_carlo_paths(
    results: MonteCarloResults,
    historical_prices: Optional[pd.Series] = None,
    max_paths: int = 200,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot Monte Carlo simulation paths.

    Args:
        results: MonteCarloResults with simulation data
        historical_prices: Optional historical prices to prepend
        max_paths: Maximum number of paths to display
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)

    fig = go.Figure()

    paths = results.simulated_paths
    if paths is None:
        raise ValueError("No simulation paths available. Run with return_paths=True")

    # Limit paths for performance
    if len(paths) > max_paths:
        indices = np.random.choice(len(paths), max_paths, replace=False)
        display_paths = paths[indices]
    else:
        display_paths = paths

    # Create x-axis (days)
    days = list(range(paths.shape[1]))

    # Plot individual paths
    for i, path in enumerate(display_paths):
        fig.add_trace(go.Scatter(
            x=days,
            y=path,
            mode='lines',
            line={"width": 0.5, "color": "rgba(100, 100, 200, 0.1)"},
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add mean path
    mean_path = np.mean(paths, axis=0)
    fig.add_trace(go.Scatter(
        x=days,
        y=mean_path,
        mode='lines',
        name='Prix moyen',
        line={"width": 3, "color": chart_config.colors["neutral"]}
    ))

    # Add confidence intervals as shaded regions
    for level, color in zip(['95%', '68%', '50%'], chart_config.colors["bands"]):
        if level in results.confidence_intervals:
            ci = results.confidence_intervals[level]
            # Calculate paths within CI at final point
            lower_pct = (1 - float(level.rstrip('%')) / 100) / 2
            upper_pct = 1 - lower_pct
            lower_bound = np.percentile(paths, lower_pct * 100, axis=0)
            upper_bound = np.percentile(paths, upper_pct * 100, axis=0)

            fig.add_trace(go.Scatter(
                x=days + days[::-1],
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill='toself',
                fillcolor=color,
                line={"width": 0},
                name=f'IC {level}',
                hoverinfo='skip'
            ))

    # Add current price line
    fig.add_hline(
        y=results.current_price,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Prix actuel: {results.current_price:.2f}"
    )

    fig.update_layout(
        **chart_config.layout_defaults,
        title=f"Simulation Monte Carlo ({results.num_simulations:,} trajectoires)",
        xaxis_title="Jours",
        yaxis_title="Prix ($)",
        height=500
    )

    return fig


def plot_monte_carlo_distribution(
    results: MonteCarloResults,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot distribution of final prices from Monte Carlo.

    Args:
        results: MonteCarloResults with final prices
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)

    final_prices = results.final_prices
    if final_prices is None:
        raise ValueError("No final prices available")

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=50,
        name='Distribution',
        marker_color=chart_config.colors["neutral"],
        opacity=0.7
    ))

    # Add vertical lines for key statistics
    fig.add_vline(
        x=results.current_price,
        line_dash="dash",
        line_color="gray",
        annotation_text="Prix actuel"
    )

    fig.add_vline(
        x=results.mean_price,
        line_dash="solid",
        line_color=chart_config.colors["positive"],
        annotation_text=f"Moyenne: {results.mean_price:.2f}"
    )

    # Add confidence interval shading
    ci_95 = results.confidence_intervals.get("95%")
    if ci_95:
        fig.add_vrect(
            x0=ci_95[0], x1=ci_95[1],
            fillcolor="rgba(0, 200, 83, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="IC 95%"
        )

    # Add probability annotation
    prob_text = f"P(hausse) = {results.probability_up:.1%}"
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=prob_text,
        showarrow=False,
        font={"size": 14, "color": chart_config.colors["positive"]},
        bgcolor="white",
        bordercolor=chart_config.colors["positive"],
        borderwidth=1
    )

    fig.update_layout(
        **chart_config.layout_defaults,
        title=f"Distribution des prix à {results.days_simulated} jours",
        xaxis_title="Prix ($)",
        yaxis_title="Fréquence",
        height=400
    )

    return fig

# ==============================================================================
# TIME SERIES VISUALIZATIONS
# ==============================================================================

def plot_arima_forecast(
    results: TimeSeriesResults,
    historical_prices: pd.Series,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot ARIMA forecast with confidence intervals.

    Args:
        results: TimeSeriesResults with ARIMA forecast
        historical_prices: Historical price data
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)
    arima = results.arima

    fig = go.Figure()

    # Historical prices (last 60 days)
    hist_display = historical_prices.tail(60)
    fig.add_trace(go.Scatter(
        x=hist_display.index,
        y=hist_display.values,
        mode='lines',
        name='Historique',
        line={"color": "black", "width": 2}
    ))

    # Forecast dates
    forecast_dates = pd.to_datetime(arima.forecast_dates)

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=arima.confidence_interval_upper + arima.confidence_interval_lower[::-1],
        fill='toself',
        fillcolor='rgba(33, 150, 243, 0.2)',
        line={"width": 0},
        name='IC 95%',
        hoverinfo='skip'
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=arima.forecast,
        mode='lines+markers',
        name='Prévision ARIMA',
        line={"color": chart_config.colors["neutral"], "width": 2, "dash": "dash"},
        marker={"size": 6}
    ))

    # Add model info annotation
    order = arima.model_order
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"ARIMA{order}<br>AIC: {arima.aic:.1f}",
        showarrow=False,
        font={"size": 12},
        bgcolor="white",
        borderwidth=1
    )

    fig.update_layout(
        **chart_config.layout_defaults,
        title="Prévision ARIMA",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        height=450
    )

    return fig


def plot_garch_volatility(
    results: TimeSeriesResults,
    historical_prices: pd.Series,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot GARCH volatility analysis.

    Args:
        results: TimeSeriesResults with GARCH results
        historical_prices: Historical price data
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)

    if not results.garch:
        raise ValueError("No GARCH results available")

    garch = results.garch

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Rendements journaliers", "Volatilité conditionnelle"),
        vertical_spacing=0.12
    )

    # Calculate returns
    returns = historical_prices.pct_change().dropna() * 100
    returns_display = returns.tail(100)

    # Plot returns
    colors = [
        chart_config.colors["positive"] if r > 0 else chart_config.colors["negative"]
        for r in returns_display.values
    ]

    fig.add_trace(
        go.Bar(
            x=returns_display.index,
            y=returns_display.values,
            marker_color=colors,
            name='Rendements (%)',
            showlegend=True
        ),
        row=1, col=1
    )

    # Plot conditional volatility if available
    if garch.conditional_volatility is not None:
        cond_vol = garch.conditional_volatility[-100:] * 100  # Convert to %
        vol_dates = historical_prices.index[-len(cond_vol):]

        fig.add_trace(
            go.Scatter(
                x=vol_dates,
                y=cond_vol,
                mode='lines',
                name='Volatilité conditionnelle',
                line={"color": chart_config.colors["negative"], "width": 2}
            ),
            row=2, col=1
        )

    # Add forecast volatility
    forecast_dates = pd.to_datetime(garch.forecast_dates)
    forecast_vol = [v * 100 for v in garch.forecast_volatility]

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_vol,
            mode='lines+markers',
            name='Prévision volatilité',
            line={"color": chart_config.colors["neutral"], "width": 2, "dash": "dash"}
        ),
        row=2, col=1
    )

    # Add regime annotation
    regime_colors = {
        "BASSE": "green",
        "MOYENNE": "orange",
        "HAUTE": "red",
        "EXTREME": "darkred"
    }
    regime = garch.volatility_regime

    fig.add_annotation(
        x=0.98, y=0.98,
        xref="paper", yref="paper",
        text=f"Régime: {regime}<br>σ annualisée: {garch.annualized_volatility:.1%}",
        showarrow=False,
        font={"size": 12, "color": regime_colors.get(regime, "black")},
        bgcolor="white",
        borderwidth=1
    )

    fig.update_layout(
        **chart_config.layout_defaults,
        title=f"Analyse GARCH - {garch.model_type}({garch.order[0]},{garch.order[1]})",
        height=600
    )

    fig.update_yaxes(title_text="Rendement (%)", row=1, col=1)
    fig.update_yaxes(title_text="Volatilité (%)", row=2, col=1)

    return fig

# ==============================================================================
# SENTIMENT VISUALIZATIONS
# ==============================================================================

def plot_sentiment_timeline(
    results: SentimentResults,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot sentiment evolution over time.

    Args:
        results: SentimentResults with news data
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Sentiment par période", "Répartition du sentiment"),
        specs=[[{"type": "bar"}], [{"type": "pie"}]],
        vertical_spacing=0.2
    )

    # Period sentiment bar chart
    periods = list(results.sentiment_by_period.keys())
    values = list(results.sentiment_by_period.values())

    colors = [
        chart_config.colors["positive"] if v > 0 else chart_config.colors["negative"]
        for v in values
    ]

    fig.add_trace(
        go.Bar(
            x=periods,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition='outside',
            name='Sentiment'
        ),
        row=1, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # Sentiment breakdown pie chart
    breakdown = results.sentiment_breakdown
    fig.add_trace(
        go.Pie(
            labels=['Positif', 'Neutre', 'Négatif'],
            values=[
                breakdown.positive_count,
                breakdown.neutral_count,
                breakdown.negative_count
            ],
            marker_colors=[
                chart_config.colors["positive"],
                "lightgray",
                chart_config.colors["negative"]
            ],
            hole=0.4,
            textinfo='percent+label'
        ),
        row=2, col=1
    )

    # Global sentiment annotation
    sentiment_color = (
        chart_config.colors["positive"] if results.sentiment_global > 0
        else chart_config.colors["negative"]
    )

    fig.add_annotation(
        x=0.5, y=0.5,
        xref="x2 domain", yref="y2 domain",
        text=f"{results.sentiment_global:.2f}",
        font={"size": 24, "color": sentiment_color},
        showarrow=False
    )

    fig.update_layout(
        **chart_config.layout_defaults,
        title=f"Analyse du Sentiment ({results.news_analyzed} articles - {results.model_used})",
        height=600
    )

    return fig


def plot_word_cloud(
    results: SentimentResults,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot word frequency as a simple bar chart (word cloud alternative).

    Args:
        results: SentimentResults with word frequencies
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)

    words = list(results.word_frequencies.keys())[:15]
    counts = list(results.word_frequencies.values())[:15]

    fig = go.Figure(go.Bar(
        x=counts[::-1],
        y=words[::-1],
        orientation='h',
        marker_color=chart_config.colors["neutral"]
    ))

    fig.update_layout(
        **chart_config.layout_defaults,
        title="Mots-clés les plus fréquents",
        xaxis_title="Fréquence",
        yaxis_title="",
        height=400
    )

    return fig

# ==============================================================================
# DASHBOARD VISUALIZATIONS
# ==============================================================================

def plot_probability_gauge(
    prediction: PredictionResult,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot probability gauge for final prediction.

    Args:
        prediction: PredictionResult with aggregated probability
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)

    prob = prediction.probability_hausse * 100

    # Determine color based on probability
    if prob >= 60:
        bar_color = chart_config.colors["positive"]
    elif prob <= 40:
        bar_color = chart_config.colors["negative"]
    else:
        bar_color = chart_config.colors["neutral"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Hausse", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(255, 23, 68, 0.3)'},
                {'range': [35, 45], 'color': 'rgba(255, 152, 0, 0.3)'},
                {'range': [45, 55], 'color': 'rgba(158, 158, 158, 0.3)'},
                {'range': [55, 65], 'color': 'rgba(139, 195, 74, 0.3)'},
                {'range': [65, 100], 'color': 'rgba(0, 200, 83, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob
            }
        }
    ))

    # Add signal annotation
    fig.add_annotation(
        x=0.5, y=-0.15,
        xref="paper", yref="paper",
        text=f"<b>{prediction.signal}</b>",
        font={'size': 18, 'color': bar_color},
        showarrow=False
    )

    fig.update_layout(
        height=350,
        margin={'t': 50, 'b': 50, 'l': 50, 'r': 50}
    )

    return fig


def plot_model_contributions(
    prediction: PredictionResult,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot contribution of each model to final prediction.

    Args:
        prediction: PredictionResult with model contributions
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)

    models = list(prediction.contributions.keys())
    contributions = list(prediction.contributions.values())

    colors = [
        chart_config.colors["positive"] if c > 0 else chart_config.colors["negative"]
        for c in contributions
    ]

    fig = go.Figure(go.Bar(
        x=models,
        y=contributions,
        marker_color=colors,
        text=[f"{c:+.2f}" for c in contributions],
        textposition='outside'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        **chart_config.layout_defaults,
        title="Contribution des modèles",
        xaxis_title="Modèle",
        yaxis_title="Contribution à P(hausse)",
        height=350
    )

    return fig


def plot_risk_assessment(
    prediction: PredictionResult,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """
    Plot risk assessment visualization.

    Args:
        prediction: PredictionResult with risk assessment
        config: Visualization configuration

    Returns:
        Plotly figure
    """
    chart_config = ChartConfig(config)
    risk = prediction.risk_assessment

    # Risk level colors
    level_colors = {
        "LOW": "#4CAF50",
        "MEDIUM": "#FF9800",
        "HIGH": "#f44336",
        "EXTREME": "#9C27B0"
    }

    fig = go.Figure()

    # Risk score gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk.risk_score,
        domain={'x': [0, 0.5], 'y': [0.3, 1]},
        title={'text': "Score de Risque"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': level_colors.get(risk.risk_level, "gray")},
            'steps': [
                {'range': [0, 25], 'color': 'rgba(76, 175, 80, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(255, 152, 0, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(244, 67, 54, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(156, 39, 176, 0.3)'}
            ]
        }
    ))

    # Risk level indicator
    fig.add_annotation(
        x=0.75, y=0.7,
        xref="paper", yref="paper",
        text=f"<b>Niveau: {risk.risk_level}</b>",
        font={'size': 20, 'color': level_colors.get(risk.risk_level, "gray")},
        showarrow=False,
        bgcolor="white",
        bordercolor=level_colors.get(risk.risk_level, "gray"),
        borderwidth=2,
        borderpad=10
    )

    # List of risks
    risks_text = "<br>".join([f"• {r}" for r in risk.identified_risks[:5]])
    fig.add_annotation(
        x=0.75, y=0.3,
        xref="paper", yref="paper",
        text=risks_text,
        font={'size': 12},
        showarrow=False,
        align="left"
    )

    fig.update_layout(
        title="Évaluation des Risques",
        height=400,
        margin={'t': 50, 'b': 20, 'l': 20, 'r': 20}
    )

    return fig


def create_prediction_dashboard(
    prediction: PredictionResult,
    monte_carlo: Optional[MonteCarloResults] = None,
    time_series: Optional[TimeSeriesResults] = None,
    sentiment: Optional[SentimentResults] = None,
    historical_prices: Optional[pd.Series] = None,
    config: Optional[VisualizationConfig] = None
) -> Dict[str, go.Figure]:
    """
    Create complete prediction dashboard with all charts.

    Args:
        prediction: PredictionResult with final prediction
        monte_carlo: Optional Monte Carlo results
        time_series: Optional time series results
        sentiment: Optional sentiment results
        historical_prices: Optional historical price data
        config: Visualization configuration

    Returns:
        Dictionary of figure name to Plotly figure
    """
    figures = {}

    # Main prediction gauge
    figures["probability_gauge"] = plot_probability_gauge(prediction, config)

    # Model contributions
    figures["contributions"] = plot_model_contributions(prediction, config)

    # Risk assessment
    figures["risk"] = plot_risk_assessment(prediction, config)

    # Monte Carlo charts
    if monte_carlo and monte_carlo.simulated_paths is not None:
        figures["monte_carlo_paths"] = plot_monte_carlo_paths(
            monte_carlo, historical_prices, config=config
        )
        figures["monte_carlo_distribution"] = plot_monte_carlo_distribution(
            monte_carlo, config
        )

    # Time series charts
    if time_series and historical_prices is not None:
        figures["arima_forecast"] = plot_arima_forecast(
            time_series, historical_prices, config
        )
        if time_series.garch:
            figures["garch_volatility"] = plot_garch_volatility(
                time_series, historical_prices, config
            )

    # Sentiment charts
    if sentiment and sentiment.news_analyzed > 0:
        figures["sentiment_timeline"] = plot_sentiment_timeline(sentiment, config)
        if sentiment.word_frequencies:
            figures["word_frequencies"] = plot_word_cloud(sentiment, config)

    return figures
