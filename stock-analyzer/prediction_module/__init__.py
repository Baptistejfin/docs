"""
Stock Prediction Module
=======================
Advanced stock price prediction using Monte Carlo, ARIMA, GARCH, and Sentiment Analysis.

Usage:
    from prediction_module import StockPredictor

    predictor = StockPredictor(ticker="AAPL")
    results = predictor.run_full_analysis(days_ahead=14)
    predictor.display_dashboard()

Author: Stock Analyzer Pro
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
import json

# Third-party imports
import yfinance as yf

# Module imports
from .config import (
    PredictionConfig,
    DEFAULT_CONFIG,
    MonteCarloConfig,
    ARIMAConfig,
    GARCHConfig,
    SentimentConfig,
    AggregationWeights,
    VisualizationConfig,
    setup_logger,
    DISCLAIMERS
)
from .monte_carlo import MonteCarloSimulator, MonteCarloResults
from .time_series import TimeSeriesAnalyzer, TimeSeriesResults
from .news_sentiment import NewsSentimentAnalyzer, SentimentResults
from .probability_engine import ProbabilityEngine, PredictionResult
from .visualization import create_prediction_dashboard

# Configure logger
logger = setup_logger(__name__)

# Version
__version__ = "1.0.0"

# Public API
__all__ = [
    "StockPredictor",
    "PredictionConfig",
    "MonteCarloResults",
    "TimeSeriesResults",
    "SentimentResults",
    "PredictionResult"
]

# ==============================================================================
# MAIN PREDICTOR CLASS
# ==============================================================================

class StockPredictor:
    """
    Main class for stock price prediction.

    Combines Monte Carlo simulations, ARIMA/GARCH time series models,
    and news sentiment analysis to generate predictions.

    Attributes:
        ticker: Stock ticker symbol
        config: PredictionConfig instance
        stock_info: yfinance Ticker info
        historical_prices: Historical price data
    """

    def __init__(
        self,
        ticker: str,
        config: Optional[PredictionConfig] = None
    ):
        """
        Initialize StockPredictor.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
            config: Optional PredictionConfig (uses defaults if None)

        Raises:
            ValueError: If ticker is invalid or no data available
        """
        self.ticker = ticker.upper()
        self.config = config or DEFAULT_CONFIG

        # Initialize components
        self._monte_carlo = MonteCarloSimulator(self.config.monte_carlo)
        self._time_series = TimeSeriesAnalyzer(
            self.config.arima,
            self.config.garch
        )
        self._sentiment = NewsSentimentAnalyzer(self.config.sentiment)
        self._probability_engine = ProbabilityEngine(self.config.weights)

        # Data storage
        self._stock = None
        self._stock_info = None
        self._historical_prices = None
        self._company_name = None

        # Results storage
        self._monte_carlo_results: Optional[MonteCarloResults] = None
        self._time_series_results: Optional[TimeSeriesResults] = None
        self._sentiment_results: Optional[SentimentResults] = None
        self._prediction_results: Optional[PredictionResult] = None
        self._dashboard_figures: Optional[Dict] = None

        # Load data
        self._load_stock_data()

    def _load_stock_data(self) -> None:
        """Load stock data from yfinance."""
        logger.info(f"Loading data for {self.ticker}")

        try:
            self._stock = yf.Ticker(self.ticker)
            info = self._stock.info

            if not info or info.get("regularMarketPrice") is None:
                # Try to get historical data as fallback
                hist = self._stock.history(period="5d")
                if hist.empty:
                    raise ValueError(f"No data found for ticker: {self.ticker}")

            self._stock_info = info
            self._company_name = info.get("longName", self.ticker)

            # Load historical prices
            hist = self._stock.history(period="1y")
            if len(hist) < self.config.min_historical_days:
                raise ValueError(
                    f"Insufficient historical data: {len(hist)} days "
                    f"(minimum {self.config.min_historical_days} required)"
                )

            self._historical_prices = hist["Close"]
            logger.info(f"Loaded {len(hist)} days of data for {self._company_name}")

        except Exception as e:
            logger.error(f"Failed to load data for {self.ticker}: {e}")
            raise ValueError(f"Failed to load data for {self.ticker}: {e}")

    @property
    def current_price(self) -> float:
        """Get current stock price."""
        if self._historical_prices is not None:
            return float(self._historical_prices.iloc[-1])
        return 0.0

    @property
    def company_name(self) -> str:
        """Get company name."""
        return self._company_name or self.ticker

    @property
    def historical_prices(self) -> pd.Series:
        """Get historical prices."""
        return self._historical_prices

    def run_monte_carlo(
        self,
        days_ahead: Optional[int] = None,
        num_simulations: Optional[int] = None,
        return_paths: bool = True
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation.

        Args:
            days_ahead: Forecast horizon in days
            num_simulations: Number of simulation paths
            return_paths: Whether to return full paths (needed for visualization)

        Returns:
            MonteCarloResults with simulation statistics
        """
        logger.info(f"Running Monte Carlo simulation for {self.ticker}")

        days = days_ahead or self.config.monte_carlo.days_ahead
        sims = num_simulations or self.config.monte_carlo.simulations

        self._monte_carlo_results = self._monte_carlo.run_simulation(
            prices=self._historical_prices,
            days_ahead=days,
            num_simulations=sims,
            return_paths=return_paths
        )

        logger.info(
            f"Monte Carlo complete: P(up)={self._monte_carlo_results.probability_up:.2%}"
        )

        return self._monte_carlo_results

    def run_time_series(
        self,
        forecast_days: Optional[int] = None,
        include_garch: bool = True
    ) -> TimeSeriesResults:
        """
        Run ARIMA and GARCH analysis.

        Args:
            forecast_days: Forecast horizon in days
            include_garch: Whether to include GARCH volatility modeling

        Returns:
            TimeSeriesResults with forecasts
        """
        logger.info(f"Running time series analysis for {self.ticker}")

        days = forecast_days or self.config.default_forecast_days

        self._time_series_results = self._time_series.analyze(
            prices=self._historical_prices,
            forecast_days=days,
            include_garch=include_garch
        )

        logger.info(
            f"Time series complete: ARIMA order={self._time_series_results.arima.model_order}"
        )

        return self._time_series_results

    def run_sentiment_analysis(
        self,
        lookback_days: Optional[int] = None
    ) -> SentimentResults:
        """
        Run news sentiment analysis.

        Args:
            lookback_days: Days of news to analyze

        Returns:
            SentimentResults with sentiment scores
        """
        logger.info(f"Running sentiment analysis for {self.ticker}")

        days = lookback_days or self.config.sentiment.lookback_days

        self._sentiment_results = self._sentiment.analyze(
            ticker=self.ticker,
            company_name=self._company_name,
            lookback_days=days
        )

        logger.info(
            f"Sentiment analysis complete: {self._sentiment_results.news_analyzed} articles, "
            f"sentiment={self._sentiment_results.sentiment_global:.2f}"
        )

        return self._sentiment_results

    def run_full_analysis(
        self,
        days_ahead: int = 14,
        monte_carlo_sims: int = 10000,
        news_lookback: int = 30,
        include_garch: bool = True
    ) -> PredictionResult:
        """
        Run complete analysis with all models.

        Args:
            days_ahead: Forecast horizon in days
            monte_carlo_sims: Number of Monte Carlo simulations
            news_lookback: Days of news to analyze
            include_garch: Whether to include GARCH modeling

        Returns:
            PredictionResult with aggregated prediction
        """
        start_time = datetime.now()
        logger.info(f"Starting full analysis for {self.ticker}")

        errors = []

        # Run Monte Carlo
        try:
            self.run_monte_carlo(
                days_ahead=days_ahead,
                num_simulations=monte_carlo_sims,
                return_paths=True
            )
        except Exception as e:
            logger.error(f"Monte Carlo failed: {e}")
            errors.append(f"Monte Carlo: {e}")

        # Run Time Series
        try:
            self.run_time_series(
                forecast_days=days_ahead,
                include_garch=include_garch
            )
        except Exception as e:
            logger.error(f"Time series failed: {e}")
            errors.append(f"Time series: {e}")

        # Run Sentiment
        try:
            self.run_sentiment_analysis(lookback_days=news_lookback)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            errors.append(f"Sentiment: {e}")

        # Aggregate results
        self._prediction_results = self._probability_engine.aggregate(
            ticker=self.ticker,
            current_price=self.current_price,
            horizon_days=days_ahead,
            monte_carlo=self._monte_carlo_results,
            time_series=self._time_series_results,
            sentiment=self._sentiment_results
        )

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Full analysis complete in {execution_time:.1f}s: "
            f"Signal={self._prediction_results.signal}, "
            f"P(up)={self._prediction_results.probability_hausse:.2%}"
        )

        if errors:
            logger.warning(f"Analysis completed with errors: {errors}")

        return self._prediction_results

    def get_dashboard_figures(self) -> Dict[str, Any]:
        """
        Generate all dashboard figures.

        Returns:
            Dictionary of figure name to Plotly figure
        """
        if self._prediction_results is None:
            raise ValueError("Run analysis first with run_full_analysis()")

        self._dashboard_figures = create_prediction_dashboard(
            prediction=self._prediction_results,
            monte_carlo=self._monte_carlo_results,
            time_series=self._time_series_results,
            sentiment=self._sentiment_results,
            historical_prices=self._historical_prices,
            config=self.config.visualization
        )

        return self._dashboard_figures

    def display_dashboard(self) -> None:
        """Display interactive dashboard (for Jupyter/Streamlit)."""
        figures = self.get_dashboard_figures()

        for name, fig in figures.items():
            fig.show()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of prediction results.

        Returns:
            Dictionary with key metrics
        """
        if self._prediction_results is None:
            raise ValueError("Run analysis first with run_full_analysis()")

        return {
            "ticker": self.ticker,
            "company": self._company_name,
            "current_price": round(self.current_price, 2),
            "signal": self._prediction_results.signal,
            "probability_up": round(self._prediction_results.probability_hausse, 4),
            "confidence_interval": self._prediction_results.confidence_interval,
            "risk_level": self._prediction_results.risk_assessment.risk_level,
            "target_price": self._prediction_results.target_price_mean,
            "recommendation": self._prediction_results.recommendation,
            "models": {
                "monte_carlo": self._monte_carlo_results is not None,
                "arima": self._time_series_results is not None,
                "garch": (
                    self._time_series_results.garch is not None
                    if self._time_series_results else False
                ),
                "sentiment": self._sentiment_results is not None
            }
        }

    def export_report(
        self,
        filepath: str,
        format: str = "json"
    ) -> str:
        """
        Export analysis report to file.

        Args:
            filepath: Output file path
            format: Output format ("json" or "html")

        Returns:
            Path to exported file
        """
        if self._prediction_results is None:
            raise ValueError("Run analysis first with run_full_analysis()")

        if format == "json":
            report = {
                "ticker": self.ticker,
                "company": self._company_name,
                "analysis_date": datetime.now().isoformat(),
                "prediction": self._prediction_results.to_dict(),
                "monte_carlo": (
                    self._monte_carlo_results.to_dict()
                    if self._monte_carlo_results else None
                ),
                "time_series": (
                    self._time_series_results.to_dict()
                    if self._time_series_results else None
                ),
                "sentiment": (
                    self._sentiment_results.to_dict()
                    if self._sentiment_results else None
                ),
                "disclaimers": DISCLAIMERS
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        elif format == "html":
            # Generate HTML report with embedded charts
            figures = self.get_dashboard_figures()
            html_parts = [
                f"<html><head><title>Rapport {self.ticker}</title></head><body>",
                f"<h1>Analyse de {self._company_name} ({self.ticker})</h1>",
                f"<p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
                f"<h2>Signal: {self._prediction_results.signal}</h2>",
                f"<p>Probabilité de hausse: {self._prediction_results.probability_hausse:.1%}</p>"
            ]

            for name, fig in figures.items():
                html_parts.append(f"<h3>{name}</h3>")
                html_parts.append(fig.to_html(include_plotlyjs='cdn', full_html=False))

            html_parts.append("<h3>Avertissements</h3><ul>")
            for d in DISCLAIMERS:
                html_parts.append(f"<li>{d}</li>")
            html_parts.append("</ul></body></html>")

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Report exported to {filepath}")
        return filepath

    def __repr__(self) -> str:
        """String representation."""
        status = "analyzed" if self._prediction_results else "not analyzed"
        return f"StockPredictor(ticker='{self.ticker}', status='{status}')"


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def quick_predict(
    ticker: str,
    days_ahead: int = 14
) -> Dict[str, Any]:
    """
    Quick prediction for a single ticker.

    Args:
        ticker: Stock ticker symbol
        days_ahead: Forecast horizon

    Returns:
        Dictionary with prediction summary
    """
    predictor = StockPredictor(ticker)
    predictor.run_full_analysis(days_ahead=days_ahead)
    return predictor.get_summary()


def compare_tickers(
    tickers: List[str],
    days_ahead: int = 14
) -> pd.DataFrame:
    """
    Compare predictions for multiple tickers.

    Args:
        tickers: List of ticker symbols
        days_ahead: Forecast horizon

    Returns:
        DataFrame with comparison
    """
    results = []

    for ticker in tickers:
        try:
            predictor = StockPredictor(ticker)
            predictor.run_full_analysis(days_ahead=days_ahead)
            summary = predictor.get_summary()
            results.append(summary)
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "error": str(e)
            })

    return pd.DataFrame(results)
