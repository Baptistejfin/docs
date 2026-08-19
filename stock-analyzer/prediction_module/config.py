"""
Configuration for Stock Prediction Module
==========================================
Centralized configuration for Monte Carlo, ARIMA, GARCH, and Sentiment analysis.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger

# ==============================================================================
# ENUMS
# ==============================================================================

class VolatilityRegime(Enum):
    """Volatility regime classification."""
    LOW = "BASSE"
    MEDIUM = "MOYENNE"
    HIGH = "HAUTE"
    EXTREME = "EXTREME"

class Signal(Enum):
    """Trading signal classification."""
    STRONG_BUY = "ACHAT FORT"
    BUY = "ACHAT"
    MODERATE_BUY = "ACHAT MODÉRÉ"
    NEUTRAL = "NEUTRE"
    MODERATE_SELL = "VENTE MODÉRÉE"
    SELL = "VENTE"
    STRONG_SELL = "VENTE FORTE"

class SentimentModel(Enum):
    """Available sentiment analysis models."""
    FINBERT = "finbert"
    VADER = "vader"
    TEXTBLOB = "textblob"

class GARCHModel(Enum):
    """Available GARCH model variants."""
    GARCH = "GARCH"
    EGARCH = "EGARCH"
    GJR_GARCH = "GJR-GARCH"

class InformationCriterion(Enum):
    """Model selection criteria."""
    AIC = "aic"
    BIC = "bic"

# ==============================================================================
# CONFIGURATION DATACLASSES
# ==============================================================================

@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation parameters."""
    simulations: int = 10000
    days_ahead: int = 30
    confidence_levels: List[float] = field(default_factory=lambda: [0.50, 0.68, 0.95])
    use_multiprocessing: bool = True
    random_seed: Optional[int] = None
    trading_days_per_year: int = 252

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.simulations < 1000:
            raise ValueError("Minimum 1000 simulations required for statistical significance")
        if self.days_ahead < 1 or self.days_ahead > 365:
            raise ValueError("days_ahead must be between 1 and 365")
        if not all(0 < cl < 1 for cl in self.confidence_levels):
            raise ValueError("Confidence levels must be between 0 and 1")

@dataclass
class ARIMAConfig:
    """ARIMA model parameters."""
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    information_criterion: str = "aic"
    seasonal: bool = False
    seasonal_period: int = 5  # Weekly seasonality for daily data
    stepwise: bool = True
    suppress_warnings: bool = True
    max_iterations: int = 100

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_p < 0 or self.max_d < 0 or self.max_q < 0:
            raise ValueError("ARIMA orders must be non-negative")
        if self.information_criterion not in ["aic", "bic"]:
            raise ValueError("information_criterion must be 'aic' or 'bic'")

@dataclass
class GARCHConfig:
    """GARCH model parameters."""
    model_type: str = "GARCH"
    p: int = 1
    q: int = 1
    distribution: str = "normal"  # or "t", "skewt", "ged"
    mean_model: str = "Constant"  # or "Zero", "AR", "ARX"
    vol_targeting: bool = False

    # Volatility regime thresholds (annualized)
    volatility_low_threshold: float = 0.15
    volatility_medium_threshold: float = 0.25
    volatility_high_threshold: float = 0.40

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.p < 1 or self.q < 1:
            raise ValueError("GARCH p and q must be at least 1")
        if self.model_type not in ["GARCH", "EGARCH", "GJR-GARCH"]:
            raise ValueError("Invalid GARCH model type")

@dataclass
class SentimentConfig:
    """Sentiment analysis parameters."""
    # API Keys (from environment variables)
    newsapi_key: str = field(default_factory=lambda: os.environ.get("NEWSAPI_KEY", ""))
    finnhub_key: str = field(default_factory=lambda: os.environ.get("FINNHUB_API_KEY", ""))

    # Analysis parameters
    model: str = "finbert"
    max_articles: int = 100
    lookback_days: int = 30
    min_articles_required: int = 5

    # Weighting for time decay
    weight_24h: float = 0.40
    weight_7d: float = 0.30
    weight_30d: float = 0.20
    weight_sector: float = 0.10

    # FinBERT model path
    finbert_model: str = "ProsusAI/finbert"

    # Rate limiting
    api_delay_seconds: float = 0.5
    max_retries: int = 3

    def validate(self) -> None:
        """Validate configuration parameters."""
        weights_sum = self.weight_24h + self.weight_7d + self.weight_30d + self.weight_sector
        if abs(weights_sum - 1.0) > 0.01:
            raise ValueError(f"Sentiment weights must sum to 1.0, got {weights_sum}")
        if self.model not in ["finbert", "vader", "textblob"]:
            raise ValueError("Invalid sentiment model")

@dataclass
class AggregationWeights:
    """Weights for probability aggregation."""
    monte_carlo: float = 0.25
    arima: float = 0.20
    garch: float = 0.15
    sentiment: float = 0.40

    def validate(self) -> None:
        """Validate weights sum to 1."""
        total = self.monte_carlo + self.arima + self.garch + self.sentiment
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Aggregation weights must sum to 1.0, got {total}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "monte_carlo": self.monte_carlo,
            "arima": self.arima,
            "garch": self.garch,
            "sentiment": self.sentiment
        }

@dataclass
class VisualizationConfig:
    """Visualization parameters."""
    theme: str = "plotly_white"
    color_positive: str = "#00C853"
    color_negative: str = "#FF1744"
    color_neutral: str = "#2196F3"
    color_confidence_bands: List[str] = field(
        default_factory=lambda: ["rgba(0, 200, 83, 0.3)", "rgba(0, 200, 83, 0.2)", "rgba(0, 200, 83, 0.1)"]
    )
    figure_width: int = 1200
    figure_height: int = 600
    font_family: str = "Arial, sans-serif"

@dataclass
class CacheConfig:
    """Caching parameters."""
    enabled: bool = True
    price_data_ttl_seconds: int = 3600  # 1 hour
    news_data_ttl_seconds: int = 1800   # 30 minutes
    model_results_ttl_seconds: int = 300  # 5 minutes
    cache_directory: str = ".prediction_cache"

# ==============================================================================
# MAIN CONFIGURATION CLASS
# ==============================================================================

@dataclass
class PredictionConfig:
    """Main configuration container for all prediction modules."""
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    weights: AggregationWeights = field(default_factory=AggregationWeights)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # General settings
    min_historical_days: int = 100
    default_forecast_days: int = 14
    max_forecast_days: int = 90
    execution_timeout_seconds: int = 30

    def validate_all(self) -> None:
        """Validate all configuration sections."""
        self.monte_carlo.validate()
        self.arima.validate()
        self.garch.validate()
        self.sentiment.validate()
        self.weights.validate()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PredictionConfig":
        """Create configuration from dictionary."""
        config = cls()

        if "monte_carlo" in config_dict:
            config.monte_carlo = MonteCarloConfig(**config_dict["monte_carlo"])
        if "arima" in config_dict:
            config.arima = ARIMAConfig(**config_dict["arima"])
        if "garch" in config_dict:
            config.garch = GARCHConfig(**config_dict["garch"])
        if "sentiment" in config_dict:
            config.sentiment = SentimentConfig(**config_dict["sentiment"])
        if "weights" in config_dict:
            config.weights = AggregationWeights(**config_dict["weights"])

        return config

# ==============================================================================
# DEFAULT CONFIGURATION INSTANCE
# ==============================================================================

DEFAULT_CONFIG = PredictionConfig()

# ==============================================================================
# LEGAL DISCLAIMERS
# ==============================================================================

DISCLAIMERS = [
    "⚠️ Aucune garantie de performance future",
    "📊 Les résultats passés ne préjugent pas des résultats futurs",
    "💼 Consultez un conseiller financier agréé avant tout investissement",
    "🎲 Les modèles prédictifs comportent une incertitude inhérente",
    "📉 Risque de perte en capital",
    "🔬 Cet outil est à usage éducatif et informatif uniquement"
]

RISK_WARNINGS = {
    "high_volatility": "⚠️ Volatilité élevée détectée - Risque accru",
    "low_confidence": "⚠️ Faible confiance dans les prévisions - Données insuffisantes",
    "contradictory_signals": "⚠️ Signaux contradictoires entre les modèles",
    "extreme_market": "⚠️ Conditions de marché extrêmes détectées",
    "limited_news": "⚠️ Peu d'actualités disponibles - Sentiment moins fiable",
    "model_convergence": "⚠️ Problème de convergence du modèle"
}
