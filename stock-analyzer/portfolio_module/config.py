"""
Portfolio Module Configuration
==============================
Configuration for portfolio tracking and Trade Republic integration.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# ==============================================================================
# LOGGING
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

class AlertType(Enum):
    """Types of portfolio alerts."""
    PRICE_DROP = "price_drop"
    PRICE_SURGE = "price_surge"
    LOSS_THRESHOLD = "loss_threshold"
    GAIN_THRESHOLD = "gain_threshold"
    VOLATILITY = "volatility"
    REBALANCE = "rebalance"

class RecommendationType(Enum):
    """Types of investment recommendations."""
    BUY = "ACHETER"
    SELL = "VENDRE"
    HOLD = "CONSERVER"
    REDUCE = "RÉDUIRE"
    INCREASE = "RENFORCER"
    WATCH = "SURVEILLER"

class PerformanceMetric(Enum):
    """Portfolio performance metrics."""
    TOTAL_RETURN = "total_return"
    DAILY_RETURN = "daily_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"

# ==============================================================================
# CONFIGURATION DATACLASSES
# ==============================================================================

@dataclass
class TradeRepublicConfig:
    """Trade Republic connection settings."""
    phone_number: str = ""
    pin: str = ""
    locale: str = "de"
    auto_refresh_interval: int = 300  # seconds
    use_cache: bool = True
    cache_duration: int = 60  # seconds

@dataclass
class AlertConfig:
    """Alert thresholds configuration."""
    # Loss alerts
    loss_threshold_warning: float = -5.0  # %
    loss_threshold_critical: float = -10.0  # %

    # Gain alerts
    gain_threshold_take_profit: float = 20.0  # %
    gain_threshold_notify: float = 10.0  # %

    # Daily change alerts
    daily_drop_threshold: float = -3.0  # %
    daily_surge_threshold: float = 5.0  # %

    # Position size alerts
    max_position_weight: float = 25.0  # % of portfolio
    min_diversification: int = 5  # minimum positions

@dataclass
class VisualizationConfig:
    """Chart and display configuration."""
    theme: str = "plotly_white"
    color_positive: str = "#00C853"
    color_negative: str = "#FF1744"
    color_neutral: str = "#2196F3"
    color_warning: str = "#FF9800"
    currency_symbol: str = "€"
    date_format: str = "%d/%m/%Y"
    number_format: str = "fr_FR"

@dataclass
class PortfolioConfig:
    """Main portfolio configuration."""
    trade_republic: TradeRepublicConfig = field(default_factory=TradeRepublicConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Data storage
    portfolio_file: str = "portfolio_data.json"
    history_file: str = "portfolio_history.json"

    # Performance calculation
    risk_free_rate: float = 0.03  # 3% annual
    benchmark_symbol: str = "^STOXX50E"  # Euro Stoxx 50

    # Refresh settings
    auto_refresh: bool = True
    refresh_interval: int = 300  # 5 minutes

# ==============================================================================
# DEFAULT INSTANCE
# ==============================================================================

DEFAULT_CONFIG = PortfolioConfig()

# ==============================================================================
# DISPLAY HELPERS
# ==============================================================================

def format_currency(value: float, symbol: str = "€") -> str:
    """Format value as currency."""
    if value >= 0:
        return f"{value:,.2f} {symbol}"
    return f"-{abs(value):,.2f} {symbol}"

def format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"

def get_color_for_value(value: float, config: VisualizationConfig = None) -> str:
    """Get color based on value (positive/negative)."""
    config = config or DEFAULT_CONFIG.visualization
    if value > 0:
        return config.color_positive
    elif value < 0:
        return config.color_negative
    return config.color_neutral

# ==============================================================================
# RECOMMENDATION MESSAGES
# ==============================================================================

RECOMMENDATION_MESSAGES = {
    RecommendationType.BUY: {
        "icon": "🟢",
        "title": "Opportunité d'achat",
        "color": "#00C853"
    },
    RecommendationType.SELL: {
        "icon": "🔴",
        "title": "Signal de vente",
        "color": "#FF1744"
    },
    RecommendationType.HOLD: {
        "icon": "🟡",
        "title": "Conserver la position",
        "color": "#FF9800"
    },
    RecommendationType.REDUCE: {
        "icon": "🟠",
        "title": "Réduire l'exposition",
        "color": "#FF5722"
    },
    RecommendationType.INCREASE: {
        "icon": "🔵",
        "title": "Renforcer la position",
        "color": "#2196F3"
    },
    RecommendationType.WATCH: {
        "icon": "👁️",
        "title": "À surveiller",
        "color": "#9E9E9E"
    }
}

ALERT_MESSAGES = {
    AlertType.PRICE_DROP: {
        "icon": "📉",
        "title": "Baisse significative",
        "severity": "warning"
    },
    AlertType.PRICE_SURGE: {
        "icon": "📈",
        "title": "Hausse significative",
        "severity": "info"
    },
    AlertType.LOSS_THRESHOLD: {
        "icon": "🔴",
        "title": "Seuil de perte atteint",
        "severity": "error"
    },
    AlertType.GAIN_THRESHOLD: {
        "icon": "🎯",
        "title": "Objectif de gain atteint",
        "severity": "success"
    },
    AlertType.VOLATILITY: {
        "icon": "⚡",
        "title": "Volatilité élevée",
        "severity": "warning"
    },
    AlertType.REBALANCE: {
        "icon": "⚖️",
        "title": "Rééquilibrage suggéré",
        "severity": "info"
    }
}
