"""
Configuration centralisée pour Stock Analyzer Pro.
Gestion des variables d'environnement, logging et constantes.

Version: 1.0
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum

# Chargement des variables d'environnement avec python-dotenv
try:
    from dotenv import load_dotenv
    # Charger .env depuis le répertoire du projet
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging(
    name: str = "stock_analyzer",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure et retourne un logger.

    Args:
        name: Nom du logger
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin vers un fichier de log optionnel

    Returns:
        Logger configuré

    Example:
        >>> logger = setup_logging("my_module", level="DEBUG")
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)

    # Éviter les duplications de handlers
    if logger.handlers:
        return logger

    log_level = getattr(logging, level or LOG_LEVEL, logging.INFO)
    logger.setLevel(log_level)

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Handler fichier optionnel
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


# Logger principal
logger = setup_logging()


# ============================================================================
# API KEYS CONFIGURATION
# ============================================================================

@dataclass
class APIConfig:
    """Configuration des clés API."""

    # Mistral AI
    mistral_api_key: str = field(
        default_factory=lambda: os.environ.get("MISTRAL_API_KEY", "")
    )
    mistral_model: str = field(
        default_factory=lambda: os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    )

    # Ollama (local)
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", "mistral:latest")
    )

    # ESG APIs
    finnhub_api_key: str = field(
        default_factory=lambda: os.environ.get("FINNHUB_API_KEY", "")
    )
    fmp_api_key: str = field(
        default_factory=lambda: os.environ.get("FMP_API_KEY", "")
    )
    newsapi_key: str = field(
        default_factory=lambda: os.environ.get("NEWSAPI_KEY", "")
    )

    # Alpha Vantage (pour taux sans risque)
    alpha_vantage_key: str = field(
        default_factory=lambda: os.environ.get("ALPHA_VANTAGE_KEY", "")
    )

    def has_mistral(self) -> bool:
        """Vérifie si la clé Mistral est configurée."""
        return bool(self.mistral_api_key)

    def has_finnhub(self) -> bool:
        """Vérifie si la clé Finnhub est configurée."""
        return bool(self.finnhub_api_key)

    def has_fmp(self) -> bool:
        """Vérifie si la clé FMP est configurée."""
        return bool(self.fmp_api_key)


# Instance globale
API_CONFIG = APIConfig()


# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

@dataclass
class AppConfig:
    """Configuration de l'application."""

    # Cache
    cache_ttl_default: int = 3600  # 1 heure
    cache_ttl_prices: int = 60  # 1 minute pour les prix
    cache_ttl_financials: int = 86400  # 24 heures pour données fondamentales
    cache_ttl_esg: int = 7200  # 2 heures pour ESG

    # API Timeouts
    api_timeout_default: int = 15
    api_timeout_extended: int = 30
    api_max_retries: int = 3
    api_retry_delay: float = 1.0

    # DCF Defaults
    dcf_risk_free_rate: float = 0.04  # 4% par défaut (Treasury 10Y)
    dcf_market_premium: float = 0.05  # 5% prime de risque marché
    dcf_terminal_growth: float = 0.025  # 2.5% croissance terminale
    dcf_projection_years: int = 5

    # UI
    max_companies_compare: int = 4
    chart_height_default: int = 500
    dark_theme: bool = True


APP_CONFIG = AppConfig()


# ============================================================================
# FINANCIAL CONSTANTS
# ============================================================================

class MarketIndex(Enum):
    """Indices de marché pour les benchmarks."""
    SP500 = "^GSPC"
    NASDAQ = "^IXIC"
    DOW_JONES = "^DJI"
    CAC40 = "^FCHI"
    DAX = "^GDAXI"
    FTSE100 = "^FTSE"
    NIKKEI = "^N225"


# Secteurs controversés ESG
CONTROVERSIAL_SECTORS = {
    'coal': 'Charbon',
    'tobacco': 'Tabac',
    'controversial_weapons': 'Armes controversées',
    'firearms': 'Armes à feu',
    'gambling': 'Jeux d\'argent',
    'nuclear': 'Nucléaire',
    'palm_oil': 'Huile de palme',
    'pesticides': 'Pesticides',
    'adult_entertainment': 'Divertissement adulte',
    'military_contract': 'Contrats militaires',
}

# Benchmarks ESG par secteur
ESG_SECTOR_BENCHMARKS: Dict[str, Dict[str, float]] = {
    'Technology': {'avg_esg': 18, 'low': 10, 'high': 28},
    'Energy': {'avg_esg': 35, 'low': 20, 'high': 50},
    'Financial Services': {'avg_esg': 22, 'low': 12, 'high': 32},
    'Healthcare': {'avg_esg': 20, 'low': 12, 'high': 30},
    'Consumer Cyclical': {'avg_esg': 24, 'low': 14, 'high': 35},
    'Consumer Defensive': {'avg_esg': 22, 'low': 12, 'high': 32},
    'Industrials': {'avg_esg': 26, 'low': 15, 'high': 38},
    'Basic Materials': {'avg_esg': 32, 'low': 18, 'high': 45},
    'Utilities': {'avg_esg': 28, 'low': 15, 'high': 42},
    'Real Estate': {'avg_esg': 24, 'low': 14, 'high': 35},
    'Communication Services': {'avg_esg': 20, 'low': 12, 'high': 30},
}


# ============================================================================
# CHART COLORS
# ============================================================================

# Couleurs pour comparaison multi-entreprises
COMPANY_COLORS = [
    '#00D4AA',  # Turquoise
    '#FF6B6B',  # Rouge corail
    '#4ECDC4',  # Cyan
    '#FFE66D',  # Jaune
    '#95E1D3',  # Vert menthe
    '#A66CFF',  # Violet
    '#FF9F45',  # Orange
    '#6BCB77',  # Vert
]

# Couleurs pour niveaux de risque
RISK_COLORS = {
    'MINIMAL': '#27ae60',
    'LOW': '#2ecc71',
    'MODERATE': '#f1c40f',
    'MEDIUM': '#f1c40f',
    'HIGH': '#e67e22',
    'CRITICAL': '#e74c3c',
    'SEVERE': '#c0392b',
}

# Couleurs pour scores ESG
ESG_COLORS = {
    'excellent': '#27ae60',
    'good': '#2ecc71',
    'average': '#f1c40f',
    'poor': '#e67e22',
    'critical': '#e74c3c',
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_ticker(ticker: str) -> bool:
    """
    Valide le format d'un ticker.

    Args:
        ticker: Symbole boursier à valider

    Returns:
        True si le format est valide

    Example:
        >>> validate_ticker("AAPL")
        True
        >>> validate_ticker("")
        False
    """
    if not ticker:
        return False
    # Accepte lettres, chiffres, points et tirets (pour marchés européens)
    import re
    pattern = r'^[A-Za-z0-9\.\-]{1,10}$'
    return bool(re.match(pattern, ticker))


def validate_api_key(key: str, provider: str = "generic") -> bool:
    """
    Valide le format d'une clé API.

    Args:
        key: Clé API à valider
        provider: Fournisseur (mistral, finnhub, etc.)

    Returns:
        True si le format semble valide
    """
    if not key:
        return False

    # Vérifications basiques par provider
    if provider == "mistral":
        return len(key) >= 20 and not key.startswith("sk-")  # Mistral n'utilise pas sk-
    elif provider == "finnhub":
        return len(key) >= 10

    return len(key) >= 8


# ============================================================================
# ENVIRONMENT INFO
# ============================================================================

def get_environment_info() -> Dict[str, Any]:
    """
    Retourne les informations sur l'environnement.

    Returns:
        Dictionnaire avec les infos d'environnement
    """
    return {
        'dotenv_available': DOTENV_AVAILABLE,
        'log_level': LOG_LEVEL,
        'mistral_configured': API_CONFIG.has_mistral(),
        'finnhub_configured': API_CONFIG.has_finnhub(),
        'fmp_configured': API_CONFIG.has_fmp(),
        'cache_ttl': APP_CONFIG.cache_ttl_default,
    }


# ============================================================================
# MODULE INFO
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Stock Analyzer Pro - Configuration")
    print("=" * 60)

    env_info = get_environment_info()
    print(f"\npython-dotenv disponible: {env_info['dotenv_available']}")
    print(f"Niveau de log: {env_info['log_level']}")
    print(f"\nAPIs configurées:")
    print(f"  - Mistral AI: {'✓' if env_info['mistral_configured'] else '✗'}")
    print(f"  - Finnhub: {'✓' if env_info['finnhub_configured'] else '✗'}")
    print(f"  - FMP: {'✓' if env_info['fmp_configured'] else '✗'}")

    print("\n✓ Configuration chargée avec succès")
