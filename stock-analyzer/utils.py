"""
Utilitaires partagés pour Stock Analyzer Pro.
Contient les fonctions et classes communes à tous les modules.

Version: 1.0
"""

import time
import hashlib
import functools
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import (
    Optional, Dict, Any, List, Tuple, Callable, TypeVar, Union
)
from dataclasses import dataclass, field

from config import setup_logging, APP_CONFIG

# Logger pour ce module
logger = setup_logging("utils")

# Type générique pour décorateurs
T = TypeVar('T')


# ============================================================================
# DECORATORS
# ============================================================================

def handle_errors(
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Callable:
    """
    Décorateur pour gérer les erreurs de manière uniforme.

    Args:
        default_return: Valeur à retourner en cas d'erreur
        log_errors: Si True, log l'erreur
        reraise: Si True, relance l'exception après logging

    Returns:
        Fonction décorée

    Example:
        >>> @handle_errors(default_return={})
        ... def fetch_data(url):
        ...     return requests.get(url).json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.Timeout as e:
                if log_errors:
                    logger.warning(f"{func.__name__}: Timeout - {e}")
                if reraise:
                    raise
                return default_return
            except requests.exceptions.ConnectionError as e:
                if log_errors:
                    logger.warning(f"{func.__name__}: Connection error - {e}")
                if reraise:
                    raise
                return default_return
            except requests.exceptions.RequestException as e:
                if log_errors:
                    logger.error(f"{func.__name__}: Request failed - {e}")
                if reraise:
                    raise
                return default_return
            except ValueError as e:
                if log_errors:
                    logger.error(f"{func.__name__}: Value error - {e}")
                if reraise:
                    raise
                return default_return
            except KeyError as e:
                if log_errors:
                    logger.warning(f"{func.__name__}: Missing key - {e}")
                if reraise:
                    raise
                return default_return
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__}: Unexpected error - {type(e).__name__}: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: Tuple = (requests.RequestException, ConnectionError)
) -> Callable:
    """
    Décorateur pour réessayer une fonction avec backoff exponentiel.

    Args:
        max_retries: Nombre maximum de tentatives
        base_delay: Délai initial en secondes
        exceptions: Tuple d'exceptions à capturer

    Returns:
        Fonction décorée

    Example:
        >>> @retry_with_backoff(max_retries=3)
        ... def call_api():
        ...     return requests.get("https://api.example.com").json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__}: Attempt {attempt + 1}/{max_retries} "
                            f"failed, retrying in {delay}s..."
                        )
                        time.sleep(delay)
            logger.error(f"{func.__name__}: All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# CACHE CLASS
# ============================================================================

class Cache:
    """
    Classe de cache générique avec TTL.

    Attributes:
        default_ttl: Durée de vie par défaut en secondes

    Example:
        >>> cache = Cache(default_ttl=3600)
        >>> cache.set("key", "value")
        >>> cache.get("key")
        'value'
    """

    def __init__(self, default_ttl: int = 3600):
        """
        Initialise le cache.

        Args:
            default_ttl: Durée de vie par défaut des entrées en secondes
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def _make_key(self, func_name: str, *args, **kwargs) -> str:
        """
        Génère une clé de cache unique.

        Args:
            func_name: Nom de la fonction
            *args: Arguments positionnels
            **kwargs: Arguments nommés

        Returns:
            Clé de cache hashée
        """
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.

        Args:
            key: Clé de cache

        Returns:
            Valeur cachée ou None si expirée/inexistante
        """
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Stocke une valeur dans le cache.

        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Durée de vie en secondes (utilise default_ttl si None)
        """
        expiry = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (value, expiry)

    def clear(self) -> None:
        """Vide le cache."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def remove(self, key: str) -> bool:
        """
        Supprime une entrée du cache.

        Args:
            key: Clé à supprimer

        Returns:
            True si la clé existait
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def size(self) -> int:
        """Retourne le nombre d'entrées dans le cache."""
        return len(self._cache)


# Alias pour compatibilité avec les modules existants
ESGCache = Cache


def cached(ttl: int = 3600, cache_instance: Optional[Cache] = None) -> Callable:
    """
    Décorateur pour mettre en cache les résultats d'une fonction.

    Args:
        ttl: Durée de vie en secondes
        cache_instance: Instance de cache à utiliser (crée une nouvelle si None)

    Returns:
        Fonction décorée

    Example:
        >>> @cached(ttl=3600)
        ... def expensive_calculation(x):
        ...     return x ** 2
    """
    _cache = cache_instance or Cache(ttl)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            key = _cache._make_key(func.__name__, *args, **kwargs)
            result = _cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            _cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator


# ============================================================================
# HTTP UTILITIES
# ============================================================================

@retry_with_backoff(max_retries=3)
def safe_request(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 15,
    method: str = "GET"
) -> Optional[requests.Response]:
    """
    Effectue une requête HTTP sécurisée avec retry.

    Args:
        url: URL à appeler
        params: Paramètres de requête
        headers: En-têtes HTTP
        timeout: Timeout en secondes
        method: Méthode HTTP (GET, POST)

    Returns:
        Objet Response ou None en cas d'échec

    Raises:
        requests.RequestException: Si toutes les tentatives échouent
    """
    default_headers = {
        'User-Agent': 'StockAnalyzerPro/1.0 (Research Purpose)'
    }
    if headers:
        default_headers.update(headers)

    if method.upper() == "GET":
        response = requests.get(
            url,
            params=params,
            headers=default_headers,
            timeout=timeout
        )
    elif method.upper() == "POST":
        response = requests.post(
            url,
            json=params,
            headers=default_headers,
            timeout=timeout
        )
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    response.raise_for_status()
    return response


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_value(value: Optional[float], currency: str = "$") -> str:
    """
    Formate une valeur monétaire avec notation abrégée.

    Args:
        value: Valeur numérique
        currency: Symbole monétaire

    Returns:
        Chaîne formatée (ex: "$1.5B")

    Example:
        >>> format_value(1500000000)
        '$1.50B'
        >>> format_value(None)
        'N/A'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1e12:
        return f"{sign}{currency}{abs_value/1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"{sign}{currency}{abs_value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"{sign}{currency}{abs_value/1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{sign}{currency}{abs_value/1e3:.1f}K"
    else:
        return f"{sign}{currency}{abs_value:.2f}"


def format_percent(
    value: Optional[float],
    decimals: int = 2,
    include_sign: bool = True
) -> str:
    """
    Formate une valeur en pourcentage.

    Args:
        value: Valeur décimale (0.15 = 15%)
        decimals: Nombre de décimales
        include_sign: Si True, ajoute + devant les valeurs positives

    Returns:
        Chaîne formatée

    Example:
        >>> format_percent(0.156, decimals=1)
        '+15.6%'
        >>> format_percent(-0.05)
        '-5.00%'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"

    percent = value * 100 if abs(value) < 1 else value
    sign = "+" if include_sign and percent > 0 else ""
    return f"{sign}{percent:.{decimals}f}%"


def format_ratio(value: Optional[float], decimals: int = 2) -> str:
    """
    Formate un ratio financier.

    Args:
        value: Valeur du ratio
        decimals: Nombre de décimales

    Returns:
        Chaîne formatée

    Example:
        >>> format_ratio(15.234)
        '15.23x'
        >>> format_ratio(None)
        'N/A'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}x"


def format_currency(
    value: Optional[float],
    currency: str = "$",
    decimals: int = 2
) -> str:
    """
    Formate une valeur monétaire simple.

    Args:
        value: Valeur numérique
        currency: Symbole monétaire
        decimals: Nombre de décimales

    Returns:
        Chaîne formatée

    Example:
        >>> format_currency(1234.567)
        '$1,234.57'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{currency}{value:,.{decimals}f}"


# ============================================================================
# DATA VALIDATION UTILITIES
# ============================================================================

def safe_get(
    data: Union[Dict, pd.DataFrame],
    key: str,
    default: Any = None
) -> Any:
    """
    Récupère une valeur de manière sécurisée.

    Args:
        data: Dictionnaire ou DataFrame
        key: Clé à récupérer
        default: Valeur par défaut si clé absente

    Returns:
        Valeur ou default
    """
    try:
        if isinstance(data, dict):
            return data.get(key, default)
        elif isinstance(data, pd.DataFrame):
            if key in data.columns:
                return data[key]
            elif key in data.index:
                return data.loc[key]
        return default
    except (KeyError, TypeError):
        return default


def is_valid_number(value: Any) -> bool:
    """
    Vérifie si une valeur est un nombre valide (non-NaN, non-None).

    Args:
        value: Valeur à vérifier

    Returns:
        True si c'est un nombre valide
    """
    if value is None:
        return False
    try:
        return not np.isnan(float(value))
    except (ValueError, TypeError):
        return False


def is_series_valid(series: Optional[pd.Series], min_values: int = 2) -> bool:
    """
    Vérifie si une Series pandas est valide pour les calculs.

    Args:
        series: Series à vérifier
        min_values: Nombre minimum de valeurs non-nulles requises

    Returns:
        True si la Series est valide
    """
    if series is None or not isinstance(series, pd.Series):
        return False
    valid_values = series.dropna()
    return len(valid_values) >= min_values


def get_valid_values(series: Optional[pd.Series]) -> List[float]:
    """
    Extrait les valeurs valides d'une Series.

    Args:
        series: Series pandas

    Returns:
        Liste des valeurs non-nulles
    """
    if series is None or not isinstance(series, pd.Series):
        return []
    return [v for v in series.dropna().tolist() if is_valid_number(v)]


# ============================================================================
# FINANCIAL UTILITIES
# ============================================================================

def calculate_cagr(
    start_value: float,
    end_value: float,
    years: int
) -> Optional[float]:
    """
    Calcule le taux de croissance annuel composé (CAGR).

    Args:
        start_value: Valeur initiale
        end_value: Valeur finale
        years: Nombre d'années

    Returns:
        CAGR en décimal (0.10 = 10%)

    Example:
        >>> calculate_cagr(100, 200, 7)
        0.10408...
    """
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
    trading_days: int = 252
) -> float:
    """
    Calcule la volatilité (écart-type des rendements).

    Args:
        returns: Series des rendements
        annualize: Si True, annualise la volatilité
        trading_days: Nombre de jours de trading par an

    Returns:
        Volatilité (annualisée si demandé)
    """
    std = returns.std()
    if annualize:
        return std * np.sqrt(trading_days)
    return std


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252
) -> float:
    """
    Calcule le ratio de Sharpe.

    Args:
        returns: Series des rendements quotidiens
        risk_free_rate: Taux sans risque annuel
        trading_days: Nombre de jours de trading par an

    Returns:
        Ratio de Sharpe annualisé

    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        >>> calculate_sharpe_ratio(returns)
        1.234...  # Exemple
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0

    return (mean_excess / std_excess) * np.sqrt(trading_days)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    trading_days: int = 252
) -> float:
    """
    Calcule le ratio de Sortino (pénalise uniquement la volatilité baissière).

    Args:
        returns: Series des rendements quotidiens
        risk_free_rate: Taux sans risque annuel
        trading_days: Nombre de jours de trading par an

    Returns:
        Ratio de Sortino annualisé
    """
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    mean_excess = excess_returns.mean()

    # Downside deviation (seulement les rendements négatifs)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.sqrt(np.mean(negative_returns ** 2))

    if downside_std == 0:
        return 0.0

    return (mean_excess / downside_std) * np.sqrt(trading_days)


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calcule la Value at Risk (VaR).

    Args:
        returns: Series des rendements
        confidence_level: Niveau de confiance (0.95 = 95%)
        method: Méthode ("historical" ou "parametric")

    Returns:
        VaR (valeur négative représentant la perte potentielle)

    Example:
        >>> returns = pd.Series(np.random.normal(0, 0.02, 252))
        >>> calculate_var(returns, 0.95)
        -0.033...  # Exemple
    """
    if len(returns) < 10:
        return 0.0

    if method == "historical":
        return np.percentile(returns, (1 - confidence_level) * 100)
    elif method == "parametric":
        from scipy import stats
        mean = returns.mean()
        std = returns.std()
        return stats.norm.ppf(1 - confidence_level, mean, std)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calcule le drawdown maximum.

    Args:
        prices: Series des prix

    Returns:
        Drawdown maximum (valeur négative)

    Example:
        >>> prices = pd.Series([100, 110, 105, 95, 100, 90])
        >>> calculate_max_drawdown(prices)
        -0.182...  # -18.2%
    """
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


# ============================================================================
# WACC CALCULATION (CAPM-BASED)
# ============================================================================

@cached(ttl=86400)  # Cache 24h pour le taux sans risque
def get_risk_free_rate(fallback: float = 0.04) -> float:
    """
    Récupère le taux sans risque actuel (Treasury 10Y).

    Args:
        fallback: Taux par défaut si échec de récupération

    Returns:
        Taux sans risque annuel

    Note:
        Utilise yfinance pour récupérer le yield du Treasury 10Y.
    """
    try:
        import yfinance as yf
        treasury = yf.Ticker("^TNX")  # 10-Year Treasury Yield
        hist = treasury.history(period="5d")
        if not hist.empty:
            # Le yield est en pourcentage, convertir en décimal
            rate = hist['Close'].iloc[-1] / 100
            logger.debug(f"Risk-free rate fetched: {rate:.4f}")
            return rate
    except Exception as e:
        logger.warning(f"Failed to fetch risk-free rate: {e}")

    return fallback


def calculate_cost_of_equity(
    beta: float,
    risk_free_rate: Optional[float] = None,
    market_premium: float = 0.05
) -> float:
    """
    Calcule le coût des fonds propres via CAPM.

    Formula: Re = Rf + β(Rm - Rf)

    Args:
        beta: Beta de l'action
        risk_free_rate: Taux sans risque (récupéré automatiquement si None)
        market_premium: Prime de risque du marché (Rm - Rf)

    Returns:
        Coût des fonds propres

    Example:
        >>> calculate_cost_of_equity(1.2, 0.04, 0.05)
        0.10  # 10%
    """
    rf = risk_free_rate if risk_free_rate is not None else get_risk_free_rate()
    return rf + beta * market_premium


def calculate_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float = 0.25
) -> float:
    """
    Calcule le WACC (Weighted Average Cost of Capital).

    Formula: WACC = (E/V × Re) + (D/V × Rd × (1-Tc))

    Args:
        market_cap: Capitalisation boursière (E)
        total_debt: Dette totale (D)
        cost_of_equity: Coût des fonds propres (Re)
        cost_of_debt: Coût de la dette (Rd)
        tax_rate: Taux d'imposition (Tc)

    Returns:
        WACC en décimal

    Example:
        >>> calculate_wacc(
        ...     market_cap=1000000000,
        ...     total_debt=500000000,
        ...     cost_of_equity=0.10,
        ...     cost_of_debt=0.05,
        ...     tax_rate=0.25
        ... )
        0.0833...  # 8.33%
    """
    if market_cap <= 0:
        logger.warning("Invalid market cap for WACC calculation")
        return cost_of_equity  # Fallback au coût des fonds propres

    total_value = market_cap + total_debt
    equity_weight = market_cap / total_value
    debt_weight = total_debt / total_value

    wacc = (equity_weight * cost_of_equity) + \
           (debt_weight * cost_of_debt * (1 - tax_rate))

    logger.debug(
        f"WACC calculated: {wacc:.4f} "
        f"(E/V={equity_weight:.2f}, D/V={debt_weight:.2f})"
    )

    return wacc


def calculate_wacc_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    """
    Calcule le WACC à partir des données yfinance.

    Args:
        info: Dictionnaire info de yfinance

    Returns:
        Dictionnaire avec WACC et composants

    Example:
        >>> import yfinance as yf
        >>> info = yf.Ticker("AAPL").info
        >>> result = calculate_wacc_from_info(info)
        >>> result['wacc']
        0.0912...
    """
    # Récupération des données
    beta = info.get('beta', 1.0) or 1.0
    market_cap = info.get('marketCap', 0) or 0
    total_debt = info.get('totalDebt', 0) or 0

    # Taux sans risque dynamique
    risk_free_rate = get_risk_free_rate()

    # Coût de la dette estimé (intérêts / dette totale, ou approximation)
    interest_expense = info.get('interestExpense', 0) or 0
    if total_debt > 0 and interest_expense > 0:
        cost_of_debt = abs(interest_expense) / total_debt
    else:
        # Estimation basée sur la notation crédit implicite
        cost_of_debt = risk_free_rate + 0.02  # Spread de 200 bps par défaut

    # Coût des fonds propres via CAPM
    cost_of_equity = calculate_cost_of_equity(
        beta=beta,
        risk_free_rate=risk_free_rate
    )

    # Taux d'imposition effectif
    income_tax = info.get('incomeTaxExpense', 0) or 0
    pretax_income = info.get('incomeBeforeTax', 0) or 0
    if pretax_income > 0 and income_tax > 0:
        effective_tax_rate = income_tax / pretax_income
    else:
        effective_tax_rate = 0.25  # Taux par défaut

    # Calcul WACC
    wacc = calculate_wacc(
        market_cap=market_cap,
        total_debt=total_debt,
        cost_of_equity=cost_of_equity,
        cost_of_debt=cost_of_debt,
        tax_rate=effective_tax_rate
    )

    return {
        'wacc': wacc,
        'cost_of_equity': cost_of_equity,
        'cost_of_debt': cost_of_debt,
        'risk_free_rate': risk_free_rate,
        'beta': beta,
        'effective_tax_rate': effective_tax_rate,
        'equity_weight': market_cap / (market_cap + total_debt) if market_cap > 0 else 1,
        'debt_weight': total_debt / (market_cap + total_debt) if market_cap > 0 else 0,
    }


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcule le Relative Strength Index (RSI).

    Args:
        prices: Series des prix de clôture
        period: Période de calcul

    Returns:
        Series du RSI

    Example:
        >>> prices = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83])
        >>> rsi = calculate_rsi(prices, period=5)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcule le MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series des prix de clôture
        fast: Période EMA rapide
        slow: Période EMA lente
        signal: Période ligne de signal

    Returns:
        Tuple (MACD line, Signal line, Histogram)

    Example:
        >>> macd, signal, hist = calculate_macd(prices)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcule les bandes de Bollinger.

    Args:
        prices: Series des prix de clôture
        period: Période de la moyenne mobile
        num_std: Nombre d'écarts-types

    Returns:
        Tuple (Upper band, Middle band, Lower band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcule l'oscillateur stochastique.

    Args:
        high: Series des plus hauts
        low: Series des plus bas
        close: Series des clôtures
        k_period: Période %K
        d_period: Période %D (signal)

    Returns:
        Tuple (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return k, d


# ============================================================================
# MODULE INFO
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Stock Analyzer Pro - Utilities Module")
    print("=" * 60)
    print("\nFonctions disponibles:")
    print("  Décorateurs:")
    print("    - @handle_errors")
    print("    - @retry_with_backoff")
    print("    - @cached")
    print("\n  Cache:")
    print("    - Cache class (alias: ESGCache)")
    print("\n  Formatage:")
    print("    - format_value, format_percent, format_ratio")
    print("\n  Finance:")
    print("    - calculate_wacc, calculate_cost_of_equity")
    print("    - calculate_sharpe_ratio, calculate_sortino_ratio")
    print("    - calculate_var, calculate_max_drawdown")
    print("\n  Technique:")
    print("    - calculate_rsi, calculate_macd")
    print("    - calculate_bollinger_bands, calculate_stochastic")
    print("\n✓ Module chargé avec succès")
