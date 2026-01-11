"""
Tests unitaires pour les indicateurs techniques.
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '..')

from utils import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic,
)


class TestRSI:
    """Tests pour le calcul du RSI."""

    def test_rsi_basic(self, sample_prices):
        """Test calcul RSI basique."""
        result = calculate_rsi(sample_prices, period=14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_prices)
        # RSI doit être entre 0 et 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()

    def test_rsi_overbought(self):
        """RSI doit être > 70 pour des hausses continues."""
        # Prix en hausse constante
        prices = pd.Series([100 + i * 2 for i in range(30)])
        result = calculate_rsi(prices, period=14)
        # Les dernières valeurs devraient être élevées
        assert result.iloc[-1] > 70

    def test_rsi_oversold(self):
        """RSI doit être < 30 pour des baisses continues."""
        # Prix en baisse constante
        prices = pd.Series([100 - i * 2 for i in range(30)])
        result = calculate_rsi(prices, period=14)
        # Les dernières valeurs devraient être basses
        assert result.iloc[-1] < 30

    def test_rsi_neutral(self):
        """RSI autour de 50 pour des mouvements mixtes."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        result = calculate_rsi(prices, period=14)
        # La moyenne devrait être proche de 50
        assert 30 < result.iloc[-10:].mean() < 70

    def test_rsi_different_periods(self, sample_prices):
        """Test avec différentes périodes."""
        rsi_7 = calculate_rsi(sample_prices, period=7)
        rsi_14 = calculate_rsi(sample_prices, period=14)
        rsi_21 = calculate_rsi(sample_prices, period=21)

        # RSI plus court devrait être plus volatil
        # (écart-type plus élevé)
        assert rsi_7.std() >= rsi_14.std()


class TestMACD:
    """Tests pour le calcul du MACD."""

    def test_macd_basic(self, sample_prices):
        """Test calcul MACD basique."""
        macd, signal, hist = calculate_macd(sample_prices)

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(sample_prices)

    def test_macd_histogram_calculation(self, sample_prices):
        """L'histogramme doit être MACD - Signal."""
        macd, signal, hist = calculate_macd(sample_prices)

        # Vérifier que histogram = macd - signal
        expected_hist = macd - signal
        pd.testing.assert_series_equal(hist, expected_hist)

    def test_macd_uptrend(self):
        """MACD positif pour une tendance haussière."""
        # Tendance haussière forte
        prices = pd.Series([100 + i * 3 for i in range(50)])
        macd, signal, hist = calculate_macd(prices)

        # MACD devrait être positif vers la fin
        assert macd.iloc[-1] > 0

    def test_macd_downtrend(self):
        """MACD négatif pour une tendance baissière."""
        # Tendance baissière forte
        prices = pd.Series([200 - i * 3 for i in range(50)])
        macd, signal, hist = calculate_macd(prices)

        # MACD devrait être négatif vers la fin
        assert macd.iloc[-1] < 0

    def test_macd_custom_periods(self, sample_prices):
        """Test avec périodes personnalisées."""
        macd1, _, _ = calculate_macd(sample_prices, fast=12, slow=26, signal=9)
        macd2, _, _ = calculate_macd(sample_prices, fast=8, slow=17, signal=9)

        # Les résultats devraient être différents
        assert not macd1.equals(macd2)


class TestBollingerBands:
    """Tests pour les bandes de Bollinger."""

    def test_bollinger_basic(self, sample_prices):
        """Test calcul Bollinger basique."""
        upper, middle, lower = calculate_bollinger_bands(sample_prices)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_bollinger_band_order(self, sample_prices):
        """Upper > Middle > Lower partout."""
        upper, middle, lower = calculate_bollinger_bands(sample_prices)

        valid_idx = upper.dropna().index
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_bollinger_middle_is_sma(self, sample_prices):
        """Middle band doit être la SMA."""
        upper, middle, lower = calculate_bollinger_bands(sample_prices, period=20)

        expected_sma = sample_prices.rolling(window=20).mean()
        pd.testing.assert_series_equal(middle, expected_sma)

    def test_bollinger_band_width(self, sample_prices):
        """Test que les bandes s'élargissent avec num_std."""
        upper1, _, lower1 = calculate_bollinger_bands(sample_prices, num_std=1.0)
        upper2, _, lower2 = calculate_bollinger_bands(sample_prices, num_std=2.0)

        # Bandes à 2 std devraient être plus larges
        width1 = (upper1 - lower1).dropna()
        width2 = (upper2 - lower2).dropna()
        assert (width2 >= width1).all()

    def test_bollinger_price_containment(self, sample_prices):
        """La plupart des prix devraient être dans les bandes (±2σ)."""
        upper, _, lower = calculate_bollinger_bands(sample_prices, num_std=2.0)

        valid_idx = upper.dropna().index
        prices_in_range = sample_prices[valid_idx]
        upper_valid = upper[valid_idx]
        lower_valid = lower[valid_idx]

        in_bands = (prices_in_range <= upper_valid) & (prices_in_range >= lower_valid)
        # ~95% devraient être dans les bandes à 2σ
        assert in_bands.mean() > 0.90


class TestStochastic:
    """Tests pour l'oscillateur stochastique."""

    def test_stochastic_basic(self, sample_ohlcv):
        """Test calcul stochastique basique."""
        k, d = calculate_stochastic(
            sample_ohlcv['High'],
            sample_ohlcv['Low'],
            sample_ohlcv['Close']
        )

        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_stochastic_range(self, sample_ohlcv):
        """Stochastique doit être entre 0 et 100."""
        k, d = calculate_stochastic(
            sample_ohlcv['High'],
            sample_ohlcv['Low'],
            sample_ohlcv['Close']
        )

        valid_k = k.dropna()
        valid_d = d.dropna()

        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_stochastic_overbought(self):
        """Stochastique > 80 près des plus hauts."""
        # Prix près des plus hauts
        high = pd.Series([100, 102, 104, 106, 108] * 5)
        low = pd.Series([90, 92, 94, 96, 98] * 5)
        close = pd.Series([99, 101, 103, 105, 107] * 5)  # Près des hauts

        k, d = calculate_stochastic(high, low, close, k_period=5)
        assert k.iloc[-1] > 80

    def test_stochastic_oversold(self):
        """Stochastique < 20 près des plus bas."""
        # Prix près des plus bas
        high = pd.Series([100, 98, 96, 94, 92] * 5)
        low = pd.Series([90, 88, 86, 84, 82] * 5)
        close = pd.Series([91, 89, 87, 85, 83] * 5)  # Près des bas

        k, d = calculate_stochastic(high, low, close, k_period=5)
        assert k.iloc[-1] < 20

    def test_stochastic_d_is_smoothed_k(self, sample_ohlcv):
        """%D est la moyenne mobile de %K."""
        k, d = calculate_stochastic(
            sample_ohlcv['High'],
            sample_ohlcv['Low'],
            sample_ohlcv['Close'],
            k_period=14,
            d_period=3
        )

        expected_d = k.rolling(window=3).mean()
        pd.testing.assert_series_equal(d, expected_d)
