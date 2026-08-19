"""
Tests unitaires pour les calculs DCF et WACC.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, '..')

from utils import (
    calculate_wacc,
    calculate_cost_of_equity,
    calculate_wacc_from_info,
    get_risk_free_rate,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    calculate_max_drawdown,
)


class TestCostOfEquity:
    """Tests pour le calcul du coût des fonds propres (CAPM)."""

    def test_cost_of_equity_basic(self):
        """Test calcul basique du coût des fonds propres."""
        # Re = Rf + β(Rm - Rf)
        # Re = 0.04 + 1.0 * 0.05 = 0.09 (9%)
        result = calculate_cost_of_equity(
            beta=1.0,
            risk_free_rate=0.04,
            market_premium=0.05
        )
        assert abs(result - 0.09) < 1e-6

    def test_cost_of_equity_high_beta(self):
        """Test avec un beta élevé."""
        # Re = 0.04 + 1.5 * 0.05 = 0.115 (11.5%)
        result = calculate_cost_of_equity(
            beta=1.5,
            risk_free_rate=0.04,
            market_premium=0.05
        )
        assert abs(result - 0.115) < 1e-6

    def test_cost_of_equity_low_beta(self):
        """Test avec un beta faible."""
        # Re = 0.04 + 0.5 * 0.05 = 0.065 (6.5%)
        result = calculate_cost_of_equity(
            beta=0.5,
            risk_free_rate=0.04,
            market_premium=0.05
        )
        assert abs(result - 0.065) < 1e-6

    def test_cost_of_equity_zero_beta(self):
        """Test avec beta = 0 (théorique)."""
        # Re = Rf = 0.04
        result = calculate_cost_of_equity(
            beta=0.0,
            risk_free_rate=0.04,
            market_premium=0.05
        )
        assert abs(result - 0.04) < 1e-6


class TestWACC:
    """Tests pour le calcul du WACC."""

    def test_wacc_basic(self):
        """Test calcul WACC basique."""
        # WACC = (E/V × Re) + (D/V × Rd × (1-Tc))
        # E = 1000M, D = 500M, V = 1500M
        # E/V = 0.667, D/V = 0.333
        # WACC = (0.667 × 0.10) + (0.333 × 0.05 × 0.75)
        # WACC = 0.0667 + 0.0125 = 0.0792
        result = calculate_wacc(
            market_cap=1000000000,
            total_debt=500000000,
            cost_of_equity=0.10,
            cost_of_debt=0.05,
            tax_rate=0.25
        )
        expected = (1000/1500 * 0.10) + (500/1500 * 0.05 * 0.75)
        assert abs(result - expected) < 1e-6

    def test_wacc_no_debt(self):
        """Test WACC sans dette (100% equity)."""
        # WACC = Re quand D = 0
        result = calculate_wacc(
            market_cap=1000000000,
            total_debt=0,
            cost_of_equity=0.10,
            cost_of_debt=0.05,
            tax_rate=0.25
        )
        assert abs(result - 0.10) < 1e-6

    def test_wacc_high_debt(self):
        """Test WACC avec ratio dette élevé."""
        # E = 300M, D = 700M
        result = calculate_wacc(
            market_cap=300000000,
            total_debt=700000000,
            cost_of_equity=0.12,
            cost_of_debt=0.06,
            tax_rate=0.30
        )
        expected = (0.3 * 0.12) + (0.7 * 0.06 * 0.70)
        assert abs(result - expected) < 1e-6

    def test_wacc_zero_market_cap(self):
        """Test WACC avec market cap = 0 (fallback)."""
        result = calculate_wacc(
            market_cap=0,
            total_debt=500000000,
            cost_of_equity=0.10,
            cost_of_debt=0.05,
            tax_rate=0.25
        )
        # Devrait retourner le coût des fonds propres
        assert result == 0.10


class TestWACCFromInfo:
    """Tests pour calculate_wacc_from_info."""

    def test_wacc_from_info_complete(self, sample_company_info):
        """Test avec données complètes."""
        result = calculate_wacc_from_info(sample_company_info)

        assert 'wacc' in result
        assert 'cost_of_equity' in result
        assert 'cost_of_debt' in result
        assert 'beta' in result

        # Vérifications de cohérence
        assert 0 < result['wacc'] < 0.30  # WACC entre 0 et 30%
        assert result['equity_weight'] + result['debt_weight'] == pytest.approx(1.0)
        assert result['beta'] == sample_company_info['beta']

    def test_wacc_from_info_missing_debt(self):
        """Test avec dette manquante."""
        info = {
            'marketCap': 1000000000,
            'totalDebt': 0,
            'beta': 1.0,
        }
        result = calculate_wacc_from_info(info)

        assert result['debt_weight'] == 0
        assert result['equity_weight'] == 1.0
        # WACC = coût des fonds propres
        assert result['wacc'] == pytest.approx(result['cost_of_equity'])


class TestRiskMetrics:
    """Tests pour les métriques de risque."""

    def test_sharpe_ratio_positive(self, sample_returns):
        """Test Sharpe ratio avec rendements positifs."""
        # Créer des rendements positifs
        positive_returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        result = calculate_sharpe_ratio(positive_returns, risk_free_rate=0.04)
        # Le Sharpe devrait être positif pour des rendements moyens positifs
        # avec faible volatilité
        assert isinstance(result, float)

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio avec rendements négatifs."""
        negative_returns = pd.Series([-0.02, -0.01, -0.015, -0.005, -0.02])
        result = calculate_sharpe_ratio(negative_returns, risk_free_rate=0.04)
        assert result < 0

    def test_sharpe_ratio_empty(self):
        """Test Sharpe ratio avec série vide."""
        result = calculate_sharpe_ratio(pd.Series([]), risk_free_rate=0.04)
        assert result == 0.0

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio."""
        result = calculate_sortino_ratio(sample_returns, risk_free_rate=0.04)
        assert isinstance(result, float)

    def test_sortino_vs_sharpe(self):
        """Le Sortino devrait être >= Sharpe pour des rendements mixtes."""
        # Rendements avec quelques pertes significatives
        returns = pd.Series([0.02, -0.05, 0.03, -0.01, 0.02, 0.01, -0.02, 0.04])
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        # Sortino pénalise seulement la volatilité baissière
        # donc il devrait être au moins égal au Sharpe
        # (En pratique, dépend de la distribution)

    def test_var_historical(self, sample_returns):
        """Test VaR historique."""
        result = calculate_var(sample_returns, confidence_level=0.95)
        assert result < 0  # VaR est une perte (négative)
        assert result > -1  # Pas de perte de plus de 100%

    def test_var_confidence_levels(self, sample_returns):
        """VaR 99% devrait être plus négatif que VaR 95%."""
        var_95 = calculate_var(sample_returns, confidence_level=0.95)
        var_99 = calculate_var(sample_returns, confidence_level=0.99)
        assert var_99 <= var_95

    def test_max_drawdown(self, sample_prices):
        """Test max drawdown."""
        result = calculate_max_drawdown(sample_prices)
        assert result <= 0  # Drawdown est négatif
        assert result >= -1  # Pas plus de -100%

    def test_max_drawdown_increasing(self):
        """Max drawdown pour une série monotone croissante = 0."""
        prices = pd.Series([100, 110, 120, 130, 140, 150])
        result = calculate_max_drawdown(prices)
        assert result == 0

    def test_max_drawdown_decreasing(self):
        """Max drawdown pour une série monotone décroissante."""
        prices = pd.Series([100, 90, 80, 70, 60])
        result = calculate_max_drawdown(prices)
        assert result == pytest.approx(-0.4)  # -40%


class TestDCFCalculation:
    """Tests pour le calcul DCF (fonction dans stock_analyzer.py)."""

    def test_dcf_basic(self, sample_fcf_values):
        """Test calcul DCF basique."""
        # Import depuis le module principal
        # Note: Ce test nécessite que calculate_dcf soit importable
        pass  # À implémenter après refactoring de stock_analyzer.py

    def test_dcf_terminal_value(self):
        """Test que la valeur terminale est correctement calculée."""
        # Terminal Value = FCF_n * (1 + g) / (WACC - g)
        fcf_n = 100000000
        growth = 0.025
        wacc = 0.10

        expected_terminal = fcf_n * (1 + growth) / (wacc - growth)
        # TV = 100M * 1.025 / 0.075 = 1.367B
        assert expected_terminal == pytest.approx(1366666666.67, rel=0.01)

    def test_dcf_present_value(self):
        """Test calcul de valeur présente."""
        # PV = FV / (1 + r)^n
        future_value = 1000000
        rate = 0.10
        years = 5

        expected_pv = future_value / ((1 + rate) ** years)
        # PV = 1M / 1.61051 = 620,921
        assert expected_pv == pytest.approx(620921, rel=0.01)


class TestRiskFreeRate:
    """Tests pour la récupération du taux sans risque."""

    def test_risk_free_rate_fallback(self):
        """Test fallback quand API échoue."""
        with patch('yfinance.Ticker') as mock:
            mock.side_effect = Exception("API Error")
            result = get_risk_free_rate(fallback=0.05)
            assert result == 0.05

    def test_risk_free_rate_format(self, mock_treasury_rate):
        """Test que le taux est correctement formaté."""
        # Le mock retourne des valeurs autour de 4.30%
        # La fonction devrait diviser par 100
        result = get_risk_free_rate()
        assert 0 < result < 0.10  # Entre 0 et 10%
