"""
Fixtures pytest pour Stock Analyzer Pro.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_prices():
    """Génère une série de prix pour les tests."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, 252)
    prices = initial_price * np.cumprod(1 + returns)
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture
def sample_returns(sample_prices):
    """Calcule les rendements à partir des prix."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_ohlcv():
    """Génère des données OHLCV pour les tests."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')

    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    open_price = close + np.random.randn(100) * 0.5
    volume = np.random.randint(1000000, 10000000, 100)

    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def sample_company_info():
    """Données d'entreprise fictives type yfinance."""
    return {
        'symbol': 'TEST',
        'shortName': 'Test Company Inc.',
        'longName': 'Test Company Incorporated',
        'sector': 'Technology',
        'industry': 'Software',
        'country': 'United States',
        'regularMarketPrice': 150.0,
        'marketCap': 500000000000,  # 500B
        'totalDebt': 100000000000,  # 100B
        'totalCash': 50000000000,   # 50B
        'sharesOutstanding': 3000000000,
        'beta': 1.2,
        'trailingPE': 25.0,
        'forwardPE': 22.0,
        'priceToBook': 8.5,
        'returnOnEquity': 0.35,
        'returnOnAssets': 0.15,
        'profitMargins': 0.25,
        'operatingMargins': 0.30,
        'grossMargins': 0.45,
        'debtToEquity': 0.8,
        'currentRatio': 1.5,
        'quickRatio': 1.2,
        'dividendYield': 0.015,
        'payoutRatio': 0.20,
        'totalRevenue': 400000000000,
        'revenueGrowth': 0.15,
        'netIncomeToCommon': 100000000000,
        'ebitda': 150000000000,
        'freeCashflow': 80000000000,
        'operatingCashflow': 120000000000,
        'interestExpense': 5000000000,
        'incomeTaxExpense': 25000000000,
        'incomeBeforeTax': 125000000000,
    }


@pytest.fixture
def sample_fcf_values():
    """Valeurs de Free Cash Flow projetées."""
    return [
        80000000000,   # Year 1
        88000000000,   # Year 2
        96800000000,   # Year 3
        106480000000,  # Year 4
        117128000000,  # Year 5
    ]


@pytest.fixture
def sample_esg_data():
    """Données ESG fictives."""
    return {
        'source': 'yfinance',
        'available': True,
        'ticker': 'TEST',
        'total_esg_score': 18.5,
        'environment_score': 15.0,
        'social_score': 20.0,
        'governance_score': 18.0,
        'controversy_level': 1,
        'risk_level': 'Low',
        'peer_group': 'Technology',
        'controversial_sectors': {
            'coal': False,
            'tobacco': False,
            'controversial_weapons': False,
        }
    }


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_yfinance_ticker(sample_company_info, sample_ohlcv):
    """Mock pour yfinance.Ticker."""
    with patch('yfinance.Ticker') as mock:
        ticker_instance = MagicMock()
        ticker_instance.info = sample_company_info
        ticker_instance.history.return_value = sample_ohlcv

        # Mock des données financières
        years = pd.date_range(end=datetime.now(), periods=5, freq='YE')
        financials = pd.DataFrame({
            year: {
                'Total Revenue': 350000000000 + i * 20000000000,
                'Net Income': 80000000000 + i * 5000000000,
                'Operating Income': 100000000000 + i * 6000000000,
                'Gross Profit': 150000000000 + i * 8000000000,
            } for i, year in enumerate(years)
        })
        ticker_instance.financials = financials
        ticker_instance.quarterly_financials = financials

        # Mock balance sheet
        balance_sheet = pd.DataFrame({
            year: {
                'Total Assets': 400000000000,
                'Total Liabilities': 150000000000,
                'Total Stockholder Equity': 250000000000,
                'Total Debt': 100000000000,
                'Cash And Cash Equivalents': 50000000000,
            } for i, year in enumerate(years)
        })
        ticker_instance.balance_sheet = balance_sheet
        ticker_instance.quarterly_balance_sheet = balance_sheet

        # Mock cashflow
        cashflow = pd.DataFrame({
            year: {
                'Operating Cash Flow': 100000000000 + i * 5000000000,
                'Free Cash Flow': 80000000000 + i * 4000000000,
                'Capital Expenditure': -20000000000,
            } for i, year in enumerate(years)
        })
        ticker_instance.cashflow = cashflow
        ticker_instance.quarterly_cashflow = cashflow

        # Mock sustainability
        sustainability = pd.DataFrame({
            'Value': {
                'totalEsg': 18.5,
                'environmentScore': 15.0,
                'socialScore': 20.0,
                'governanceScore': 18.0,
                'highestControversy': 1,
            }
        })
        ticker_instance.sustainability = sustainability

        mock.return_value = ticker_instance
        yield mock


@pytest.fixture
def mock_treasury_rate():
    """Mock pour le taux sans risque."""
    with patch('yfinance.Ticker') as mock:
        ticker_instance = MagicMock()
        hist = pd.DataFrame({
            'Close': [4.25, 4.30, 4.28, 4.32, 4.30]
        }, index=pd.date_range(end=datetime.now(), periods=5, freq='B'))
        ticker_instance.history.return_value = hist
        mock.return_value = ticker_instance
        yield mock


@pytest.fixture
def mock_requests():
    """Mock pour les requêtes HTTP."""
    with patch('requests.get') as mock:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {}
        mock.return_value = response
        yield mock


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def temp_cache():
    """Cache temporaire pour les tests."""
    from utils import Cache
    cache = Cache(default_ttl=60)
    yield cache
    cache.clear()


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset des caches entre les tests."""
    yield
    # Cleanup after each test if needed


# ============================================================================
# TOLERANCE VALUES
# ============================================================================

FLOAT_TOLERANCE = 1e-6
PERCENT_TOLERANCE = 0.001  # 0.1%
