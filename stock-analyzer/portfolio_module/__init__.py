"""
Portfolio Tracking Module
=========================
Track and analyze your Trade Republic portfolio in real-time.

Features:
- Trade Republic integration (via PyTR or CSV import)
- Real-time price updates
- Global and individual performance metrics
- Declining position alerts
- Investment recommendations

Usage:
    from portfolio_module import TradeRepublicClient, PortfolioAnalyzer

    # Create portfolio manually
    client = TradeRepublicClient()
    portfolio = client.create_manual_portfolio([
        {"symbol": "AAPL", "quantity": 10, "average_buy_price": 150.00},
        {"symbol": "MSFT", "quantity": 5, "average_buy_price": 320.00},
    ])

    # Analyze
    analyzer = PortfolioAnalyzer()
    metrics = analyzer.calculate_performance(portfolio)
    declining = analyzer.get_declining_positions(portfolio, threshold=-5.0)

Author: Stock Analyzer Pro
Version: 1.0.0
"""

from .config import (
    PortfolioConfig,
    DEFAULT_CONFIG,
    AlertType,
    RecommendationType,
    format_currency,
    format_percent
)

from .trade_republic import (
    TradeRepublicClient,
    Portfolio,
    Position,
    create_sample_portfolio
)

from .analytics import (
    PortfolioAnalyzer,
    PerformanceMetrics,
    AllocationAnalysis,
    PositionAnalysis
)

from .alerts import (
    AlertManager,
    RecommendationEngine,
    Alert,
    InvestmentRecommendation,
    check_portfolio_alerts,
    get_investment_recommendations
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "PortfolioConfig",
    "DEFAULT_CONFIG",
    "AlertType",
    "RecommendationType",
    "format_currency",
    "format_percent",

    # Trade Republic
    "TradeRepublicClient",
    "Portfolio",
    "Position",
    "create_sample_portfolio",

    # Analytics
    "PortfolioAnalyzer",
    "PerformanceMetrics",
    "AllocationAnalysis",
    "PositionAnalysis",

    # Alerts
    "AlertManager",
    "RecommendationEngine",
    "Alert",
    "InvestmentRecommendation",
    "check_portfolio_alerts",
    "get_investment_recommendations"
]
