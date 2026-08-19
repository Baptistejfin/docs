"""
Portfolio Analytics Module
==========================
Performance calculations and portfolio analysis.

Features:
- Global and individual performance metrics
- Risk-adjusted returns (Sharpe, Sortino)
- Sector/asset allocation analysis
- Correlation analysis
- Benchmark comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yfinance as yf

from .trade_republic import Portfolio, Position
from .config import (
    PortfolioConfig,
    DEFAULT_CONFIG,
    PerformanceMetric,
    setup_logger,
    format_currency,
    format_percent
)

logger = setup_logger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_value: float
    total_invested: float
    total_profit_loss: float
    total_profit_loss_percent: float
    daily_change: float
    daily_change_percent: float

    # Risk metrics
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None

    # Time-based returns
    return_1d: Optional[float] = None
    return_1w: Optional[float] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_ytd: Optional[float] = None
    return_1y: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_value": round(self.total_value, 2),
            "total_invested": round(self.total_invested, 2),
            "total_profit_loss": round(self.total_profit_loss, 2),
            "total_profit_loss_percent": round(self.total_profit_loss_percent, 2),
            "daily_change": round(self.daily_change, 2),
            "daily_change_percent": round(self.daily_change_percent, 2) if self.daily_change_percent else 0,
            "volatility": round(self.volatility, 4) if self.volatility else None,
            "sharpe_ratio": round(self.sharpe_ratio, 2) if self.sharpe_ratio else None,
            "sortino_ratio": round(self.sortino_ratio, 2) if self.sortino_ratio else None,
            "max_drawdown": round(self.max_drawdown, 2) if self.max_drawdown else None,
            "beta": round(self.beta, 2) if self.beta else None,
            "alpha": round(self.alpha, 2) if self.alpha else None,
            "returns": {
                "1d": round(self.return_1d, 2) if self.return_1d else None,
                "1w": round(self.return_1w, 2) if self.return_1w else None,
                "1m": round(self.return_1m, 2) if self.return_1m else None,
                "3m": round(self.return_3m, 2) if self.return_3m else None,
                "ytd": round(self.return_ytd, 2) if self.return_ytd else None,
                "1y": round(self.return_1y, 2) if self.return_1y else None
            }
        }


@dataclass
class AllocationAnalysis:
    """Portfolio allocation breakdown."""
    by_sector: Dict[str, float] = field(default_factory=dict)
    by_asset_type: Dict[str, float] = field(default_factory=dict)
    by_currency: Dict[str, float] = field(default_factory=dict)
    concentration: float = 0.0  # Herfindahl index
    top_holdings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "by_sector": self.by_sector,
            "by_asset_type": self.by_asset_type,
            "by_currency": self.by_currency,
            "concentration_index": round(self.concentration, 4),
            "top_holdings": self.top_holdings
        }


@dataclass
class PositionAnalysis:
    """Individual position analysis."""
    symbol: str
    name: str
    weight: float
    profit_loss: float
    profit_loss_percent: float
    daily_change_percent: float
    contribution_to_return: float
    risk_contribution: float
    recommendation: str
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "weight": round(self.weight, 2),
            "profit_loss": round(self.profit_loss, 2),
            "profit_loss_percent": round(self.profit_loss_percent, 2),
            "daily_change_percent": round(self.daily_change_percent, 2),
            "contribution_to_return": round(self.contribution_to_return, 4),
            "risk_contribution": round(self.risk_contribution, 4),
            "recommendation": self.recommendation,
            "signals": self.signals
        }


# ==============================================================================
# PORTFOLIO ANALYZER
# ==============================================================================

class PortfolioAnalyzer:
    """
    Analyzes portfolio performance and provides insights.

    Features:
    - Global performance metrics
    - Individual position analysis
    - Allocation analysis
    - Risk metrics
    - Benchmark comparison
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """Initialize analyzer."""
        self.config = config or DEFAULT_CONFIG

    def calculate_performance(self, portfolio: Portfolio) -> PerformanceMetrics:
        """
        Calculate overall portfolio performance metrics.

        Args:
            portfolio: Portfolio to analyze

        Returns:
            PerformanceMetrics with all calculations
        """
        # Basic metrics
        total_value = portfolio.total_value
        total_invested = portfolio.total_invested
        total_pl = portfolio.total_profit_loss
        total_pl_pct = portfolio.total_profit_loss_percent

        # Daily change
        daily_change = portfolio.daily_change
        daily_change_pct = (daily_change / (total_value - daily_change)) * 100 if total_value != daily_change else 0

        metrics = PerformanceMetrics(
            total_value=total_value,
            total_invested=total_invested,
            total_profit_loss=total_pl,
            total_profit_loss_percent=total_pl_pct,
            daily_change=daily_change,
            daily_change_percent=daily_change_pct
        )

        # Calculate advanced metrics if we have historical data
        try:
            hist_data = self._get_portfolio_history(portfolio)
            if hist_data is not None and len(hist_data) > 20:
                metrics.volatility = self._calculate_volatility(hist_data)
                metrics.sharpe_ratio = self._calculate_sharpe(hist_data)
                metrics.sortino_ratio = self._calculate_sortino(hist_data)
                metrics.max_drawdown = self._calculate_max_drawdown(hist_data)

                # Benchmark comparison
                benchmark_data = self._get_benchmark_data()
                if benchmark_data is not None:
                    metrics.beta = self._calculate_beta(hist_data, benchmark_data)
                    metrics.alpha = self._calculate_alpha(hist_data, benchmark_data, metrics.beta)

        except Exception as e:
            logger.warning(f"Could not calculate advanced metrics: {e}")

        return metrics

    def analyze_positions(self, portfolio: Portfolio) -> List[PositionAnalysis]:
        """
        Analyze each position individually.

        Args:
            portfolio: Portfolio to analyze

        Returns:
            List of PositionAnalysis for each position
        """
        analyses = []
        total_value = portfolio.total_value

        for pos in portfolio.positions:
            # Weight in portfolio
            weight = (pos.market_value / total_value * 100) if total_value > 0 else 0

            # Contribution to total return
            contribution = (pos.profit_loss / portfolio.total_invested * 100) if portfolio.total_invested > 0 else 0

            # Risk contribution (simplified - based on weight and volatility)
            risk_contrib = weight / 100  # Placeholder

            # Generate signals
            signals = self._generate_position_signals(pos, weight)

            # Recommendation
            recommendation = self._get_position_recommendation(pos, weight, signals)

            analysis = PositionAnalysis(
                symbol=pos.symbol,
                name=pos.name,
                weight=weight,
                profit_loss=pos.profit_loss,
                profit_loss_percent=pos.profit_loss_percent,
                daily_change_percent=pos.daily_change_percent,
                contribution_to_return=contribution,
                risk_contribution=risk_contrib,
                recommendation=recommendation,
                signals=signals
            )
            analyses.append(analysis)

        # Sort by profit/loss percentage
        analyses.sort(key=lambda x: x.profit_loss_percent, reverse=True)

        return analyses

    def analyze_allocation(self, portfolio: Portfolio) -> AllocationAnalysis:
        """
        Analyze portfolio allocation.

        Args:
            portfolio: Portfolio to analyze

        Returns:
            AllocationAnalysis with breakdowns
        """
        total_value = portfolio.total_value
        if total_value == 0:
            return AllocationAnalysis()

        # By sector
        sector_values = {}
        for pos in portfolio.positions:
            sector = pos.sector or "Non classé"
            sector_values[sector] = sector_values.get(sector, 0) + pos.market_value

        by_sector = {k: (v / total_value * 100) for k, v in sector_values.items()}

        # By asset type
        type_values = {}
        for pos in portfolio.positions:
            asset_type = pos.asset_type
            type_values[asset_type] = type_values.get(asset_type, 0) + pos.market_value

        by_asset_type = {k: (v / total_value * 100) for k, v in type_values.items()}

        # By currency
        currency_values = {}
        for pos in portfolio.positions:
            currency = pos.currency
            currency_values[currency] = currency_values.get(currency, 0) + pos.market_value

        # Add cash
        currency_values[portfolio.currency] = currency_values.get(portfolio.currency, 0) + portfolio.cash_balance
        by_currency = {k: (v / total_value * 100) for k, v in currency_values.items()}

        # Concentration (Herfindahl-Hirschman Index)
        weights = [pos.market_value / total_value for pos in portfolio.positions]
        concentration = sum(w ** 2 for w in weights)

        # Top holdings
        sorted_positions = sorted(portfolio.positions, key=lambda p: p.market_value, reverse=True)
        top_holdings = [
            {
                "symbol": p.symbol,
                "name": p.name,
                "value": p.market_value,
                "weight": p.market_value / total_value * 100
            }
            for p in sorted_positions[:10]
        ]

        return AllocationAnalysis(
            by_sector=by_sector,
            by_asset_type=by_asset_type,
            by_currency=by_currency,
            concentration=concentration,
            top_holdings=top_holdings
        )

    def get_declining_positions(
        self,
        portfolio: Portfolio,
        threshold: float = -5.0
    ) -> List[Dict[str, Any]]:
        """
        Get positions that are declining below threshold.

        Args:
            portfolio: Portfolio to analyze
            threshold: P&L percentage threshold (default -5%)

        Returns:
            List of declining positions with details
        """
        declining = []

        for pos in portfolio.positions:
            if pos.profit_loss_percent < threshold:
                declining.append({
                    "symbol": pos.symbol,
                    "name": pos.name,
                    "quantity": pos.quantity,
                    "average_buy_price": pos.average_buy_price,
                    "current_price": pos.current_price,
                    "profit_loss": pos.profit_loss,
                    "profit_loss_percent": pos.profit_loss_percent,
                    "daily_change_percent": pos.daily_change_percent,
                    "market_value": pos.market_value,
                    "severity": self._get_loss_severity(pos.profit_loss_percent),
                    "recommendation": self._get_loss_recommendation(pos)
                })

        # Sort by loss (most negative first)
        declining.sort(key=lambda x: x["profit_loss_percent"])

        return declining

    def get_investment_signals(self, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """
        Generate investment signals for portfolio positions.

        Args:
            portfolio: Portfolio to analyze

        Returns:
            List of investment signals with recommendations
        """
        signals = []

        for pos in portfolio.positions:
            # Analyze position
            analysis = self._analyze_single_position(pos, portfolio)

            if analysis["signals"]:
                signals.append({
                    "symbol": pos.symbol,
                    "name": pos.name,
                    "current_price": pos.current_price,
                    "profit_loss_percent": pos.profit_loss_percent,
                    "signals": analysis["signals"],
                    "recommendation": analysis["recommendation"],
                    "priority": analysis["priority"]
                })

        # Sort by priority
        signals.sort(key=lambda x: x["priority"], reverse=True)

        return signals

    def _analyze_single_position(
        self,
        pos: Position,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Analyze a single position for signals."""
        signals = []
        priority = 0

        weight = (pos.market_value / portfolio.total_value * 100) if portfolio.total_value > 0 else 0

        # Check for significant loss
        if pos.profit_loss_percent < -20:
            signals.append("⚠️ Perte importante (>20%)")
            priority += 3
        elif pos.profit_loss_percent < -10:
            signals.append("📉 Perte modérée (>10%)")
            priority += 2

        # Check for significant gain
        if pos.profit_loss_percent > 50:
            signals.append("🎯 Gain significatif (>50%) - Prise de profit?")
            priority += 2
        elif pos.profit_loss_percent > 25:
            signals.append("📈 Bonne performance (>25%)")
            priority += 1

        # Check daily change
        if pos.daily_change_percent < -5:
            signals.append("🔴 Forte baisse aujourd'hui")
            priority += 2
        elif pos.daily_change_percent > 5:
            signals.append("🟢 Forte hausse aujourd'hui")
            priority += 1

        # Check concentration
        if weight > 25:
            signals.append(f"⚖️ Concentration élevée ({weight:.1f}%)")
            priority += 1
        elif weight > 15:
            signals.append(f"📊 Position importante ({weight:.1f}%)")

        # Determine recommendation
        if pos.profit_loss_percent < -15 and pos.daily_change_percent < -3:
            recommendation = "VENDRE ou STOP-LOSS"
        elif pos.profit_loss_percent > 30 and weight > 15:
            recommendation = "PRENDRE DES PROFITS"
        elif pos.profit_loss_percent < -10:
            recommendation = "SURVEILLER / RENFORCER?"
        elif pos.profit_loss_percent > 10:
            recommendation = "CONSERVER"
        else:
            recommendation = "NEUTRE"

        return {
            "signals": signals,
            "recommendation": recommendation,
            "priority": priority
        }

    def _generate_position_signals(
        self,
        pos: Position,
        weight: float
    ) -> List[str]:
        """Generate signals for a position."""
        signals = []

        # Loss signals
        if pos.profit_loss_percent < -20:
            signals.append("PERTE_CRITIQUE")
        elif pos.profit_loss_percent < -10:
            signals.append("PERTE_IMPORTANTE")
        elif pos.profit_loss_percent < -5:
            signals.append("PERTE_MODEREE")

        # Gain signals
        if pos.profit_loss_percent > 50:
            signals.append("GAIN_EXCEPTIONNEL")
        elif pos.profit_loss_percent > 25:
            signals.append("GAIN_IMPORTANT")
        elif pos.profit_loss_percent > 10:
            signals.append("GAIN_MODERE")

        # Daily change signals
        if pos.daily_change_percent < -5:
            signals.append("CHUTE_JOURNALIERE")
        elif pos.daily_change_percent > 5:
            signals.append("HAUSSE_JOURNALIERE")

        # Weight signals
        if weight > 25:
            signals.append("SURPONDERATION")
        elif weight < 2:
            signals.append("SOUS_PONDERATION")

        return signals

    def _get_position_recommendation(
        self,
        pos: Position,
        weight: float,
        signals: List[str]
    ) -> str:
        """Get recommendation for a position."""
        if "PERTE_CRITIQUE" in signals:
            return "VENDRE / STOP-LOSS"
        if "PERTE_IMPORTANTE" in signals and "CHUTE_JOURNALIERE" in signals:
            return "SURVEILLER ATTENTIVEMENT"
        if "GAIN_EXCEPTIONNEL" in signals and "SURPONDERATION" in signals:
            return "PRENDRE DES PROFITS"
        if "GAIN_IMPORTANT" in signals:
            return "CONSERVER / TRAILING STOP"
        if "PERTE_MODEREE" in signals:
            return "RENFORCER SI CONVICTION"
        if "SURPONDERATION" in signals:
            return "RÉDUIRE LA POSITION"
        return "CONSERVER"

    def _get_loss_severity(self, loss_percent: float) -> str:
        """Classify loss severity."""
        if loss_percent < -20:
            return "CRITIQUE"
        elif loss_percent < -10:
            return "IMPORTANT"
        elif loss_percent < -5:
            return "MODÉRÉ"
        return "FAIBLE"

    def _get_loss_recommendation(self, pos: Position) -> str:
        """Get recommendation for losing position."""
        if pos.profit_loss_percent < -20:
            return "Évaluer si la thèse d'investissement est toujours valide. Considérer un stop-loss."
        elif pos.profit_loss_percent < -10:
            return "Surveiller de près. Renforcer uniquement si forte conviction."
        else:
            return "Position à surveiller. Volatilité normale possible."

    def _get_portfolio_history(self, portfolio: Portfolio) -> Optional[pd.Series]:
        """Get historical portfolio value (approximation)."""
        # Simplified: use weighted average of position histories
        if not portfolio.positions:
            return None

        try:
            dfs = []
            weights = []

            for pos in portfolio.positions:
                ticker = yf.Ticker(pos.symbol)
                hist = ticker.history(period="6mo")['Close']
                if not hist.empty:
                    # Normalize to current weight
                    weight = pos.market_value / portfolio.total_value if portfolio.total_value > 0 else 0
                    dfs.append(hist)
                    weights.append(weight)

            if not dfs:
                return None

            # Combine with weights
            combined = pd.concat(dfs, axis=1)
            combined = combined.dropna()

            # Weighted portfolio value
            portfolio_value = (combined * weights).sum(axis=1)
            return portfolio_value

        except Exception as e:
            logger.warning(f"Could not get portfolio history: {e}")
            return None

    def _get_benchmark_data(self) -> Optional[pd.Series]:
        """Get benchmark historical data."""
        try:
            ticker = yf.Ticker(self.config.benchmark_symbol)
            hist = ticker.history(period="6mo")['Close']
            return hist if not hist.empty else None
        except Exception as e:
            logger.warning(f"Could not get benchmark data: {e}")
            return None

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        daily_returns = returns.pct_change().dropna()
        return daily_returns.std() * np.sqrt(252)

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        daily_returns = returns.pct_change().dropna()
        excess_returns = daily_returns - (self.config.risk_free_rate / 252)
        if excess_returns.std() == 0:
            return 0
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        daily_returns = returns.pct_change().dropna()
        excess_returns = daily_returns - (self.config.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        if downside_returns.std() == 0:
            return 0
        return (excess_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100

    def _calculate_beta(self, portfolio: pd.Series, benchmark: pd.Series) -> float:
        """Calculate portfolio beta."""
        aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
        if len(aligned) < 20:
            return 1.0

        port_returns = aligned.iloc[:, 0].pct_change().dropna()
        bench_returns = aligned.iloc[:, 1].pct_change().dropna()

        covariance = port_returns.cov(bench_returns)
        variance = bench_returns.var()

        return covariance / variance if variance != 0 else 1.0

    def _calculate_alpha(
        self,
        portfolio: pd.Series,
        benchmark: pd.Series,
        beta: float
    ) -> float:
        """Calculate Jensen's alpha."""
        aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
        if len(aligned) < 20:
            return 0.0

        port_return = (aligned.iloc[-1, 0] / aligned.iloc[0, 0] - 1) * 100
        bench_return = (aligned.iloc[-1, 1] / aligned.iloc[0, 1] - 1) * 100
        rf = self.config.risk_free_rate * 100 * (len(aligned) / 252)

        return port_return - (rf + beta * (bench_return - rf))
