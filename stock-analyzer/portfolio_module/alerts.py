"""
Portfolio Alerts and Recommendations Module
===========================================
Monitors portfolio and generates alerts and investment recommendations.

Features:
- Price drop/surge alerts
- Loss threshold alerts
- Rebalancing suggestions
- Investment recommendations based on portfolio analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import yfinance as yf

from .trade_republic import Portfolio, Position
from .config import (
    PortfolioConfig,
    DEFAULT_CONFIG,
    AlertType,
    RecommendationType,
    ALERT_MESSAGES,
    RECOMMENDATION_MESSAGES,
    setup_logger
)

logger = setup_logger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Alert:
    """Represents a portfolio alert."""
    alert_type: AlertType
    severity: str  # info, warning, error, success
    title: str
    message: str
    symbol: Optional[str] = None
    position_name: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    action_required: bool = False
    suggested_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.alert_type.value,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "symbol": self.symbol,
            "position_name": self.position_name,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "action_required": self.action_required,
            "suggested_action": self.suggested_action
        }


@dataclass
class InvestmentRecommendation:
    """Investment recommendation for a position."""
    symbol: str
    name: str
    recommendation_type: RecommendationType
    confidence: float  # 0-100
    reasoning: List[str]
    target_action: Optional[str] = None
    target_amount: Optional[float] = None
    priority: int = 0  # Higher = more urgent
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        rec_info = RECOMMENDATION_MESSAGES.get(self.recommendation_type, {})
        return {
            "symbol": self.symbol,
            "name": self.name,
            "recommendation": self.recommendation_type.value,
            "icon": rec_info.get("icon", ""),
            "title": rec_info.get("title", ""),
            "color": rec_info.get("color", "#9E9E9E"),
            "confidence": round(self.confidence, 1),
            "reasoning": self.reasoning,
            "target_action": self.target_action,
            "target_amount": self.target_amount,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat()
        }


# ==============================================================================
# ALERT MANAGER
# ==============================================================================

class AlertManager:
    """
    Manages portfolio alerts and monitors positions.

    Checks for:
    - Daily price changes exceeding thresholds
    - Cumulative losses exceeding thresholds
    - Concentration risks
    - Portfolio rebalancing needs
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """Initialize alert manager."""
        self.config = config or DEFAULT_CONFIG
        self.alerts_config = self.config.alerts

    def check_all_alerts(self, portfolio: Portfolio) -> List[Alert]:
        """
        Run all alert checks on portfolio.

        Args:
            portfolio: Portfolio to check

        Returns:
            List of triggered alerts
        """
        alerts = []

        # Check each position
        for pos in portfolio.positions:
            alerts.extend(self._check_position_alerts(pos))

        # Check portfolio-level alerts
        alerts.extend(self._check_portfolio_alerts(portfolio))

        # Sort by severity
        severity_order = {"error": 0, "warning": 1, "info": 2, "success": 3}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 4))

        return alerts

    def _check_position_alerts(self, pos: Position) -> List[Alert]:
        """Check alerts for a single position."""
        alerts = []

        # Daily change alerts
        if pos.daily_change_percent <= self.alerts_config.daily_drop_threshold:
            alerts.append(Alert(
                alert_type=AlertType.PRICE_DROP,
                severity="warning",
                title=f"📉 {pos.symbol} en baisse",
                message=f"{pos.name} a chuté de {pos.daily_change_percent:.1f}% aujourd'hui",
                symbol=pos.symbol,
                position_name=pos.name,
                value=pos.daily_change_percent,
                threshold=self.alerts_config.daily_drop_threshold,
                action_required=pos.daily_change_percent < -5,
                suggested_action="Surveiller la position et les actualités"
            ))

        if pos.daily_change_percent >= self.alerts_config.daily_surge_threshold:
            alerts.append(Alert(
                alert_type=AlertType.PRICE_SURGE,
                severity="info",
                title=f"📈 {pos.symbol} en hausse",
                message=f"{pos.name} a gagné {pos.daily_change_percent:.1f}% aujourd'hui",
                symbol=pos.symbol,
                position_name=pos.name,
                value=pos.daily_change_percent,
                threshold=self.alerts_config.daily_surge_threshold,
                suggested_action="Envisager une prise de profit partielle"
            ))

        # Cumulative loss alerts
        if pos.profit_loss_percent <= self.alerts_config.loss_threshold_critical:
            alerts.append(Alert(
                alert_type=AlertType.LOSS_THRESHOLD,
                severity="error",
                title=f"🔴 Perte critique sur {pos.symbol}",
                message=f"{pos.name} est en perte de {pos.profit_loss_percent:.1f}% ({pos.profit_loss:.2f}€)",
                symbol=pos.symbol,
                position_name=pos.name,
                value=pos.profit_loss_percent,
                threshold=self.alerts_config.loss_threshold_critical,
                action_required=True,
                suggested_action="Évaluer la thèse d'investissement. Stop-loss recommandé."
            ))
        elif pos.profit_loss_percent <= self.alerts_config.loss_threshold_warning:
            alerts.append(Alert(
                alert_type=AlertType.LOSS_THRESHOLD,
                severity="warning",
                title=f"⚠️ Perte importante sur {pos.symbol}",
                message=f"{pos.name} est en perte de {pos.profit_loss_percent:.1f}% ({pos.profit_loss:.2f}€)",
                symbol=pos.symbol,
                position_name=pos.name,
                value=pos.profit_loss_percent,
                threshold=self.alerts_config.loss_threshold_warning,
                suggested_action="Surveiller la position. Renforcer uniquement si forte conviction."
            ))

        # Gain threshold alerts
        if pos.profit_loss_percent >= self.alerts_config.gain_threshold_take_profit:
            alerts.append(Alert(
                alert_type=AlertType.GAIN_THRESHOLD,
                severity="success",
                title=f"🎯 Objectif atteint sur {pos.symbol}",
                message=f"{pos.name} est en gain de {pos.profit_loss_percent:.1f}% ({pos.profit_loss:.2f}€)",
                symbol=pos.symbol,
                position_name=pos.name,
                value=pos.profit_loss_percent,
                threshold=self.alerts_config.gain_threshold_take_profit,
                suggested_action="Envisager une prise de profit ou mettre un trailing stop"
            ))

        return alerts

    def _check_portfolio_alerts(self, portfolio: Portfolio) -> List[Alert]:
        """Check portfolio-level alerts."""
        alerts = []
        total_value = portfolio.total_value

        if total_value == 0:
            return alerts

        # Concentration risk
        for pos in portfolio.positions:
            weight = (pos.market_value / total_value) * 100
            if weight > self.alerts_config.max_position_weight:
                alerts.append(Alert(
                    alert_type=AlertType.REBALANCE,
                    severity="warning",
                    title=f"⚖️ Concentration élevée: {pos.symbol}",
                    message=f"{pos.name} représente {weight:.1f}% du portefeuille (max recommandé: {self.alerts_config.max_position_weight}%)",
                    symbol=pos.symbol,
                    position_name=pos.name,
                    value=weight,
                    threshold=self.alerts_config.max_position_weight,
                    suggested_action="Envisager une réduction de la position pour diversifier"
                ))

        # Diversification check
        if len(portfolio.positions) < self.alerts_config.min_diversification:
            alerts.append(Alert(
                alert_type=AlertType.REBALANCE,
                severity="info",
                title="📊 Diversification faible",
                message=f"Vous avez {len(portfolio.positions)} positions (minimum recommandé: {self.alerts_config.min_diversification})",
                value=len(portfolio.positions),
                threshold=self.alerts_config.min_diversification,
                suggested_action="Envisager d'ajouter des positions pour réduire le risque"
            ))

        return alerts


# ==============================================================================
# RECOMMENDATION ENGINE
# ==============================================================================

class RecommendationEngine:
    """
    Generates investment recommendations based on portfolio analysis.

    Analyzes:
    - Position performance
    - Technical indicators
    - Portfolio balance
    - Risk factors
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """Initialize recommendation engine."""
        self.config = config or DEFAULT_CONFIG

    def generate_recommendations(
        self,
        portfolio: Portfolio,
        include_technical: bool = True
    ) -> List[InvestmentRecommendation]:
        """
        Generate investment recommendations for all positions.

        Args:
            portfolio: Portfolio to analyze
            include_technical: Include technical analysis in recommendations

        Returns:
            List of InvestmentRecommendation
        """
        recommendations = []
        total_value = portfolio.total_value

        for pos in portfolio.positions:
            rec = self._analyze_position(pos, portfolio, include_technical)
            if rec:
                recommendations.append(rec)

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        return recommendations

    def _analyze_position(
        self,
        pos: Position,
        portfolio: Portfolio,
        include_technical: bool
    ) -> Optional[InvestmentRecommendation]:
        """Analyze a position and generate recommendation."""
        reasoning = []
        confidence = 50.0
        priority = 0

        weight = (pos.market_value / portfolio.total_value * 100) if portfolio.total_value > 0 else 0

        # Performance analysis
        if pos.profit_loss_percent < -20:
            reasoning.append(f"Perte importante de {pos.profit_loss_percent:.1f}%")
            confidence -= 20
            priority += 3
        elif pos.profit_loss_percent < -10:
            reasoning.append(f"Perte modérée de {pos.profit_loss_percent:.1f}%")
            confidence -= 10
            priority += 2
        elif pos.profit_loss_percent > 30:
            reasoning.append(f"Gain significatif de {pos.profit_loss_percent:.1f}%")
            confidence += 15
            priority += 2
        elif pos.profit_loss_percent > 15:
            reasoning.append(f"Bonne performance de {pos.profit_loss_percent:.1f}%")
            confidence += 10
            priority += 1

        # Daily momentum
        if pos.daily_change_percent < -5:
            reasoning.append(f"Forte baisse aujourd'hui ({pos.daily_change_percent:.1f}%)")
            confidence -= 15
            priority += 2
        elif pos.daily_change_percent < -2:
            reasoning.append(f"Baisse aujourd'hui ({pos.daily_change_percent:.1f}%)")
            confidence -= 5
        elif pos.daily_change_percent > 5:
            reasoning.append(f"Forte hausse aujourd'hui ({pos.daily_change_percent:.1f}%)")
            confidence += 10
            priority += 1

        # Weight analysis
        if weight > 25:
            reasoning.append(f"Position très concentrée ({weight:.1f}% du portefeuille)")
            priority += 1
        elif weight > 15:
            reasoning.append(f"Position importante ({weight:.1f}% du portefeuille)")

        # Technical analysis
        if include_technical:
            tech_signals = self._get_technical_signals(pos.symbol)
            reasoning.extend(tech_signals.get("reasoning", []))
            confidence += tech_signals.get("confidence_adjustment", 0)

        # Determine recommendation type
        rec_type = self._determine_recommendation(pos, confidence, weight)

        # Calculate target action
        target_action, target_amount = self._calculate_target(pos, rec_type, portfolio)

        return InvestmentRecommendation(
            symbol=pos.symbol,
            name=pos.name,
            recommendation_type=rec_type,
            confidence=max(0, min(100, confidence)),
            reasoning=reasoning,
            target_action=target_action,
            target_amount=target_amount,
            priority=priority
        )

    def _determine_recommendation(
        self,
        pos: Position,
        confidence: float,
        weight: float
    ) -> RecommendationType:
        """Determine recommendation type based on analysis."""
        # Critical loss - suggest sell
        if pos.profit_loss_percent < -20 and pos.daily_change_percent < -3:
            return RecommendationType.SELL

        # Large gain with high weight - take profits
        if pos.profit_loss_percent > 30 and weight > 15:
            return RecommendationType.REDUCE

        # Good performance - hold
        if pos.profit_loss_percent > 10:
            return RecommendationType.HOLD

        # Moderate loss but good company - maybe increase
        if -15 < pos.profit_loss_percent < -5 and confidence > 40:
            return RecommendationType.INCREASE

        # Significant loss - watch closely
        if pos.profit_loss_percent < -10:
            return RecommendationType.WATCH

        # Over-concentrated - reduce
        if weight > 25:
            return RecommendationType.REDUCE

        # Default - hold
        return RecommendationType.HOLD

    def _get_technical_signals(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis signals for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")

            if hist.empty or len(hist) < 20:
                return {"reasoning": [], "confidence_adjustment": 0}

            reasoning = []
            confidence_adj = 0

            close = hist['Close']

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            if current_rsi < 30:
                reasoning.append(f"RSI survendu ({current_rsi:.0f})")
                confidence_adj += 10
            elif current_rsi > 70:
                reasoning.append(f"RSI suracheté ({current_rsi:.0f})")
                confidence_adj -= 10

            # Moving averages
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20
            current_price = close.iloc[-1]

            if current_price > ma20 > ma50:
                reasoning.append("Tendance haussière (prix > MA20 > MA50)")
                confidence_adj += 5
            elif current_price < ma20 < ma50:
                reasoning.append("Tendance baissière (prix < MA20 < MA50)")
                confidence_adj -= 5

            return {
                "reasoning": reasoning,
                "confidence_adjustment": confidence_adj
            }

        except Exception as e:
            logger.warning(f"Could not get technical signals for {symbol}: {e}")
            return {"reasoning": [], "confidence_adjustment": 0}

    def _calculate_target(
        self,
        pos: Position,
        rec_type: RecommendationType,
        portfolio: Portfolio
    ) -> Tuple[Optional[str], Optional[float]]:
        """Calculate target action and amount."""
        if rec_type == RecommendationType.SELL:
            return "Vendre toute la position", pos.quantity

        if rec_type == RecommendationType.REDUCE:
            # Suggest reducing to 10% of portfolio
            target_weight = 0.10
            target_value = portfolio.total_value * target_weight
            current_value = pos.market_value
            if current_value > target_value:
                sell_value = current_value - target_value
                sell_qty = sell_value / pos.current_price
                return f"Réduire de {sell_qty:.0f} actions", round(sell_qty, 2)

        if rec_type == RecommendationType.INCREASE:
            # Suggest increasing by 25%
            increase_qty = pos.quantity * 0.25
            return f"Renforcer de {increase_qty:.0f} actions", round(increase_qty, 2)

        return None, None


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def check_portfolio_alerts(portfolio: Portfolio) -> List[Alert]:
    """Quick function to check all portfolio alerts."""
    manager = AlertManager()
    return manager.check_all_alerts(portfolio)


def get_investment_recommendations(portfolio: Portfolio) -> List[InvestmentRecommendation]:
    """Quick function to get investment recommendations."""
    engine = RecommendationEngine()
    return engine.generate_recommendations(portfolio)
