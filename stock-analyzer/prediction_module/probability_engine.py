"""
Probability Aggregation Engine
==============================
Combines signals from multiple models into unified predictions.

Features:
- Weighted probability aggregation
- Confidence interval calculation
- Signal classification (BUY/SELL/NEUTRAL)
- Risk identification and scoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .config import (
    AggregationWeights,
    Signal,
    DEFAULT_CONFIG,
    setup_logger,
    DISCLAIMERS,
    RISK_WARNINGS
)
from .monte_carlo import MonteCarloResults
from .time_series import TimeSeriesResults, GARCHResults
from .news_sentiment import SentimentResults

logger = setup_logger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ModelContribution:
    """Contribution of a single model to the final probability."""
    model_name: str
    probability: float
    weight: float
    contribution: float  # weight * (probability - 0.5)
    available: bool = True
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "probability": round(self.probability, 4),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 4),
            "available": self.available,
            "warning": self.warning
        }


@dataclass
class RiskAssessment:
    """Risk factors identified in the analysis."""
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    risk_score: float  # 0-100
    identified_risks: List[str]
    volatility_warning: bool
    confidence_warning: bool
    contradiction_warning: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.risk_level,
            "score": round(self.risk_score, 1),
            "risks": self.identified_risks,
            "high_volatility": self.volatility_warning,
            "low_confidence": self.confidence_warning,
            "contradictory_signals": self.contradiction_warning
        }


@dataclass
class PredictionResult:
    """Final aggregated prediction result."""
    # Core prediction
    signal: str
    probability_hausse: float
    confidence_interval: Tuple[float, float]
    horizon_days: int

    # Model contributions
    contributions: Dict[str, float]
    model_details: List[ModelContribution]

    # Risk assessment
    risk_assessment: RiskAssessment

    # Price targets
    target_price_mean: Optional[float]
    target_price_range: Optional[Tuple[float, float]]
    current_price: float

    # Recommendations
    recommendation: str
    disclaimers: List[str]

    # Metadata
    ticker: str
    analysis_timestamp: str
    execution_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_global": self.signal,
            "probabilite_hausse": round(self.probability_hausse, 4),
            "intervalle_confiance": (
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4)
            ),
            "horizon": f"{self.horizon_days} jours",
            "contributions": {
                k: round(v, 4) for k, v in self.contributions.items()
            },
            "model_details": [m.to_dict() for m in self.model_details],
            "risques_identifies": self.risk_assessment.identified_risks,
            "risk_assessment": self.risk_assessment.to_dict(),
            "target_price": {
                "mean": round(self.target_price_mean, 2) if self.target_price_mean else None,
                "range": (
                    round(self.target_price_range[0], 2),
                    round(self.target_price_range[1], 2)
                ) if self.target_price_range else None
            },
            "current_price": round(self.current_price, 2),
            "recommandation": self.recommendation,
            "disclaimers": self.disclaimers,
            "ticker": self.ticker,
            "timestamp": self.analysis_timestamp,
            "execution_time_seconds": round(self.execution_time_seconds, 2)
        }

# ==============================================================================
# SIGNAL CLASSIFICATION
# ==============================================================================

class SignalClassifier:
    """Classifies probability into trading signals."""

    # Thresholds for signal classification
    THRESHOLDS = {
        Signal.STRONG_BUY: 0.75,
        Signal.BUY: 0.65,
        Signal.MODERATE_BUY: 0.55,
        Signal.NEUTRAL: 0.45,
        Signal.MODERATE_SELL: 0.35,
        Signal.SELL: 0.25,
        Signal.STRONG_SELL: 0.0
    }

    @classmethod
    def classify(cls, probability: float) -> str:
        """
        Classify probability into signal.

        Args:
            probability: Probability of price going up [0, 1]

        Returns:
            Signal string
        """
        if probability >= cls.THRESHOLDS[Signal.STRONG_BUY]:
            return Signal.STRONG_BUY.value
        elif probability >= cls.THRESHOLDS[Signal.BUY]:
            return Signal.BUY.value
        elif probability >= cls.THRESHOLDS[Signal.MODERATE_BUY]:
            return Signal.MODERATE_BUY.value
        elif probability >= cls.THRESHOLDS[Signal.NEUTRAL]:
            return Signal.NEUTRAL.value
        elif probability >= cls.THRESHOLDS[Signal.MODERATE_SELL]:
            return Signal.MODERATE_SELL.value
        elif probability >= cls.THRESHOLDS[Signal.SELL]:
            return Signal.SELL.value
        else:
            return Signal.STRONG_SELL.value

    @classmethod
    def get_recommendation(cls, signal: str, risk_level: str) -> str:
        """Generate recommendation based on signal and risk."""
        if risk_level in ["HIGH", "EXTREME"]:
            return f"⚠️ SIGNAL {signal} - PRUDENCE RECOMMANDÉE (Risque élevé)"

        if signal in [Signal.STRONG_BUY.value, Signal.BUY.value]:
            return f"📈 SIGNAL {signal}"
        elif signal in [Signal.STRONG_SELL.value, Signal.SELL.value]:
            return f"📉 SIGNAL {signal}"
        elif signal == Signal.NEUTRAL.value:
            return "➡️ SIGNAL NEUTRE - Pas de tendance claire"
        else:
            return f"📊 SIGNAL {signal}"

# ==============================================================================
# RISK ANALYZER
# ==============================================================================

class RiskAnalyzer:
    """Analyzes and scores risk factors."""

    def analyze(
        self,
        monte_carlo: Optional[MonteCarloResults],
        time_series: Optional[TimeSeriesResults],
        sentiment: Optional[SentimentResults],
        model_contributions: List[ModelContribution]
    ) -> RiskAssessment:
        """
        Analyze risk factors from all models.

        Returns:
            RiskAssessment with identified risks
        """
        risks = []
        risk_score = 0.0

        volatility_warning = False
        confidence_warning = False
        contradiction_warning = False

        # Check volatility from Monte Carlo
        if monte_carlo:
            # High VaR indicates high risk
            if monte_carlo.var_5 < -10:
                risks.append(f"VaR élevé ({monte_carlo.var_5:.1f}%)")
                risk_score += 20

            # Wide confidence intervals
            ci_95 = monte_carlo.confidence_intervals.get("95%")
            if ci_95:
                ci_width = (ci_95[1] - ci_95[0]) / monte_carlo.current_price * 100
                if ci_width > 30:
                    risks.append(f"Grande incertitude (±{ci_width/2:.1f}%)")
                    risk_score += 15
                    confidence_warning = True

        # Check GARCH volatility regime
        if time_series and time_series.garch:
            regime = time_series.garch.volatility_regime
            if regime in ["HAUTE", "EXTREME"]:
                ann_vol = time_series.garch.annualized_volatility * 100
                risks.append(f"Volatilité {regime.lower()} (σ={ann_vol:.1f}%)")
                risk_score += 25 if regime == "EXTREME" else 15
                volatility_warning = True

        # Check sentiment warnings
        if sentiment:
            if sentiment.news_analyzed < 10:
                risks.append("Peu d'actualités analysées")
                risk_score += 10
                confidence_warning = True

            # Mixed sentiment
            breakdown = sentiment.sentiment_breakdown
            if breakdown.total_articles > 0:
                pos_ratio = breakdown.positive_ratio
                neg_ratio = breakdown.negative_ratio
                if 0.3 < pos_ratio < 0.7 and 0.3 < neg_ratio < 0.7:
                    risks.append("Actualités contradictoires")
                    risk_score += 10

        # Check model contradictions
        probabilities = [
            m.probability for m in model_contributions
            if m.available
        ]
        if len(probabilities) >= 2:
            prob_std = np.std(probabilities)
            if prob_std > 0.2:
                risks.append("Signaux contradictoires entre modèles")
                risk_score += 15
                contradiction_warning = True

        # Check for unavailable models
        unavailable = [m.model_name for m in model_contributions if not m.available]
        if unavailable:
            risks.append(f"Modèles indisponibles: {', '.join(unavailable)}")
            risk_score += 5 * len(unavailable)

        # Classify risk level
        if risk_score >= 60:
            risk_level = "EXTREME"
        elif risk_score >= 40:
            risk_level = "HIGH"
        elif risk_score >= 20:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=min(100, risk_score),
            identified_risks=risks if risks else ["Aucun risque majeur identifié"],
            volatility_warning=volatility_warning,
            confidence_warning=confidence_warning,
            contradiction_warning=contradiction_warning
        )

# ==============================================================================
# MAIN PROBABILITY ENGINE
# ==============================================================================

class ProbabilityEngine:
    """
    Aggregates probabilities from multiple models.

    Combines Monte Carlo, ARIMA, GARCH, and Sentiment signals
    into a unified prediction.
    """

    def __init__(self, weights: Optional[AggregationWeights] = None):
        """Initialize with aggregation weights."""
        self.weights = weights or DEFAULT_CONFIG.weights
        self.risk_analyzer = RiskAnalyzer()
        self.signal_classifier = SignalClassifier()

    def aggregate(
        self,
        ticker: str,
        current_price: float,
        horizon_days: int,
        monte_carlo: Optional[MonteCarloResults] = None,
        time_series: Optional[TimeSeriesResults] = None,
        sentiment: Optional[SentimentResults] = None
    ) -> PredictionResult:
        """
        Aggregate all model results into final prediction.

        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            horizon_days: Prediction horizon
            monte_carlo: Monte Carlo simulation results
            time_series: ARIMA/GARCH results
            sentiment: News sentiment results

        Returns:
            PredictionResult with aggregated prediction
        """
        start_time = datetime.now()

        # Extract probabilities from each model
        contributions = self._calculate_contributions(
            monte_carlo, time_series, sentiment
        )

        # Calculate weighted probability
        final_probability = self._calculate_weighted_probability(contributions)

        # Calculate confidence interval (bootstrap-like approach)
        confidence_interval = self._calculate_confidence_interval(
            contributions, final_probability
        )

        # Classify signal
        signal = self.signal_classifier.classify(final_probability)

        # Assess risks
        risk_assessment = self.risk_analyzer.analyze(
            monte_carlo, time_series, sentiment, contributions
        )

        # Get recommendation
        recommendation = self.signal_classifier.get_recommendation(
            signal, risk_assessment.risk_level
        )

        # Extract price targets from Monte Carlo
        target_price_mean = None
        target_price_range = None
        if monte_carlo:
            target_price_mean = monte_carlo.mean_price
            ci_95 = monte_carlo.confidence_intervals.get("95%")
            if ci_95:
                target_price_range = ci_95

        # Build contributions dict for output
        contributions_dict = {
            "Monte Carlo": next(
                (c.contribution for c in contributions if c.model_name == "Monte Carlo"),
                0.0
            ),
            "ARIMA": next(
                (c.contribution for c in contributions if c.model_name == "ARIMA"),
                0.0
            ),
            "GARCH": next(
                (c.contribution for c in contributions if c.model_name == "GARCH"),
                0.0
            ),
            "Sentiment": next(
                (c.contribution for c in contributions if c.model_name == "Sentiment"),
                0.0
            )
        }

        execution_time = (datetime.now() - start_time).total_seconds()

        return PredictionResult(
            signal=signal,
            probability_hausse=final_probability,
            confidence_interval=confidence_interval,
            horizon_days=horizon_days,
            contributions=contributions_dict,
            model_details=contributions,
            risk_assessment=risk_assessment,
            target_price_mean=target_price_mean,
            target_price_range=target_price_range,
            current_price=current_price,
            recommendation=recommendation,
            disclaimers=DISCLAIMERS,
            ticker=ticker,
            analysis_timestamp=datetime.now().isoformat(),
            execution_time_seconds=execution_time
        )

    def _calculate_contributions(
        self,
        monte_carlo: Optional[MonteCarloResults],
        time_series: Optional[TimeSeriesResults],
        sentiment: Optional[SentimentResults]
    ) -> List[ModelContribution]:
        """Calculate contribution from each model."""
        contributions = []

        # Monte Carlo contribution
        if monte_carlo:
            prob = monte_carlo.probability_up
            weight = self.weights.monte_carlo
            contributions.append(ModelContribution(
                model_name="Monte Carlo",
                probability=prob,
                weight=weight,
                contribution=weight * (prob - 0.5),
                available=True
            ))
        else:
            contributions.append(ModelContribution(
                model_name="Monte Carlo",
                probability=0.5,
                weight=self.weights.monte_carlo,
                contribution=0.0,
                available=False,
                warning="Monte Carlo simulation not available"
            ))

        # ARIMA contribution
        if time_series:
            prob = time_series.price_direction_probability
            weight = self.weights.arima
            contributions.append(ModelContribution(
                model_name="ARIMA",
                probability=prob,
                weight=weight,
                contribution=weight * (prob - 0.5),
                available=True
            ))
        else:
            contributions.append(ModelContribution(
                model_name="ARIMA",
                probability=0.5,
                weight=self.weights.arima,
                contribution=0.0,
                available=False,
                warning="ARIMA model not available"
            ))

        # GARCH contribution (adjusts based on volatility)
        if time_series and time_series.garch:
            garch = time_series.garch
            # High volatility = lower confidence in direction
            # This reduces probability towards 0.5
            vol_factor = 1.0 - min(garch.annualized_volatility, 1.0)
            prob = 0.5 + (time_series.price_direction_probability - 0.5) * vol_factor
            weight = self.weights.garch
            contributions.append(ModelContribution(
                model_name="GARCH",
                probability=prob,
                weight=weight,
                contribution=weight * (prob - 0.5),
                available=True
            ))
        else:
            contributions.append(ModelContribution(
                model_name="GARCH",
                probability=0.5,
                weight=self.weights.garch,
                contribution=0.0,
                available=False,
                warning="GARCH model not available"
            ))

        # Sentiment contribution
        if sentiment and sentiment.news_analyzed > 0:
            prob = sentiment.probability_hausse
            weight = self.weights.sentiment
            contributions.append(ModelContribution(
                model_name="Sentiment",
                probability=prob,
                weight=weight,
                contribution=weight * (prob - 0.5),
                available=True
            ))
        else:
            contributions.append(ModelContribution(
                model_name="Sentiment",
                probability=0.5,
                weight=self.weights.sentiment,
                contribution=0.0,
                available=False,
                warning="No news data available"
            ))

        return contributions

    def _calculate_weighted_probability(
        self,
        contributions: List[ModelContribution]
    ) -> float:
        """Calculate weighted average probability."""
        # Only use available models
        available = [c for c in contributions if c.available]

        if not available:
            logger.warning("No models available, returning neutral probability")
            return 0.5

        # Renormalize weights for available models
        total_weight = sum(c.weight for c in available)

        if total_weight == 0:
            return 0.5

        # Weighted average
        weighted_sum = sum(
            c.probability * (c.weight / total_weight)
            for c in available
        )

        return weighted_sum

    def _calculate_confidence_interval(
        self,
        contributions: List[ModelContribution],
        final_probability: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the probability estimate."""
        available = [c for c in contributions if c.available]

        if len(available) < 2:
            # Wide interval with few models
            return (max(0, final_probability - 0.2), min(1, final_probability + 0.2))

        # Calculate standard error from model disagreement
        probs = [c.probability for c in available]
        std = np.std(probs)

        # 95% confidence interval (approximately)
        margin = 1.96 * std / np.sqrt(len(probs))

        lower = max(0, final_probability - margin)
        upper = min(1, final_probability + margin)

        return (lower, upper)

    def recalibrate_weights(
        self,
        new_weights: Dict[str, float]
    ) -> None:
        """Update aggregation weights."""
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.weights = AggregationWeights(
            monte_carlo=new_weights.get("monte_carlo", 0.25),
            arima=new_weights.get("arima", 0.20),
            garch=new_weights.get("garch", 0.15),
            sentiment=new_weights.get("sentiment", 0.40)
        )


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def aggregate_predictions(
    ticker: str,
    current_price: float,
    horizon_days: int = 14,
    monte_carlo: Optional[MonteCarloResults] = None,
    time_series: Optional[TimeSeriesResults] = None,
    sentiment: Optional[SentimentResults] = None,
    weights: Optional[AggregationWeights] = None
) -> PredictionResult:
    """
    Convenience function to aggregate model predictions.

    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        horizon_days: Prediction horizon
        monte_carlo: Monte Carlo results
        time_series: Time series results
        sentiment: Sentiment results
        weights: Optional custom weights

    Returns:
        PredictionResult with aggregated prediction
    """
    engine = ProbabilityEngine(weights)
    return engine.aggregate(
        ticker=ticker,
        current_price=current_price,
        horizon_days=horizon_days,
        monte_carlo=monte_carlo,
        time_series=time_series,
        sentiment=sentiment
    )
