"""
Time Series Analysis Module
===========================
Implements ARIMA and GARCH models for price and volatility forecasting.

Features:
- Auto-ARIMA with AIC/BIC model selection
- GARCH(p,q) volatility modeling
- Stationarity tests (ADF, KPSS)
- Confidence intervals and diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Statistical libraries
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import acf, pacf

# Auto-ARIMA
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not installed. Auto-ARIMA will use fallback method.")

# GARCH models
try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, ConstantMean, ZeroMean
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch package not installed. GARCH models will be unavailable.")

from .config import (
    ARIMAConfig,
    GARCHConfig,
    VolatilityRegime,
    DEFAULT_CONFIG,
    setup_logger,
    RISK_WARNINGS
)

logger = setup_logger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class StationarityTest:
    """Results of stationarity tests."""
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: Dict[str, float]
    is_stationary_adf: bool
    kpss_statistic: Optional[float] = None
    kpss_pvalue: Optional[float] = None
    is_stationary_kpss: Optional[bool] = None
    differencing_required: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adf": {
                "statistic": round(self.adf_statistic, 4),
                "p_value": round(self.adf_pvalue, 4),
                "critical_values": {k: round(v, 4) for k, v in self.adf_critical_values.items()},
                "is_stationary": self.is_stationary_adf
            },
            "kpss": {
                "statistic": round(self.kpss_statistic, 4) if self.kpss_statistic else None,
                "p_value": round(self.kpss_pvalue, 4) if self.kpss_pvalue else None,
                "is_stationary": self.is_stationary_kpss
            },
            "differencing_required": self.differencing_required
        }


@dataclass
class ARIMAResults:
    """Container for ARIMA model results."""
    model_order: Tuple[int, int, int]
    aic: float
    bic: float
    forecast: List[float]
    confidence_interval_lower: List[float]
    confidence_interval_upper: List[float]
    forecast_dates: List[str]
    residuals: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    stationarity: Optional[StationarityTest] = None
    model_summary: Optional[str] = None
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_order": self.model_order,
            "aic": round(self.aic, 2),
            "bic": round(self.bic, 2),
            "forecast": [round(f, 2) for f in self.forecast],
            "confidence_interval": [
                (round(l, 2), round(u, 2))
                for l, u in zip(self.confidence_interval_lower, self.confidence_interval_upper)
            ],
            "forecast_dates": self.forecast_dates,
            "stationarity": self.stationarity.to_dict() if self.stationarity else None,
            "execution_time_seconds": round(self.execution_time_seconds, 2)
        }


@dataclass
class GARCHResults:
    """Container for GARCH model results."""
    model_type: str
    order: Tuple[int, int]
    current_volatility: float
    annualized_volatility: float
    forecast_volatility: List[float]
    forecast_dates: List[str]
    volatility_regime: str
    parameters: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    standardized_residuals: Optional[np.ndarray] = None
    conditional_volatility: Optional[np.ndarray] = None
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": f"{self.model_type}({self.order[0]},{self.order[1]})",
            "current_volatility": round(self.current_volatility, 6),
            "annualized_volatility": round(self.annualized_volatility, 4),
            "forecast_volatility": [round(v, 6) for v in self.forecast_volatility],
            "forecast_dates": self.forecast_dates,
            "volatility_regime": self.volatility_regime,
            "parameters": {k: round(v, 6) for k, v in self.parameters.items()},
            "log_likelihood": round(self.log_likelihood, 2),
            "aic": round(self.aic, 2),
            "bic": round(self.bic, 2),
            "execution_time_seconds": round(self.execution_time_seconds, 2)
        }


@dataclass
class TimeSeriesResults:
    """Combined results from ARIMA and GARCH models."""
    arima: ARIMAResults
    garch: Optional[GARCHResults]
    price_direction_probability: float
    combined_forecast: List[float]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arima": self.arima.to_dict(),
            "garch": self.garch.to_dict() if self.garch else None,
            "price_direction_probability": round(self.price_direction_probability, 4),
            "combined_forecast": [round(f, 2) for f in self.combined_forecast],
            "warnings": self.warnings
        }

# ==============================================================================
# STATIONARITY TESTS
# ==============================================================================

def test_stationarity(
    series: pd.Series,
    significance_level: float = 0.05
) -> StationarityTest:
    """
    Test series stationarity using ADF and KPSS tests.

    Args:
        series: Time series to test
        significance_level: Significance level for tests

    Returns:
        StationarityTest with test results
    """
    # Clean series
    series_clean = series.dropna()

    # ADF Test (null: series has unit root = non-stationary)
    adf_result = adfuller(series_clean, autolag='AIC')
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_critical = adf_result[4]
    is_stationary_adf = adf_pvalue < significance_level

    # KPSS Test (null: series is stationary)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(series_clean, regression='c', nlags='auto')
        kpss_statistic = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        is_stationary_kpss = kpss_pvalue > significance_level
    except Exception:
        kpss_statistic = None
        kpss_pvalue = None
        is_stationary_kpss = None

    # Determine differencing required
    diff_required = 0
    test_series = series_clean.copy()
    for d in range(3):
        adf_test = adfuller(test_series, autolag='AIC')
        if adf_test[1] < significance_level:
            diff_required = d
            break
        test_series = test_series.diff().dropna()
        diff_required = d + 1

    return StationarityTest(
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        adf_critical_values=adf_critical,
        is_stationary_adf=is_stationary_adf,
        kpss_statistic=kpss_statistic,
        kpss_pvalue=kpss_pvalue,
        is_stationary_kpss=is_stationary_kpss,
        differencing_required=diff_required
    )

# ==============================================================================
# ARIMA MODEL
# ==============================================================================

class ARIMAForecaster:
    """
    ARIMA forecaster with automatic order selection.

    Attributes:
        config: ARIMAConfig instance
    """

    def __init__(self, config: Optional[ARIMAConfig] = None):
        """Initialize ARIMA forecaster."""
        self.config = config or DEFAULT_CONFIG.arima
        self.model = None
        self.fitted_model = None

    def fit_and_forecast(
        self,
        prices: pd.Series,
        forecast_days: int = 14
    ) -> ARIMAResults:
        """
        Fit ARIMA model and generate forecasts.

        Args:
            prices: Historical price series
            forecast_days: Number of days to forecast

        Returns:
            ARIMAResults with forecasts and model info
        """
        start_time = datetime.now()

        # Use log prices for better stationarity
        log_prices = np.log(prices)

        # Test stationarity
        stationarity = test_stationarity(log_prices)
        logger.info(f"Stationarity test: ADF p-value={stationarity.adf_pvalue:.4f}")

        # Find optimal order
        if PMDARIMA_AVAILABLE:
            order = self._find_order_auto(log_prices)
        else:
            order = self._find_order_manual(log_prices, stationarity.differencing_required)

        logger.info(f"Selected ARIMA order: {order}")

        # Fit model
        try:
            model = ARIMA(log_prices, order=order)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit()

            self.model = model
            self.fitted_model = fitted

            # Generate forecast
            forecast_result = fitted.get_forecast(steps=forecast_days)
            forecast_log = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)

            # Convert back from log
            forecast = np.exp(forecast_log).tolist()
            ci_lower = np.exp(conf_int.iloc[:, 0]).tolist()
            ci_upper = np.exp(conf_int.iloc[:, 1]).tolist()

            # Generate forecast dates
            last_date = prices.index[-1]
            forecast_dates = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days
            ).strftime('%Y-%m-%d').tolist()

            execution_time = (datetime.now() - start_time).total_seconds()

            return ARIMAResults(
                model_order=order,
                aic=fitted.aic,
                bic=fitted.bic,
                forecast=forecast,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                forecast_dates=forecast_dates,
                residuals=fitted.resid.values,
                fitted_values=np.exp(fitted.fittedvalues).values,
                stationarity=stationarity,
                model_summary=str(fitted.summary()),
                execution_time_seconds=execution_time
            )

        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise ValueError(f"ARIMA model failed to converge: {e}")

    def _find_order_auto(self, series: pd.Series) -> Tuple[int, int, int]:
        """Find optimal ARIMA order using auto_arima."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto_model = auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=self.config.max_p,
                max_d=self.config.max_d,
                max_q=self.config.max_q,
                seasonal=self.config.seasonal,
                m=self.config.seasonal_period if self.config.seasonal else 1,
                information_criterion=self.config.information_criterion,
                stepwise=self.config.stepwise,
                suppress_warnings=self.config.suppress_warnings,
                error_action='ignore',
                trace=False
            )
        return auto_model.order

    def _find_order_manual(
        self,
        series: pd.Series,
        d: int
    ) -> Tuple[int, int, int]:
        """Find ARIMA order using grid search (fallback)."""
        best_aic = np.inf
        best_order = (1, d, 1)

        for p in range(self.config.max_p + 1):
            for q in range(self.config.max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                except Exception:
                    continue

        return best_order

    def calculate_direction_probability(
        self,
        current_price: float,
        forecast: List[float],
        ci_lower: List[float],
        ci_upper: List[float]
    ) -> float:
        """Calculate probability of price going up based on forecast."""
        # Use final forecast vs current price
        final_forecast = forecast[-1]

        # Calculate z-score
        ci_width = ci_upper[-1] - ci_lower[-1]
        std_estimate = ci_width / (2 * 1.96)  # Approximate std from 95% CI

        if std_estimate > 0:
            z_score = (final_forecast - current_price) / std_estimate
            prob_up = stats.norm.cdf(z_score)
        else:
            prob_up = 0.5 if final_forecast > current_price else 0.5

        return float(prob_up)

# ==============================================================================
# GARCH MODEL
# ==============================================================================

class GARCHForecaster:
    """
    GARCH forecaster for volatility modeling.

    Attributes:
        config: GARCHConfig instance
    """

    def __init__(self, config: Optional[GARCHConfig] = None):
        """Initialize GARCH forecaster."""
        self.config = config or DEFAULT_CONFIG.garch
        self.model = None
        self.fitted_model = None

    def fit_and_forecast(
        self,
        prices: pd.Series,
        forecast_days: int = 14
    ) -> GARCHResults:
        """
        Fit GARCH model and forecast volatility.

        Args:
            prices: Historical price series
            forecast_days: Number of days to forecast

        Returns:
            GARCHResults with volatility forecasts
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required for GARCH models")

        start_time = datetime.now()

        # Calculate returns (percentage)
        returns = 100 * prices.pct_change().dropna()

        # Build model
        if self.config.model_type == "EGARCH":
            vol_model = arch_model(
                returns,
                mean=self.config.mean_model,
                vol='EGARCH',
                p=self.config.p,
                q=self.config.q,
                dist=self.config.distribution
            )
        elif self.config.model_type == "GJR-GARCH":
            vol_model = arch_model(
                returns,
                mean=self.config.mean_model,
                vol='GARCH',
                p=self.config.p,
                o=1,  # GJR term
                q=self.config.q,
                dist=self.config.distribution
            )
        else:  # Standard GARCH
            vol_model = arch_model(
                returns,
                mean=self.config.mean_model,
                vol='GARCH',
                p=self.config.p,
                q=self.config.q,
                dist=self.config.distribution
            )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = vol_model.fit(disp='off', show_warning=False)

        self.model = vol_model
        self.fitted_model = fitted

        # Get current conditional volatility
        cond_vol = fitted.conditional_volatility
        current_volatility = cond_vol.iloc[-1] / 100  # Convert back from percentage

        # Annualize
        annualized_vol = current_volatility * np.sqrt(252)

        # Forecast volatility
        forecast = fitted.forecast(horizon=forecast_days)
        forecast_variance = forecast.variance.iloc[-1].values
        forecast_volatility = (np.sqrt(forecast_variance) / 100).tolist()

        # Generate forecast dates
        last_date = prices.index[-1]
        forecast_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days
        ).strftime('%Y-%m-%d').tolist()

        # Determine volatility regime
        regime = self._classify_volatility_regime(annualized_vol)

        # Extract parameters
        params = {name: float(value) for name, value in fitted.params.items()}

        execution_time = (datetime.now() - start_time).total_seconds()

        return GARCHResults(
            model_type=self.config.model_type,
            order=(self.config.p, self.config.q),
            current_volatility=current_volatility,
            annualized_volatility=annualized_vol,
            forecast_volatility=forecast_volatility,
            forecast_dates=forecast_dates,
            volatility_regime=regime,
            parameters=params,
            log_likelihood=float(fitted.loglikelihood),
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            standardized_residuals=fitted.std_resid.values,
            conditional_volatility=cond_vol.values / 100,
            execution_time_seconds=execution_time
        )

    def _classify_volatility_regime(self, annualized_vol: float) -> str:
        """Classify volatility into regimes."""
        if annualized_vol < self.config.volatility_low_threshold:
            return VolatilityRegime.LOW.value
        elif annualized_vol < self.config.volatility_medium_threshold:
            return VolatilityRegime.MEDIUM.value
        elif annualized_vol < self.config.volatility_high_threshold:
            return VolatilityRegime.HIGH.value
        else:
            return VolatilityRegime.EXTREME.value

    def calculate_volatility_impact(self, garch_results: GARCHResults) -> float:
        """
        Calculate probability adjustment based on volatility regime.

        High volatility = higher uncertainty = probability closer to 0.5
        Low volatility = more confidence in direction
        """
        regime = garch_results.volatility_regime

        if regime == VolatilityRegime.LOW.value:
            return 0.0  # No penalty
        elif regime == VolatilityRegime.MEDIUM.value:
            return -0.05  # Small penalty
        elif regime == VolatilityRegime.HIGH.value:
            return -0.10  # Moderate penalty
        else:  # EXTREME
            return -0.15  # Significant penalty

# ==============================================================================
# COMBINED TIME SERIES ANALYZER
# ==============================================================================

class TimeSeriesAnalyzer:
    """
    Combined ARIMA and GARCH analyzer.

    Provides unified interface for time series forecasting.
    """

    def __init__(
        self,
        arima_config: Optional[ARIMAConfig] = None,
        garch_config: Optional[GARCHConfig] = None
    ):
        """Initialize analyzer with configurations."""
        self.arima_forecaster = ARIMAForecaster(arima_config)
        self.garch_forecaster = GARCHForecaster(garch_config) if ARCH_AVAILABLE else None

    def analyze(
        self,
        prices: pd.Series,
        forecast_days: int = 14,
        include_garch: bool = True
    ) -> TimeSeriesResults:
        """
        Run complete time series analysis.

        Args:
            prices: Historical price series
            forecast_days: Number of days to forecast
            include_garch: Whether to include GARCH analysis

        Returns:
            TimeSeriesResults with combined forecasts
        """
        warnings_list = []
        current_price = float(prices.iloc[-1])

        # Run ARIMA
        try:
            arima_results = self.arima_forecaster.fit_and_forecast(
                prices, forecast_days
            )
        except Exception as e:
            logger.error(f"ARIMA failed: {e}")
            raise

        # Calculate direction probability from ARIMA
        arima_prob = self.arima_forecaster.calculate_direction_probability(
            current_price,
            arima_results.forecast,
            arima_results.confidence_interval_lower,
            arima_results.confidence_interval_upper
        )

        # Run GARCH if requested and available
        garch_results = None
        garch_adjustment = 0.0

        if include_garch and self.garch_forecaster:
            try:
                garch_results = self.garch_forecaster.fit_and_forecast(
                    prices, forecast_days
                )
                garch_adjustment = self.garch_forecaster.calculate_volatility_impact(
                    garch_results
                )

                if garch_results.volatility_regime in [
                    VolatilityRegime.HIGH.value,
                    VolatilityRegime.EXTREME.value
                ]:
                    warnings_list.append(RISK_WARNINGS["high_volatility"])

            except Exception as e:
                logger.warning(f"GARCH failed: {e}")
                warnings_list.append(RISK_WARNINGS["model_convergence"])

        # Calculate combined probability
        # Adjust probability towards 0.5 based on volatility
        if arima_prob > 0.5:
            adjusted_prob = arima_prob + garch_adjustment
        else:
            adjusted_prob = arima_prob - garch_adjustment

        # Ensure probability is in valid range
        adjusted_prob = max(0.0, min(1.0, adjusted_prob))

        return TimeSeriesResults(
            arima=arima_results,
            garch=garch_results,
            price_direction_probability=adjusted_prob,
            combined_forecast=arima_results.forecast,
            warnings=warnings_list
        )


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def run_time_series_analysis(
    ticker: str,
    forecast_days: int = 14,
    historical_days: int = 252
) -> TimeSeriesResults:
    """
    Convenience function to run time series analysis for a ticker.

    Args:
        ticker: Stock ticker symbol
        forecast_days: Number of days to forecast
        historical_days: Days of historical data to use

    Returns:
        TimeSeriesResults with forecasts
    """
    import yfinance as yf

    # Download data
    stock = yf.Ticker(ticker)
    hist = stock.history(period=f"{historical_days}d")

    if hist.empty or len(hist) < 100:
        raise ValueError(f"Insufficient data for ticker: {ticker}")

    prices = hist["Close"]

    # Run analysis
    analyzer = TimeSeriesAnalyzer()
    return analyzer.analyze(prices, forecast_days)
