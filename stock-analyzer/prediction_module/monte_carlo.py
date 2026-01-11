"""
Monte Carlo Simulation Module
=============================
Implements Geometric Brownian Motion (GBM) for stock price simulations.

Features:
- N-path price simulations (default 10,000)
- Confidence intervals (50%, 68%, 95%)
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Multiprocessing for performance optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import warnings

from .config import (
    MonteCarloConfig,
    DEFAULT_CONFIG,
    setup_logger,
    RISK_WARNINGS
)

logger = setup_logger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results."""
    current_price: float
    mean_price: float
    median_price: float
    std_price: float
    min_price: float
    max_price: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    probability_up: float
    probability_down: float
    var_5: float  # Value at Risk at 5%
    var_1: float  # Value at Risk at 1%
    cvar_5: float  # Conditional VaR at 5%
    cvar_1: float  # Conditional VaR at 1%
    simulated_paths: Optional[np.ndarray] = None
    final_prices: Optional[np.ndarray] = None
    daily_returns_used: float = 0.0
    daily_volatility_used: float = 0.0
    days_simulated: int = 0
    num_simulations: int = 0
    execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            "current_price": round(self.current_price, 2),
            "mean_price": round(self.mean_price, 2),
            "median_price": round(self.median_price, 2),
            "std_price": round(self.std_price, 2),
            "min_price": round(self.min_price, 2),
            "max_price": round(self.max_price, 2),
            "confidence_intervals": {
                k: (round(v[0], 2), round(v[1], 2))
                for k, v in self.confidence_intervals.items()
            },
            "probability_up": round(self.probability_up, 4),
            "probability_down": round(self.probability_down, 4),
            "var_5%": round(self.var_5, 2),
            "var_1%": round(self.var_1, 2),
            "cvar_5%": round(self.cvar_5, 2),
            "cvar_1%": round(self.cvar_1, 2),
            "parameters": {
                "daily_return": round(self.daily_returns_used, 6),
                "daily_volatility": round(self.daily_volatility_used, 6),
                "days_simulated": self.days_simulated,
                "num_simulations": self.num_simulations
            },
            "execution_time_seconds": round(self.execution_time_seconds, 2)
        }

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def calculate_historical_parameters(
    prices: pd.Series,
    trading_days: int = 252
) -> Tuple[float, float, float]:
    """
    Calculate historical drift and volatility from price series.

    Args:
        prices: Historical price series
        trading_days: Number of trading days per year

    Returns:
        Tuple of (daily_return, daily_volatility, annualized_volatility)
    """
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 20:
        raise ValueError("Insufficient data: need at least 20 days of returns")

    # Daily statistics
    daily_return = log_returns.mean()
    daily_volatility = log_returns.std()

    # Annualized volatility
    annualized_volatility = daily_volatility * np.sqrt(trading_days)

    return daily_return, daily_volatility, annualized_volatility


def _simulate_gbm_batch(args: Tuple) -> np.ndarray:
    """
    Worker function for parallel GBM simulation.

    Args:
        args: Tuple of (S0, mu, sigma, T, n_sims, seed)

    Returns:
        Array of simulated final prices
    """
    S0, mu, sigma, T, n_sims, seed = args

    if seed is not None:
        np.random.seed(seed)

    # Generate random shocks
    dt = 1  # Daily steps
    Z = np.random.standard_normal((n_sims, T))

    # GBM formula: S(t) = S(0) * exp((mu - 0.5*sigma^2)*t + sigma*sqrt(t)*Z)
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # Cumulative returns
    cumulative_returns = np.cumsum(drift + diffusion, axis=1)

    # Final prices
    final_prices = S0 * np.exp(cumulative_returns[:, -1])

    return final_prices


def _simulate_gbm_paths(args: Tuple) -> np.ndarray:
    """
    Worker function for parallel GBM path simulation (full paths).

    Args:
        args: Tuple of (S0, mu, sigma, T, n_sims, seed)

    Returns:
        Array of simulated price paths (n_sims x T)
    """
    S0, mu, sigma, T, n_sims, seed = args

    if seed is not None:
        np.random.seed(seed)

    dt = 1
    Z = np.random.standard_normal((n_sims, T))

    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # Build full paths
    paths = np.zeros((n_sims, T + 1))
    paths[:, 0] = S0

    for t in range(T):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion[:, t])

    return paths

# ==============================================================================
# MAIN MONTE CARLO CLASS
# ==============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulator for stock price prediction using Geometric Brownian Motion.

    Attributes:
        config: MonteCarloConfig instance with simulation parameters
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            config: MonteCarloConfig instance (uses default if None)
        """
        self.config = config or DEFAULT_CONFIG.monte_carlo

    def run_simulation(
        self,
        prices: pd.Series,
        days_ahead: Optional[int] = None,
        num_simulations: Optional[int] = None,
        return_paths: bool = False
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation for price prediction.

        Args:
            prices: Historical price series (must have DatetimeIndex)
            days_ahead: Number of days to simulate forward
            num_simulations: Number of simulation paths
            return_paths: If True, include all simulated paths in results

        Returns:
            MonteCarloResults with simulation statistics
        """
        start_time = datetime.now()

        # Use config defaults if not specified
        days = days_ahead or self.config.days_ahead
        n_sims = num_simulations or self.config.simulations

        # Validate inputs
        if len(prices) < 30:
            raise ValueError("Minimum 30 days of historical data required")

        # Get current price
        current_price = float(prices.iloc[-1])

        # Calculate historical parameters
        mu, sigma, ann_vol = calculate_historical_parameters(
            prices,
            self.config.trading_days_per_year
        )

        logger.info(
            f"Monte Carlo: S0={current_price:.2f}, mu={mu:.6f}, "
            f"sigma={sigma:.6f}, T={days}, N={n_sims}"
        )

        # Run simulations
        if return_paths:
            paths, final_prices = self._run_with_paths(
                current_price, mu, sigma, days, n_sims
            )
        else:
            paths = None
            final_prices = self._run_final_prices_only(
                current_price, mu, sigma, days, n_sims
            )

        # Calculate statistics
        results = self._calculate_statistics(
            current_price=current_price,
            final_prices=final_prices,
            paths=paths,
            mu=mu,
            sigma=sigma,
            days=days,
            n_sims=n_sims,
            start_time=start_time
        )

        return results

    def _run_final_prices_only(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: int,
        n_sims: int
    ) -> np.ndarray:
        """Run simulation returning only final prices (memory efficient)."""
        if self.config.use_multiprocessing and n_sims >= 10000:
            return self._run_parallel(S0, mu, sigma, T, n_sims, return_paths=False)
        else:
            return self._run_sequential(S0, mu, sigma, T, n_sims, return_paths=False)

    def _run_with_paths(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: int,
        n_sims: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation returning full paths."""
        if self.config.use_multiprocessing and n_sims >= 10000:
            paths = self._run_parallel(S0, mu, sigma, T, n_sims, return_paths=True)
        else:
            paths = self._run_sequential(S0, mu, sigma, T, n_sims, return_paths=True)

        final_prices = paths[:, -1]
        return paths, final_prices

    def _run_sequential(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: int,
        n_sims: int,
        return_paths: bool
    ) -> np.ndarray:
        """Run sequential simulation."""
        seed = self.config.random_seed

        if return_paths:
            args = (S0, mu, sigma, T, n_sims, seed)
            return _simulate_gbm_paths(args)
        else:
            args = (S0, mu, sigma, T, n_sims, seed)
            return _simulate_gbm_batch(args)

    def _run_parallel(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: int,
        n_sims: int,
        return_paths: bool
    ) -> np.ndarray:
        """Run parallel simulation using ProcessPoolExecutor."""
        n_workers = min(mp.cpu_count(), 4)
        sims_per_worker = n_sims // n_workers

        # Create tasks with different seeds
        base_seed = self.config.random_seed or 42
        tasks = [
            (S0, mu, sigma, T, sims_per_worker, base_seed + i)
            for i in range(n_workers)
        ]

        # Add remaining simulations to last task
        remaining = n_sims - (sims_per_worker * n_workers)
        if remaining > 0:
            tasks[-1] = (
                S0, mu, sigma, T,
                sims_per_worker + remaining,
                base_seed + n_workers - 1
            )

        worker_func = _simulate_gbm_paths if return_paths else _simulate_gbm_batch

        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker_func, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        return np.concatenate(results, axis=0)

    def _calculate_statistics(
        self,
        current_price: float,
        final_prices: np.ndarray,
        paths: Optional[np.ndarray],
        mu: float,
        sigma: float,
        days: int,
        n_sims: int,
        start_time: datetime
    ) -> MonteCarloResults:
        """Calculate all statistics from simulation results."""
        # Basic statistics
        mean_price = float(np.mean(final_prices))
        median_price = float(np.median(final_prices))
        std_price = float(np.std(final_prices))
        min_price = float(np.min(final_prices))
        max_price = float(np.max(final_prices))

        # Confidence intervals
        confidence_intervals = {}
        for level in self.config.confidence_levels:
            lower_pct = (1 - level) / 2 * 100
            upper_pct = (1 + level) / 2 * 100
            ci_lower = float(np.percentile(final_prices, lower_pct))
            ci_upper = float(np.percentile(final_prices, upper_pct))
            confidence_intervals[f"{int(level * 100)}%"] = (ci_lower, ci_upper)

        # Probability calculations
        returns = (final_prices - current_price) / current_price
        probability_up = float(np.mean(returns > 0))
        probability_down = float(np.mean(returns < 0))

        # Value at Risk (percentage loss)
        var_5 = float(np.percentile(returns, 5) * 100)
        var_1 = float(np.percentile(returns, 1) * 100)

        # Conditional VaR (Expected Shortfall)
        cvar_5 = float(np.mean(returns[returns <= np.percentile(returns, 5)]) * 100)
        cvar_1 = float(np.mean(returns[returns <= np.percentile(returns, 1)]) * 100)

        execution_time = (datetime.now() - start_time).total_seconds()

        return MonteCarloResults(
            current_price=current_price,
            mean_price=mean_price,
            median_price=median_price,
            std_price=std_price,
            min_price=min_price,
            max_price=max_price,
            confidence_intervals=confidence_intervals,
            probability_up=probability_up,
            probability_down=probability_down,
            var_5=var_5,
            var_1=var_1,
            cvar_5=cvar_5,
            cvar_1=cvar_1,
            simulated_paths=paths,
            final_prices=final_prices,
            daily_returns_used=mu,
            daily_volatility_used=sigma,
            days_simulated=days,
            num_simulations=n_sims,
            execution_time_seconds=execution_time
        )

    def calculate_probability_target(
        self,
        prices: pd.Series,
        target_price: float,
        days_ahead: int = 30,
        num_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Calculate probability of reaching a target price.

        Args:
            prices: Historical price series
            target_price: Target price to evaluate
            days_ahead: Simulation horizon
            num_simulations: Number of simulations

        Returns:
            Dictionary with probability statistics
        """
        results = self.run_simulation(
            prices=prices,
            days_ahead=days_ahead,
            num_simulations=num_simulations,
            return_paths=False
        )

        current_price = results.current_price
        final_prices = results.final_prices

        prob_above_target = float(np.mean(final_prices >= target_price))
        prob_below_target = float(np.mean(final_prices < target_price))

        # Expected return if target is hit
        above_target_prices = final_prices[final_prices >= target_price]
        expected_if_hit = (
            float(np.mean(above_target_prices))
            if len(above_target_prices) > 0
            else target_price
        )

        return {
            "target_price": target_price,
            "current_price": current_price,
            "probability_above_target": round(prob_above_target, 4),
            "probability_below_target": round(prob_below_target, 4),
            "expected_price_if_target_hit": round(expected_if_hit, 2),
            "required_return": round((target_price / current_price - 1) * 100, 2)
        }


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def run_monte_carlo(
    ticker: str,
    days_ahead: int = 30,
    num_simulations: int = 10000,
    config: Optional[MonteCarloConfig] = None
) -> MonteCarloResults:
    """
    Convenience function to run Monte Carlo simulation for a ticker.

    Args:
        ticker: Stock ticker symbol
        days_ahead: Number of days to simulate
        num_simulations: Number of simulation paths
        config: Optional MonteCarloConfig

    Returns:
        MonteCarloResults with simulation statistics
    """
    import yfinance as yf

    # Download historical data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    prices = hist["Close"]

    # Run simulation
    simulator = MonteCarloSimulator(config)
    return simulator.run_simulation(
        prices=prices,
        days_ahead=days_ahead,
        num_simulations=num_simulations,
        return_paths=True
    )
