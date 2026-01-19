"""
Trade Republic Integration Module
=================================
Connects to Trade Republic to fetch portfolio data.

Methods:
1. PyTR library (unofficial API) - Real-time WebSocket
2. CSV Import - Manual export from Trade Republic app
3. Manual entry - Add positions manually

Note: Trade Republic doesn't have an official API.
PyTR is an unofficial library that may break with updates.
"""

import os
import json
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import warnings

# Try to import pytr (unofficial Trade Republic API)
try:
    from pytr.api import TradeRepublicApi
    from pytr.account import Account
    PYTR_AVAILABLE = True
except ImportError:
    PYTR_AVAILABLE = False
    warnings.warn("pytr not installed. Use: pip install pytr")

import yfinance as yf
import requests

from .config import PortfolioConfig, setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Position:
    """Represents a single portfolio position."""
    isin: str
    symbol: str
    name: str
    quantity: float
    average_buy_price: float
    current_price: float
    currency: str = "EUR"

    # Calculated fields
    market_value: float = 0.0
    total_cost: float = 0.0
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0
    daily_change: float = 0.0
    daily_change_percent: float = 0.0

    # Metadata
    sector: Optional[str] = None
    country: Optional[str] = None  # Pays de l'entreprise
    region: Optional[str] = None   # Région géographique
    asset_type: str = "stock"  # stock, etf, crypto
    last_updated: Optional[datetime] = None
    purchase_date: Optional[datetime] = None  # Date d'achat pour suivi performance

    def __post_init__(self):
        """Calculate derived values."""
        self.total_cost = self.quantity * self.average_buy_price
        self.market_value = self.quantity * self.current_price
        self.profit_loss = self.market_value - self.total_cost
        if self.total_cost > 0:
            self.profit_loss_percent = (self.profit_loss / self.total_cost) * 100
        self.last_updated = datetime.now()

    def update_price(self, new_price: float, daily_change: float = 0.0):
        """Update position with new price."""
        previous_value = self.market_value
        self.current_price = new_price
        self.market_value = self.quantity * new_price
        self.profit_loss = self.market_value - self.total_cost
        if self.total_cost > 0:
            self.profit_loss_percent = (self.profit_loss / self.total_cost) * 100
        self.daily_change = daily_change
        self.daily_change_percent = (daily_change / (self.current_price - daily_change)) * 100 if self.current_price != daily_change else 0
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "isin": self.isin,
            "symbol": self.symbol,
            "name": self.name,
            "quantity": self.quantity,
            "average_buy_price": round(self.average_buy_price, 2),
            "current_price": round(self.current_price, 2),
            "currency": self.currency,
            "market_value": round(self.market_value, 2),
            "total_cost": round(self.total_cost, 2),
            "profit_loss": round(self.profit_loss, 2),
            "profit_loss_percent": round(self.profit_loss_percent, 2),
            "daily_change": round(self.daily_change, 2),
            "daily_change_percent": round(self.daily_change_percent, 2),
            "sector": self.sector,
            "country": self.country,
            "region": self.region,
            "asset_type": self.asset_type,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "purchase_date": self.purchase_date.isoformat() if self.purchase_date else None
        }


@dataclass
class Portfolio:
    """Represents the complete portfolio."""
    positions: List[Position] = field(default_factory=list)
    cash_balance: float = 0.0
    currency: str = "EUR"
    last_sync: Optional[datetime] = None
    source: str = "manual"  # trade_republic, csv, manual
    creation_date: Optional[datetime] = None  # Date de création pour suivi performance
    historical_values: List[Dict[str, Any]] = field(default_factory=list)  # Historique des valeurs

    @property
    def total_value(self) -> float:
        """Total portfolio value including cash."""
        return sum(p.market_value for p in self.positions) + self.cash_balance

    @property
    def total_invested(self) -> float:
        """Total amount invested."""
        return sum(p.total_cost for p in self.positions)

    @property
    def total_profit_loss(self) -> float:
        """Total profit/loss."""
        return sum(p.profit_loss for p in self.positions)

    @property
    def total_profit_loss_percent(self) -> float:
        """Total profit/loss percentage."""
        if self.total_invested > 0:
            return (self.total_profit_loss / self.total_invested) * 100
        return 0.0

    @property
    def daily_change(self) -> float:
        """Total daily change."""
        return sum(p.daily_change * p.quantity for p in self.positions)

    def get_declining_positions(self, threshold: float = 0.0) -> List[Position]:
        """Get positions with negative P&L below threshold."""
        return [p for p in self.positions if p.profit_loss_percent < threshold]

    def get_top_performers(self, n: int = 5) -> List[Position]:
        """Get top N performing positions."""
        return sorted(self.positions, key=lambda p: p.profit_loss_percent, reverse=True)[:n]

    def get_worst_performers(self, n: int = 5) -> List[Position]:
        """Get worst N performing positions."""
        return sorted(self.positions, key=lambda p: p.profit_loss_percent)[:n]

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Find position by symbol."""
        for p in self.positions:
            if p.symbol.upper() == symbol.upper():
                return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "positions": [p.to_dict() for p in self.positions],
            "cash_balance": round(self.cash_balance, 2),
            "currency": self.currency,
            "total_value": round(self.total_value, 2),
            "total_invested": round(self.total_invested, 2),
            "total_profit_loss": round(self.total_profit_loss, 2),
            "total_profit_loss_percent": round(self.total_profit_loss_percent, 2),
            "daily_change": round(self.daily_change, 2),
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "source": self.source,
            "position_count": len(self.positions),
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "historical_values": self.historical_values
        }

    def record_value(self):
        """Enregistre la valeur actuelle dans l'historique."""
        self.historical_values.append({
            "date": datetime.now().isoformat(),
            "total_value": round(self.total_value, 2),
            "total_invested": round(self.total_invested, 2),
            "profit_loss": round(self.total_profit_loss, 2),
            "profit_loss_percent": round(self.total_profit_loss_percent, 2)
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Convert positions to DataFrame."""
        if not self.positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.positions])


# ==============================================================================
# TRADE REPUBLIC API CLIENT
# ==============================================================================

class TradeRepublicClient:
    """
    Client for Trade Republic integration.

    Supports multiple connection methods:
    1. PyTR library (requires phone number + PIN)
    2. CSV import from Trade Republic export
    3. Manual position entry
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        """Initialize Trade Republic client."""
        self.config = config or PortfolioConfig()
        self.api = None
        self.connected = False
        self._portfolio_cache: Optional[Portfolio] = None
        self._cache_time: Optional[datetime] = None

    async def connect_pytr(
        self,
        phone_number: str,
        pin: str,
        locale: str = "de"
    ) -> bool:
        """
        Connect using PyTR library.

        Args:
            phone_number: Trade Republic phone number (+49...)
            pin: 4-digit PIN
            locale: Country locale (de, fr, etc.)

        Returns:
            True if connected successfully
        """
        if not PYTR_AVAILABLE:
            logger.error("PyTR not installed. Run: pip install pytr")
            return False

        try:
            self.api = TradeRepublicApi(
                phone_number=phone_number,
                pin=pin,
                locale=locale
            )
            await self.api.login()
            self.connected = True
            logger.info("Connected to Trade Republic via PyTR")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Trade Republic: {e}")
            self.connected = False
            return False

    async def fetch_portfolio_pytr(self) -> Optional[Portfolio]:
        """Fetch portfolio data using PyTR."""
        if not self.connected or not self.api:
            logger.error("Not connected to Trade Republic")
            return None

        try:
            # Get portfolio positions
            positions_data = await self.api.portfolio()

            positions = []
            for item in positions_data.get("positions", []):
                pos = Position(
                    isin=item.get("instrumentId", ""),
                    symbol=item.get("symbol", item.get("instrumentId", "")[:6]),
                    name=item.get("name", "Unknown"),
                    quantity=float(item.get("quantity", 0)),
                    average_buy_price=float(item.get("averageBuyIn", 0)),
                    current_price=float(item.get("currentPrice", 0)),
                    currency="EUR"
                )
                positions.append(pos)

            # Get cash balance
            cash = await self.api.cash()
            cash_balance = float(cash.get("available", 0))

            portfolio = Portfolio(
                positions=positions,
                cash_balance=cash_balance,
                currency="EUR",
                last_sync=datetime.now(),
                source="trade_republic"
            )

            self._portfolio_cache = portfolio
            self._cache_time = datetime.now()

            return portfolio

        except Exception as e:
            logger.error(f"Failed to fetch portfolio from Trade Republic: {e}")
            return None

    def import_from_csv(self, csv_path: str) -> Optional[Portfolio]:
        """
        Import portfolio from Trade Republic CSV export.

        Args:
            csv_path: Path to CSV file exported from Trade Republic

        Returns:
            Portfolio object
        """
        try:
            df = pd.read_csv(csv_path, sep=';', decimal=',')

            # Trade Republic CSV columns may vary
            # Common columns: ISIN, Name, Quantity, Average Price, etc.
            column_mapping = {
                'ISIN': 'isin',
                'Instrument': 'name',
                'Name': 'name',
                'Anzahl': 'quantity',
                'Quantity': 'quantity',
                'Stück': 'quantity',
                'Durchschnittspreis': 'average_buy_price',
                'Average Price': 'average_buy_price',
                'Einstandskurs': 'average_buy_price'
            }

            df = df.rename(columns=column_mapping)

            positions = []
            for _, row in df.iterrows():
                isin = str(row.get('isin', ''))
                if not isin or isin == 'nan':
                    continue

                # Try to get current price from yfinance
                symbol = self._isin_to_symbol(isin)
                current_price = self._get_current_price(symbol) if symbol else row.get('average_buy_price', 0)

                pos = Position(
                    isin=isin,
                    symbol=symbol or isin[:6],
                    name=str(row.get('name', 'Unknown')),
                    quantity=float(row.get('quantity', 0)),
                    average_buy_price=float(row.get('average_buy_price', 0)),
                    current_price=current_price,
                    currency="EUR"
                )
                positions.append(pos)

            portfolio = Portfolio(
                positions=positions,
                cash_balance=0.0,
                currency="EUR",
                last_sync=datetime.now(),
                source="csv"
            )

            self._portfolio_cache = portfolio
            self._cache_time = datetime.now()

            logger.info(f"Imported {len(positions)} positions from CSV")
            return portfolio

        except Exception as e:
            logger.error(f"Failed to import CSV: {e}")
            return None

    def _get_region_from_country(self, country: str) -> str:
        """Détermine la région géographique à partir du pays."""
        region_mapping = {
            # Amérique du Nord
            'United States': 'Amérique du Nord', 'Canada': 'Amérique du Nord', 'Mexico': 'Amérique du Nord',
            # Europe
            'France': 'Europe', 'Germany': 'Europe', 'United Kingdom': 'Europe', 'Switzerland': 'Europe',
            'Netherlands': 'Europe', 'Spain': 'Europe', 'Italy': 'Europe', 'Belgium': 'Europe',
            'Sweden': 'Europe', 'Norway': 'Europe', 'Denmark': 'Europe', 'Finland': 'Europe',
            'Ireland': 'Europe', 'Austria': 'Europe', 'Portugal': 'Europe', 'Luxembourg': 'Europe',
            # Asie-Pacifique
            'Japan': 'Asie-Pacifique', 'China': 'Asie-Pacifique', 'Hong Kong': 'Asie-Pacifique',
            'South Korea': 'Asie-Pacifique', 'Taiwan': 'Asie-Pacifique', 'Singapore': 'Asie-Pacifique',
            'Australia': 'Asie-Pacifique', 'India': 'Asie-Pacifique', 'New Zealand': 'Asie-Pacifique',
            # Amérique Latine
            'Brazil': 'Amérique Latine', 'Argentina': 'Amérique Latine', 'Chile': 'Amérique Latine',
            # Moyen-Orient & Afrique
            'Israel': 'Moyen-Orient', 'Saudi Arabia': 'Moyen-Orient', 'United Arab Emirates': 'Moyen-Orient',
            'South Africa': 'Afrique',
        }
        return region_mapping.get(country, 'Autres')

    def create_manual_portfolio(
        self,
        positions_data: List[Dict[str, Any]],
        cash_balance: float = 0.0
    ) -> Portfolio:
        """
        Create portfolio from manual entry.

        Args:
            positions_data: List of position dicts with keys:
                - symbol: Ticker symbol
                - quantity: Number of shares
                - average_buy_price: Average purchase price
                - name (optional): Company name
            cash_balance: Cash balance in account

        Returns:
            Portfolio object
        """
        positions = []

        for data in positions_data:
            symbol = data.get('symbol', '').upper()
            if not symbol:
                continue

            # Get current price and info from yfinance
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
                name = data.get('name') or info.get('longName', symbol)
                sector = info.get('sector')
                country = info.get('country')
                region = self._get_region_from_country(country) if country else None

                # Determine asset type
                quote_type = info.get('quoteType', 'EQUITY')
                if quote_type == 'ETF':
                    asset_type = 'etf'
                elif quote_type == 'CRYPTOCURRENCY':
                    asset_type = 'crypto'
                else:
                    asset_type = 'stock'

                # Get daily change
                prev_close = info.get('previousClose', current_price)
                daily_change = current_price - prev_close if prev_close else 0

            except Exception as e:
                logger.warning(f"Could not fetch data for {symbol}: {e}")
                current_price = data.get('current_price', data.get('average_buy_price', 0))
                name = data.get('name', symbol)
                sector = None
                country = None
                region = None
                asset_type = 'stock'
                daily_change = 0

            pos = Position(
                isin=data.get('isin', ''),
                symbol=symbol,
                name=name,
                quantity=float(data.get('quantity', 0)),
                average_buy_price=float(data.get('average_buy_price', 0)),
                current_price=float(current_price),
                currency=data.get('currency', 'EUR'),
                sector=sector,
                country=country,
                region=region,
                asset_type=asset_type,
                purchase_date=datetime.now()
            )
            pos.daily_change = daily_change
            if current_price and daily_change:
                pos.daily_change_percent = (daily_change / (current_price - daily_change)) * 100

            positions.append(pos)

        portfolio = Portfolio(
            positions=positions,
            cash_balance=cash_balance,
            currency="EUR",
            last_sync=datetime.now(),
            source="manual",
            creation_date=datetime.now()
        )

        # Enregistrer la valeur initiale
        portfolio.record_value()

        self._portfolio_cache = portfolio
        self._cache_time = datetime.now()

        logger.info(f"Created portfolio with {len(positions)} positions")
        return portfolio

    def refresh_prices(self, portfolio: Optional[Portfolio] = None) -> Portfolio:
        """
        Refresh current prices for all positions.

        Args:
            portfolio: Portfolio to refresh (uses cache if None)

        Returns:
            Updated portfolio
        """
        portfolio = portfolio or self._portfolio_cache
        if not portfolio:
            raise ValueError("No portfolio to refresh")

        for pos in portfolio.positions:
            try:
                ticker = yf.Ticker(pos.symbol)
                info = ticker.info
                current_price = info.get('regularMarketPrice') or info.get('currentPrice', pos.current_price)
                prev_close = info.get('previousClose', current_price)
                daily_change = current_price - prev_close if prev_close else 0

                pos.update_price(current_price, daily_change)

            except Exception as e:
                logger.warning(f"Could not refresh price for {pos.symbol}: {e}")

        portfolio.last_sync = datetime.now()
        self._portfolio_cache = portfolio
        self._cache_time = datetime.now()

        return portfolio

    def _isin_to_symbol(self, isin: str) -> Optional[str]:
        """Convert ISIN to ticker symbol."""
        # Common ISIN prefixes
        # DE = Germany, US = USA, FR = France, etc.

        # Try to look up symbol from ISIN
        # This is a simplified lookup - in production use a proper ISIN database
        try:
            # Try searching via yfinance
            search_result = yf.Ticker(isin)
            if search_result.info.get('symbol'):
                return search_result.info['symbol']
        except:
            pass

        # Fallback: return first 6 chars
        return None

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('regularMarketPrice', 0)
        except:
            return 0.0

    def save_portfolio(self, filepath: str, portfolio: Optional[Portfolio] = None) -> bool:
        """Save portfolio to JSON file."""
        portfolio = portfolio or self._portfolio_cache
        if not portfolio:
            return False

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(portfolio.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Portfolio saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")
            return False

    def load_portfolio(self, filepath: str) -> Optional[Portfolio]:
        """Load portfolio from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            positions = []
            for pos_data in data.get('positions', []):
                pos = Position(
                    isin=pos_data.get('isin', ''),
                    symbol=pos_data.get('symbol', ''),
                    name=pos_data.get('name', ''),
                    quantity=pos_data.get('quantity', 0),
                    average_buy_price=pos_data.get('average_buy_price', 0),
                    current_price=pos_data.get('current_price', 0),
                    currency=pos_data.get('currency', 'EUR'),
                    sector=pos_data.get('sector'),
                    country=pos_data.get('country'),
                    region=pos_data.get('region'),
                    asset_type=pos_data.get('asset_type', 'stock'),
                    purchase_date=datetime.fromisoformat(pos_data['purchase_date']) if pos_data.get('purchase_date') else None
                )
                positions.append(pos)

            portfolio = Portfolio(
                positions=positions,
                cash_balance=data.get('cash_balance', 0),
                currency=data.get('currency', 'EUR'),
                last_sync=datetime.fromisoformat(data['last_sync']) if data.get('last_sync') else None,
                source=data.get('source', 'loaded'),
                creation_date=datetime.fromisoformat(data['creation_date']) if data.get('creation_date') else None,
                historical_values=data.get('historical_values', [])
            )

            self._portfolio_cache = portfolio
            self._cache_time = datetime.now()

            logger.info(f"Portfolio loaded from {filepath}")
            return portfolio

        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            return None


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_sample_portfolio() -> Portfolio:
    """Create a sample portfolio for testing."""
    client = TradeRepublicClient()

    sample_positions = [
        {"symbol": "AAPL", "quantity": 10, "average_buy_price": 150.00},
        {"symbol": "MSFT", "quantity": 5, "average_buy_price": 320.00},
        {"symbol": "GOOGL", "quantity": 3, "average_buy_price": 140.00},
        {"symbol": "AMZN", "quantity": 8, "average_buy_price": 175.00},
        {"symbol": "NVDA", "quantity": 15, "average_buy_price": 450.00},
        {"symbol": "TSLA", "quantity": 12, "average_buy_price": 220.00},
    ]

    return client.create_manual_portfolio(sample_positions, cash_balance=1500.00)
