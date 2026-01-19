"""
Module d'Analyse Comparative pour Stock Analyzer Pro
Compare jusqu'à 4 entreprises sur performance, ratios financiers et scoring.
Version 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from yahooquery import search
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Couleurs distinctives pour chaque entreprise
COMPANY_COLORS = [
    '#00D4AA',  # Turquoise
    '#FF6B6B',  # Rouge corail
    '#4ECDC4',  # Cyan
    '#FFE66D',  # Jaune
    '#95E1D3',  # Vert menthe
]

# Périodes disponibles pour l'analyse
PERIODS_CONFIG = {
    '1J': {'period': '1d', 'interval': '5m', 'label': '1 Jour'},
    '1S': {'period': '5d', 'interval': '15m', 'label': '1 Semaine'},
    '1M': {'period': '1mo', 'interval': '1h', 'label': '1 Mois'},
    '3M': {'period': '3mo', 'interval': '1d', 'label': '3 Mois'},
    '6M': {'period': '6mo', 'interval': '1d', 'label': '6 Mois'},
    'YTD': {'period': 'ytd', 'interval': '1d', 'label': 'Année en cours'},
    '1A': {'period': '1y', 'interval': '1d', 'label': '1 An'},
    '3A': {'period': '3y', 'interval': '1wk', 'label': '3 Ans'},
    '5A': {'period': '5y', 'interval': '1wk', 'label': '5 Ans'},
    'MAX': {'period': 'max', 'interval': '1mo', 'label': 'Maximum'},
}

# Catégories de ratios financiers
class RatioCategory(Enum):
    VALORISATION = "Valorisation"
    RENTABILITE = "Rentabilité"
    CROISSANCE = "Croissance"
    SOLIDITE = "Solidité Financière"
    DIVIDENDES = "Dividendes"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CompanyData:
    """Données d'une entreprise pour la comparaison."""
    ticker: str
    name: str
    info: Dict[str, Any] = field(default_factory=dict)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    color: str = '#00D4AA'
    valid: bool = True
    error_message: str = ""


@dataclass
class RatioDefinition:
    """Définition d'un ratio financier."""
    name: str
    key: str
    category: RatioCategory
    format_type: str = "ratio"  # ratio, percent, currency, number
    higher_is_better: bool = True
    description: str = ""
    fallback_keys: List[str] = field(default_factory=list)


# ============================================================================
# RATIO DEFINITIONS
# ============================================================================

RATIO_DEFINITIONS = [
    # Valorisation
    RatioDefinition("PER (Price/Earnings)", "trailingPE", RatioCategory.VALORISATION,
                   "ratio", False, "Cours / Bénéfice par action", ["forwardPE"]),
    RatioDefinition("P/Book Value", "priceToBook", RatioCategory.VALORISATION,
                   "ratio", False, "Cours / Valeur comptable"),
    RatioDefinition("P/Sales", "priceToSalesTrailing12Months", RatioCategory.VALORISATION,
                   "ratio", False, "Cours / Chiffre d'affaires"),
    RatioDefinition("EV/EBITDA", "enterpriseToEbitda", RatioCategory.VALORISATION,
                   "ratio", False, "Valeur d'entreprise / EBITDA"),
    RatioDefinition("EV/Revenue", "enterpriseToRevenue", RatioCategory.VALORISATION,
                   "ratio", False, "Valeur d'entreprise / CA"),
    RatioDefinition("PEG Ratio", "pegRatio", RatioCategory.VALORISATION,
                   "ratio", False, "PER / Croissance bénéfices"),

    # Rentabilité
    RatioDefinition("ROE", "returnOnEquity", RatioCategory.RENTABILITE,
                   "percent", True, "Résultat net / Capitaux propres"),
    RatioDefinition("ROA", "returnOnAssets", RatioCategory.RENTABILITE,
                   "percent", True, "Résultat net / Total actifs"),
    RatioDefinition("Marge brute", "grossMargins", RatioCategory.RENTABILITE,
                   "percent", True, "Marge brute / CA"),
    RatioDefinition("Marge opérationnelle", "operatingMargins", RatioCategory.RENTABILITE,
                   "percent", True, "Résultat opérationnel / CA"),
    RatioDefinition("Marge nette", "profitMargins", RatioCategory.RENTABILITE,
                   "percent", True, "Résultat net / CA"),
    RatioDefinition("Marge EBITDA", "ebitdaMargins", RatioCategory.RENTABILITE,
                   "percent", True, "EBITDA / CA"),

    # Croissance
    RatioDefinition("Croissance CA", "revenueGrowth", RatioCategory.CROISSANCE,
                   "percent", True, "Croissance du chiffre d'affaires"),
    RatioDefinition("Croissance bénéfices", "earningsGrowth", RatioCategory.CROISSANCE,
                   "percent", True, "Croissance des bénéfices"),
    RatioDefinition("Croissance BPA", "earningsQuarterlyGrowth", RatioCategory.CROISSANCE,
                   "percent", True, "Croissance du bénéfice par action"),

    # Solidité financière
    RatioDefinition("Dette/Equity", "debtToEquity", RatioCategory.SOLIDITE,
                   "ratio", False, "Dette totale / Capitaux propres"),
    RatioDefinition("Current Ratio", "currentRatio", RatioCategory.SOLIDITE,
                   "ratio", True, "Actifs courants / Passifs courants"),
    RatioDefinition("Quick Ratio", "quickRatio", RatioCategory.SOLIDITE,
                   "ratio", True, "Liquidités / Passifs courants"),
    RatioDefinition("Cash/Debt", "totalCashPerShare", RatioCategory.SOLIDITE,
                   "ratio", True, "Trésorerie par action"),

    # Dividendes
    RatioDefinition("Dividend Yield", "dividendYield", RatioCategory.DIVIDENDES,
                   "percent", True, "Dividende / Cours"),
    RatioDefinition("Payout Ratio", "payoutRatio", RatioCategory.DIVIDENDES,
                   "percent", False, "Dividendes / Bénéfices"),
    RatioDefinition("Dividende/Action", "dividendRate", RatioCategory.DIVIDENDES,
                   "currency", True, "Dividende annuel par action"),
]


# ============================================================================
# MAIN CLASS: ComparativeAnalysis
# ============================================================================

class ComparativeAnalysis:
    """
    Module d'analyse comparative pour Stock Analyzer Pro.
    Compare jusqu'à 4 entreprises sur performance, ratios et scoring.
    """

    def __init__(self, tickers: List[str], max_companies: int = 4):
        """
        Initialise l'analyse comparative.

        Args:
            tickers: Liste des tickers à comparer (2-4 entreprises)
            max_companies: Nombre maximum d'entreprises (défaut: 4)
        """
        self.tickers = tickers[:max_companies]
        self.max_companies = max_companies
        self.companies: List[CompanyData] = []
        self.valid_companies: List[CompanyData] = []
        self._data_loaded = False

    def load_data(self) -> bool:
        """
        Charge les données pour toutes les entreprises.

        Returns:
            True si au moins 2 entreprises ont été chargées avec succès
        """
        self.companies = []

        for i, ticker in enumerate(self.tickers):
            color = COMPANY_COLORS[i % len(COMPANY_COLORS)]
            company = self._fetch_company_data(ticker, color)
            self.companies.append(company)

        self.valid_companies = [c for c in self.companies if c.valid]
        self._data_loaded = len(self.valid_companies) >= 2

        return self._data_loaded

    def _fetch_company_data(self, ticker: str, color: str) -> CompanyData:
        """Récupère les données d'une entreprise."""
        try:
            tkr = yf.Ticker(ticker)
            info = tkr.info

            if not info or 'symbol' not in info:
                return CompanyData(
                    ticker=ticker,
                    name=ticker,
                    color=color,
                    valid=False,
                    error_message=f"Ticker '{ticker}' introuvable"
                )

            name = info.get('longName') or info.get('shortName') or ticker

            # Récupérer l'historique des prix (1 an par défaut)
            history = tkr.history(period='1y', interval='1d')

            # Récupérer les données financières
            financials = tkr.financials if hasattr(tkr, 'financials') else pd.DataFrame()
            balance_sheet = tkr.balance_sheet if hasattr(tkr, 'balance_sheet') else pd.DataFrame()
            cashflow = tkr.cashflow if hasattr(tkr, 'cashflow') else pd.DataFrame()

            return CompanyData(
                ticker=ticker,
                name=name,
                info=info,
                history=history,
                financials=financials if isinstance(financials, pd.DataFrame) else pd.DataFrame(),
                balance_sheet=balance_sheet if isinstance(balance_sheet, pd.DataFrame) else pd.DataFrame(),
                cashflow=cashflow if isinstance(cashflow, pd.DataFrame) else pd.DataFrame(),
                color=color,
                valid=True
            )

        except Exception as e:
            return CompanyData(
                ticker=ticker,
                name=ticker,
                color=color,
                valid=False,
                error_message=str(e)
            )

    def fetch_price_data(self, period_key: str = "1A") -> Dict[str, pd.DataFrame]:
        """
        Récupère les données de cours pour toutes les entreprises.

        Args:
            period_key: Clé de période (1J, 1S, 1M, 3M, 6M, YTD, 1A, 3A, 5A, MAX)

        Returns:
            Dictionnaire {ticker: DataFrame avec prix}
        """
        config = PERIODS_CONFIG.get(period_key, PERIODS_CONFIG['1A'])
        price_data = {}

        for company in self.valid_companies:
            try:
                tkr = yf.Ticker(company.ticker)
                hist = tkr.history(period=config['period'], interval=config['interval'])
                if not hist.empty:
                    price_data[company.ticker] = hist
            except Exception:
                continue

        return price_data

    def calculate_relative_performance(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calcule la performance relative (base 100) pour toutes les entreprises.

        Args:
            price_data: Dictionnaire des prix par ticker

        Returns:
            DataFrame avec performance relative pour chaque ticker
        """
        if not price_data:
            return pd.DataFrame()

        # Trouver la date de début commune
        min_dates = [df.index.min() for df in price_data.values() if not df.empty]
        max_dates = [df.index.max() for df in price_data.values() if not df.empty]

        if not min_dates:
            return pd.DataFrame()

        start_date = max(min_dates)
        end_date = min(max_dates)

        # Créer DataFrame avec performance relative
        result = pd.DataFrame()

        for ticker, df in price_data.items():
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            if not df_filtered.empty and 'Close' in df_filtered.columns:
                base_price = df_filtered['Close'].iloc[0]
                if base_price > 0:
                    result[ticker] = (df_filtered['Close'] / base_price) * 100

        return result

    def calculate_performance_stats(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Calcule les statistiques de performance pour chaque entreprise.

        Returns:
            Dictionnaire avec performance, volatilité, etc. par ticker
        """
        stats = {}

        for ticker, df in price_data.items():
            if df.empty or 'Close' not in df.columns:
                continue

            close_prices = df['Close']
            returns = close_prices.pct_change().dropna()

            total_return = ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100  # Annualisée
            max_drawdown = self._calculate_max_drawdown(close_prices)

            stats[ticker] = {
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'start_price': close_prices.iloc[0],
                'end_price': close_prices.iloc[-1],
                'high': close_prices.max(),
                'low': close_prices.min()
            }

        return stats

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calcule le drawdown maximum."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min() * 100

    def fetch_financial_ratios(self) -> pd.DataFrame:
        """
        Récupère les ratios financiers pour comparaison.

        Returns:
            DataFrame avec ratios par entreprise et catégorie
        """
        data = []

        for ratio_def in RATIO_DEFINITIONS:
            row = {
                'Ratio': ratio_def.name,
                'Catégorie': ratio_def.category.value,
                'Description': ratio_def.description,
                'higher_is_better': ratio_def.higher_is_better,
                'format_type': ratio_def.format_type
            }

            values = []
            for company in self.valid_companies:
                value = company.info.get(ratio_def.key)

                # Essayer les clés alternatives
                if value is None:
                    for fallback in ratio_def.fallback_keys:
                        value = company.info.get(fallback)
                        if value is not None:
                            break

                row[company.ticker] = value
                if value is not None and not np.isnan(value):
                    values.append(value)

            # Calculer la moyenne
            row['Moyenne'] = np.mean(values) if values else None

            # Déterminer le meilleur
            if values:
                if ratio_def.higher_is_better:
                    best_value = max(values)
                else:
                    best_value = min(values)
                row['best_value'] = best_value
            else:
                row['best_value'] = None

            data.append(row)

        return pd.DataFrame(data)

    def calculate_scores(self) -> Dict[str, Dict]:
        """
        Calcule les scores par catégorie et global pour chaque entreprise.

        Returns:
            Dictionnaire avec scores par entreprise
        """
        ratios_df = self.fetch_financial_ratios()
        scores = {}

        for company in self.valid_companies:
            company_scores = {cat.value: [] for cat in RatioCategory}

            for _, row in ratios_df.iterrows():
                value = row.get(company.ticker)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                category = row['Catégorie']
                higher_is_better = row['higher_is_better']

                # Collecter toutes les valeurs valides pour ce ratio
                all_values = []
                for c in self.valid_companies:
                    v = row.get(c.ticker)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        all_values.append(v)

                if len(all_values) < 2:
                    continue

                # Calculer le percentile (0-100)
                rank = sorted(all_values).index(value)
                percentile = (rank / (len(all_values) - 1)) * 100 if len(all_values) > 1 else 50

                # Inverser si lower is better
                if not higher_is_better:
                    percentile = 100 - percentile

                # Convertir en score 1-5
                score = 1 + (percentile / 100) * 4
                company_scores[category].append(score)

            # Moyennes par catégorie
            category_scores = {}
            for cat, cat_scores in company_scores.items():
                if cat_scores:
                    category_scores[cat] = np.mean(cat_scores)
                else:
                    category_scores[cat] = None

            # Score global
            valid_scores = [s for s in category_scores.values() if s is not None]
            global_score = np.mean(valid_scores) if valid_scores else None

            scores[company.ticker] = {
                'categories': category_scores,
                'global': global_score,
                'name': company.name
            }

        return scores

    def get_earnings_history(self, years: int = 5) -> pd.DataFrame:
        """
        Récupère l'historique des bénéfices pour chaque entreprise.

        Args:
            years: Nombre d'années d'historique

        Returns:
            DataFrame avec bénéfices par année et entreprise
        """
        all_earnings = {}

        for company in self.valid_companies:
            if company.financials.empty:
                continue

            earnings = {}
            for col in company.financials.columns[:years]:
                if isinstance(col, (datetime, pd.Timestamp)):
                    year = col.year
                else:
                    year = col

                # Chercher le résultat net
                net_income = None
                for key in ['Net Income', 'Net Income Common Stockholders',
                           'Net Income From Continuing Operations']:
                    if key in company.financials.index:
                        net_income = company.financials.loc[key, col]
                        break

                if net_income is not None:
                    earnings[year] = net_income

            if earnings:
                all_earnings[company.ticker] = earnings

        if not all_earnings:
            return pd.DataFrame()

        # Créer DataFrame
        df = pd.DataFrame(all_earnings)
        df.index.name = 'Année'
        df = df.sort_index()

        return df

    def get_dividend_history(self) -> Dict[str, pd.DataFrame]:
        """
        Récupère l'historique des dividendes pour chaque entreprise.

        Returns:
            Dictionnaire avec historique dividendes par ticker
        """
        dividends = {}

        for company in self.valid_companies:
            try:
                tkr = yf.Ticker(company.ticker)
                div_hist = tkr.dividends
                if not div_hist.empty:
                    dividends[company.ticker] = div_hist
            except Exception:
                continue

        return dividends

    # =========================================================================
    # PLOTTING METHODS
    # =========================================================================

    def plot_performance_chart(self, period_key: str = "1A",
                               relative: bool = True) -> go.Figure:
        """
        Génère le graphique de performance comparative.

        Args:
            period_key: Clé de période
            relative: Si True, affiche performance base 100

        Returns:
            Figure Plotly
        """
        price_data = self.fetch_price_data(period_key)

        if not price_data:
            return self._empty_figure("Aucune donnée disponible")

        if relative:
            df = self.calculate_relative_performance(price_data)
            y_title = "Performance (base 100)"
        else:
            df = pd.DataFrame({t: d['Close'] for t, d in price_data.items() if 'Close' in d.columns})
            y_title = "Prix"

        if df.empty:
            return self._empty_figure("Aucune donnée disponible")

        stats = self.calculate_performance_stats(price_data)

        fig = go.Figure()

        for company in self.valid_companies:
            if company.ticker not in df.columns:
                continue

            stat = stats.get(company.ticker, {})
            perf = stat.get('total_return', 0)
            perf_str = f"+{perf:.1f}%" if perf >= 0 else f"{perf:.1f}%"

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[company.ticker],
                mode='lines',
                name=f"{company.ticker} ({perf_str})",
                line=dict(color=company.color, width=2.5),
                hovertemplate=(
                    f"<b>{company.name}</b><br>"
                    f"Date: %{{x|%d/%m/%Y}}<br>"
                    f"{'Performance' if relative else 'Prix'}: %{{y:.2f}}"
                    f"{'%' if relative else ''}<br>"
                    "<extra></extra>"
                )
            ))

        # Ligne de référence à 100 si relatif
        if relative:
            fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)")

        period_label = PERIODS_CONFIG.get(period_key, {}).get('label', period_key)

        fig.update_layout(
            title=dict(
                text=f"📈 Performance Boursière Comparée ({period_label})",
                font=dict(size=18, color='white')
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            ),
            xaxis=dict(
                title="",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickformat='%d/%m/%y'
            ),
            yaxis=dict(
                title=y_title,
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            height=500,
            margin=dict(t=80, b=40, l=60, r=40)
        )

        return fig

    def plot_ratios_heatmap(self, category: RatioCategory = None) -> go.Figure:
        """
        Génère un heatmap des ratios financiers.

        Args:
            category: Catégorie spécifique ou None pour toutes

        Returns:
            Figure Plotly
        """
        ratios_df = self.fetch_financial_ratios()

        if category:
            ratios_df = ratios_df[ratios_df['Catégorie'] == category.value]

        if ratios_df.empty:
            return self._empty_figure("Aucun ratio disponible")

        # Préparer les données pour le heatmap
        tickers = [c.ticker for c in self.valid_companies]
        ratio_names = ratios_df['Ratio'].tolist()

        z_values = []
        text_values = []

        for _, row in ratios_df.iterrows():
            z_row = []
            text_row = []

            for ticker in tickers:
                value = row.get(ticker)
                format_type = row['format_type']

                if value is None or (isinstance(value, float) and np.isnan(value)):
                    z_row.append(0.5)  # Valeur neutre
                    text_row.append("N/A")
                else:
                    # Normaliser pour le heatmap
                    all_vals = [row.get(t) for t in tickers
                               if row.get(t) is not None and not np.isnan(row.get(t))]
                    if all_vals:
                        min_val, max_val = min(all_vals), max(all_vals)
                        if max_val != min_val:
                            norm = (value - min_val) / (max_val - min_val)
                        else:
                            norm = 0.5

                        # Inverser si lower is better
                        if not row['higher_is_better']:
                            norm = 1 - norm
                    else:
                        norm = 0.5

                    z_row.append(norm)

                    # Formatage du texte
                    if format_type == 'percent':
                        text_row.append(f"{value*100:.1f}%")
                    elif format_type == 'ratio':
                        text_row.append(f"{value:.2f}x")
                    elif format_type == 'currency':
                        text_row.append(f"{value:.2f}")
                    else:
                        text_row.append(f"{value:.2f}")

            z_values.append(z_row)
            text_values.append(text_row)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=tickers,
            y=ratio_names,
            text=text_values,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            colorscale=[[0, '#FF6B6B'], [0.5, '#FFE66D'], [1, '#00D4AA']],
            showscale=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}: %{text}<br>"
                "<extra></extra>"
            )
        ))

        title = f"📊 Ratios: {category.value}" if category else "📊 Ratios Financiers Comparés"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='white')),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=max(300, len(ratio_names) * 35 + 100),
            margin=dict(t=60, b=40, l=180, r=40),
            xaxis=dict(side='top', tickangle=0),
            yaxis=dict(autorange='reversed')
        )

        return fig

    def plot_radar_chart(self) -> go.Figure:
        """
        Génère le radar chart comparatif des scores par catégorie.

        Returns:
            Figure Plotly
        """
        scores = self.calculate_scores()

        if not scores:
            return self._empty_figure("Aucun score disponible")

        categories = [cat.value for cat in RatioCategory]

        fig = go.Figure()

        for company in self.valid_companies:
            company_scores = scores.get(company.ticker, {}).get('categories', {})

            values = []
            for cat in categories:
                score = company_scores.get(cat)
                values.append(score if score is not None else 0)

            # Fermer le polygone
            values.append(values[0])
            cats_closed = categories + [categories[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=cats_closed,
                fill='toself',
                fillcolor=f"rgba{tuple(list(int(company.color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                line=dict(color=company.color, width=2),
                name=company.ticker,
                hovertemplate=(
                    f"<b>{company.name}</b><br>"
                    "%{theta}: %{r:.1f}/5<br>"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=dict(
                text="🎯 Radar Comparatif",
                font=dict(size=16, color='white')
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['1', '2', '3', '4', '5'],
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                bgcolor='rgba(0,0,0,0.1)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=500,
            margin=dict(t=80, b=80, l=80, r=80)
        )

        return fig

    def plot_earnings_comparison(self, years: int = 5) -> go.Figure:
        """
        Génère le graphique de comparaison des bénéfices.

        Args:
            years: Nombre d'années

        Returns:
            Figure Plotly
        """
        earnings_df = self.get_earnings_history(years)

        if earnings_df.empty:
            return self._empty_figure("Aucune donnée de bénéfices disponible")

        fig = go.Figure()

        bar_width = 0.8 / len(self.valid_companies)

        for i, company in enumerate(self.valid_companies):
            if company.ticker not in earnings_df.columns:
                continue

            values = earnings_df[company.ticker].values
            years_list = earnings_df.index.tolist()

            # Convertir en milliards
            values_b = values / 1e9

            fig.add_trace(go.Bar(
                x=[str(y) for y in years_list],
                y=values_b,
                name=company.ticker,
                marker_color=company.color,
                offsetgroup=i,
                hovertemplate=(
                    f"<b>{company.name}</b><br>"
                    "Année: %{x}<br>"
                    "Bénéfice: %{y:.2f} Mds<br>"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=dict(
                text="💰 Évolution des Bénéfices (en Mds)",
                font=dict(size=16, color='white')
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(
                title="Année",
                showgrid=False
            ),
            yaxis=dict(
                title="Bénéfice net (Mds)",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            height=400,
            margin=dict(t=80, b=40, l=60, r=40)
        )

        return fig

    def plot_dividend_comparison(self) -> go.Figure:
        """
        Génère le graphique de comparaison des dividendes.

        Returns:
            Figure Plotly
        """
        fig = go.Figure()

        for company in self.valid_companies:
            div_yield = company.info.get('dividendYield')
            payout = company.info.get('payoutRatio')
            div_rate = company.info.get('dividendRate')

            if div_yield is not None:
                fig.add_trace(go.Bar(
                    x=[company.ticker],
                    y=[div_yield * 100],
                    name=f"{company.ticker}",
                    marker_color=company.color,
                    text=[f"{div_yield*100:.2f}%"],
                    textposition='outside',
                    hovertemplate=(
                        f"<b>{company.name}</b><br>"
                        f"Rendement: {div_yield*100:.2f}%<br>"
                        f"Dividende/action: {div_rate or 'N/A'}<br>"
                        f"Payout ratio: {payout*100 if payout else 'N/A'}%<br>"
                        "<extra></extra>"
                    )
                ))

        fig.update_layout(
            title=dict(
                text="📈 Rendement du Dividende (%)",
                font=dict(size=16, color='white')
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            showlegend=False,
            xaxis=dict(title=""),
            yaxis=dict(
                title="Dividend Yield (%)",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            height=350,
            margin=dict(t=60, b=40, l=60, r=40)
        )

        return fig

    def generate_scorecard(self) -> Dict[str, Any]:
        """
        Génère le scorecard récapitulatif complet.

        Returns:
            Dictionnaire avec toutes les données du scorecard
        """
        scores = self.calculate_scores()
        price_data = self.fetch_price_data("1A")
        stats = self.calculate_performance_stats(price_data)

        # YTD performance
        ytd_data = self.fetch_price_data("YTD")
        ytd_stats = self.calculate_performance_stats(ytd_data)

        scorecard = {}

        for company in self.valid_companies:
            ticker = company.ticker
            info = company.info
            company_scores = scores.get(ticker, {})
            perf_stats = stats.get(ticker, {})
            ytd_perf = ytd_stats.get(ticker, {})

            scorecard[ticker] = {
                'name': company.name,
                'color': company.color,
                'market_cap': info.get('marketCap'),
                'current_price': info.get('regularMarketPrice') or info.get('currentPrice'),
                'currency': info.get('currency', 'USD'),
                'perf_ytd': ytd_perf.get('total_return'),
                'perf_1y': perf_stats.get('total_return'),
                'volatility': perf_stats.get('volatility'),
                'category_scores': company_scores.get('categories', {}),
                'global_score': company_scores.get('global'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }

        # Classement
        sorted_companies = sorted(
            scorecard.items(),
            key=lambda x: x[1].get('global_score') or 0,
            reverse=True
        )

        for rank, (ticker, data) in enumerate(sorted_companies, 1):
            scorecard[ticker]['rank'] = rank
            if rank == 1:
                scorecard[ticker]['medal'] = '🥇'
            elif rank == 2:
                scorecard[ticker]['medal'] = '🥈'
            elif rank == 3:
                scorecard[ticker]['medal'] = '🥉'
            else:
                scorecard[ticker]['medal'] = ''

        return scorecard

    def generate_insight(self) -> str:
        """
        Génère un insight textuel sur la comparaison.

        Returns:
            Texte d'analyse
        """
        scorecard = self.generate_scorecard()

        if not scorecard:
            return "Données insuffisantes pour générer une analyse."

        # Trouver le meilleur
        best = max(scorecard.items(), key=lambda x: x[1].get('global_score') or 0)
        best_ticker, best_data = best

        # Analyser les forces
        categories = best_data.get('category_scores', {})
        best_category = max(categories.items(), key=lambda x: x[1] or 0) if categories else (None, 0)

        insight = f"💡 **{best_data['name']} ({best_ticker})** "

        if best_data.get('global_score'):
            insight += f"obtient le meilleur score global ({best_data['global_score']:.1f}/5). "

        if best_category[0]:
            insight += f"Son point fort est la **{best_category[0]}** "
            insight += f"(score: {best_category[1]:.1f}/5). "

        perf_1y = best_data.get('perf_1y')
        if perf_1y:
            if perf_1y > 0:
                insight += f"Performance sur 1 an: **+{perf_1y:.1f}%**."
            else:
                insight += f"Performance sur 1 an: **{perf_1y:.1f}%**."

        return insight

    def _empty_figure(self, message: str) -> go.Figure:
        """Crée une figure vide avec un message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='white')
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        return fig

    # =========================================================================
    # CORRELATION METHODS
    # =========================================================================

    def calculate_correlation_matrix(self, period_key: str = "1A") -> pd.DataFrame:
        """
        Calcule la matrice de corrélation entre les entreprises.

        Args:
            period_key: Clé de période pour les données historiques

        Returns:
            DataFrame avec la matrice de corrélation
        """
        if not self.valid_companies:
            return pd.DataFrame()

        # Récupérer les données de prix
        price_data = self.fetch_price_data(period_key)

        # Créer un DataFrame avec les rendements quotidiens
        returns_data = {}

        for company in self.valid_companies:
            if company.ticker in price_data:
                df = price_data[company.ticker]
                if not df.empty and 'Close' in df.columns:
                    # Calculer les rendements quotidiens
                    returns = df['Close'].pct_change().dropna()
                    returns_data[company.ticker] = returns

        if len(returns_data) < 2:
            return pd.DataFrame()

        # Créer un DataFrame aligné
        returns_df = pd.DataFrame(returns_data)

        # Calculer la matrice de corrélation
        correlation_matrix = returns_df.corr()

        return correlation_matrix

    def plot_correlation_matrix(self, period_key: str = "1A") -> go.Figure:
        """
        Affiche la matrice de corrélation sous forme de heatmap.

        Args:
            period_key: Clé de période pour les données historiques

        Returns:
            Figure Plotly avec la heatmap de corrélation
        """
        corr_matrix = self.calculate_correlation_matrix(period_key)

        if corr_matrix.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Données insuffisantes pour calculer la corrélation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Créer les labels avec les noms des entreprises
        labels = []
        for ticker in corr_matrix.columns:
            company = next((c for c in self.valid_companies if c.ticker == ticker), None)
            if company:
                labels.append(f"{ticker}<br>({company.name[:15]}...)" if len(company.name) > 15 else f"{ticker}<br>({company.name})")
            else:
                labels.append(ticker)

        # Créer la heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=labels,
            y=labels,
            colorscale=[
                [0, '#e74c3c'],      # Rouge (corrélation négative)
                [0.5, '#f5f5f5'],    # Blanc (pas de corrélation)
                [1, '#2ecc71']       # Vert (corrélation positive)
            ],
            zmin=-1,
            zmax=1,
            text=[[f"{val:.3f}" for val in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Corrélation: %{z:.4f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="Corrélation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1 (Inverse)", "-0.5", "0 (Aucune)", "0.5", "1 (Parfaite)"]
            )
        ))

        fig.update_layout(
            title=dict(
                text="🔗 Matrice de Corrélation des Rendements",
                font=dict(size=18)
            ),
            xaxis=dict(
                title="",
                tickangle=0,
                side="bottom"
            ),
            yaxis=dict(
                title="",
                autorange="reversed"
            ),
            height=500,
            template='plotly_dark'
        )

        return fig

    def get_correlation_insights(self, period_key: str = "1A") -> List[str]:
        """
        Génère des insights sur les corrélations.

        Returns:
            Liste de strings avec les insights
        """
        corr_matrix = self.calculate_correlation_matrix(period_key)

        if corr_matrix.empty:
            return ["Données insuffisantes pour analyser les corrélations."]

        insights = []
        n = len(corr_matrix)

        # Analyser chaque paire
        pairs_analyzed = set()

        for i, ticker1 in enumerate(corr_matrix.columns):
            for j, ticker2 in enumerate(corr_matrix.columns):
                if i >= j:  # Éviter les doublons et la diagonale
                    continue

                pair_key = tuple(sorted([ticker1, ticker2]))
                if pair_key in pairs_analyzed:
                    continue
                pairs_analyzed.add(pair_key)

                corr = corr_matrix.loc[ticker1, ticker2]

                if corr > 0.8:
                    insights.append(f"🟢 **{ticker1}** et **{ticker2}** sont fortement corrélés ({corr:.2f}). Ils évoluent généralement dans le même sens.")
                elif corr > 0.5:
                    insights.append(f"🔵 **{ticker1}** et **{ticker2}** sont modérément corrélés ({corr:.2f}). Tendance similaire mais avec des divergences.")
                elif corr > -0.5:
                    insights.append(f"⚪ **{ticker1}** et **{ticker2}** ont une faible corrélation ({corr:.2f}). Bonne diversification potentielle.")
                elif corr > -0.8:
                    insights.append(f"🟡 **{ticker1}** et **{ticker2}** sont négativement corrélés ({corr:.2f}). Tendances opposées modérées.")
                else:
                    insights.append(f"🔴 **{ticker1}** et **{ticker2}** sont fortement inversement corrélés ({corr:.2f}). Excellent pour la couverture.")

        # Ajouter un résumé
        all_corrs = []
        for i in range(n):
            for j in range(i+1, n):
                all_corrs.append(corr_matrix.iloc[i, j])

        if all_corrs:
            avg_corr = np.mean(all_corrs)
            if avg_corr > 0.7:
                insights.insert(0, f"📊 **Corrélation moyenne élevée ({avg_corr:.2f})**: Ces actions évoluent souvent ensemble. Diversification limitée.")
            elif avg_corr > 0.3:
                insights.insert(0, f"📊 **Corrélation moyenne modérée ({avg_corr:.2f})**: Mix équilibré entre actions corrélées et indépendantes.")
            else:
                insights.insert(0, f"📊 **Corrélation moyenne faible ({avg_corr:.2f})**: Bonne diversification - les actions évoluent de manière relativement indépendante.")

        return insights

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export_to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Exporte toutes les données en DataFrames.

        Returns:
            Dictionnaire de DataFrames
        """
        exports = {}

        # Ratios
        exports['ratios'] = self.fetch_financial_ratios()

        # Scorecard
        scorecard = self.generate_scorecard()
        scorecard_data = []
        for ticker, data in scorecard.items():
            row = {
                'Ticker': ticker,
                'Nom': data['name'],
                'Capitalisation': data['market_cap'],
                'Prix actuel': data['current_price'],
                'Devise': data['currency'],
                'Perf YTD (%)': data['perf_ytd'],
                'Perf 1A (%)': data['perf_1y'],
                'Volatilité (%)': data['volatility'],
                'Score Global': data['global_score'],
                'Rang': data['rank']
            }
            # Ajouter scores par catégorie
            for cat, score in data.get('category_scores', {}).items():
                row[f'Score {cat}'] = score
            scorecard_data.append(row)

        exports['scorecard'] = pd.DataFrame(scorecard_data)

        # Bénéfices
        exports['earnings'] = self.get_earnings_history()

        return exports


# ============================================================================
# STREAMLIT UI FUNCTIONS
# ============================================================================

def format_market_cap(value: float) -> str:
    """Formate la capitalisation boursière."""
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value/1e12:.2f} T"
    elif value >= 1e9:
        return f"{value/1e9:.2f} Mds"
    elif value >= 1e6:
        return f"{value/1e6:.2f} M"
    else:
        return f"{value:,.0f}"


def format_percent(value: float) -> str:
    """Formate un pourcentage."""
    if value is None:
        return "N/A"
    if value >= 0:
        return f"+{value:.1f}%"
    return f"{value:.1f}%"


def format_score_stars(score: float) -> str:
    """Convertit un score en étoiles."""
    if score is None:
        return "N/A"
    full_stars = int(score)
    half_star = 1 if score - full_stars >= 0.5 else 0
    return "⭐" * full_stars + ("½" if half_star else "")


def render_comparative_analysis_tab(current_ticker: str = None):
    """
    Fonction principale pour afficher l'onglet Analyse Comparative dans Streamlit.

    Args:
        current_ticker: Ticker actuellement analysé (pré-rempli)
    """
    import streamlit as st

    st.markdown("## 📊 Analyse Comparative")
    st.markdown("Comparez jusqu'à 4 entreprises sur leur performance et leurs ratios financiers.")

    # =========================================================================
    # SECTION 1: Sélection des entreprises
    # =========================================================================

    st.markdown("### 🔍 Sélection des entreprises")

    cols = st.columns(4)
    tickers = []

    for i in range(4):
        with cols[i]:
            default_val = current_ticker if i == 0 and current_ticker else ""
            label = f"Entreprise {i+1}" + (" (optionnel)" if i >= 2 else " *")
            ticker = st.text_input(
                label,
                value=default_val,
                key=f"comp_ticker_{i}",
                placeholder="Ex: AAPL, BNP.PA"
            ).strip().upper()
            if ticker:
                tickers.append(ticker)

    col_btn1, col_btn2, _ = st.columns([1, 1, 2])

    with col_btn1:
        compare_btn = st.button("🔄 Comparer", type="primary", use_container_width=True)

    with col_btn2:
        reset_btn = st.button("🗑️ Réinitialiser", use_container_width=True)

    if reset_btn:
        for i in range(4):
            st.session_state[f"comp_ticker_{i}"] = ""
        st.rerun()

    # =========================================================================
    # SECTION 2: Analyse comparative
    # =========================================================================

    if compare_btn or st.session_state.get('comparative_data'):
        if compare_btn:
            if len(tickers) < 2:
                st.error("⚠️ Veuillez sélectionner au moins 2 entreprises pour la comparaison.")
                return

            with st.spinner("🔄 Chargement des données..."):
                analysis = ComparativeAnalysis(tickers)
                success = analysis.load_data()

                if not success:
                    st.error("❌ Impossible de charger les données pour au moins 2 entreprises.")
                    # Afficher les erreurs
                    for company in analysis.companies:
                        if not company.valid:
                            st.warning(f"⚠️ {company.ticker}: {company.error_message}")
                    return

                st.session_state['comparative_data'] = analysis
                st.session_state['comparative_tickers'] = tickers
        else:
            analysis = st.session_state.get('comparative_data')
            if not analysis:
                return

        # Afficher les entreprises validées
        valid_tickers = [c.ticker for c in analysis.valid_companies]
        st.success(f"✅ Comparaison de {len(valid_tickers)} entreprises: {', '.join(valid_tickers)}")

        # Afficher erreurs éventuelles
        for company in analysis.companies:
            if not company.valid:
                st.warning(f"⚠️ {company.ticker}: {company.error_message}")

        # =====================================================================
        # Graphique de performance
        # =====================================================================

        st.markdown("---")
        st.markdown("### 📈 Performance Boursière")

        col_period, col_type = st.columns([3, 2])

        with col_period:
            period_options = list(PERIODS_CONFIG.keys())
            period = st.select_slider(
                "Période",
                options=period_options,
                value="1A",
                key="comp_period"
            )

        with col_type:
            relative = st.radio(
                "Type d'affichage",
                options=["Performance relative (base 100)", "Prix absolu"],
                horizontal=True,
                key="comp_display_type"
            ) == "Performance relative (base 100)"

        # Graphique
        fig_perf = analysis.plot_performance_chart(period, relative)
        st.plotly_chart(fig_perf, use_container_width=True)

        # Stats de performance
        price_data = analysis.fetch_price_data(period)
        stats = analysis.calculate_performance_stats(price_data)

        if stats:
            st.markdown("**📊 Résumé de la période:**")
            stat_cols = st.columns(len(stats))

            for i, (ticker, stat) in enumerate(stats.items()):
                with stat_cols[i]:
                    company = next((c for c in analysis.valid_companies if c.ticker == ticker), None)
                    color = company.color if company else '#FFFFFF'

                    perf = stat.get('total_return', 0)
                    vol = stat.get('volatility', 0)
                    dd = stat.get('max_drawdown', 0)

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}22, {color}11);
                                border-left: 4px solid {color};
                                padding: 15px; border-radius: 8px;">
                        <h4 style="margin:0; color:{color};">{ticker}</h4>
                        <p style="margin:5px 0;">Performance: <b>{'+' if perf >= 0 else ''}{perf:.1f}%</b></p>
                        <p style="margin:5px 0;">Volatilité: {vol:.1f}%</p>
                        <p style="margin:5px 0;">Max Drawdown: {dd:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

        # =====================================================================
        # Ratios financiers
        # =====================================================================

        st.markdown("---")
        st.markdown("### 📊 Ratios Financiers Comparés")

        # Tabs par catégorie
        ratio_tabs = st.tabs([cat.value for cat in RatioCategory])

        for tab, category in zip(ratio_tabs, RatioCategory):
            with tab:
                fig_ratios = analysis.plot_ratios_heatmap(category)
                st.plotly_chart(fig_ratios, use_container_width=True)

        # =====================================================================
        # Graphiques détaillés
        # =====================================================================

        st.markdown("---")
        st.markdown("### 📈 Analyses Détaillées")

        detail_tabs = st.tabs(["🎯 Radar", "💰 Bénéfices", "📈 Dividendes", "🔗 Corrélation"])

        with detail_tabs[0]:
            fig_radar = analysis.plot_radar_chart()
            st.plotly_chart(fig_radar, use_container_width=True)

        with detail_tabs[1]:
            years = st.slider("Nombre d'années", 3, 10, 5, key="earnings_years")
            fig_earnings = analysis.plot_earnings_comparison(years)
            st.plotly_chart(fig_earnings, use_container_width=True)

        with detail_tabs[2]:
            fig_div = analysis.plot_dividend_comparison()
            st.plotly_chart(fig_div, use_container_width=True)

        with detail_tabs[3]:
            st.markdown("#### 🔗 Corrélation des Rendements")
            st.markdown("""
            La matrice de corrélation montre comment les rendements des différentes actions
            évoluent les uns par rapport aux autres. Une corrélation proche de **1** indique
            que les actions évoluent ensemble, proche de **-1** qu'elles évoluent en sens inverse,
            et proche de **0** qu'elles sont indépendantes.
            """)

            corr_period = st.selectbox(
                "Période d'analyse",
                options=["1M", "3M", "6M", "1A", "3A", "5A"],
                index=3,
                key="corr_period",
                help="Période sur laquelle calculer la corrélation"
            )

            fig_corr = analysis.plot_correlation_matrix(corr_period)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Afficher les insights
            st.markdown("#### 💡 Analyse des Corrélations")
            insights = analysis.get_correlation_insights(corr_period)
            for insight in insights:
                st.markdown(insight)

            # Recommandation de diversification
            corr_matrix = analysis.calculate_correlation_matrix(corr_period)
            if not corr_matrix.empty:
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                if avg_corr > 0.7:
                    st.warning("⚠️ **Attention**: Ces actions sont fortement corrélées. Pour une meilleure diversification, envisagez d'ajouter des actifs de secteurs différents.")
                elif avg_corr < 0.3:
                    st.success("✅ **Bonne diversification**: Ces actions ont une corrélation faible, ce qui réduit le risque global du portefeuille.")

        # =====================================================================
        # Scorecard
        # =====================================================================

        st.markdown("---")
        st.markdown("### 🏆 Scorecard Comparatif")

        scorecard = analysis.generate_scorecard()

        if scorecard:
            # Trier par score global
            sorted_items = sorted(
                scorecard.items(),
                key=lambda x: x[1].get('global_score') or 0,
                reverse=True
            )

            score_cols = st.columns(len(sorted_items))

            for i, (ticker, data) in enumerate(sorted_items):
                with score_cols[i]:
                    color = data['color']
                    medal = data.get('medal', '')

                    st.markdown(f"""
                    <div style="background: linear-gradient(180deg, {color}33, {color}11);
                                border: 2px solid {color};
                                border-radius: 12px;
                                padding: 20px;
                                text-align: center;">
                        <h2 style="margin:0; font-size:2.5em;">{medal}</h2>
                        <h3 style="margin:10px 0; color:{color};">{ticker}</h3>
                        <p style="margin:5px 0; font-size:0.9em; opacity:0.8;">{data['name'][:25]}...</p>
                        <hr style="border-color:{color}44;">
                        <p style="margin:5px 0;">Cap: <b>{format_market_cap(data.get('market_cap'))}</b></p>
                        <p style="margin:5px 0;">Prix: <b>{data.get('current_price', 'N/A'):.2f} {data.get('currency', '')}</b></p>
                        <p style="margin:5px 0;">YTD: <b>{format_percent(data.get('perf_ytd'))}</b></p>
                        <p style="margin:5px 0;">1A: <b>{format_percent(data.get('perf_1y'))}</b></p>
                        <hr style="border-color:{color}44;">
                    </div>
                    """, unsafe_allow_html=True)

                    # Scores par catégorie
                    for cat, score in data.get('category_scores', {}).items():
                        if score is not None:
                            st.markdown(f"**{cat[:12]}:** {format_score_stars(score)} ({score:.1f})")

                    # Score global
                    global_score = data.get('global_score')
                    if global_score:
                        st.markdown(f"""
                        <div style="background:{color}; color:white; padding:10px;
                                    border-radius:8px; text-align:center; margin-top:10px;">
                            <b>SCORE: {global_score:.1f}/5</b>
                        </div>
                        """, unsafe_allow_html=True)

            # Insight
            st.markdown("---")
            insight = analysis.generate_insight()
            st.info(insight)

        # =====================================================================
        # Export
        # =====================================================================

        st.markdown("---")
        st.markdown("### 📥 Exporter les données")

        export_cols = st.columns(3)

        with export_cols[0]:
            if st.button("📊 Exporter en CSV", use_container_width=True):
                exports = analysis.export_to_dataframe()
                csv = exports['scorecard'].to_csv(index=False)
                st.download_button(
                    "💾 Télécharger CSV",
                    csv,
                    "comparative_analysis.csv",
                    "text/csv",
                    key="download_csv"
                )


# ============================================================================
# HELPER FUNCTION FOR INTEGRATION
# ============================================================================

def get_comparative_analysis_available() -> bool:
    """Vérifie si le module est disponible."""
    return True


if __name__ == "__main__":
    # Test standalone
    import streamlit as st
    st.set_page_config(page_title="Test Analyse Comparative", layout="wide")
    render_comparative_analysis_tab()
