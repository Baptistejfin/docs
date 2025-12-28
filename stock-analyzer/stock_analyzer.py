"""
Stock Analysis Tool - Analyse financière d'actions boursières
Application Streamlit portable pour analyser n'importe quelle action.
Version 2.0 - Comparaison multi-actifs + Fondamentaux enrichis + Sélection graphique
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import search
import yfinance as yf
from datetime import datetime, timedelta

# ================= CONFIGURATION PAGE =================

st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Analyseur d'Actions Boursières Pro")
st.markdown("---")


# ============ FONCTIONS UTILITAIRES ============

@st.cache_data(ttl=300)
def get_ticker_from_name(name: str) -> str | None:
    """Recherche le symbole boursier à partir du nom de l'entreprise."""
    try:
        name = name.strip()
        if not name:
            return None
        results = search(name).get('quotes')
        if results:
            return results[0]['symbol']
        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_stock_data(ticker_symbol: str, period: str):
    """Récupère les données boursières via yfinance."""
    try:
        tkr = yf.Ticker(ticker_symbol)
        info = tkr.info

        # Pour YTD, utiliser le dernier jour de bourse de l'année précédente comme référence
        if period == 'ytd':
            # Commencer quelques jours avant le 1er janvier pour capturer le dernier cours de l'année précédente
            start_date = datetime(datetime.now().year - 1, 12, 28)
            hist = tkr.history(start=start_date)
        else:
            hist = tkr.history(period=period)

        financials = tkr.financials
        balance_sheet = tkr.balance_sheet
        cashflow = tkr.cashflow
        recommendations = tkr.recommendations_summary
        return info, hist, financials, balance_sheet, cashflow, recommendations
    except Exception:
        return None, None, None, None, None, None


def format_value(val, unit: str = '', decimal: int = 2) -> str:
    """Formate les valeurs numériques en notation lisible (K, M, B)."""
    try:
        if val is None:
            return 'N/A'
        val = float(val)
        if abs(val) >= 1e12:
            formatted = f'{val/1e12:,.{decimal}f} T'
        elif abs(val) >= 1e9:
            formatted = f'{val/1e9:,.{decimal}f} B'
        elif abs(val) >= 1e6:
            formatted = f'{val/1e6:,.{decimal}f} M'
        elif abs(val) >= 1e3:
            formatted = f'{val/1e3:,.{decimal}f} K'
        else:
            formatted = f'{val:,.{decimal}f}'
        return f'{formatted} {unit}'.strip() if unit else formatted
    except (ValueError, TypeError):
        return 'N/A'


def format_percent(val) -> str:
    """Formate un ratio en pourcentage."""
    try:
        if val is None:
            return 'N/A'
        return f"{float(val) * 100:.2f}%"
    except:
        return 'N/A'


def format_ratio(val) -> str:
    """Formate un ratio simple."""
    try:
        if val is None:
            return 'N/A'
        return f"{float(val):.2f}"
    except:
        return 'N/A'


def monte_carlo_simulation(hist: pd.DataFrame, days_forward: int = 252, num_simulations: int = 1000):
    """Simulation Monte Carlo pour projeter les prix futurs."""
    returns = hist['Close'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    last_price = hist['Close'].iloc[-1]

    random_returns = np.random.normal(mean_return, std_return, (days_forward, num_simulations))
    price_paths = last_price * np.cumprod(1 + random_returns, axis=0)

    percentile_5 = np.percentile(price_paths, 5, axis=1)
    percentile_50 = np.percentile(price_paths, 50, axis=1)
    percentile_95 = np.percentile(price_paths, 95, axis=1)

    return price_paths, percentile_5, percentile_50, percentile_95, last_price


def arima_forecast(hist: pd.DataFrame, days_forward: int = 30):
    """Prévision ARIMA des rendements futurs."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        returns = hist['Close'].pct_change().dropna()
        last_price = hist['Close'].iloc[-1]

        best_aic = np.inf
        best_model = None
        best_order = (1, 0, 1)

        for p in range(0, 4):
            for q in range(0, 4):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(returns, order=(p, 0, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_model = fitted
                        best_order = (p, 0, q)
                except Exception:
                    continue

        if best_model is None:
            return None, None, None, None, None

        forecast_res = best_model.get_forecast(steps=days_forward)
        forecast = forecast_res.predicted_mean
        ci = forecast_res.conf_int()

        prices = [last_price]
        upper = [last_price]
        lower = [last_price]

        for i in range(days_forward):
            r = forecast.iloc[i]
            r_low = ci.iloc[i, 0]
            r_high = ci.iloc[i, 1]

            prices.append(prices[-1] * (1 + r))
            upper.append(upper[-1] * (1 + r_high))
            lower.append(lower[-1] * (1 + r_low))

        return np.array(prices[1:]), np.array(upper[1:]), np.array(lower[1:]), best_order, best_aic

    except Exception as e:
        return None, None, None, None, None


def garch_forecast(hist: pd.DataFrame, days_forward: int = 30):
    """Prévision GARCH(1,1) de la volatilité."""
    try:
        from arch import arch_model
        returns = hist['Close'].pct_change().dropna() * 100

        model = arch_model(returns, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
        fitted = model.fit(disp='off')

        forecast = fitted.forecast(horizon=days_forward)
        volatility = np.sqrt(forecast.variance.values[-1, :])

        return volatility, fitted

    except Exception as e:
        return None, None


def arima_garch_combined(hist: pd.DataFrame, days_forward: int = 30):
    """Combinaison ARIMA + GARCH pour la prévision des prix."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from arch import arch_model

        returns = hist['Close'].pct_change().dropna()
        last_price = hist['Close'].iloc[-1]

        best_aic = np.inf
        best_model = None
        best_order = (1, 0, 1)

        for p in range(0, 4):
            for q in range(0, 4):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(returns, order=(p, 0, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_model = fitted
                        best_order = (p, 0, q)
                except Exception:
                    continue

        if best_model is None:
            return None, None, None, None

        arima_fc = best_model.forecast(steps=days_forward)

        returns_pct = returns * 100
        garch = arch_model(returns_pct, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
        garch_fitted = garch.fit(disp='off')
        garch_fc = garch_fitted.forecast(horizon=days_forward)
        vol = np.sqrt(garch_fc.variance.values[-1, :]) / 100

        prices = [last_price]
        upper = [last_price]
        lower = [last_price]

        for i in range(days_forward):
            mu = arima_fc.iloc[i]
            sigma = vol[i]

            prices.append(prices[-1] * (1 + mu))
            upper.append(prices[-1] * (1 + 1.96 * sigma))
            lower.append(prices[-1] * (1 - 1.96 * sigma))

        return np.array(prices[1:]), np.array(upper[1:]), np.array(lower[1:]), best_order

    except Exception as e:
        return None, None, None, None


def get_fundamental_metrics(info, balance_sheet, cashflow):
    """Extrait les métriques fondamentales avancées."""
    metrics = {}

    # Valorisation
    metrics['P/E Ratio'] = info.get('trailingPE')
    metrics['Forward P/E'] = info.get('forwardPE')
    metrics['PEG Ratio'] = info.get('pegRatio')
    metrics['P/B Ratio'] = info.get('priceToBook')
    metrics['P/S Ratio'] = info.get('priceToSalesTrailing12Months')
    metrics['EV/EBITDA'] = info.get('enterpriseToEbitda')
    metrics['EV/Revenue'] = info.get('enterpriseToRevenue')

    # Rentabilité
    metrics['ROE'] = info.get('returnOnEquity')
    metrics['ROA'] = info.get('returnOnAssets')
    metrics['Marge Brute'] = info.get('grossMargins')
    metrics['Marge Opérationnelle'] = info.get('operatingMargins')
    metrics['Marge Nette'] = info.get('profitMargins')

    # Croissance
    metrics['Croissance CA'] = info.get('revenueGrowth')
    metrics['Croissance Bénéfices'] = info.get('earningsGrowth')
    metrics['Croissance CA (5Y)'] = info.get('revenueGrowth')

    # Santé financière
    metrics['Current Ratio'] = info.get('currentRatio')
    metrics['Quick Ratio'] = info.get('quickRatio')
    metrics['Debt/Equity'] = info.get('debtToEquity')

    # Dividendes
    metrics['Dividend Yield'] = info.get('dividendYield')
    metrics['Payout Ratio'] = info.get('payoutRatio')

    # Cash Flow
    metrics['Operating Cash Flow'] = info.get('operatingCashflow')
    metrics['Free Cash Flow'] = info.get('freeCashflow')

    return metrics


def calculate_period_change(hist, start_date, end_date):
    """Calcule le changement entre deux dates."""
    try:
        hist_filtered = hist.loc[start_date:end_date]
        if len(hist_filtered) < 2:
            return None, None, None, None

        start_price = hist_filtered['Close'].iloc[0]
        end_price = hist_filtered['Close'].iloc[-1]
        change = end_price - start_price
        change_pct = (change / start_price) * 100

        return start_price, end_price, change, change_pct
    except:
        return None, None, None, None


def calculate_all_period_returns(ticker_symbol: str):
    """Calcule les rendements pour toutes les périodes standard."""
    periods = {
        'YTD': 'ytd',
        '1 mois': '1mo',
        '3 mois': '3mo',
        '6 mois': '6mo',
        '1 an': '1y',
        '2 ans': '2y',
        '5 ans': '5y'
    }

    results = {}
    tkr = yf.Ticker(ticker_symbol)

    for label, period_code in periods.items():
        try:
            # Pour YTD, utiliser le dernier jour de bourse de l'année précédente comme référence
            if period_code == 'ytd':
                start_date = datetime(datetime.now().year - 1, 12, 28)
                hist = tkr.history(start=start_date)
            else:
                hist = tkr.history(period=period_code)

            if hist is not None and len(hist) >= 2:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                change_pct = ((end_price - start_price) / start_price) * 100
                results[label] = change_pct
            else:
                results[label] = None
        except:
            results[label] = None

    return results


def calculate_dcf(fcf, growth_rate, terminal_growth, discount_rate, projection_years, shares_outstanding, total_debt, total_cash):
    """
    Calcule la valeur intrinsèque par action via le modèle DCF.

    Args:
        fcf: Free Cash Flow actuel
        growth_rate: Taux de croissance du FCF pendant la période de projection (%)
        terminal_growth: Taux de croissance perpétuel après la période de projection (%)
        discount_rate: Taux d'actualisation / WACC (%)
        projection_years: Nombre d'années de projection
        shares_outstanding: Nombre d'actions en circulation
        total_debt: Dette totale
        total_cash: Trésorerie totale

    Returns:
        dict avec tous les détails du calcul DCF
    """
    # Convertir les pourcentages en décimales
    g = growth_rate / 100
    tg = terminal_growth / 100
    r = discount_rate / 100

    # Projeter les FCF futurs
    projected_fcf = []
    pv_fcf = []  # Present Value des FCF

    current_fcf = fcf
    for year in range(1, projection_years + 1):
        future_fcf = current_fcf * (1 + g)
        current_fcf = future_fcf
        projected_fcf.append(future_fcf)

        # Actualiser le FCF
        pv = future_fcf / ((1 + r) ** year)
        pv_fcf.append(pv)

    # Valeur terminale (Gordon Growth Model)
    terminal_fcf = projected_fcf[-1] * (1 + tg)
    terminal_value = terminal_fcf / (r - tg)

    # Actualiser la valeur terminale
    pv_terminal = terminal_value / ((1 + r) ** projection_years)

    # Valeur d'entreprise (Enterprise Value)
    sum_pv_fcf = sum(pv_fcf)
    enterprise_value = sum_pv_fcf + pv_terminal

    # Valeur des capitaux propres (Equity Value)
    equity_value = enterprise_value - total_debt + total_cash

    # Valeur intrinsèque par action
    intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        'projected_fcf': projected_fcf,
        'pv_fcf': pv_fcf,
        'sum_pv_fcf': sum_pv_fcf,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'intrinsic_value': intrinsic_value_per_share
    }


def estimate_wacc(info):
    """
    Estime le WACC (Weighted Average Cost of Capital) basé sur les données disponibles.
    Utilise le modèle CAPM pour le coût des capitaux propres.
    """
    # Taux sans risque (approximation - rendement obligations 10 ans)
    risk_free_rate = 0.04  # 4%

    # Prime de risque du marché
    market_risk_premium = 0.05  # 5%

    # Beta de l'action
    beta = info.get('beta', 1.0) or 1.0

    # Coût des capitaux propres (CAPM)
    cost_of_equity = risk_free_rate + beta * market_risk_premium

    # Coût de la dette (approximation)
    cost_of_debt = 0.05  # 5% par défaut
    tax_rate = 0.25  # Taux d'imposition 25%

    # Structure du capital
    market_cap = info.get('marketCap', 0) or 0
    total_debt = info.get('totalDebt', 0) or 0

    total_capital = market_cap + total_debt

    if total_capital > 0:
        weight_equity = market_cap / total_capital
        weight_debt = total_debt / total_capital

        # WACC
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
    else:
        wacc = cost_of_equity

    return wacc * 100  # Retourner en pourcentage


def estimate_growth_rate(info, cashflow):
    """
    Estime le taux de croissance basé sur les données historiques et les prévisions.
    """
    # Essayer d'utiliser la croissance des revenus
    revenue_growth = info.get('revenueGrowth')
    earnings_growth = info.get('earningsGrowth')

    if revenue_growth and revenue_growth > 0:
        # Utiliser une moyenne pondérée
        base_growth = revenue_growth * 100
    elif earnings_growth and earnings_growth > 0:
        base_growth = earnings_growth * 100
    else:
        base_growth = 5.0  # Défaut conservateur

    # Limiter à des valeurs raisonnables
    return min(max(base_growth, 0), 30)


def simulate_dca(hist: pd.DataFrame, investment_amount: float, frequency: str = 'monthly'):
    """
    Simule une stratégie DCA (Dollar Cost Averaging) sur les données historiques.

    Args:
        hist: DataFrame avec les données historiques (doit avoir une colonne 'Close')
        investment_amount: Montant investi à chaque période
        frequency: Fréquence d'investissement ('weekly', 'bi-weekly', 'monthly')

    Returns:
        dict avec les résultats de la simulation
    """
    # Définir la fréquence
    if frequency == 'weekly':
        freq_days = 7
    elif frequency == 'bi-weekly':
        freq_days = 14
    else:  # monthly
        freq_days = 30

    # Initialiser les variables
    total_invested = 0
    total_shares = 0
    investments = []

    # Parcourir les données historiques
    last_investment_date = None

    for date, row in hist.iterrows():
        price = row['Close']

        # Vérifier si c'est le moment d'investir
        should_invest = False
        if last_investment_date is None:
            should_invest = True
        else:
            days_since_last = (date - last_investment_date).days
            if days_since_last >= freq_days:
                should_invest = True

        if should_invest:
            shares_bought = investment_amount / price
            total_shares += shares_bought
            total_invested += investment_amount
            last_investment_date = date

            # Valeur du portefeuille à cette date
            portfolio_value = total_shares * price

            investments.append({
                'date': date,
                'price': price,
                'shares_bought': shares_bought,
                'total_shares': total_shares,
                'total_invested': total_invested,
                'portfolio_value': portfolio_value,
                'cost_basis': total_invested / total_shares if total_shares > 0 else 0
            })

    # Calculer les résultats finaux
    if len(investments) > 0 and len(hist) > 0:
        final_price = hist['Close'].iloc[-1]
        final_portfolio_value = total_shares * final_price
        total_return = ((final_portfolio_value - total_invested) / total_invested) * 100 if total_invested > 0 else 0
        avg_cost_basis = total_invested / total_shares if total_shares > 0 else 0

        # Simulation Lump Sum (investir tout au début)
        initial_price = hist['Close'].iloc[0]
        lump_sum_shares = total_invested / initial_price
        lump_sum_final_value = lump_sum_shares * final_price
        lump_sum_return = ((lump_sum_final_value - total_invested) / total_invested) * 100 if total_invested > 0 else 0

        return {
            'investments': pd.DataFrame(investments),
            'total_invested': total_invested,
            'total_shares': total_shares,
            'final_value': final_portfolio_value,
            'total_return': total_return,
            'avg_cost_basis': avg_cost_basis,
            'num_investments': len(investments),
            'final_price': final_price,
            'lump_sum_value': lump_sum_final_value,
            'lump_sum_return': lump_sum_return,
            'dca_vs_lump': total_return - lump_sum_return
        }

    return None


# ============ INTERFACE UTILISATEUR ============

with st.sidebar:
    st.header("⚙️ Paramètres")

    # Mode d'analyse
    analysis_mode = st.radio(
        "Mode d'analyse",
        options=['Analyse simple', 'Comparaison'],
        index=0
    )

    st.markdown("---")

    if analysis_mode == 'Analyse simple':
        company_name = st.text_input(
            "Nom de l'entreprise",
            value="Apple",
            placeholder="Ex: Amazon, Apple, Microsoft..."
        )
        companies_to_analyze = [company_name] if company_name else []
    else:
        st.subheader("📊 Entreprises à comparer")
        company_1 = st.text_input("Entreprise 1", value="Apple", key="comp1")
        company_2 = st.text_input("Entreprise 2", value="Microsoft", key="comp2")
        company_3 = st.text_input("Entreprise 3 (optionnel)", value="", key="comp3")
        company_4 = st.text_input("Entreprise 4 (optionnel)", value="", key="comp4")

        companies_to_analyze = [c for c in [company_1, company_2, company_3, company_4] if c.strip()]

        st.markdown("---")
        st.subheader("📊 Indice de référence")
        reference_index = st.selectbox(
            "Comparer avec un indice",
            options=['Aucun', 'S&P 500', 'NASDAQ', 'Dow Jones', 'CAC 40', 'DAX', 'FTSE 100'],
            index=1
        )

        index_symbols = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Dow Jones': '^DJI',
            'CAC 40': '^FCHI',
            'DAX': '^GDAXI',
            'FTSE 100': '^FTSE'
        }

    st.markdown("---")

    period = st.selectbox(
        "Période d'analyse",
        options=['ytd', '6mo', '1y', '2y', '5y', '10y', 'max'],
        index=2,
        format_func=lambda x: {
            'ytd': 'YTD (Depuis 1er janv.)', '6mo': '6 mois', '1y': '1 an', '2y': '2 ans',
            '5y': '5 ans', '10y': '10 ans', 'max': 'Maximum'
        }.get(x, x)
    )

    st.markdown("---")

    # Options de prévision (seulement en mode simple)
    if analysis_mode == 'Analyse simple':
        st.subheader("🔮 Modèle de prévision")
        model_choice = st.selectbox(
            "Choisir le modèle",
            options=['Monte Carlo', 'ARIMA', 'GARCH', 'ARIMA + GARCH'],
            index=0
        )

        if model_choice == 'Monte Carlo':
            mc_simulations = st.slider("Nombre de simulations", 100, 5000, 1000, 100)
            mc_days = st.slider("Jours à projeter", 30, 504, 252, 21)
        else:
            forecast_days = st.slider("Jours de prévision", 5, 90, 30, 5)

    st.markdown("---")
    analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)


# ============ ANALYSE PRINCIPALE ============

# Initialiser le session_state pour conserver les données
if 'all_data' not in st.session_state:
    st.session_state.all_data = {}
if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'current_period' not in st.session_state:
    st.session_state.current_period = None

# Lancer l'analyse si le bouton est cliqué
if analyze_button and companies_to_analyze:
    # Réinitialiser les données
    st.session_state.all_data = {}
    st.session_state.valid_tickers = []
    st.session_state.current_period = period

    for company in companies_to_analyze:
        with st.spinner(f"Recherche de {company}..."):
            ticker_symbol = get_ticker_from_name(company)

        if ticker_symbol:
            with st.spinner(f"Chargement des données pour {ticker_symbol}..."):
                info, hist, financials, balance_sheet, cashflow, recommendations = get_stock_data(ticker_symbol, period)

            if info is not None and hist is not None and not hist.empty:
                st.session_state.all_data[ticker_symbol] = {
                    'name': info.get('shortName', ticker_symbol),
                    'info': info,
                    'hist': hist,
                    'financials': financials,
                    'balance_sheet': balance_sheet,
                    'cashflow': cashflow,
                    'recommendations': recommendations
                }
                st.session_state.valid_tickers.append(ticker_symbol)
            else:
                st.warning(f"⚠️ Données indisponibles pour {company}")
        else:
            st.warning(f"⚠️ Entreprise '{company}' non trouvée")

    if not st.session_state.valid_tickers:
        st.error("Aucune entreprise valide trouvée.")
        st.session_state.analysis_done = False
        st.stop()
    else:
        st.session_state.analysis_done = True

# Afficher les résultats si l'analyse a été effectuée
if st.session_state.analysis_done and st.session_state.valid_tickers:
    all_data = st.session_state.all_data
    valid_tickers = st.session_state.valid_tickers

    # ==================== MODE COMPARAISON ====================
    if analysis_mode == 'Comparaison' and len(valid_tickers) >= 2:
        st.header("📊 Comparaison des Entreprises")

        # Tableau comparatif des indicateurs clés
        st.subheader("🔑 Indicateurs Clés")

        comparison_data = []
        for ticker in valid_tickers:
            data = all_data[ticker]
            info = data['info']
            metrics = get_fundamental_metrics(info, data['balance_sheet'], data['cashflow'])

            comparison_data.append({
                'Entreprise': f"{data['name']} ({ticker})",
                'Prix': f"${info.get('regularMarketPrice', 0):.2f}",
                'Market Cap': format_value(info.get('marketCap'), 'USD'),
                'P/E': format_ratio(metrics['P/E Ratio']),
                'PEG': format_ratio(metrics['PEG Ratio']),
                'P/B': format_ratio(metrics['P/B Ratio']),
                'EV/EBITDA': format_ratio(metrics['EV/EBITDA']),
                'ROE': format_percent(metrics['ROE']),
                'ROA': format_percent(metrics['ROA']),
                'Marge Nette': format_percent(metrics['Marge Nette']),
                'Debt/Equity': format_ratio(metrics['Debt/Equity']),
                'Dividend Yield': format_percent(metrics['Dividend Yield']),
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Graphique comparatif des cours (normalisé à 100)
        st.subheader("📈 Évolution Comparée des Cours (Base 100)")

        fig_compare = go.Figure()

        # Charger l'indice de référence si sélectionné
        index_hist = None
        if reference_index != 'Aucun':
            try:
                index_symbol = index_symbols[reference_index]
                # Pour YTD, utiliser le dernier jour de bourse de l'année précédente comme référence
                if period == 'ytd':
                    start_date = datetime(datetime.now().year - 1, 12, 28)
                    index_hist = yf.Ticker(index_symbol).history(start=start_date)
                else:
                    index_hist = yf.Ticker(index_symbol).history(period=period)
            except Exception:
                st.warning(f"Impossible de charger l'indice {reference_index}")

        for ticker in valid_tickers:
            data = all_data[ticker]
            hist = data['hist']
            # Normalisation à base 100
            normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100

            fig_compare.add_trace(go.Scatter(
                x=hist.index,
                y=normalized,
                mode='lines',
                name=f"{data['name']} ({ticker})",
                line=dict(width=2)
            ))

        # Ajouter l'indice de référence au graphique
        if index_hist is not None and not index_hist.empty:
            index_normalized = (index_hist['Close'] / index_hist['Close'].iloc[0]) * 100
            fig_compare.add_trace(go.Scatter(
                x=index_hist.index,
                y=index_normalized,
                mode='lines',
                name=f"{reference_index}",
                line=dict(width=2, dash='dash', color='white')
            ))

        fig_compare.update_layout(
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Performance (Base 100)',
            hovermode='x unified',
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # Calcul des performances
        st.subheader("📊 Performance sur la Période")

        # Calculer le nombre de colonnes (entreprises + indice si sélectionné)
        num_cols = len(valid_tickers) + (1 if index_hist is not None and not index_hist.empty else 0)
        perf_cols = st.columns(num_cols)

        for idx, ticker in enumerate(valid_tickers):
            data = all_data[ticker]
            hist = data['hist']
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            perf = ((end_price - start_price) / start_price) * 100

            with perf_cols[idx]:
                st.metric(
                    f"{data['name']}",
                    f"${end_price:.2f}",
                    delta=f"{perf:.2f}%"
                )

        # Ajouter la performance de l'indice de référence
        if index_hist is not None and not index_hist.empty:
            idx_start = index_hist['Close'].iloc[0]
            idx_end = index_hist['Close'].iloc[-1]
            idx_perf = ((idx_end - idx_start) / idx_start) * 100

            with perf_cols[-1]:
                st.metric(
                    f"{reference_index}",
                    f"{idx_end:.2f}",
                    delta=f"{idx_perf:.2f}%"
                )

        st.markdown("---")

        # ===== HEATMAP DE CORRÉLATION =====
        st.subheader("🔥 Matrice de Corrélation")

        # Construire un DataFrame avec les rendements de chaque action
        returns_df = pd.DataFrame()

        for ticker in valid_tickers:
            data = all_data[ticker]
            hist = data['hist']
            returns = hist['Close'].pct_change().dropna()
            returns_df[data['name']] = returns

        # Ajouter l'indice de référence si sélectionné
        if reference_index != 'Aucun':
            try:
                index_symbol = index_symbols[reference_index]
                # Pour YTD, utiliser le dernier jour de bourse de l'année précédente comme référence
                if period == 'ytd':
                    start_date = datetime(datetime.now().year - 1, 12, 28)
                    index_data = yf.Ticker(index_symbol).history(start=start_date)
                else:
                    index_data = yf.Ticker(index_symbol).history(period=period)

                if not index_data.empty:
                    index_returns = index_data['Close'].pct_change().dropna()
                    # Aligner les dates
                    common_dates = returns_df.index.intersection(index_returns.index)
                    returns_df = returns_df.loc[common_dates]
                    returns_df[reference_index] = index_returns.loc[common_dates]
            except Exception:
                st.warning(f"Impossible de charger l'indice {reference_index}")

        # Calculer la matrice de corrélation
        correlation_matrix = returns_df.corr()

        # Créer la heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdYlGn',
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 14},
            hoverongaps=False,
            colorbar=dict(title='Corrélation')
        ))

        fig_heatmap.update_layout(
            title='Corrélation des Rendements Journaliers',
            template='plotly_dark',
            height=450,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Interprétation
        with st.expander("📖 Comment interpréter la corrélation ?"):
            st.markdown("""
            - **+1.00** : Corrélation parfaite positive (évoluent exactement ensemble)
            - **+0.70 à +0.99** : Forte corrélation positive
            - **+0.40 à +0.69** : Corrélation modérée positive
            - **+0.10 à +0.39** : Faible corrélation positive
            - **-0.10 à +0.10** : Pas de corrélation significative
            - **-0.40 à -0.10** : Faible corrélation négative
            - **-0.70 à -0.40** : Corrélation modérée négative
            - **-1.00 à -0.70** : Forte corrélation négative

            **Pour la diversification** : Choisissez des actions avec une faible corrélation entre elles pour réduire le risque global du portefeuille.
            """)

        # Afficher le Beta si un indice est sélectionné
        if reference_index != 'Aucun' and reference_index in returns_df.columns:
            st.markdown("---")
            st.subheader(f"📈 Beta par rapport au {reference_index}")

            beta_cols = st.columns(len(valid_tickers))
            index_var = returns_df[reference_index].var()

            for idx, ticker in enumerate(valid_tickers):
                name = all_data[ticker]['name']
                if name in returns_df.columns:
                    cov = returns_df[name].cov(returns_df[reference_index])
                    beta = cov / index_var if index_var != 0 else 0

                    with beta_cols[idx]:
                        if beta > 1:
                            interpretation = "Plus volatile que le marché"
                            delta_color = "normal"
                        elif beta < 1:
                            interpretation = "Moins volatile que le marché"
                            delta_color = "inverse"
                        else:
                            interpretation = "Suit le marché"
                            delta_color = "off"

                        st.metric(
                            f"{name}",
                            f"β = {beta:.2f}",
                            delta=interpretation,
                            delta_color=delta_color
                        )

        st.markdown("---")

        # Comparaison des fondamentaux en graphiques
        st.subheader("📊 Comparaison des Ratios de Valorisation")

        # Données pour les graphiques
        tickers_names = [all_data[t]['name'] for t in valid_tickers]

        pe_values = []
        pb_values = []
        roe_values = []
        margin_values = []

        for ticker in valid_tickers:
            info = all_data[ticker]['info']
            pe_values.append(info.get('trailingPE') or 0)
            pb_values.append(info.get('priceToBook') or 0)
            roe_values.append((info.get('returnOnEquity') or 0) * 100)
            margin_values.append((info.get('profitMargins') or 0) * 100)

        fig_ratios = make_subplots(rows=2, cols=2, subplot_titles=('P/E Ratio', 'P/B Ratio', 'ROE (%)', 'Marge Nette (%)'))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig_ratios.add_trace(go.Bar(x=tickers_names, y=pe_values, marker_color=colors[:len(tickers_names)], showlegend=False), row=1, col=1)
        fig_ratios.add_trace(go.Bar(x=tickers_names, y=pb_values, marker_color=colors[:len(tickers_names)], showlegend=False), row=1, col=2)
        fig_ratios.add_trace(go.Bar(x=tickers_names, y=roe_values, marker_color=colors[:len(tickers_names)], showlegend=False), row=2, col=1)
        fig_ratios.add_trace(go.Bar(x=tickers_names, y=margin_values, marker_color=colors[:len(tickers_names)], showlegend=False), row=2, col=2)

        fig_ratios.update_layout(template='plotly_dark', height=500, showlegend=False)
        st.plotly_chart(fig_ratios, use_container_width=True)

    # ==================== MODE ANALYSE SIMPLE ====================
    else:
        ticker = valid_tickers[0]
        data = all_data[ticker]
        info = data['info']
        hist = data['hist']
        financials = data['financials']
        recommendations = data['recommendations']

        # En-tête
        st.header(f"🏢 {data['name']} ({ticker})")

        if info.get('longBusinessSummary'):
            with st.expander("📋 Description de l'entreprise", expanded=False):
                st.write(info['longBusinessSummary'])

        # Indicateurs clés
        st.subheader("📊 Indicateurs Clés")

        price = info.get('regularMarketPrice')
        eps = info.get('trailingEps')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Prix actuel", f"${price:.2f}" if price else "N/A")
        with col2:
            pe = price / eps if price and eps else None
            st.metric("📈 P/E Ratio", f"{pe:.2f}" if pe else "N/A")
        with col3:
            st.metric("🏦 Market Cap", format_value(info.get('marketCap'), 'USD'))
        with col4:
            change = info.get('regularMarketChangePercent')
            st.metric("📊 Variation jour", f"{change:.2f}%" if change else "N/A",
                      delta=f"{change:.2f}%" if change else None)

        st.markdown("---")

        # ===== FONDAMENTAUX ENRICHIS =====
        st.subheader("📈 Analyse Fondamentale Complète")

        metrics = get_fundamental_metrics(info, data['balance_sheet'], data['cashflow'])

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**📘 Valorisation**")
            valuation_data = {
                'Indicateur': ['P/E Ratio', 'Forward P/E', 'PEG Ratio', 'P/B Ratio', 'P/S Ratio', 'EV/EBITDA', 'EV/Revenue'],
                'Valeur': [
                    format_ratio(metrics['P/E Ratio']),
                    format_ratio(metrics['Forward P/E']),
                    format_ratio(metrics['PEG Ratio']),
                    format_ratio(metrics['P/B Ratio']),
                    format_ratio(metrics['P/S Ratio']),
                    format_ratio(metrics['EV/EBITDA']),
                    format_ratio(metrics['EV/Revenue']),
                ]
            }
            st.dataframe(pd.DataFrame(valuation_data), hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**📙 Rentabilité**")
            profitability_data = {
                'Indicateur': ['ROE', 'ROA', 'Marge Brute', 'Marge Opérationnelle', 'Marge Nette'],
                'Valeur': [
                    format_percent(metrics['ROE']),
                    format_percent(metrics['ROA']),
                    format_percent(metrics['Marge Brute']),
                    format_percent(metrics['Marge Opérationnelle']),
                    format_percent(metrics['Marge Nette']),
                ]
            }
            st.dataframe(pd.DataFrame(profitability_data), hide_index=True, use_container_width=True)

        with col3:
            st.markdown("**📗 Santé Financière**")
            health_data = {
                'Indicateur': ['Current Ratio', 'Quick Ratio', 'Debt/Equity', 'Dette Totale', 'Trésorerie'],
                'Valeur': [
                    format_ratio(metrics['Current Ratio']),
                    format_ratio(metrics['Quick Ratio']),
                    format_ratio(metrics['Debt/Equity']),
                    format_value(info.get('totalDebt'), 'USD'),
                    format_value(info.get('totalCash'), 'USD'),
                ]
            }
            st.dataframe(pd.DataFrame(health_data), hide_index=True, use_container_width=True)

        with col4:
            st.markdown("**📕 Dividendes & Cash Flow**")
            dividend_data = {
                'Indicateur': ['Dividend Yield', 'Payout Ratio', 'Operating CF', 'Free Cash Flow', 'FCF Yield'],
                'Valeur': [
                    format_percent(metrics['Dividend Yield']),
                    format_percent(metrics['Payout Ratio']),
                    format_value(metrics['Operating Cash Flow'], 'USD'),
                    format_value(metrics['Free Cash Flow'], 'USD'),
                    format_percent(info.get('freeCashflow', 0) / info.get('marketCap', 1) if info.get('marketCap') else None),
                ]
            }
            st.dataframe(pd.DataFrame(dividend_data), hide_index=True, use_container_width=True)

        st.markdown("---")

        # ===== CALCULATEUR DCF =====
        st.subheader("💰 Valorisation DCF (Discounted Cash Flow)")

        # Récupérer les données nécessaires pour le DCF
        fcf = info.get('freeCashflow', 0) or 0
        shares_outstanding = info.get('sharesOutstanding', 0) or 0
        total_debt = info.get('totalDebt', 0) or 0
        total_cash = info.get('totalCash', 0) or 0
        current_price = info.get('regularMarketPrice', 0) or 0

        # Vérifier si le DCF est possible
        if fcf > 0 and shares_outstanding > 0:
            # Estimer les valeurs par défaut
            default_wacc = estimate_wacc(info)
            default_growth = estimate_growth_rate(info, data['cashflow'])

            with st.expander("⚙️ Paramètres du modèle DCF (ajustables)", expanded=True):
                st.markdown("**Ajustez les paramètres selon vos hypothèses :**")

                col_dcf1, col_dcf2 = st.columns(2)

                with col_dcf1:
                    # FCF de base (modifiable)
                    dcf_fcf = st.number_input(
                        "💵 Free Cash Flow de base (USD)",
                        value=float(fcf),
                        min_value=0.0,
                        step=1000000.0,
                        format="%.0f",
                        help="FCF actuel de l'entreprise. Vous pouvez l'ajuster selon vos estimations.",
                        key="dcf_fcf"
                    )

                    # Taux de croissance
                    dcf_growth = st.slider(
                        "📈 Taux de croissance annuel (%)",
                        min_value=0.0,
                        max_value=30.0,
                        value=min(default_growth, 25.0),
                        step=0.5,
                        help="Croissance estimée du FCF pendant la période de projection",
                        key="dcf_growth"
                    )

                    # Nombre d'années
                    dcf_years = st.slider(
                        "📅 Années de projection",
                        min_value=5,
                        max_value=15,
                        value=10,
                        step=1,
                        help="Nombre d'années pour projeter les flux de trésorerie",
                        key="dcf_years"
                    )

                with col_dcf2:
                    # Taux d'actualisation (WACC)
                    dcf_discount = st.slider(
                        "🏦 Taux d'actualisation / WACC (%)",
                        min_value=5.0,
                        max_value=20.0,
                        value=min(max(default_wacc, 6.0), 15.0),
                        step=0.25,
                        help="Coût moyen pondéré du capital (WACC)",
                        key="dcf_discount"
                    )

                    # Taux de croissance terminal
                    dcf_terminal = st.slider(
                        "♾️ Taux de croissance perpétuel (%)",
                        min_value=0.0,
                        max_value=4.0,
                        value=2.5,
                        step=0.25,
                        help="Croissance à long terme (généralement 2-3%)",
                        key="dcf_terminal"
                    )

                    # Marge de sécurité
                    dcf_margin = st.slider(
                        "🛡️ Marge de sécurité (%)",
                        min_value=0,
                        max_value=50,
                        value=20,
                        step=5,
                        help="Réduction appliquée à la valeur intrinsèque pour plus de prudence",
                        key="dcf_margin"
                    )

                # Afficher les données utilisées
                st.markdown("---")
                st.markdown("**📊 Données de l'entreprise :**")
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("Actions en circulation", format_value(shares_outstanding))
                with col_info2:
                    st.metric("Dette totale", format_value(total_debt, 'USD'))
                with col_info3:
                    st.metric("Trésorerie", format_value(total_cash, 'USD'))
                with col_info4:
                    st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A")

            # Calculer le DCF
            if dcf_discount > dcf_terminal:  # Éviter division par zéro
                dcf_result = calculate_dcf(
                    fcf=dcf_fcf,
                    growth_rate=dcf_growth,
                    terminal_growth=dcf_terminal,
                    discount_rate=dcf_discount,
                    projection_years=dcf_years,
                    shares_outstanding=shares_outstanding,
                    total_debt=total_debt,
                    total_cash=total_cash
                )

                intrinsic_value = dcf_result['intrinsic_value']
                intrinsic_with_margin = intrinsic_value * (1 - dcf_margin / 100)

                # Calcul du potentiel
                if current_price > 0:
                    upside = ((intrinsic_value - current_price) / current_price) * 100
                    upside_with_margin = ((intrinsic_with_margin - current_price) / current_price) * 100
                else:
                    upside = 0
                    upside_with_margin = 0

                # Affichage des résultats
                st.markdown("### 🎯 Résultats de la Valorisation DCF")

                col_res1, col_res2, col_res3 = st.columns(3)

                with col_res1:
                    st.metric(
                        "💵 Prix actuel",
                        f"${current_price:.2f}"
                    )

                with col_res2:
                    delta_color = "normal" if upside >= 0 else "inverse"
                    st.metric(
                        "🎯 Valeur intrinsèque",
                        f"${intrinsic_value:.2f}",
                        delta=f"{upside:+.1f}%",
                        delta_color=delta_color
                    )

                with col_res3:
                    st.metric(
                        f"🛡️ Avec marge {dcf_margin}%",
                        f"${intrinsic_with_margin:.2f}",
                        delta=f"{upside_with_margin:+.1f}%",
                        delta_color="normal" if upside_with_margin >= 0 else "inverse"
                    )

                # Indicateur visuel
                if upside >= 20:
                    st.success(f"🟢 **SOUS-ÉVALUÉ** - Potentiel de hausse de {upside:.1f}% selon ce modèle DCF")
                elif upside >= 0:
                    st.info(f"🟡 **CORRECTEMENT VALORISÉ** - Proche de la valeur intrinsèque estimée")
                else:
                    st.warning(f"🔴 **SURÉVALUÉ** - Le prix actuel est {abs(upside):.1f}% au-dessus de la valeur DCF estimée")

                # Détail des calculs dans un expander
                with st.expander("📋 Détail des calculs DCF"):
                    # Tableau des FCF projetés
                    years = list(range(1, dcf_years + 1))
                    df_fcf = pd.DataFrame({
                        'Année': years,
                        'FCF Projeté': [format_value(v, 'USD') for v in dcf_result['projected_fcf']],
                        'FCF Actualisé': [format_value(v, 'USD') for v in dcf_result['pv_fcf']]
                    })
                    st.dataframe(df_fcf, hide_index=True, use_container_width=True)

                    st.markdown("---")

                    col_detail1, col_detail2 = st.columns(2)
                    with col_detail1:
                        st.markdown("**Valeur des flux actualisés:**")
                        st.write(f"Somme des FCF actualisés: {format_value(dcf_result['sum_pv_fcf'], 'USD')}")
                        st.write(f"Valeur terminale: {format_value(dcf_result['terminal_value'], 'USD')}")
                        st.write(f"Valeur terminale actualisée: {format_value(dcf_result['pv_terminal'], 'USD')}")

                    with col_detail2:
                        st.markdown("**Valeur de l'entreprise:**")
                        st.write(f"Enterprise Value: {format_value(dcf_result['enterprise_value'], 'USD')}")
                        st.write(f"- Dette: {format_value(total_debt, 'USD')}")
                        st.write(f"+ Trésorerie: {format_value(total_cash, 'USD')}")
                        st.write(f"**Equity Value: {format_value(dcf_result['equity_value'], 'USD')}**")

                # Avertissement
                st.caption("⚠️ Ce modèle DCF est une estimation basée sur des hypothèses. Les résultats dépendent fortement des paramètres choisis. Faites vos propres recherches avant d'investir.")

            else:
                st.error("⚠️ Le taux d'actualisation doit être supérieur au taux de croissance perpétuel.")

        else:
            st.warning("⚠️ Données insuffisantes pour calculer le DCF (Free Cash Flow négatif ou non disponible).")
            if fcf <= 0:
                st.info(f"Free Cash Flow actuel: {format_value(fcf, 'USD')} - Le DCF nécessite un FCF positif.")

        st.markdown("---")

        # ===== GRAPHIQUE INTERACTIF AVEC SÉLECTION =====
        st.subheader(f"📈 Évolution du Cours - {period}")
        st.info("💡 **Astuce:** Utilisez les sélecteurs de date ci-dessous pour calculer la variation entre deux points")

        # Sélecteurs de dates
        col_date1, col_date2, col_calc = st.columns([2, 2, 1])

        min_date = hist.index.min().date()
        max_date = hist.index.max().date()

        with col_date1:
            start_date = st.date_input(
                "📅 Date de début",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="start_date"
            )

        with col_date2:
            end_date = st.date_input(
                "📅 Date de fin",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date"
            )

        # Calcul de la variation
        if start_date and end_date and start_date < end_date:
            start_price, end_price, change, change_pct = calculate_period_change(
                hist,
                pd.Timestamp(start_date),
                pd.Timestamp(end_date)
            )

            if start_price is not None:
                st.markdown("### 📊 Variation sur la période sélectionnée")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Prix début", f"${start_price:.2f}")
                with col_m2:
                    st.metric("Prix fin", f"${end_price:.2f}")
                with col_m3:
                    st.metric("Variation ($)", f"${change:.2f}", delta=f"{change_pct:.2f}%")
                with col_m4:
                    st.metric("Variation (%)", f"{change_pct:.2f}%")

        # Graphique avec zone sélectionnée
        fig_price = go.Figure()

        fig_price.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            mode='lines',
            name='Prix de clôture',
            line=dict(color='#1f77b4', width=2)
        ))

        # Moyenne mobile 50 jours
        if len(hist) > 50:
            ma50 = hist['Close'].rolling(window=50).mean()
            fig_price.add_trace(go.Scatter(
                x=hist.index,
                y=ma50,
                mode='lines',
                name='MM 50 jours',
                line=dict(color='orange', width=1, dash='dash')
            ))

        # Moyenne mobile 200 jours
        if len(hist) > 200:
            ma200 = hist['Close'].rolling(window=200).mean()
            fig_price.add_trace(go.Scatter(
                x=hist.index,
                y=ma200,
                mode='lines',
                name='MM 200 jours',
                line=dict(color='red', width=1, dash='dot')
            ))

        # Marquer les points sélectionnés
        if start_date and end_date and start_price is not None:
            fig_price.add_trace(go.Scatter(
                x=[pd.Timestamp(start_date), pd.Timestamp(end_date)],
                y=[start_price, end_price],
                mode='markers+lines',
                name='Période sélectionnée',
                marker=dict(size=12, color='yellow', symbol='star'),
                line=dict(color='yellow', width=2, dash='dash')
            ))

            # Zone colorée
            fig_price.add_vrect(
                x0=pd.Timestamp(start_date),
                x1=pd.Timestamp(end_date),
                fillcolor="yellow",
                opacity=0.1,
                line_width=0
            )

        fig_price.update_layout(
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Prix (USD)',
            hovermode='x unified',
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        st.plotly_chart(fig_price, use_container_width=True)

        # ===== PERFORMANCE DE LA PÉRIODE SÉLECTIONNÉE =====
        st.subheader("📈 Performance sur la Période")

        # Calculer la performance sur la période d'analyse complète
        period_start_price = hist['Close'].iloc[0]
        period_end_price = hist['Close'].iloc[-1]
        period_change = period_end_price - period_start_price
        period_change_pct = ((period_end_price - period_start_price) / period_start_price) * 100

        # Afficher le nom de la période
        period_names = {
            'ytd': 'YTD (Depuis 1er janv.)', '6mo': '6 mois', '1y': '1 an', '2y': '2 ans',
            '5y': '5 ans', '10y': '10 ans', 'max': 'Maximum'
        }
        period_display = period_names.get(period, period)

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.metric(f"📅 Période: {period_display}", f"${period_start_price:.2f}", delta="Prix début")
        with col_p2:
            st.metric("Prix actuel", f"${period_end_price:.2f}")
        with col_p3:
            color = "normal" if period_change_pct >= 0 else "inverse"
            st.metric("Variation ($)", f"${period_change:+.2f}", delta=f"{period_change_pct:+.2f}%", delta_color=color)
        with col_p4:
            st.metric(
                "Performance",
                f"{period_change_pct:+.2f}%",
                delta=f"{'↑ Hausse' if period_change_pct >= 0 else '↓ Baisse'}",
                delta_color="normal" if period_change_pct >= 0 else "inverse"
            )

        # ===== MODÈLES DE PRÉVISION =====
        if len(hist) > 30:
            st.markdown("---")

            if model_choice == 'Monte Carlo':
                st.subheader("🎲 Simulation Monte Carlo - Projection des Prix")

                with st.spinner("Exécution de la simulation Monte Carlo..."):
                    simulations, p5, p50, p95, last_price = monte_carlo_simulation(
                        hist, days_forward=mc_days, num_simulations=mc_simulations
                    )

                last_date = hist.index[-1]
                future_dates = pd.bdate_range(start=last_date, periods=mc_days + 1)[1:]

                fig_mc = go.Figure()

                sample_size = min(100, mc_simulations)
                indices = np.linspace(0, mc_simulations - 1, sample_size, dtype=int)
                for i in indices:
                    fig_mc.add_trace(go.Scatter(
                        x=future_dates, y=simulations[:, i], mode='lines',
                        line=dict(width=0.3, color='rgba(150,150,150,0.3)'),
                        showlegend=False, hoverinfo='skip'
                    ))

                fig_mc.add_trace(go.Scatter(
                    x=future_dates, y=p95, mode='lines', name='95e Percentile (Optimiste)',
                    line=dict(color='rgba(0,255,0,0.7)', width=2, dash='dash')
                ))
                fig_mc.add_trace(go.Scatter(
                    x=future_dates, y=p50, mode='lines', name='Médiane (50e)',
                    line=dict(color='#FFD700', width=3)
                ))
                fig_mc.add_trace(go.Scatter(
                    x=future_dates, y=p5, mode='lines', name='5e Percentile (Pessimiste)',
                    line=dict(color='rgba(255,0,0,0.7)', width=2, dash='dash')
                ))
                fig_mc.add_trace(go.Scatter(
                    x=[last_date], y=[last_price], mode='markers', name='Prix actuel',
                    marker=dict(size=12, color='cyan', symbol='star')
                ))

                fig_mc.update_layout(
                    title=f"Projection sur {mc_days} jours ({mc_simulations} scénarios)",
                    template='plotly_dark', xaxis_title='Date',
                    yaxis_title='Prix (USD)', hovermode='x unified', height=500
                )
                st.plotly_chart(fig_mc, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Prix Médian Projeté", f"${p50[-1]:.2f}",
                              delta=f"{((p50[-1]/last_price - 1)*100):.2f}%")
                with col2:
                    st.metric("🟢 Scénario Optimiste (95%)", f"${p95[-1]:.2f}",
                              delta=f"{((p95[-1]/last_price - 1)*100):.2f}%")
                with col3:
                    st.metric("🔴 Scénario Pessimiste (5%)", f"${p5[-1]:.2f}",
                              delta=f"{((p5[-1]/last_price - 1)*100):.2f}%")

            elif model_choice == 'ARIMA':
                st.subheader("📉 Prévision ARIMA")
                with st.spinner("Calcul du modèle ARIMA..."):
                    prices_arima, upper, lower, best_order, aic = arima_forecast(hist, days_forward=forecast_days)

                if prices_arima is not None:
                    last_price = hist['Close'].iloc[-1]
                    last_date = hist.index[-1]
                    future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

                    fig_arima = go.Figure()
                    fig_arima.add_trace(go.Scatter(
                        x=future_dates, y=upper, mode='lines',
                        name='IC 95% Supérieur', line=dict(color='rgba(0,255,0,0.5)', dash='dash')
                    ))
                    fig_arima.add_trace(go.Scatter(
                        x=future_dates, y=prices_arima, mode='lines',
                        name='Prévision ARIMA', line=dict(color='#FFD700', width=3)
                    ))
                    fig_arima.add_trace(go.Scatter(
                        x=future_dates, y=lower, mode='lines',
                        name='IC 95% Inférieur', line=dict(color='rgba(255,0,0,0.5)', dash='dash')
                    ))
                    fig_arima.add_trace(go.Scatter(
                        x=[last_date], y=[last_price], mode='markers', name='Prix actuel',
                        marker=dict(size=12, color='cyan', symbol='star')
                    ))
                    fig_arima.update_layout(
                        title=f"Prévision ARIMA{best_order} sur {forecast_days} jours (AIC: {aic:.2f})",
                        template='plotly_dark', xaxis_title='Date',
                        yaxis_title='Prix (USD)', hovermode='x unified', height=500
                    )
                    st.plotly_chart(fig_arima, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Prix Prévu", f"${prices_arima[-1]:.2f}",
                                  delta=f"{((prices_arima[-1]/last_price - 1)*100):.2f}%")
                    with col2:
                        st.info(f"Modèle optimal: ARIMA{best_order} | AIC: {aic:.2f}")
                else:
                    st.error("Impossible de calculer le modèle ARIMA. Essayez avec plus de données.")

            elif model_choice == 'GARCH':
                st.subheader("📊 Prévision de Volatilité GARCH")
                with st.spinner("Calcul du modèle GARCH..."):
                    volatility, fitted = garch_forecast(hist, days_forward=forecast_days)

                if volatility is not None:
                    last_date = hist.index[-1]
                    future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

                    fig_garch = go.Figure()
                    fig_garch.add_trace(go.Scatter(
                        x=future_dates, y=volatility, mode='lines+markers',
                        name='Volatilité Prévue', line=dict(color='#FF6B6B', width=2)
                    ))
                    fig_garch.update_layout(
                        title=f"Prévision de Volatilité GARCH(1,1) sur {forecast_days} jours",
                        template='plotly_dark', xaxis_title='Date',
                        yaxis_title='Volatilité (%)', hovermode='x unified', height=400
                    )
                    st.plotly_chart(fig_garch, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Volatilité Moyenne", f"{np.mean(volatility):.2f}%")
                    with col2:
                        st.metric("📈 Volatilité Finale", f"{volatility[-1]:.2f}%")

                    st.info("Le modèle GARCH prédit la volatilité future, pas le prix. Utile pour estimer le risque.")
                else:
                    st.error("Impossible de calculer le modèle GARCH. Installez le package 'arch': pip install arch")

            elif model_choice == 'ARIMA + GARCH':
                st.subheader("🔬 Prévision Combinée ARIMA + GARCH")
                with st.spinner("Calcul du modèle combiné..."):
                    prices_combined, upper, lower, best_order = arima_garch_combined(hist, days_forward=forecast_days)

                if prices_combined is not None:
                    last_price = hist['Close'].iloc[-1]
                    last_date = hist.index[-1]
                    future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

                    fig_combined = go.Figure()
                    fig_combined.add_trace(go.Scatter(
                        x=future_dates, y=upper, mode='lines',
                        name='Borne Supérieure (95%)', line=dict(color='rgba(0,255,0,0.5)', dash='dash')
                    ))
                    fig_combined.add_trace(go.Scatter(
                        x=future_dates, y=lower, mode='lines',
                        name='Borne Inférieure (95%)', line=dict(color='rgba(255,0,0,0.5)', dash='dash'),
                        fill='tonexty', fillcolor='rgba(128,128,128,0.15)'
                    ))
                    fig_combined.add_trace(go.Scatter(
                        x=future_dates, y=prices_combined, mode='lines',
                        name='Prévision ARIMA+GARCH', line=dict(color='#FFD700', width=3)
                    ))
                    fig_combined.add_trace(go.Scatter(
                        x=[last_date], y=[last_price], mode='markers', name='Prix actuel',
                        marker=dict(size=12, color='cyan', symbol='star')
                    ))
                    fig_combined.update_layout(
                        title=f"Prévision ARIMA{best_order} + GARCH(1,1) sur {forecast_days} jours",
                        template='plotly_dark', xaxis_title='Date',
                        yaxis_title='Prix (USD)', hovermode='x unified', height=500
                    )
                    st.plotly_chart(fig_combined, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Prix Prévu", f"${prices_combined[-1]:.2f}",
                                  delta=f"{((prices_combined[-1]/last_price - 1)*100):.2f}%")
                    with col2:
                        st.metric("🟢 Borne Supérieure", f"${upper[-1]:.2f}",
                                  delta=f"{((upper[-1]/last_price - 1)*100):.2f}%")
                    with col3:
                        st.metric("🔴 Borne Inférieure", f"${lower[-1]:.2f}",
                                  delta=f"{((lower[-1]/last_price - 1)*100):.2f}%")

                    st.success(f"ARIMA{best_order} prédit la tendance, GARCH(1,1) ajuste l'incertitude.")
                else:
                    st.error("Impossible de calculer le modèle. Installez les packages: pip install statsmodels arch")

        # ===== EBITDA VS REVENUE =====
        if financials is not None and not financials.empty:
            try:
                if 'Total Revenue' in financials.index and 'EBITDA' in financials.index:
                    st.markdown("---")
                    st.subheader("💹 EBITDA vs Chiffre d'Affaires")

                    rev = financials.loc['Total Revenue']
                    ebi = financials.loc['EBITDA']

                    fig_fin = go.Figure()
                    fig_fin.add_trace(go.Bar(
                        x=rev.index.astype(str), y=rev.values,
                        name="Chiffre d'Affaires", marker_color='orange'
                    ))
                    fig_fin.add_trace(go.Bar(
                        x=ebi.index.astype(str), y=ebi.values,
                        name='EBITDA', marker_color='green'
                    ))
                    fig_fin.update_layout(
                        barmode='group', template='plotly_dark',
                        xaxis_title='Année', yaxis_title='USD', height=400
                    )
                    st.plotly_chart(fig_fin, use_container_width=True)
            except Exception:
                pass

        # ===== RECOMMANDATIONS ANALYSTES =====
        if recommendations is not None and not recommendations.empty:
            try:
                st.markdown("---")
                st.subheader("🎯 Recommandations des Analystes")

                counts = recommendations.iloc[0].dropna()
                if len(counts) > 0:
                    fig_rec = go.Figure(go.Pie(
                        labels=counts.index, values=counts.values, hole=0.5,
                        marker_colors=['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#c0392b']
                    ))
                    fig_rec.update_layout(title="Distribution des recommandations", height=400)
                    st.plotly_chart(fig_rec, use_container_width=True)
            except Exception:
                pass

        # ===== SIMULATEUR DCA =====
        st.markdown("---")
        st.subheader("📊 Simulateur DCA (Dollar Cost Averaging)")
        st.markdown("Simulez une stratégie d'investissement programmé sur les données historiques.")

        with st.expander("⚙️ Paramètres de la simulation DCA", expanded=True):
            col_dca1, col_dca2, col_dca3 = st.columns(3)

            with col_dca1:
                dca_amount = st.number_input(
                    "💵 Montant par investissement ($)",
                    min_value=10.0,
                    max_value=100000.0,
                    value=500.0,
                    step=50.0,
                    help="Montant investi à chaque période",
                    key="dca_amount"
                )

            with col_dca2:
                dca_frequency = st.selectbox(
                    "📅 Fréquence d'investissement",
                    options=['monthly', 'bi-weekly', 'weekly'],
                    index=0,
                    format_func=lambda x: {
                        'weekly': 'Hebdomadaire',
                        'bi-weekly': 'Bi-mensuel',
                        'monthly': 'Mensuel'
                    }.get(x, x),
                    help="À quelle fréquence investir",
                    key="dca_frequency"
                )

            with col_dca3:
                # Calculer la période en fonction des données disponibles
                data_start = hist.index.min()
                data_end = hist.index.max()
                data_years = (data_end - data_start).days / 365

                st.metric(
                    "📆 Période analysée",
                    f"{data_years:.1f} ans",
                    delta=f"{data_start.strftime('%Y-%m-%d')} → {data_end.strftime('%Y-%m-%d')}"
                )

        # Lancer la simulation
        if len(hist) > 30:  # Minimum de données requises
            dca_result = simulate_dca(hist, dca_amount, dca_frequency)

            if dca_result:
                st.markdown("### 🎯 Résultats de la Simulation")

                # Métriques principales
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)

                with col_r1:
                    st.metric(
                        "💰 Total investi",
                        f"${dca_result['total_invested']:,.0f}",
                        delta=f"{dca_result['num_investments']} versements"
                    )

                with col_r2:
                    st.metric(
                        "📈 Valeur finale (DCA)",
                        f"${dca_result['final_value']:,.0f}",
                        delta=f"{dca_result['total_return']:+.1f}%",
                        delta_color="normal" if dca_result['total_return'] >= 0 else "inverse"
                    )

                with col_r3:
                    st.metric(
                        "💵 Valeur finale (Lump Sum)",
                        f"${dca_result['lump_sum_value']:,.0f}",
                        delta=f"{dca_result['lump_sum_return']:+.1f}%",
                        delta_color="normal" if dca_result['lump_sum_return'] >= 0 else "inverse"
                    )

                with col_r4:
                    gain = dca_result['final_value'] - dca_result['total_invested']
                    st.metric(
                        "🏆 Gain/Perte (DCA)",
                        f"${gain:+,.0f}",
                        delta=f"PRU: ${dca_result['avg_cost_basis']:.2f}"
                    )

                # Comparaison DCA vs Lump Sum
                st.markdown("### ⚔️ DCA vs Lump Sum")

                if dca_result['dca_vs_lump'] > 0:
                    st.success(f"🏆 **DCA gagnant !** La stratégie DCA a surperformé de **{dca_result['dca_vs_lump']:.1f}%** par rapport à un investissement unique.")
                elif dca_result['dca_vs_lump'] < 0:
                    st.info(f"📊 **Lump Sum gagnant.** L'investissement unique a surperformé de **{abs(dca_result['dca_vs_lump']):.1f}%** par rapport au DCA.")
                else:
                    st.info("🤝 **Égalité.** Les deux stratégies ont obtenu des résultats similaires.")

                # Graphique d'évolution
                st.markdown("### 📈 Évolution du Portefeuille")

                investments_df = dca_result['investments']

                fig_dca = go.Figure()

                # Valeur du portefeuille
                fig_dca.add_trace(go.Scatter(
                    x=investments_df['date'],
                    y=investments_df['portfolio_value'],
                    mode='lines',
                    name='Valeur du portefeuille (DCA)',
                    line=dict(color='#2ecc71', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(46, 204, 113, 0.2)'
                ))

                # Total investi
                fig_dca.add_trace(go.Scatter(
                    x=investments_df['date'],
                    y=investments_df['total_invested'],
                    mode='lines',
                    name='Total investi',
                    line=dict(color='#3498db', width=2, dash='dash')
                ))

                # Ligne de valeur Lump Sum
                initial_price = hist['Close'].iloc[0]
                lump_sum_shares = dca_result['total_invested'] / initial_price

                # Calculer la valeur Lump Sum pour chaque date d'investissement
                lump_values = []
                for _, row in investments_df.iterrows():
                    invested_at_date = row['total_invested']
                    shares_lump = invested_at_date / initial_price
                    lump_value = shares_lump * row['price']
                    lump_values.append(lump_value)

                fig_dca.add_trace(go.Scatter(
                    x=investments_df['date'],
                    y=lump_values,
                    mode='lines',
                    name='Valeur Lump Sum (comparaison)',
                    line=dict(color='#e74c3c', width=2, dash='dot')
                ))

                fig_dca.update_layout(
                    template='plotly_dark',
                    xaxis_title='Date',
                    yaxis_title='Valeur ($)',
                    hovermode='x unified',
                    height=450,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )

                st.plotly_chart(fig_dca, use_container_width=True)

                # Détails des investissements
                with st.expander("📋 Détail des investissements"):
                    # Créer un DataFrame formaté
                    display_df = investments_df.copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
                    display_df['shares_bought'] = display_df['shares_bought'].apply(lambda x: f"{x:.4f}")
                    display_df['total_shares'] = display_df['total_shares'].apply(lambda x: f"{x:.4f}")
                    display_df['total_invested'] = display_df['total_invested'].apply(lambda x: f"${x:,.0f}")
                    display_df['portfolio_value'] = display_df['portfolio_value'].apply(lambda x: f"${x:,.0f}")
                    display_df['cost_basis'] = display_df['cost_basis'].apply(lambda x: f"${x:.2f}")

                    display_df.columns = ['Date', 'Prix', 'Actions achetées', 'Total actions',
                                          'Total investi', 'Valeur portefeuille', 'PRU']

                    st.dataframe(display_df, hide_index=True, use_container_width=True)

                # Informations supplémentaires
                st.markdown("---")
                col_info1, col_info2, col_info3 = st.columns(3)

                with col_info1:
                    st.markdown("**📊 Statistiques DCA:**")
                    st.write(f"• Nombre d'actions: {dca_result['total_shares']:.4f}")
                    st.write(f"• Prix moyen d'achat: ${dca_result['avg_cost_basis']:.2f}")
                    st.write(f"• Prix actuel: ${dca_result['final_price']:.2f}")

                with col_info2:
                    st.markdown("**💡 Avantages du DCA:**")
                    st.write("• Réduit l'impact de la volatilité")
                    st.write("• Discipline d'investissement")
                    st.write("• Pas de timing de marché")

                with col_info3:
                    st.markdown("**⚠️ Limites:**")
                    st.write("• Moins performant en marché haussier")
                    st.write("• Frais de transaction multiples")
                    st.write("• Simulation sur données passées")

                st.caption("⚠️ Cette simulation est basée sur des données historiques. Les performances passées ne préjugent pas des performances futures.")

        else:
            st.warning("⚠️ Pas assez de données historiques pour simuler une stratégie DCA. Sélectionnez une période plus longue.")

# Footer
st.markdown("---")
st.caption("📊 Stock Analyzer Pro v2.0 - Données Yahoo Finance. Analyses et simulations à titre indicatif uniquement.")
