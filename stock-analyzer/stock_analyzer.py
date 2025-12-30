"""
Stock Analyzer Pro v6.2 - Correctif Robuste & Auto-Correction
Analyse ESG + Fondamentale Avancée + Historique 10 ans + DCF + Free Form
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import search
import yfinance as yf
from datetime import datetime, timedelta

# Gestion du module ESG (Fallback si absent)
try:
    from esg_module_pro import (
        get_simple_esg_display,
        analyze_greenwashing_pro,
    )
except ImportError:
    def get_simple_esg_display(ticker): return {}
    def analyze_greenwashing_pro(ticker, name, info):
        return {
            'risk_score': 50, 'risk_level': 'MEDIUM', 'risk_color': '#f1c40f',
            'detailed_scores': {},
            'esg_data': {'summary': {'scandals_found': 0, 'total_fines_usd': 0, 'sanctions_matches': 0, 'governance_issues': 0, 'eu_facilities': 0, 'jurisdictions': 0}},
            'flags': [], 'recommendations': []
        }

st.set_page_config(page_title="Stock Analyzer Pro", page_icon="📈", layout="wide")
st.title("📊 Analyseur d'Actions - Niveau Institutionnel")
st.markdown("---")


# ============ FONCTIONS UTILITAIRES ============

@st.cache_data(ttl=300)
def get_ticker_from_name(name: str) -> str | None:
    try:
        results = search(name.strip()).get('quotes')
        return results[0]['symbol'] if results else None
    except:
        return None

@st.cache_data(ttl=300)
def get_stock_data_full(ticker: str, period: str = '10y'):
    """Récupère toutes les données incluant l'historique 10 ans."""
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        hist = tkr.history(period=period)

        # Données financières historiques
        financials = tkr.financials
        quarterly_financials = tkr.quarterly_financials
        balance_sheet = tkr.balance_sheet
        quarterly_balance = tkr.quarterly_balance_sheet
        cashflow = tkr.cashflow
        quarterly_cashflow = tkr.quarterly_cashflow
        income_stmt = tkr.income_stmt

        return {
            'info': info,
            'hist': hist,
            'financials': financials,
            'quarterly_financials': quarterly_financials,
            'balance_sheet': balance_sheet,
            'quarterly_balance': quarterly_balance,
            'cashflow': cashflow,
            'quarterly_cashflow': quarterly_cashflow,
            'income_stmt': income_stmt,
        }
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

def format_value(val, decimal=2):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        val = float(val)
        if abs(val) >= 1e12: return f'{val/1e12:,.{decimal}f} T'
        if abs(val) >= 1e9: return f'{val/1e9:,.{decimal}f} B'
        if abs(val) >= 1e6: return f'{val/1e6:,.{decimal}f} M'
        return f'{val:,.{decimal}f}'
    except:
        return 'N/A'

def format_percent(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f"{float(val) * 100:.2f}%"
    except:
        return 'N/A'

def format_ratio(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 'N/A'
        return f"{float(val):.2f}"
    except:
        return 'N/A'

def safe_get(df, key, default=None):
    """Récupère une ligne d'un DataFrame de manière sécurisée."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return default
    try:
        if key in df.index:
            return df.loc[key]
        return default
    except:
        return default

def is_series_valid(series):
    """Vérifie si une Series pandas contient des données valides non-nulles."""
    if series is None:
        return False
    if isinstance(series, pd.Series):
        if series.empty:
            return False
        # Vérifie si au moins une valeur est valide et positive
        valid_values = series.apply(lambda x: pd.notna(x) and x != 0 and x > 0)
        return valid_values.any()
    elif isinstance(series, list):
        return bool(series) and any(x is not None and x != 0 and x > 0 for x in series)
    return False

def get_valid_values(series):
    """Extrait les valeurs valides d'une Series pandas."""
    if series is None:
        return []
    if isinstance(series, pd.Series):
        return [x for x in series.values if pd.notna(x) and x > 0]
    elif isinstance(series, list):
        return [x for x in series if x is not None and x > 0]
    return []

def calculate_growth(series):
    """Calcule le taux de croissance annuel."""
    if series is None:
        return None
    if isinstance(series, pd.Series) and series.empty:
        return None
    if len(series) < 2:
        return None
    try:
        first = series.iloc[-1]
        last = series.iloc[0]
        years = len(series) - 1
        if pd.notna(first) and pd.notna(last) and first > 0 and last > 0 and years > 0:
            return (last / first) ** (1/years) - 1
        return None
    except:
        return None


# ============ ANALYSE HISTORIQUE 10 ANS ============

def get_historical_metrics(data):
    """Extrait les métriques historiques sur 10 ans."""
    metrics = {
        'years': [],
        'revenue': None,
        'net_income': None,
        'gross_profit': None,
        'operating_income': None,
        'ebitda': None,
        'free_cash_flow': None,
        'total_debt': None,
        'total_cash': None,
        'total_equity': None,
        'total_assets': None,
        'dividends_paid': None,
        'shares_outstanding': None,
    }

    income = data.get('income_stmt')
    balance = data.get('balance_sheet')
    cashflow = data.get('cashflow')

    if income is not None and not income.empty:
        # Années disponibles
        years = [col.year if hasattr(col, 'year') else str(col)[:4] for col in income.columns]
        metrics['years'] = years

        # Revenus
        metrics['revenue'] = safe_get(income, 'Total Revenue')
        if metrics['revenue'] is None:
            metrics['revenue'] = safe_get(income, 'Operating Revenue')

        # Bénéfice net
        metrics['net_income'] = safe_get(income, 'Net Income')

        # Bénéfice brut
        metrics['gross_profit'] = safe_get(income, 'Gross Profit')

        # Résultat opérationnel
        metrics['operating_income'] = safe_get(income, 'Operating Income')

        # EBITDA
        metrics['ebitda'] = safe_get(income, 'EBITDA')
        if metrics['ebitda'] is None:
            metrics['ebitda'] = safe_get(income, 'Normalized EBITDA')

    if balance is not None and not balance.empty:
        # Dette totale
        metrics['total_debt'] = safe_get(balance, 'Total Debt')

        # Trésorerie
        metrics['total_cash'] = safe_get(balance, 'Cash And Cash Equivalents')

        # Capitaux propres
        metrics['total_equity'] = safe_get(balance, 'Total Equity Gross Minority Interest')
        if metrics['total_equity'] is None:
            metrics['total_equity'] = safe_get(balance, 'Stockholders Equity')

        # Actifs totaux
        metrics['total_assets'] = safe_get(balance, 'Total Assets')

        # Actions en circulation
        metrics['shares_outstanding'] = safe_get(balance, 'Ordinary Shares Number')

    if cashflow is not None and not cashflow.empty:
        # Free Cash Flow
        operating_cf = safe_get(cashflow, 'Operating Cash Flow')
        capex = safe_get(cashflow, 'Capital Expenditure')

        if operating_cf is not None and capex is not None:
            try:
                metrics['free_cash_flow'] = operating_cf + capex  # capex est négatif
            except:
                metrics['free_cash_flow'] = None

        # Dividendes versés
        metrics['dividends_paid'] = safe_get(cashflow, 'Common Stock Dividend Paid')

    return metrics


# ============ OUTIL DCF INTERACTIF ============

def calculate_dcf(fcf_values, wacc, terminal_growth_rate, shares_outstanding):
    """Calcule la valeur DCF."""
    if not fcf_values or shares_outstanding <= 0:
        return None

    pv_fcfs = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcf_values)])
    terminal_fcf = fcf_values[-1] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
    pv_terminal = terminal_value / ((1 + wacc) ** len(fcf_values))
    enterprise_value = pv_fcfs + pv_terminal
    equity_value = enterprise_value
    value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
    return {
        'pv_fcfs': pv_fcfs,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'value_per_share': value_per_share
    }

def display_dcf_tool(data, info):
    """Affiche l'outil DCF interactif avec explications."""
    st.subheader("🧮 Outil DCF - Valorisation par Flux de Trésorerie")

    metrics = get_historical_metrics(data)
    fcf = metrics.get('free_cash_flow')

    # CORRECTION: Utiliser la fonction de validation
    if not is_series_valid(fcf):
        st.warning("Données de Free Cash Flow insuffisantes pour le DCF")
        return

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### ⚙️ Paramètres DCF")

        # WACC
        default_wacc = info.get('returnOnEquity', 0.10) if info.get('returnOnEquity') else 0.10
        if pd.isna(default_wacc) or default_wacc <= 0:
            default_wacc = 0.10
        wacc = st.slider(
            "WACC (Coût du Capital)",
            min_value=0.02,
            max_value=0.20,
            value=min(max(float(default_wacc), 0.02), 0.20),
            step=0.01,
            help="Coût moyen pondéré du capital - généralement entre 5% et 12%"
        )

        # Taux de croissance terminal
        terminal_growth = st.slider(
            "Croissance Terminal",
            min_value=0.01,
            max_value=0.05,
            value=0.025,
            step=0.005,
            help="Croissance perpétuelle estimée - généralement 2-3%"
        )

        # Horizon projection
        projection_years = st.slider(
            "Années à projeter",
            min_value=3,
            max_value=10,
            value=5,
            help="Nombre d'années à projeter"
        )

        st.markdown("---")

        # Affichage du mode éducation
        show_education = st.checkbox("📚 Mode Éducation", value=False)

    with col1:
        st.markdown("### 📊 Projection DCF")

        # Préparation des données - utiliser la fonction helper
        fcf_hist = get_valid_values(fcf)

        if not fcf_hist:
            st.warning("Pas de Free Cash Flow positif disponible")
            return

        # Calcul du CAGR FCF pour projection
        fcf_series = pd.Series(fcf_hist)
        cagr_fcf = calculate_growth(fcf_series) or 0.05

        # Projection future
        last_fcf = fcf_hist[-1]
        projected_fcfs = []
        for year in range(1, projection_years + 1):
            projected_fcf = last_fcf * ((1 + cagr_fcf) ** year)
            projected_fcfs.append(projected_fcf)

        # Calcul DCF
        shares_out = info.get('sharesOutstanding', 1) or 1
        dcf_result = calculate_dcf(projected_fcfs, wacc, terminal_growth, shares_out)

        if dcf_result is None:
            st.warning("Impossible de calculer le DCF")
            return

        # Graphique historique + projection
        years = metrics['years']
        all_fcf_values = list(fcf.values) if isinstance(fcf, pd.Series) else list(fcf)
        all_fcf = all_fcf_values + projected_fcfs

        # Générer les années futures
        if years:
            try:
                last_year = int(years[-1]) if isinstance(years[-1], (int, str)) else years[-1]
                future_years = [str(last_year + i) for i in range(1, projection_years + 1)]
            except:
                future_years = [f"+{i}" for i in range(1, projection_years + 1)]
        else:
            future_years = [f"+{i}" for i in range(1, projection_years + 1)]

        all_years = list(years) + future_years

        colors = ['#3498db'] * len(years) + ['#2ecc71'] * projection_years

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(y) for y in all_years],
            y=[x/1e9 if pd.notna(x) else 0 for x in all_fcf],
            marker=dict(color=colors),
            text=[f"${x/1e9:.1f}B" if pd.notna(x) else "" for x in all_fcf],
            textposition='outside',
            name='Free Cash Flow'
        ))

        fig.add_vline(x=len(years)-0.5, line_dash="dash", line_color="red",
                      annotation_text="Projection", annotation_position="top right")

        fig.update_layout(
            template='plotly_dark',
            height=400,
            title=f"FCF Historique + Projection ({projection_years} ans)",
            yaxis_title="Milliards $",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Résultats DCF
        st.markdown("### 💰 Résultats DCF")

        col_res1, col_res2, col_res3 = st.columns(3)

        current_price = info.get('regularMarketPrice', 0) or 0

        with col_res1:
            st.metric(
                "Valeur par Action",
                f"${dcf_result['value_per_share']:.2f}",
                delta=f"vs ${current_price:.2f} actuellement"
            )

        with col_res2:
            if current_price > 0:
                upside = ((dcf_result['value_per_share'] - current_price) / current_price * 100)
                color = "🟢" if upside > 0 else "🔴"
                st.metric(f"{color} Potentiel", f"{upside:+.1f}%")
            else:
                st.metric("Potentiel", "N/A")

        with col_res3:
            st.metric(
                "Valeur Entreprise",
                format_value(dcf_result['enterprise_value']),
            )

        # Tableau détaillé
        with st.expander("📋 Détail des calculs"):
            detail_data = {
                'Composant': [
                    'PV Flux 5 ans',
                    'Valeur Terminal',
                    'PV Terminal',
                    'Valeur Entreprise',
                    'Valeur par Action'
                ],
                'Montant': [
                    format_value(dcf_result['pv_fcfs']),
                    format_value(dcf_result['terminal_value']),
                    format_value(dcf_result['pv_terminal']),
                    format_value(dcf_result['enterprise_value']),
                    f"${dcf_result['value_per_share']:.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

    # Mode éducation
    if show_education:
        st.markdown("---")
        st.markdown("### 📚 Comprendre le DCF")

        col_edu1, col_edu2, col_edu3, col_edu4 = st.columns(4)

        with col_edu1:
            st.info("""
            **WACC**

            Coût moyen du capital. Plus il est bas, plus l'entreprise est
            finançable à bon marché.

            💡 Typiquement 5-12%
            """)

        with col_edu2:
            st.info("""
            **Free Cash Flow**

            Flux de trésorerie disponible après dépenses
            d'exploitation. C'est l'argent que peut distribuer
            l'entreprise.

            💡 Doit être positif et croissant
            """)

        with col_edu3:
            st.info("""
            **Croissance Terminal**

            Croissance perpétuelle supposée. Ne dépasse pas
            la croissance économique mondiale.

            💡 Généralement 2-3% max
            """)

        with col_edu4:
            st.info("""
            **Valeur DCF**

            Somme des valeurs actualisées de tous les flux
            futurs. Compare-la au prix actuel.

            💡 Si > prix = potentiel d'appréciation
            """)


# ============ FREE FORM TOOL ============

def display_free_form_tool(data, info):
    """Outil de création de graphiques personnalisés avec multi-métriques."""
    st.subheader("📊 Free Form Tool - Créer Vos Graphiques")

    metrics = get_historical_metrics(data)
    years = metrics['years']

    if not years:
        st.warning("Données insuffisantes")
        return

    # Dictionnaire de toutes les métriques disponibles
    available_metrics = {
        'Revenus': metrics['revenue'],
        'Bénéfice Net': metrics['net_income'],
        'Profit Brut': metrics['gross_profit'],
        'Résultat Opérationnel': metrics['operating_income'],
        'EBITDA': metrics['ebitda'],
        'Free Cash Flow': metrics['free_cash_flow'],
        'Dette Totale': metrics['total_debt'],
        'Trésorerie': metrics['total_cash'],
        'Capitaux Propres': metrics['total_equity'],
        'Actifs Totaux': metrics['total_assets'],
    }

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📋 Sélection des métriques")

        selected_metrics = st.multiselect(
            "Choisir jusqu'à 5 métriques",
            options=list(available_metrics.keys()),
            max_selections=5,
            default=["Revenus", "Bénéfice Net"],
            help="Sélectionnez les métriques à afficher"
        )

        st.markdown("---")

        # Normalisation
        normalize = st.checkbox("Normaliser (index 100)", value=False,
                                help="Affiche tous les graphiques à partir de 100")

        # Type de graphique
        chart_type = st.radio("Type de graphique", ["Ligne", "Barre", "Ligne+Points"])

    with col2:
        if selected_metrics:
            fig = go.Figure()

            colors_palette = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
            y_label = "Milliards $"

            for idx, metric_name in enumerate(selected_metrics):
                metric_data = available_metrics[metric_name]

                # Validation avec fonction helper
                if not is_series_valid(metric_data):
                    continue

                # Nettoyage des données
                if isinstance(metric_data, pd.Series):
                    cleaned_data = [x if pd.notna(x) and x > 0 else None for x in metric_data.values]
                else:
                    cleaned_data = [x if x is not None and x > 0 else None for x in metric_data]

                if normalize and any(x for x in cleaned_data if x):
                    first_val = next((x for x in cleaned_data if x), 1)
                    normalized_data = [(x / first_val * 100) if x else None for x in cleaned_data]
                    y_data = normalized_data
                    y_label = "Index (100 = année 1)"
                else:
                    y_data = [x/1e9 if x else None for x in cleaned_data]
                    y_label = "Milliards $"

                # Ajout de la trace selon le type
                if chart_type == "Ligne":
                    fig.add_trace(go.Scatter(
                        x=[str(y) for y in years], y=y_data, name=metric_name,
                        mode='lines',
                        line=dict(color=colors_palette[idx], width=2)
                    ))
                elif chart_type == "Barre":
                    fig.add_trace(go.Bar(
                        x=[str(y) for y in years], y=y_data, name=metric_name,
                        marker_color=colors_palette[idx]
                    ))
                else:  # Ligne + Points
                    fig.add_trace(go.Scatter(
                        x=[str(y) for y in years], y=y_data, name=metric_name,
                        mode='lines+markers',
                        line=dict(color=colors_palette[idx], width=2),
                        marker=dict(size=8)
                    ))

            fig.update_layout(
                template='plotly_dark',
                height=450,
                title="Graphique Personnalisé",
                yaxis_title=y_label,
                hovermode='x unified',
                legend=dict(orientation="v", yanchor="top", xanchor="right")
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Sélectionnez au moins une métrique")


# ============ SÉLECTEUR D'ANNÉES INTERACTIF ============

def display_year_selector_view(data, info):
    """Affiche les données pour une année spécifique sélectionnée."""
    st.subheader("📅 Analyse par Année - Vue Détaillée")

    metrics = get_historical_metrics(data)
    years = metrics['years']

    if not years:
        st.warning("Données insuffisantes")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        # Convertir en liste de strings pour select_slider
        years_options = [str(y) for y in years]
        years_options_sorted = sorted(years_options)

        selected_year_str = st.select_slider(
            "Sélectionner l'année",
            options=years_options_sorted,
            value=years_options_sorted[-1] if years_options_sorted else None
        )

        # Retrouver l'index correspondant
        try:
            selected_year_idx = years_options.index(selected_year_str)
        except (ValueError, IndexError):
            selected_year_idx = 0

    if not years:
        return

    selected_year = years[selected_year_idx]

    with col2:
        st.markdown(f"### 📊 Métriques pour l'année **{selected_year}**")

    # Fonction helper pour extraire la valeur d'une métrique à un index
    def get_metric_value(metric_series, idx):
        if metric_series is None:
            return None
        if isinstance(metric_series, pd.Series):
            if metric_series.empty or idx >= len(metric_series):
                return None
            val = metric_series.iloc[idx]
            return val if pd.notna(val) else None
        elif isinstance(metric_series, list):
            if idx >= len(metric_series):
                return None
            return metric_series[idx]
        return None

    # Affichage des métriques pour l'année sélectionnée
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    data_points = [
        (col1, "Revenus", metrics['revenue']),
        (col2, "Bénéfice Net", metrics['net_income']),
        (col3, "EBITDA", metrics['ebitda']),
        (col4, "FCF", metrics['free_cash_flow']),
        (col5, "Dette", metrics['total_debt']),
        (col6, "Trésorerie", metrics['total_cash']),
    ]

    for col, label, metric_series in data_points:
        with col:
            val = get_metric_value(metric_series, selected_year_idx)
            st.metric(label, format_value(val))

    # Comparaison année précédente / suivante
    st.markdown("---")
    st.markdown("### 📈 Comparaison Année vs Année")

    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)

    with comp_col1:
        if selected_year_idx > 0:
            prev_idx = selected_year_idx - 1
            prev_year = years[prev_idx]

            rev_curr = get_metric_value(metrics['revenue'], selected_year_idx)
            rev_prev = get_metric_value(metrics['revenue'], prev_idx)

            if rev_curr and rev_prev and rev_prev > 0:
                growth = ((rev_curr - rev_prev) / rev_prev) * 100
                st.metric(f"Growth vs {prev_year}", f"{growth:+.1f}%")
            else:
                st.metric(f"Growth vs {prev_year}", "N/A")
        else:
            st.metric("vs Année Précédente", "1ère année")

    with comp_col2:
        if selected_year_idx < len(years) - 1:
            next_idx = selected_year_idx + 1
            next_year = years[next_idx]

            rev_curr = get_metric_value(metrics['revenue'], selected_year_idx)
            rev_next = get_metric_value(metrics['revenue'], next_idx)

            if rev_curr and rev_next and rev_curr > 0:
                growth = ((rev_next - rev_curr) / rev_curr) * 100
                st.metric(f"Growth vs {next_year}", f"{growth:+.1f}%")
            else:
                st.metric(f"Growth vs {next_year}", "N/A")
        else:
            st.metric("vs Année Suivante", "Dernière année")

    with comp_col3:
        # Marge nette
        rev = get_metric_value(metrics['revenue'], selected_year_idx)
        ni = get_metric_value(metrics['net_income'], selected_year_idx)
        if rev and ni and rev > 0:
            net_margin = (ni / rev) * 100
            st.metric("Marge Nette %", f"{net_margin:.1f}%")
        else:
            st.metric("Marge Nette %", "N/A")

    with comp_col4:
        # ROE
        ni = get_metric_value(metrics['net_income'], selected_year_idx)
        eq = get_metric_value(metrics['total_equity'], selected_year_idx)
        if ni and eq and eq > 0:
            roe = (ni / eq) * 100
            st.metric("ROE %", f"{roe:.1f}%")
        else:
            st.metric("ROE %", "N/A")


def display_historical_analysis(data, info):
    """Affiche l'analyse historique sur 10 ans avec tous les nouveaux outils."""

    metrics = get_historical_metrics(data)

    if not metrics['years']:
        st.warning("Données historiques non disponibles")
        return

    st.subheader("📅 Analyse Fondamentale Avancée")

    # === ONGLETS PRINCIPAUX ===
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Revenus & Profits",
        "💰 Rentabilité",
        "🏦 Bilan",
        "💵 Cash Flow",
        "📊 Ratios",
        "🧮 DCF",
        "📋 Free Form"
    ])

    years = metrics['years']

    # TAB 1: REVENUS & PROFITS
    with tab1:
        st.markdown("### 📈 Évolution des Revenus et Profits")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Revenus
            if is_series_valid(metrics['revenue']):
                try:
                    rev_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['revenue'].values]
                    fig.add_trace(
                        go.Bar(x=[str(y) for y in years], y=rev_values, name="Revenus", marker_color='#3498db'),
                        secondary_y=False
                    )
                except:
                    pass

            # Bénéfice net
            if is_series_valid(metrics['net_income']):
                try:
                    ni_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['net_income'].values]
                    fig.add_trace(
                        go.Scatter(x=[str(y) for y in years], y=ni_values, name="Bénéfice Net",
                                   mode='lines+markers', line=dict(color='#2ecc71', width=3)),
                        secondary_y=False
                    )
                except:
                    pass

            # Résultat opérationnel
            if is_series_valid(metrics['operating_income']):
                try:
                    op_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['operating_income'].values]
                    fig.add_trace(
                        go.Scatter(x=[str(y) for y in years], y=op_values, name="Résultat Opérationnel",
                                   mode='lines+markers', line=dict(color='#f1c40f', width=2, dash='dash')),
                        secondary_y=False
                    )
                except:
                    pass

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Revenus et Profits (Milliards $)",
                yaxis_title="Milliards $",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                barmode='overlay',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Tableau récapitulatif
            st.markdown("**Croissance annuelle moyenne (CAGR)**")

            if is_series_valid(metrics['revenue']):
                cagr_rev = calculate_growth(metrics['revenue'])
                st.metric("📊 Revenus", format_percent(cagr_rev) if cagr_rev else "N/A")

            if is_series_valid(metrics['net_income']):
                cagr_ni = calculate_growth(metrics['net_income'])
                st.metric("💰 Bénéfice Net", format_percent(cagr_ni) if cagr_ni else "N/A")

            if is_series_valid(metrics['operating_income']):
                cagr_op = calculate_growth(metrics['operating_income'])
                st.metric("📈 Résultat Opé.", format_percent(cagr_op) if cagr_op else "N/A")

            # Dernier vs Premier
            st.markdown("---")
            st.markdown("**Dernière année vs Première**")
            if is_series_valid(metrics['revenue']):
                try:
                    first = metrics['revenue'].iloc[-1]
                    last = metrics['revenue'].iloc[0]
                    if pd.notna(first) and pd.notna(last) and first > 0:
                        mult = last / first
                        st.metric("Multiplicateur CA", f"x{mult:.1f}")
                except:
                    pass

        # Tableau détaillé
        with st.expander("📋 Données détaillées"):
            df_data = {'Année': [str(y) for y in years]}
            if is_series_valid(metrics['revenue']):
                df_data['Revenus'] = [format_value(v) for v in metrics['revenue'].values]
            if is_series_valid(metrics['gross_profit']):
                df_data['Profit Brut'] = [format_value(v) for v in metrics['gross_profit'].values]
            if is_series_valid(metrics['operating_income']):
                df_data['Résultat Opé.'] = [format_value(v) for v in metrics['operating_income'].values]
            if is_series_valid(metrics['net_income']):
                df_data['Bénéfice Net'] = [format_value(v) for v in metrics['net_income'].values]

            st.dataframe(pd.DataFrame(df_data), hide_index=True, use_container_width=True)

    # TAB 2: RENTABILITÉ
    with tab2:
        st.markdown("### 💰 Évolution de la Rentabilité")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()

            # Calcul des marges
            if is_series_valid(metrics['revenue']) and is_series_valid(metrics['gross_profit']):
                try:
                    gross_margin = [(gp/rev)*100 if pd.notna(rev) and pd.notna(gp) and rev > 0 else 0
                                    for rev, gp in zip(metrics['revenue'].values, metrics['gross_profit'].values)]
                    fig.add_trace(go.Scatter(x=[str(y) for y in years], y=gross_margin, name="Marge Brute %",
                                             mode='lines+markers', line=dict(color='#2ecc71', width=2)))
                except:
                    pass

            if is_series_valid(metrics['revenue']) and is_series_valid(metrics['operating_income']):
                try:
                    op_margin = [(oi/rev)*100 if pd.notna(rev) and pd.notna(oi) and rev > 0 else 0
                                 for rev, oi in zip(metrics['revenue'].values, metrics['operating_income'].values)]
                    fig.add_trace(go.Scatter(x=[str(y) for y in years], y=op_margin, name="Marge Opérationnelle %",
                                             mode='lines+markers', line=dict(color='#3498db', width=2)))
                except:
                    pass

            if is_series_valid(metrics['revenue']) and is_series_valid(metrics['net_income']):
                try:
                    net_margin = [(ni/rev)*100 if pd.notna(rev) and pd.notna(ni) and rev > 0 else 0
                                  for rev, ni in zip(metrics['revenue'].values, metrics['net_income'].values)]
                    fig.add_trace(go.Scatter(x=[str(y) for y in years], y=net_margin, name="Marge Nette %",
                                             mode='lines+markers', line=dict(color='#9b59b6', width=2)))
                except:
                    pass

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Évolution des Marges (%)",
                yaxis_title="Marge (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # ROE / ROA historique
            st.markdown("**Rentabilité actuelle**")
            st.metric("ROE", format_percent(info.get('returnOnEquity')))
            st.metric("ROA", format_percent(info.get('returnOnAssets')))

            # Calcul ROE historique si possible
            if is_series_valid(metrics['net_income']) and is_series_valid(metrics['total_equity']):
                try:
                    roe_values = [(ni/eq)*100 if pd.notna(eq) and pd.notna(ni) and eq > 0 else None
                                  for ni, eq in zip(metrics['net_income'].values, metrics['total_equity'].values)]
                    valid_roe = [r for r in roe_values if r is not None]
                    if valid_roe:
                        avg_roe = np.mean(valid_roe)
                        st.metric("ROE moyen 10Y", f"{avg_roe:.1f}%")
                except:
                    pass

        # Graphique ROE/ROA historique
        if is_series_valid(metrics['net_income']) and is_series_valid(metrics['total_equity']) and is_series_valid(metrics['total_assets']):
            with st.expander("📈 ROE & ROA historiques"):
                fig2 = go.Figure()

                try:
                    roe_hist = [(ni/eq)*100 if pd.notna(eq) and pd.notna(ni) and eq > 0 else None
                                for ni, eq in zip(metrics['net_income'].values, metrics['total_equity'].values)]
                    roa_hist = [(ni/ta)*100 if pd.notna(ta) and pd.notna(ni) and ta > 0 else None
                                for ni, ta in zip(metrics['net_income'].values, metrics['total_assets'].values)]

                    fig2.add_trace(go.Scatter(x=[str(y) for y in years], y=roe_hist, name="ROE %",
                                              mode='lines+markers', line=dict(color='#e74c3c')))
                    fig2.add_trace(go.Scatter(x=[str(y) for y in years], y=roa_hist, name="ROA %",
                                              mode='lines+markers', line=dict(color='#3498db')))

                    fig2.update_layout(template='plotly_dark', height=300, yaxis_title="%", hovermode='x unified')
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

    # TAB 3: BILAN
    with tab3:
        st.markdown("### 🏦 Évolution du Bilan")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()

            # Dette vs Cash
            if is_series_valid(metrics['total_debt']):
                try:
                    debt_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['total_debt'].values]
                    fig.add_trace(go.Bar(x=[str(y) for y in years], y=debt_values, name="Dette Totale", marker_color='#e74c3c'))
                except:
                    pass

            if is_series_valid(metrics['total_cash']):
                try:
                    cash_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['total_cash'].values]
                    fig.add_trace(go.Bar(x=[str(y) for y in years], y=cash_values, name="Trésorerie", marker_color='#2ecc71'))
                except:
                    pass

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Dette vs Trésorerie (Milliards $)",
                yaxis_title="Milliards $",
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Situation actuelle**")
            st.metric("Dette Totale", format_value(info.get('totalDebt')))
            st.metric("Trésorerie", format_value(info.get('totalCash')))

            debt = info.get('totalDebt', 0) or 0
            cash = info.get('totalCash', 0) or 0
            st.metric("Dette Nette", format_value(debt - cash))

            st.markdown("---")
            st.metric("Debt/Equity", format_ratio(info.get('debtToEquity')))
            st.metric("Current Ratio", format_ratio(info.get('currentRatio')))

        # Évolution Debt/Equity
        if is_series_valid(metrics['total_debt']) and is_series_valid(metrics['total_equity']):
            with st.expander("📈 Évolution Debt/Equity"):
                try:
                    de_ratio = [(d/e)*100 if pd.notna(e) and pd.notna(d) and e > 0 else None
                                for d, e in zip(metrics['total_debt'].values, metrics['total_equity'].values)]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=[str(y) for y in years], y=de_ratio, mode='lines+markers+text',
                                              line=dict(color='#e74c3c'),
                                              text=[f"{v:.0f}%" if v else "" for v in de_ratio],
                                              textposition="top center"))
                    fig2.add_hline(y=100, line_dash="dash", line_color="white",
                                   annotation_text="Seuil 100%")
                    fig2.update_layout(template='plotly_dark', height=300, yaxis_title="Debt/Equity %", hovermode='x unified')
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

    # TAB 4: CASH FLOW
    with tab4:
        st.markdown("### 💵 Évolution des Cash Flows")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()

            # Free Cash Flow
            if is_series_valid(metrics['free_cash_flow']):
                try:
                    fcf_values = [v/1e9 if pd.notna(v) else 0 for v in metrics['free_cash_flow'].values]
                    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in fcf_values]
                    fig.add_trace(go.Bar(x=[str(y) for y in years], y=fcf_values, name="Free Cash Flow", marker_color=colors))
                except:
                    pass

            # Dividendes
            if is_series_valid(metrics['dividends_paid']):
                try:
                    div_values = [abs(v)/1e9 if pd.notna(v) else 0 for v in metrics['dividends_paid'].values]
                    fig.add_trace(go.Scatter(x=[str(y) for y in years], y=div_values, name="Dividendes Versés",
                                             mode='lines+markers', line=dict(color='#f1c40f', width=2)))
                except:
                    pass

            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Free Cash Flow & Dividendes (Milliards $)",
                yaxis_title="Milliards $",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Cash Flow actuel**")
            st.metric("Operating CF", format_value(info.get('operatingCashflow')))
            st.metric("Free Cash Flow", format_value(info.get('freeCashflow')))

            # FCF Yield
            fcf = info.get('freeCashflow')
            mcap = info.get('marketCap')
            if fcf and mcap and mcap > 0:
                fcf_yield = (fcf / mcap) * 100
                st.metric("FCF Yield", f"{fcf_yield:.2f}%")

            # CAGR FCF
            if is_series_valid(metrics['free_cash_flow']):
                cagr_fcf = calculate_growth(metrics['free_cash_flow'])
                if cagr_fcf:
                    st.metric("CAGR FCF", format_percent(cagr_fcf))

        # Payout ratio historique
        if is_series_valid(metrics['dividends_paid']) and is_series_valid(metrics['net_income']):
            with st.expander("📈 Évolution Payout Ratio"):
                try:
                    payout = [(abs(d)/ni)*100 if pd.notna(ni) and pd.notna(d) and ni > 0 else None
                              for d, ni in zip(metrics['dividends_paid'].values, metrics['net_income'].values)]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=[str(y) for y in years], y=payout, mode='lines+markers',
                                              line=dict(color='#f1c40f')))
                    fig2.add_hline(y=60, line_dash="dash", line_color="orange",
                                   annotation_text="Seuil 60%")
                    fig2.update_layout(template='plotly_dark', height=300, yaxis_title="Payout %", hovermode='x unified')
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

    # TAB 5: RATIOS
    with tab5:
        st.markdown("### 📊 Évolution des Ratios Clés")

        # Calcul EPS historique
        if is_series_valid(metrics['net_income']) and is_series_valid(metrics['shares_outstanding']):
            col1, col2 = st.columns(2)

            with col1:
                try:
                    eps_hist = [ni/shares if pd.notna(shares) and pd.notna(ni) and shares > 0 else None
                                for ni, shares in zip(metrics['net_income'].values, metrics['shares_outstanding'].values)]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=[str(y) for y in years], y=eps_hist, name="EPS", marker_color='#3498db',
                                         text=[f"${v:.2f}" if v else "" for v in eps_hist],
                                         textposition='outside'))
                    fig.update_layout(template='plotly_dark', height=350, title="Bénéfice par Action (EPS)",
                                      yaxis_title="$ par action", hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

                    # CAGR EPS
                    eps_series = pd.Series([e for e in eps_hist if e is not None])
                    if len(eps_series) >= 2:
                        cagr_eps = calculate_growth(eps_series)
                        st.metric("CAGR EPS", format_percent(cagr_eps) if cagr_eps else "N/A")
                except:
                    st.info("Données EPS insuffisantes")

            with col2:
                # Book Value per Share
                if is_series_valid(metrics['total_equity']):
                    try:
                        bvps = [eq/shares if pd.notna(shares) and pd.notna(eq) and shares > 0 else None
                                for eq, shares in zip(metrics['total_equity'].values, metrics['shares_outstanding'].values)]

                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=[str(y) for y in years], y=bvps, name="Book Value", marker_color='#2ecc71',
                                             text=[f"${v:.2f}" if v else "" for v in bvps],
                                             textposition='outside'))
                        fig.update_layout(template='plotly_dark', height=350,
                                          title="Valeur Comptable par Action",
                                          yaxis_title="$ par action", hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Données BVPS insuffisantes")

        # Revenue per Share
        if is_series_valid(metrics['revenue']) and is_series_valid(metrics['shares_outstanding']):
            with st.expander("📈 Revenu par Action"):
                try:
                    rps = [rev/shares if pd.notna(shares) and pd.notna(rev) and shares > 0 else None
                           for rev, shares in zip(metrics['revenue'].values, metrics['shares_outstanding'].values)]

                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=[str(y) for y in years], y=rps, marker_color='#9b59b6'))
                    fig.update_layout(template='plotly_dark', height=300, yaxis_title="$ par action", hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Données insuffisantes")

        # Tableau récapitulatif des CAGR
        st.markdown("### 📊 Résumé des Croissances (CAGR)")

        cagr_data = []

        if is_series_valid(metrics['revenue']):
            cagr = calculate_growth(metrics['revenue'])
            cagr_data.append({'Métrique': 'Revenus', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                              'Rating': '🟢' if cagr and cagr > 0.10 else '🟡' if cagr and cagr > 0.05 else '🔴'})

        if is_series_valid(metrics['net_income']):
            cagr = calculate_growth(metrics['net_income'])
            cagr_data.append({'Métrique': 'Bénéfice Net', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                              'Rating': '🟢' if cagr and cagr > 0.12 else '🟡' if cagr and cagr > 0.05 else '🔴'})

        if is_series_valid(metrics['free_cash_flow']):
            cagr = calculate_growth(metrics['free_cash_flow'])
            cagr_data.append({'Métrique': 'Free Cash Flow', 'CAGR': format_percent(cagr) if cagr else 'N/A',
                              'Rating': '🟢' if cagr and cagr > 0.10 else '🟡' if cagr and cagr > 0.05 else '🔴'})

        if cagr_data:
            st.dataframe(pd.DataFrame(cagr_data), hide_index=True, use_container_width=True)

    # TAB 6: DCF
    with tab6:
        display_dcf_tool(data, info)

    # TAB 7: FREE FORM
    with tab7:
        display_free_form_tool(data, info)

    # BONUS: Sélecteur d'années
    st.markdown("---")
    display_year_selector_view(data, info)


# ============ SIDEBAR ============

with st.sidebar:
    st.header("⚙️ Paramètres")
    analysis_mode = st.radio("Mode", ['📈 Analyse Fondamentale', '🌱 Analyse ESG Pro'], index=0)
    st.markdown("---")

    company_name = st.text_input("Entreprise", value="Apple" if analysis_mode == '📈 Analyse Fondamentale' else "TotalEnergies")

    st.markdown("---")
    analyze_button = st.button("🔍 Analyser", type="primary", use_container_width=True)

    if analysis_mode == '📈 Analyse Fondamentale':
        st.markdown("---")
        st.markdown("### 📊 Données incluses")
        st.caption("""
        - Valorisation (P/E, P/B, EV/EBITDA...)
        - Rentabilité (ROE, ROA, Marges)
        - Croissance (CA, Bénéfices, FCF)
        - Bilan (Dette, Cash, Equity)
        - **Historique 10 ans**
        - **🆕 Outil DCF interactif**
        - **🆕 Free Form Tool**
        - **🆕 Sélecteur d'années**
        """)


# ============ ANALYSE FONDAMENTALE ============

if analysis_mode == '📈 Analyse Fondamentale':
    st.header("📈 Analyse Fondamentale Avancée")

    if analyze_button and company_name:
        with st.spinner("Recherche..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            with st.spinner("Chargement des données (10 ans)..."):
                data = get_stock_data_full(ticker, '10y')

            if data and data['info'] and data['hist'] is not None and not data['hist'].empty:
                info = data['info']
                hist = data['hist']

                st.subheader(f"🏢 {info.get('shortName', ticker)} ({ticker})")

                if info.get('longBusinessSummary'):
                    with st.expander("📋 Description"):
                        st.write(info['longBusinessSummary'])

                # Métriques clés
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Prix", f"${info.get('regularMarketPrice', 0):.2f}")
                with col2:
                    change = info.get('regularMarketChangePercent', 0) or 0
                    st.metric("Var. Jour", f"{change:.2f}%")
                with col3:
                    st.metric("Market Cap", format_value(info.get('marketCap')))
                with col4:
                    st.metric("P/E", format_ratio(info.get('trailingPE')))
                with col5:
                    st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                with col6:
                    st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")

                st.markdown("---")

                # Graphique prix 10 ans
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Prix',
                                         line=dict(color='#3498db', width=2)))
                if len(hist) > 200:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(200).mean(),
                                             mode='lines', name='MM200', line=dict(color='red', width=1)))
                fig.update_layout(template='plotly_dark', height=400, title="Évolution du cours (10 ans)", hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)

                # Performance 10 ans
                perf_10y = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prix il y a 10 ans", f"${hist['Close'].iloc[0]:.2f}")
                with col2:
                    st.metric("Prix actuel", f"${hist['Close'].iloc[-1]:.2f}")
                with col3:
                    st.metric("Performance 10 ans", f"{perf_10y:+.0f}%")

                st.markdown("---")

                # === ANALYSE HISTORIQUE 10 ANS ===
                display_historical_analysis(data, info)

                st.markdown("---")

                # ESG rapide
                st.subheader("🌱 ESG Rapide")
                esg = get_simple_esg_display(ticker)
                if esg.get('available'):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Environnement", f"{esg.get('environment', 0):.0f}/100")
                    with col2:
                        st.metric("Social", f"{esg.get('social', 0):.0f}/100")
                    with col3:
                        st.metric("Gouvernance", f"{esg.get('governance', 0):.0f}/100")
                    with col4:
                        st.metric("Controverse", f"{esg.get('controversy_level', 0)}/5")
                else:
                    st.info("Données ESG non disponibles")
        else:
            st.error(f"❌ Entreprise '{company_name}' non trouvée")


# ============ ANALYSE ESG PRO ============

elif analysis_mode == '🌱 Analyse ESG Pro':
    st.header("🌱 Analyse ESG - Niveau Institutionnel")

    if analyze_button and company_name:
        with st.spinner("Recherche..."):
            ticker = get_ticker_from_name(company_name)

        if ticker:
            with st.spinner("Chargement..."):
                data = get_stock_data_full(ticker, '1y')

            if data:
                info = data['info']
                full_name = info.get('shortName', company_name) if info else company_name

                st.subheader(f"🏢 {full_name} ({ticker})")

                with st.spinner("🔍 Collecte ESG (10 sources)..."):
                    analysis = analyze_greenwashing_pro(ticker, full_name, info)

                st.markdown("---")

                # Score de risque
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Score Global", f"{analysis['risk_score']:.0f}/100",
                        delta=analysis['risk_level'], delta_color="inverse" if analysis['risk_score'] < 30 else "normal")
                with col2:
                    st.metric("🗞️ Médias", f"{analysis['detailed_scores'].get('media', 0):.0f}/100")
                with col3:
                    st.metric("🌍 Environnement", f"{analysis['detailed_scores'].get('environmental', 0):.0f}/100")
                with col4:
                    st.metric("⚖️ Sanctions", f"{analysis['detailed_scores'].get('sanctions', 0):.0f}/100")
                with col5:
                    st.metric("🏛️ Gouvernance", f"{analysis['detailed_scores'].get('governance', 0):.0f}/100")

                # Jauge
                st.markdown(f"""
                <div style="background: linear-gradient(to right, #27ae60, #f1c40f, #e74c3c);
                            height: 25px; border-radius: 12px; position: relative; margin: 20px 0;">
                    <div style="position: absolute; left: {analysis['risk_score']}%; top: -8px;
                                width: 40px; height: 40px; background: white; border-radius: 50%;
                                border: 4px solid {analysis['risk_color']}; transform: translateX(-50%);
                                display: flex; align-items: center; justify-content: center;
                                font-weight: bold;">{analysis['risk_score']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Résumé
                summary = analysis['esg_data']['summary']
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1: st.metric("🗞️ Scandales", summary['scandals_found'])
                with col2: st.metric("💰 Amendes", f"${summary['total_fines_usd']:,.0f}")
                with col3: st.metric("🚫 Sanctions", summary['sanctions_matches'])
                with col4: st.metric("⚠️ Gouv.", summary['governance_issues'])
                with col5: st.metric("🏭 Sites EU", summary['eu_facilities'])
                with col6: st.metric("🌐 Pays", summary['jurisdictions'])

                st.markdown("---")

                # Alertes
                if analysis['flags']:
                    st.subheader("🚩 Alertes")
                    for flag in analysis['flags']:
                        sev = flag['severity']
                        emoji = {'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟡'}.get(sev, '⚪')
                        msg = f"{emoji} **{flag['title']}** - {flag['description']}"
                        if sev == 'HIGH': st.error(msg)
                        elif sev == 'MEDIUM': st.warning(msg)
                        else: st.info(msg)

                # Recommandations
                st.subheader("💡 Recommandations")
                for rec in analysis['recommendations']:
                    st.info(f"→ {rec}")
        else:
            st.error(f"❌ Entreprise non trouvée")


# ============ FOOTER ============

st.markdown("---")
st.caption("📊 Stock Analyzer Pro v6.2 - Données: Yahoo Finance (10 ans), GDELT, EPA, E-PRTR, OpenSanctions, SEC | 🆕 DCF Tool | 🆕 Free Form | 🆕 Year Selector")
