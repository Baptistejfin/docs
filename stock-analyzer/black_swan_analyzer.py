"""
Black Swan Analyzer - Module d'analyse des risques extrêmes
Inspiré de la théorie du Cygne Noir de Nassim Taleb

Ce module analyse les risques majeurs pouvant impacter une action :
- Identification des risques par secteur
- Scoring dynamique basé sur l'actualité
- Estimation de l'impact financier
- Visualisation interactive
"""

import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from dataclasses import dataclass, asdict
from enum import Enum


# ============ CONFIGURATION ============

class RiskCategory(Enum):
    """Catégories de risques Cygne Noir."""
    REGULATORY = "regulatory"
    GEOPOLITICAL = "geopolitical"
    OPERATIONAL = "operational"
    ESG_CLIMATE = "esg_climate"
    MACROECONOMIC = "macroeconomic"
    COMPANY_SPECIFIC = "company_specific"
    TECHNOLOGICAL = "technological"
    FINANCIAL = "financial"


@dataclass
class RiskDefinition:
    """Définition d'un risque type."""
    id: str
    name: str
    name_fr: str
    category: RiskCategory
    keywords: List[str]
    base_probability: float  # Probabilité de base (0-1)
    base_impact: Tuple[float, float]  # Impact range (min, max)
    sectors: List[str]  # Secteurs concernés ("all" = tous)
    recovery_months: int  # Temps de récupération estimé en mois


# ============ BASE DE DONNÉES DES RISQUES PAR SECTEUR ============

RISK_DATABASE: List[RiskDefinition] = [
    # === RISQUES RÉGLEMENTAIRES ===
    RiskDefinition(
        id="reg_carbon",
        name="Carbon regulation tightening",
        name_fr="Régulation carbone renforcée",
        category=RiskCategory.REGULATORY,
        keywords=["carbon tax", "emission regulation", "climate policy", "carbon border", "net zero mandate", "taxe carbone", "régulation émissions"],
        base_probability=0.35,
        base_impact=(-0.25, -0.45),
        sectors=["Energy", "Basic Materials", "Industrials", "Utilities"],
        recovery_months=18
    ),
    RiskDefinition(
        id="reg_antitrust",
        name="Antitrust investigation/breakup",
        name_fr="Enquête antitrust / démantèlement",
        category=RiskCategory.REGULATORY,
        keywords=["antitrust", "monopoly", "competition law", "market dominance", "breakup", "démantèlement", "abus position dominante"],
        base_probability=0.20,
        base_impact=(-0.30, -0.50),
        sectors=["Technology", "Communication Services", "Consumer Cyclical"],
        recovery_months=24
    ),
    RiskDefinition(
        id="reg_data_privacy",
        name="Data privacy regulation",
        name_fr="Régulation protection des données",
        category=RiskCategory.REGULATORY,
        keywords=["GDPR", "data privacy", "data breach fine", "privacy regulation", "CCPA", "protection données", "amende CNIL"],
        base_probability=0.30,
        base_impact=(-0.15, -0.30),
        sectors=["Technology", "Communication Services", "Financial Services", "Healthcare"],
        recovery_months=12
    ),
    RiskDefinition(
        id="reg_pharma",
        name="Drug pricing regulation",
        name_fr="Régulation des prix des médicaments",
        category=RiskCategory.REGULATORY,
        keywords=["drug pricing", "medicare negotiation", "pharma regulation", "price cap", "prix médicaments", "régulation pharma"],
        base_probability=0.40,
        base_impact=(-0.20, -0.40),
        sectors=["Healthcare"],
        recovery_months=12
    ),
    RiskDefinition(
        id="reg_banking",
        name="Banking regulation tightening",
        name_fr="Durcissement régulation bancaire",
        category=RiskCategory.REGULATORY,
        keywords=["Basel IV", "capital requirements", "stress test", "banking regulation", "ratio fonds propres", "régulation bancaire"],
        base_probability=0.25,
        base_impact=(-0.15, -0.30),
        sectors=["Financial Services"],
        recovery_months=18
    ),

    # === RISQUES GÉOPOLITIQUES ===
    RiskDefinition(
        id="geo_war",
        name="War / Armed conflict",
        name_fr="Guerre / Conflit armé",
        category=RiskCategory.GEOPOLITICAL,
        keywords=["war", "military conflict", "invasion", "armed conflict", "guerre", "conflit armé", "invasion"],
        base_probability=0.15,
        base_impact=(-0.30, -0.60),
        sectors=["all"],
        recovery_months=36
    ),
    RiskDefinition(
        id="geo_sanctions",
        name="Economic sanctions",
        name_fr="Sanctions économiques",
        category=RiskCategory.GEOPOLITICAL,
        keywords=["sanctions", "embargo", "trade ban", "OFAC", "export controls", "sanctions économiques", "embargo"],
        base_probability=0.25,
        base_impact=(-0.25, -0.50),
        sectors=["Energy", "Technology", "Financial Services", "Industrials", "Basic Materials"],
        recovery_months=24
    ),
    RiskDefinition(
        id="geo_trade_war",
        name="Trade war / Tariffs",
        name_fr="Guerre commerciale / Tarifs douaniers",
        category=RiskCategory.GEOPOLITICAL,
        keywords=["trade war", "tariff", "import duty", "trade tension", "guerre commerciale", "droits de douane"],
        base_probability=0.35,
        base_impact=(-0.15, -0.35),
        sectors=["Technology", "Consumer Cyclical", "Industrials", "Basic Materials"],
        recovery_months=18
    ),
    RiskDefinition(
        id="geo_nationalization",
        name="Nationalization / Expropriation",
        name_fr="Nationalisation / Expropriation",
        category=RiskCategory.GEOPOLITICAL,
        keywords=["nationalization", "expropriation", "state takeover", "nationalisation", "expropriation"],
        base_probability=0.10,
        base_impact=(-0.50, -0.80),
        sectors=["Energy", "Basic Materials", "Utilities", "Financial Services"],
        recovery_months=48
    ),

    # === RISQUES OPÉRATIONNELS ===
    RiskDefinition(
        id="op_cyberattack",
        name="Major cyberattack",
        name_fr="Cyberattaque majeure",
        category=RiskCategory.OPERATIONAL,
        keywords=["cyberattack", "ransomware", "data breach", "hacking", "cyber incident", "cyberattaque", "piratage", "rançongiciel"],
        base_probability=0.30,
        base_impact=(-0.10, -0.30),
        sectors=["all"],
        recovery_months=6
    ),
    RiskDefinition(
        id="op_fraud",
        name="Accounting fraud / Scandal",
        name_fr="Fraude comptable / Scandale",
        category=RiskCategory.OPERATIONAL,
        keywords=["accounting fraud", "financial scandal", "SEC investigation", "fraud", "fraude comptable", "scandale financier"],
        base_probability=0.08,
        base_impact=(-0.40, -0.70),
        sectors=["all"],
        recovery_months=36
    ),
    RiskDefinition(
        id="op_product_recall",
        name="Major product recall",
        name_fr="Rappel produit majeur",
        category=RiskCategory.OPERATIONAL,
        keywords=["product recall", "safety issue", "defect", "recall", "rappel produit", "défaut", "sécurité produit"],
        base_probability=0.20,
        base_impact=(-0.15, -0.35),
        sectors=["Consumer Cyclical", "Consumer Defensive", "Healthcare", "Industrials"],
        recovery_months=12
    ),
    RiskDefinition(
        id="op_supply_chain",
        name="Supply chain disruption",
        name_fr="Rupture chaîne d'approvisionnement",
        category=RiskCategory.OPERATIONAL,
        keywords=["supply chain", "shortage", "logistics crisis", "chip shortage", "rupture approvisionnement", "pénurie"],
        base_probability=0.35,
        base_impact=(-0.15, -0.30),
        sectors=["Technology", "Consumer Cyclical", "Industrials", "Healthcare"],
        recovery_months=12
    ),

    # === RISQUES ESG / CLIMATIQUES ===
    RiskDefinition(
        id="esg_climate_lawsuit",
        name="Climate litigation",
        name_fr="Procès climatique",
        category=RiskCategory.ESG_CLIMATE,
        keywords=["climate lawsuit", "climate litigation", "environmental lawsuit", "procès climatique", "poursuite environnementale"],
        base_probability=0.25,
        base_impact=(-0.20, -0.40),
        sectors=["Energy", "Basic Materials", "Industrials", "Utilities"],
        recovery_months=24
    ),
    RiskDefinition(
        id="esg_stranded_assets",
        name="Stranded assets",
        name_fr="Actifs échoués (stranded assets)",
        category=RiskCategory.ESG_CLIMATE,
        keywords=["stranded assets", "write-off", "asset impairment", "reserves write-down", "actifs échoués", "dépréciation"],
        base_probability=0.30,
        base_impact=(-0.25, -0.50),
        sectors=["Energy", "Utilities"],
        recovery_months=36
    ),
    RiskDefinition(
        id="esg_controversy",
        name="Major ESG controversy",
        name_fr="Controverse ESG majeure",
        category=RiskCategory.ESG_CLIMATE,
        keywords=["ESG scandal", "greenwashing", "environmental damage", "pollution scandal", "controverse ESG", "greenwashing", "pollution"],
        base_probability=0.25,
        base_impact=(-0.15, -0.35),
        sectors=["all"],
        recovery_months=18
    ),
    RiskDefinition(
        id="esg_natural_disaster",
        name="Natural disaster impact",
        name_fr="Impact catastrophe naturelle",
        category=RiskCategory.ESG_CLIMATE,
        keywords=["natural disaster", "hurricane", "flood", "wildfire", "earthquake", "catastrophe naturelle", "ouragan", "inondation"],
        base_probability=0.20,
        base_impact=(-0.10, -0.30),
        sectors=["all"],
        recovery_months=12
    ),

    # === RISQUES MACROÉCONOMIQUES ===
    RiskDefinition(
        id="macro_recession",
        name="Economic recession",
        name_fr="Récession économique",
        category=RiskCategory.MACROECONOMIC,
        keywords=["recession", "economic downturn", "GDP decline", "economic crisis", "récession", "crise économique"],
        base_probability=0.25,
        base_impact=(-0.25, -0.45),
        sectors=["all"],
        recovery_months=24
    ),
    RiskDefinition(
        id="macro_financial_crisis",
        name="Financial crisis",
        name_fr="Crise financière",
        category=RiskCategory.MACROECONOMIC,
        keywords=["financial crisis", "banking crisis", "credit crunch", "systemic risk", "crise financière", "crise bancaire"],
        base_probability=0.10,
        base_impact=(-0.40, -0.60),
        sectors=["all"],
        recovery_months=36
    ),
    RiskDefinition(
        id="macro_inflation",
        name="Hyperinflation / Currency crisis",
        name_fr="Hyperinflation / Crise monétaire",
        category=RiskCategory.MACROECONOMIC,
        keywords=["hyperinflation", "currency crisis", "currency collapse", "monetary crisis", "hyperinflation", "crise monétaire"],
        base_probability=0.15,
        base_impact=(-0.30, -0.50),
        sectors=["all"],
        recovery_months=24
    ),
    RiskDefinition(
        id="macro_interest_rates",
        name="Interest rate shock",
        name_fr="Choc de taux d'intérêt",
        category=RiskCategory.MACROECONOMIC,
        keywords=["rate hike", "interest rate shock", "fed rate", "monetary tightening", "hausse taux", "resserrement monétaire"],
        base_probability=0.30,
        base_impact=(-0.15, -0.30),
        sectors=["Real Estate", "Financial Services", "Utilities", "Technology"],
        recovery_months=12
    ),

    # === RISQUES SPÉCIFIQUES À L'ENTREPRISE ===
    RiskDefinition(
        id="spec_key_person",
        name="Key person departure",
        name_fr="Départ d'un dirigeant clé",
        category=RiskCategory.COMPANY_SPECIFIC,
        keywords=["CEO departure", "executive resignation", "founder leaves", "management change", "départ PDG", "démission dirigeant"],
        base_probability=0.20,
        base_impact=(-0.10, -0.25),
        sectors=["all"],
        recovery_months=12
    ),
    RiskDefinition(
        id="spec_customer_loss",
        name="Major customer loss",
        name_fr="Perte d'un client majeur",
        category=RiskCategory.COMPANY_SPECIFIC,
        keywords=["contract loss", "customer churn", "major client", "contract termination", "perte contrat", "perte client"],
        base_probability=0.25,
        base_impact=(-0.15, -0.30),
        sectors=["all"],
        recovery_months=18
    ),
    RiskDefinition(
        id="spec_patent_loss",
        name="Patent expiry / IP loss",
        name_fr="Expiration brevet / Perte PI",
        category=RiskCategory.COMPANY_SPECIFIC,
        keywords=["patent expiry", "patent cliff", "generic competition", "IP dispute", "expiration brevet", "génériques"],
        base_probability=0.30,
        base_impact=(-0.20, -0.40),
        sectors=["Healthcare", "Technology"],
        recovery_months=24
    ),
    RiskDefinition(
        id="spec_competitor_disruption",
        name="Disruptive competitor",
        name_fr="Concurrent disruptif",
        category=RiskCategory.COMPANY_SPECIFIC,
        keywords=["disruptive competitor", "market disruption", "new entrant", "innovation threat", "concurrent disruptif", "disruption"],
        base_probability=0.35,
        base_impact=(-0.20, -0.40),
        sectors=["all"],
        recovery_months=24
    ),

    # === RISQUES TECHNOLOGIQUES ===
    RiskDefinition(
        id="tech_obsolescence",
        name="Technology obsolescence",
        name_fr="Obsolescence technologique",
        category=RiskCategory.TECHNOLOGICAL,
        keywords=["obsolete technology", "tech disruption", "legacy systems", "obsolescence technologique", "disruption tech"],
        base_probability=0.25,
        base_impact=(-0.25, -0.45),
        sectors=["Technology", "Communication Services", "Consumer Cyclical"],
        recovery_months=24
    ),
    RiskDefinition(
        id="tech_ai_disruption",
        name="AI disruption",
        name_fr="Disruption par l'IA",
        category=RiskCategory.TECHNOLOGICAL,
        keywords=["AI disruption", "artificial intelligence threat", "automation", "ChatGPT competitor", "disruption IA", "intelligence artificielle"],
        base_probability=0.30,
        base_impact=(-0.20, -0.40),
        sectors=["Technology", "Communication Services", "Financial Services", "Healthcare"],
        recovery_months=18
    ),

    # === RISQUES FINANCIERS ===
    RiskDefinition(
        id="fin_bankruptcy",
        name="Bankruptcy risk",
        name_fr="Risque de faillite",
        category=RiskCategory.FINANCIAL,
        keywords=["bankruptcy", "insolvency", "debt default", "chapter 11", "faillite", "insolvabilité", "défaut de paiement"],
        base_probability=0.05,
        base_impact=(-0.70, -0.95),
        sectors=["all"],
        recovery_months=48
    ),
    RiskDefinition(
        id="fin_dividend_cut",
        name="Dividend cut",
        name_fr="Coupe du dividende",
        category=RiskCategory.FINANCIAL,
        keywords=["dividend cut", "dividend suspension", "payout reduction", "coupe dividende", "suspension dividende"],
        base_probability=0.15,
        base_impact=(-0.15, -0.30),
        sectors=["all"],
        recovery_months=12
    ),
    RiskDefinition(
        id="fin_credit_downgrade",
        name="Credit rating downgrade",
        name_fr="Dégradation notation crédit",
        category=RiskCategory.FINANCIAL,
        keywords=["credit downgrade", "rating downgrade", "junk status", "dégradation notation", "downgrade Moody's S&P"],
        base_probability=0.20,
        base_impact=(-0.10, -0.25),
        sectors=["all"],
        recovery_months=12
    ),
]


# ============ CLASSE PRINCIPALE ============

class BlackSwanAnalyzer:
    """
    Analyseur de risques Cygne Noir pour actions.

    Identifie, évalue et quantifie les risques extrêmes
    pouvant impacter le cours d'une action.
    """

    def __init__(self, news_api_key: Optional[str] = None, cache_dir: str = ".black_swan_cache"):
        """
        Initialise l'analyseur.

        Args:
            news_api_key: Clé API pour NewsAPI (optionnel, utilise RSS sinon)
            cache_dir: Répertoire pour le cache des données
        """
        self.news_api_key = news_api_key
        self.cache_dir = cache_dir
        self.risk_database = RISK_DATABASE

        # Créer le répertoire de cache si nécessaire
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Charger le cache des évaluations précédentes
        self.previous_evaluations = self._load_cache()

    def _load_cache(self) -> Dict:
        """Charge le cache des évaluations précédentes."""
        cache_file = os.path.join(self.cache_dir, "evaluations.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Sauvegarde le cache des évaluations."""
        cache_file = os.path.join(self.cache_dir, "evaluations.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.previous_evaluations, f)
        except:
            pass

    def _get_company_info(self, ticker: str) -> Dict:
        """Récupère les informations de l'entreprise via yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")

            # Calculer la volatilité historique
            if not hist.empty and len(hist) > 20:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualisée
                volatility_20d = returns.tail(20).std() * np.sqrt(252)
            else:
                volatility = 0.30  # Valeur par défaut
                volatility_20d = 0.30

            return {
                "name": info.get("shortName", ticker),
                "ticker": ticker,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "country": info.get("country", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("regularMarketPrice", 0),
                "beta": info.get("beta", 1.0),
                "volatility_1y": volatility,
                "volatility_20d": volatility_20d,
                "debt_to_equity": info.get("debtToEquity", 0),
                "free_cash_flow": info.get("freeCashflow", 0),
                "info": info
            }
        except Exception as e:
            return {
                "name": ticker,
                "ticker": ticker,
                "sector": "Unknown",
                "industry": "Unknown",
                "country": "Unknown",
                "market_cap": 0,
                "current_price": 0,
                "beta": 1.0,
                "volatility_1y": 0.30,
                "volatility_20d": 0.30,
                "error": str(e)
            }

    def _get_relevant_risks(self, sector: str) -> List[RiskDefinition]:
        """Filtre les risques pertinents pour un secteur donné."""
        relevant = []
        for risk in self.risk_database:
            if "all" in risk.sectors or sector in risk.sectors:
                relevant.append(risk)
        return relevant

    def _fetch_news_google_rss(self, query: str, days: int = 7) -> List[Dict]:
        """Récupère les actualités via Google News RSS (gratuit)."""
        import xml.etree.ElementTree as ET

        try:
            # Encoder la requête
            encoded_query = requests.utils.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []

            root = ET.fromstring(response.content)
            news_items = []

            for item in root.findall('.//item'):
                title = item.find('title')
                pub_date = item.find('pubDate')
                link = item.find('link')

                if title is not None:
                    news_items.append({
                        "title": title.text,
                        "date": pub_date.text if pub_date is not None else "",
                        "url": link.text if link is not None else "",
                        "source": "Google News"
                    })

            return news_items[:20]  # Limiter à 20 articles

        except Exception as e:
            return []

    def _fetch_news_newsapi(self, query: str, days: int = 7) -> List[Dict]:
        """Récupère les actualités via NewsAPI (nécessite clé API)."""
        if not self.news_api_key:
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "language": "en",
                "apiKey": self.news_api_key
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()
            news_items = []

            for article in data.get("articles", [])[:20]:
                news_items.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "date": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", "Unknown")
                })

            return news_items

        except Exception as e:
            return []

    def _analyze_sentiment_simple(self, text: str) -> float:
        """
        Analyse le sentiment d'un texte de manière simple.
        Retourne un score entre -1 (très négatif) et +1 (très positif).

        Note: Pour une meilleure précision, utiliser TextBlob ou VADER.
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        # Mots très négatifs (score -1)
        very_negative = [
            "crash", "collapse", "disaster", "bankruptcy", "fraud", "scandal",
            "crisis", "catastrophe", "plunge", "plummet", "devastating",
            "effondrement", "faillite", "fraude", "scandale", "catastrophe"
        ]

        # Mots négatifs (score -0.5)
        negative = [
            "decline", "fall", "drop", "loss", "risk", "warning", "concern",
            "trouble", "problem", "issue", "lawsuit", "investigation", "fine",
            "penalty", "cut", "layoff", "downturn", "weak", "disappointing",
            "baisse", "chute", "perte", "risque", "problème", "procès", "amende"
        ]

        # Mots positifs (score +0.5)
        positive = [
            "growth", "gain", "rise", "profit", "success", "strong", "beat",
            "exceed", "improve", "recovery", "opportunity", "innovation",
            "croissance", "hausse", "profit", "succès", "amélioration"
        ]

        # Mots très positifs (score +1)
        very_positive = [
            "surge", "soar", "breakthrough", "record", "excellent", "outstanding",
            "boom", "explosion", "record", "percée", "excellent"
        ]

        score = 0.0
        word_count = 0

        for word in very_negative:
            if word in text_lower:
                score -= 1.0
                word_count += 1

        for word in negative:
            if word in text_lower:
                score -= 0.5
                word_count += 1

        for word in positive:
            if word in text_lower:
                score += 0.5
                word_count += 1

        for word in very_positive:
            if word in text_lower:
                score += 1.0
                word_count += 1

        if word_count > 0:
            return max(-1.0, min(1.0, score / word_count))

        return 0.0

    def _evaluate_risk_from_news(self, company_name: str, risk: RiskDefinition) -> Dict:
        """Évalue un risque basé sur l'analyse des actualités."""

        # Construire la requête de recherche
        search_terms = [company_name] + risk.keywords[:3]  # Limiter les mots-clés
        query = " ".join(search_terms[:4])

        # Récupérer les actualités
        if self.news_api_key:
            news = self._fetch_news_newsapi(query)
        else:
            news = self._fetch_news_google_rss(query)

        if not news:
            return {
                "news_count": 0,
                "sentiment_avg": 0.0,
                "sentiment_trend": "neutral",
                "relevant_headlines": []
            }

        # Analyser le sentiment
        sentiments = []
        relevant_headlines = []

        for article in news:
            text = article.get("title", "") + " " + article.get("description", "")
            sentiment = self._analyze_sentiment_simple(text)
            sentiments.append(sentiment)

            # Vérifier si l'article est pertinent pour ce risque
            text_lower = text.lower()
            if any(kw.lower() in text_lower for kw in risk.keywords):
                relevant_headlines.append({
                    "title": article.get("title", "")[:100],
                    "sentiment": sentiment,
                    "source": article.get("source", "Unknown")
                })

        avg_sentiment = np.mean(sentiments) if sentiments else 0.0

        # Déterminer la tendance
        if avg_sentiment < -0.3:
            trend = "negative"
        elif avg_sentiment > 0.3:
            trend = "positive"
        else:
            trend = "neutral"

        return {
            "news_count": len(news),
            "relevant_count": len(relevant_headlines),
            "sentiment_avg": float(avg_sentiment),
            "sentiment_trend": trend,
            "relevant_headlines": relevant_headlines[:5]
        }

    def _calculate_probability_adjustment(self, news_data: Dict, volatility_ratio: float) -> float:
        """
        Calcule l'ajustement de probabilité basé sur les actualités et la volatilité.

        Args:
            news_data: Données d'analyse des actualités
            volatility_ratio: Ratio volatilité récente / volatilité historique

        Returns:
            Ajustement de probabilité (peut être négatif ou positif)
        """
        adjustment = 0.0

        # Ajustement basé sur le sentiment des news (impact: -0.15 à +0.20)
        sentiment = news_data.get("sentiment_avg", 0)
        if sentiment < 0:
            # Sentiment négatif augmente la probabilité
            adjustment += abs(sentiment) * 0.20
        else:
            # Sentiment positif diminue légèrement la probabilité
            adjustment -= sentiment * 0.10

        # Ajustement basé sur la fréquence des mentions (impact: 0 à +0.15)
        relevant_count = news_data.get("relevant_count", 0)
        if relevant_count > 5:
            adjustment += 0.15
        elif relevant_count > 2:
            adjustment += 0.08
        elif relevant_count > 0:
            adjustment += 0.03

        # Ajustement basé sur la volatilité (impact: -0.05 à +0.15)
        if volatility_ratio > 1.5:
            # Volatilité élevée = stress du marché
            adjustment += 0.15
        elif volatility_ratio > 1.2:
            adjustment += 0.08
        elif volatility_ratio < 0.8:
            # Volatilité basse = calme
            adjustment -= 0.05

        return adjustment

    def _calculate_impact_scenarios(self, risk: RiskDefinition, company_info: Dict) -> Dict:
        """Calcule les scénarios d'impact financier."""

        base_min, base_max = risk.base_impact
        current_price = company_info.get("current_price", 100)
        beta = company_info.get("beta", 1.0) or 1.0

        # Ajuster l'impact en fonction du beta
        beta_multiplier = 0.8 + (beta * 0.2)  # Beta > 1 augmente l'impact

        # Scénarios
        moderate_impact = base_min * beta_multiplier
        severe_impact = (base_min + base_max) / 2 * beta_multiplier
        extreme_impact = base_max * beta_multiplier

        return {
            "moderate": {
                "percentage": round(moderate_impact * 100, 1),
                "price_target": round(current_price * (1 + moderate_impact), 2),
                "loss_per_share": round(current_price * abs(moderate_impact), 2)
            },
            "severe": {
                "percentage": round(severe_impact * 100, 1),
                "price_target": round(current_price * (1 + severe_impact), 2),
                "loss_per_share": round(current_price * abs(severe_impact), 2)
            },
            "extreme": {
                "percentage": round(extreme_impact * 100, 1),
                "price_target": round(current_price * (1 + extreme_impact), 2),
                "loss_per_share": round(current_price * abs(extreme_impact), 2)
            },
            "recovery_months": risk.recovery_months
        }

    def analyze(self, ticker: str, custom_risks: Optional[List[Dict]] = None) -> Dict:
        """
        Analyse complète des risques Cygne Noir pour une action.

        Args:
            ticker: Symbole de l'action (ex: "TTE.PA", "AAPL")
            custom_risks: Liste de risques personnalisés (optionnel)

        Returns:
            Dictionnaire contenant l'analyse complète
        """
        timestamp = datetime.now().isoformat()

        # 1. Récupérer les informations de l'entreprise
        company_info = self._get_company_info(ticker)
        sector = company_info.get("sector", "Unknown")

        # 2. Filtrer les risques pertinents pour ce secteur
        relevant_risks = self._get_relevant_risks(sector)

        # 3. Calculer le ratio de volatilité
        vol_1y = company_info.get("volatility_1y", 0.30)
        vol_20d = company_info.get("volatility_20d", 0.30)
        volatility_ratio = vol_20d / vol_1y if vol_1y > 0 else 1.0

        # 4. Analyser chaque risque
        analyzed_risks = []

        for risk in relevant_risks:
            # Analyse des actualités
            news_data = self._evaluate_risk_from_news(company_info["name"], risk)

            # Calculer la probabilité ajustée
            prob_adjustment = self._calculate_probability_adjustment(news_data, volatility_ratio)
            final_probability = max(0.01, min(0.95, risk.base_probability + prob_adjustment))

            # Calculer la tendance par rapport à l'évaluation précédente
            prev_key = f"{ticker}_{risk.id}"
            prev_prob = self.previous_evaluations.get(prev_key, {}).get("probability", risk.base_probability)
            prob_change = final_probability - prev_prob

            if prob_change > 0.05:
                prob_trend = "up"
            elif prob_change < -0.05:
                prob_trend = "down"
            else:
                prob_trend = "stable"

            # Calculer les scénarios d'impact
            impact_scenarios = self._calculate_impact_scenarios(risk, company_info)

            # Stocker pour le cache
            self.previous_evaluations[prev_key] = {
                "probability": final_probability,
                "timestamp": timestamp
            }

            analyzed_risk = {
                "id": risk.id,
                "name": risk.name,
                "name_fr": risk.name_fr,
                "category": risk.category.value,
                "probability": round(final_probability, 3),
                "probability_percent": round(final_probability * 100, 1),
                "probability_trend": prob_trend,
                "probability_change": round(prob_change, 3),
                "impact_range": [round(x * 100, 1) for x in risk.base_impact],
                "impact_scenarios": impact_scenarios,
                "news_sentiment": round(news_data.get("sentiment_avg", 0), 2),
                "news_count_7d": news_data.get("news_count", 0),
                "relevant_news_count": news_data.get("relevant_count", 0),
                "relevant_headlines": news_data.get("relevant_headlines", []),
                "last_updated": timestamp
            }

            analyzed_risks.append(analyzed_risk)

        # 5. Trier par probabilité décroissante
        analyzed_risks.sort(key=lambda x: x["probability"], reverse=True)

        # 6. Calculer le score de risque global
        if analyzed_risks:
            # Score pondéré par probabilité et impact
            weighted_scores = []
            for risk in analyzed_risks:
                prob = risk["probability"]
                impact = abs(np.mean(risk["impact_range"]) / 100)
                weighted_scores.append(prob * impact)

            # Normaliser sur une échelle de 0-10
            global_score = min(10, sum(weighted_scores) * 20)
        else:
            global_score = 5.0

        # 7. Déterminer le niveau de risque
        if global_score >= 7:
            risk_level = "Très Élevé"
            risk_level_en = "Very High"
            risk_color = "#e74c3c"
        elif global_score >= 5:
            risk_level = "Élevé"
            risk_level_en = "High"
            risk_color = "#e67e22"
        elif global_score >= 3:
            risk_level = "Modéré"
            risk_level_en = "Moderate"
            risk_color = "#f1c40f"
        else:
            risk_level = "Faible"
            risk_level_en = "Low"
            risk_color = "#27ae60"

        # 8. Sauvegarder le cache
        self._save_cache()

        # 9. Construire le résultat final
        result = {
            "company_info": {
                "name": company_info.get("name"),
                "ticker": ticker,
                "sector": sector,
                "industry": company_info.get("industry"),
                "country": company_info.get("country"),
                "market_cap": company_info.get("market_cap"),
                "current_price": company_info.get("current_price"),
                "volatility_1y": round(company_info.get("volatility_1y", 0) * 100, 1),
                "volatility_20d": round(company_info.get("volatility_20d", 0) * 100, 1),
                "volatility_ratio": round(volatility_ratio, 2)
            },
            "risks": analyzed_risks,
            "top_risks": analyzed_risks[:5],  # Top 5 risques
            "global_risk_score": round(global_score, 1),
            "risk_level": risk_level,
            "risk_level_en": risk_level_en,
            "risk_color": risk_color,
            "total_risks_analyzed": len(analyzed_risks),
            "last_updated": timestamp,
            "methodology": {
                "news_source": "NewsAPI" if self.news_api_key else "Google News RSS",
                "sentiment_analysis": "Keyword-based",
                "update_frequency": "On-demand with 6h cache"
            }
        }

        return result

    def plot_risk_matrix(self, analysis: Dict) -> go.Figure:
        """
        Génère une matrice risque/impact interactive.

        Args:
            analysis: Résultat de la fonction analyze()

        Returns:
            Figure Plotly
        """
        risks = analysis.get("risks", [])

        if not risks:
            fig = go.Figure()
            fig.add_annotation(text="Aucun risque à afficher", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        # Préparer les données
        probabilities = [r["probability"] * 100 for r in risks]
        impacts = [abs(np.mean(r["impact_range"])) for r in risks]
        names = [r["name_fr"] for r in risks]
        categories = [r["category"] for r in risks]

        # Calculer la taille des bulles (basée sur le score combiné)
        scores = [p * i / 100 for p, i in zip(probabilities, impacts)]
        sizes = [max(20, min(80, s * 3)) for s in scores]

        # Couleurs par catégorie
        category_colors = {
            "regulatory": "#3498db",
            "geopolitical": "#e74c3c",
            "operational": "#f39c12",
            "esg_climate": "#27ae60",
            "macroeconomic": "#9b59b6",
            "company_specific": "#1abc9c",
            "technological": "#e91e63",
            "financial": "#795548"
        }
        colors = [category_colors.get(c, "#95a5a6") for c in categories]

        # Créer le graphique
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=probabilities,
            y=impacts,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=[n[:20] + "..." if len(n) > 20 else n for n in names],
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                "Probabilité: %{x:.1f}%<br>" +
                "Impact: %{y:.1f}%<br>" +
                "Catégorie: %{customdata[1]}<br>" +
                "<extra></extra>"
            ),
            customdata=list(zip(names, categories))
        ))

        # Zones de risque (background)
        fig.add_shape(type="rect", x0=0, y0=0, x1=30, y1=25,
                      fillcolor="rgba(39, 174, 96, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=30, y0=0, x1=60, y1=25,
                      fillcolor="rgba(241, 196, 15, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=60, y0=0, x1=100, y1=25,
                      fillcolor="rgba(231, 76, 60, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=0, y0=25, x1=30, y1=50,
                      fillcolor="rgba(241, 196, 15, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=30, y0=25, x1=60, y1=50,
                      fillcolor="rgba(230, 126, 34, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=60, y0=25, x1=100, y1=50,
                      fillcolor="rgba(231, 76, 60, 0.2)", line=dict(width=0))
        fig.add_shape(type="rect", x0=0, y0=50, x1=100, y1=100,
                      fillcolor="rgba(231, 76, 60, 0.15)", line=dict(width=0))

        # Labels des zones
        fig.add_annotation(x=15, y=12, text="Risque Faible", showarrow=False,
                           font=dict(size=12, color="rgba(39, 174, 96, 0.5)"))
        fig.add_annotation(x=80, y=75, text="Risque Critique", showarrow=False,
                           font=dict(size=14, color="rgba(231, 76, 60, 0.5)"))

        fig.update_layout(
            template="plotly_dark",
            title=f"Matrice des Risques - {analysis['company_info']['name']}",
            xaxis_title="Probabilité (%)",
            yaxis_title="Impact potentiel (%)",
            xaxis=dict(range=[0, 100], dtick=20),
            yaxis=dict(range=[0, 100], dtick=20),
            height=600,
            showlegend=False
        )

        return fig

    def plot_risk_bars(self, analysis: Dict, top_n: int = 10) -> go.Figure:
        """
        Génère un graphique à barres des probabilités de risque.

        Args:
            analysis: Résultat de la fonction analyze()
            top_n: Nombre de risques à afficher

        Returns:
            Figure Plotly
        """
        risks = analysis.get("risks", [])[:top_n]

        if not risks:
            fig = go.Figure()
            fig.add_annotation(text="Aucun risque à afficher", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        names = [r["name_fr"] for r in risks]
        probabilities = [r["probability_percent"] for r in risks]
        trends = [r["probability_trend"] for r in risks]

        # Couleurs basées sur la probabilité
        colors = []
        for prob in probabilities:
            if prob >= 50:
                colors.append("#e74c3c")
            elif prob >= 30:
                colors.append("#e67e22")
            elif prob >= 15:
                colors.append("#f1c40f")
            else:
                colors.append("#27ae60")

        # Indicateurs de tendance
        trend_symbols = {"up": " ↗", "down": " ↘", "stable": ""}
        names_with_trend = [n + trend_symbols.get(t, "") for n, t in zip(names, trends)]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=names_with_trend[::-1],  # Inverser pour avoir le plus haut en premier
            x=probabilities[::-1],
            orientation='h',
            marker=dict(color=colors[::-1]),
            text=[f"{p:.1f}%" for p in probabilities[::-1]],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Probabilité: %{x:.1f}%<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark",
            title=f"Top {top_n} Risques Cygne Noir - {analysis['company_info']['name']}",
            xaxis_title="Probabilité (%)",
            xaxis=dict(range=[0, 100]),
            height=max(400, top_n * 50),
            margin=dict(l=250)
        )

        return fig

    def plot_impact_scenarios(self, analysis: Dict, top_n: int = 5) -> go.Figure:
        """
        Génère un graphique des scénarios d'impact pour les principaux risques.

        Args:
            analysis: Résultat de la fonction analyze()
            top_n: Nombre de risques à afficher

        Returns:
            Figure Plotly
        """
        risks = analysis.get("top_risks", [])[:top_n]
        current_price = analysis["company_info"]["current_price"]

        if not risks or not current_price:
            fig = go.Figure()
            fig.add_annotation(text="Données insuffisantes", xref="paper", yref="paper", x=0.5, y=0.5)
            return fig

        fig = make_subplots(rows=1, cols=1)

        names = [r["name_fr"][:25] for r in risks]

        # Prix pour chaque scénario
        moderate_prices = [r["impact_scenarios"]["moderate"]["price_target"] for r in risks]
        severe_prices = [r["impact_scenarios"]["severe"]["price_target"] for r in risks]
        extreme_prices = [r["impact_scenarios"]["extreme"]["price_target"] for r in risks]

        x = np.arange(len(names))
        width = 0.25

        fig.add_trace(go.Bar(
            name='Modéré',
            x=[n + " " for n in names],
            y=moderate_prices,
            marker_color='#f1c40f',
            text=[f"${p:.0f}" for p in moderate_prices],
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='Sévère',
            x=[n + "  " for n in names],
            y=severe_prices,
            marker_color='#e67e22',
            text=[f"${p:.0f}" for p in severe_prices],
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='Extrême',
            x=[n + "   " for n in names],
            y=extreme_prices,
            marker_color='#e74c3c',
            text=[f"${p:.0f}" for p in extreme_prices],
            textposition='outside'
        ))

        # Ligne du prix actuel
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Prix actuel: ${current_price:.2f}",
            annotation_position="right"
        )

        fig.update_layout(
            template="plotly_dark",
            title=f"Scénarios d'Impact - {analysis['company_info']['name']}",
            yaxis_title="Prix Cible ($)",
            barmode='group',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        return fig

    def print_report(self, analysis: Dict):
        """Affiche un rapport formaté dans la console."""

        print("\n" + "=" * 70)
        print(f"🦢 ANALYSE CYGNE NOIR - {analysis['company_info']['name']}")
        print("=" * 70)

        print(f"\n📊 INFORMATIONS ENTREPRISE")
        print(f"   Ticker: {analysis['company_info']['ticker']}")
        print(f"   Secteur: {analysis['company_info']['sector']}")
        print(f"   Prix actuel: ${analysis['company_info']['current_price']:.2f}")
        print(f"   Volatilité 1Y: {analysis['company_info']['volatility_1y']:.1f}%")
        print(f"   Volatilité 20J: {analysis['company_info']['volatility_20d']:.1f}%")

        print(f"\n🎯 SCORE DE RISQUE GLOBAL: {analysis['global_risk_score']}/10")
        print(f"   Niveau: {analysis['risk_level']}")
        print(f"   Risques analysés: {analysis['total_risks_analyzed']}")

        print(f"\n⚠️ TOP 5 RISQUES:")
        print("-" * 70)

        for i, risk in enumerate(analysis['top_risks'], 1):
            trend_symbol = {"up": "↗", "down": "↘", "stable": "→"}.get(risk['probability_trend'], "→")
            print(f"\n{i}. {risk['name_fr']}")
            print(f"   Catégorie: {risk['category']}")
            print(f"   Probabilité: {risk['probability_percent']:.1f}% {trend_symbol}")
            print(f"   Impact potentiel: {risk['impact_range'][0]}% à {risk['impact_range'][1]}%")
            print(f"   Sentiment news: {risk['news_sentiment']:.2f}")

            if risk.get('relevant_headlines'):
                print(f"   Actualités pertinentes:")
                for headline in risk['relevant_headlines'][:2]:
                    print(f"      • {headline['title'][:60]}...")

        print("\n" + "=" * 70)
        print(f"📅 Dernière mise à jour: {analysis['last_updated']}")
        print("=" * 70 + "\n")

    def export_to_json(self, analysis: Dict, filepath: str):
        """Exporte l'analyse en JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"✅ Analyse exportée vers {filepath}")


# ============ FONCTIONS POUR INTÉGRATION STREAMLIT ============

def get_risk_category_emoji(category: str) -> str:
    """Retourne l'emoji correspondant à une catégorie de risque."""
    emojis = {
        "regulatory": "⚖️",
        "geopolitical": "🌍",
        "operational": "⚙️",
        "esg_climate": "🌱",
        "macroeconomic": "📉",
        "company_specific": "🏢",
        "technological": "💻",
        "financial": "💰"
    }
    return emojis.get(category, "⚠️")


def get_risk_category_name_fr(category: str) -> str:
    """Retourne le nom français d'une catégorie."""
    names = {
        "regulatory": "Réglementaire",
        "geopolitical": "Géopolitique",
        "operational": "Opérationnel",
        "esg_climate": "ESG / Climat",
        "macroeconomic": "Macroéconomique",
        "company_specific": "Spécifique",
        "technological": "Technologique",
        "financial": "Financier"
    }
    return names.get(category, category)


def get_trend_display(trend: str, change: float) -> Tuple[str, str]:
    """Retourne l'affichage de la tendance (emoji, couleur)."""
    if trend == "up":
        return f"↗ +{abs(change)*100:.1f}%", "#e74c3c"
    elif trend == "down":
        return f"↘ -{abs(change)*100:.1f}%", "#27ae60"
    else:
        return "→ Stable", "#95a5a6"


# ============ EXEMPLE D'UTILISATION ============

if __name__ == "__main__":
    # Test du module
    analyzer = BlackSwanAnalyzer()

    # Analyser TotalEnergies
    print("Analyse de TotalEnergies...")
    results = analyzer.analyze("TTE.PA")

    # Afficher le rapport
    analyzer.print_report(results)

    # Générer les visualisations
    fig_matrix = analyzer.plot_risk_matrix(results)
    fig_matrix.show()

    fig_bars = analyzer.plot_risk_bars(results)
    fig_bars.show()
