"""
News Sentiment Analysis Module
==============================
Implements NLP-based sentiment analysis for financial news.

Features:
- Multiple sentiment models (FinBERT, VADER, TextBlob)
- News aggregation from multiple sources
- Time-weighted sentiment scoring
- Probability calculation from sentiment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import re
import warnings
from collections import Counter

# HTTP requests
import requests

# Sentiment libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .config import (
    SentimentConfig,
    SentimentModel,
    DEFAULT_CONFIG,
    setup_logger,
    RISK_WARNINGS
)

logger = setup_logger(__name__)

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class NewsArticle:
    """Container for a single news article."""
    title: str
    description: Optional[str]
    source: str
    url: str
    published_at: datetime
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description[:200] if self.description else None,
            "source": self.source,
            "url": self.url,
            "date": self.published_at.strftime("%Y-%m-%d"),
            "sentiment": round(self.sentiment_score, 2),
            "label": self.sentiment_label
        }


@dataclass
class SentimentBreakdown:
    """Sentiment distribution statistics."""
    positive_count: int
    neutral_count: int
    negative_count: int
    total_articles: int

    @property
    def positive_ratio(self) -> float:
        return self.positive_count / max(1, self.total_articles)

    @property
    def negative_ratio(self) -> float:
        return self.negative_count / max(1, self.total_articles)

    def to_dict(self) -> Dict[str, int]:
        return {
            "positif": self.positive_count,
            "neutre": self.neutral_count,
            "negatif": self.negative_count
        }


@dataclass
class SentimentResults:
    """Container for sentiment analysis results."""
    sentiment_global: float  # [-1, 1]
    probability_hausse: float  # [0, 1]
    news_analyzed: int
    top_headlines: List[Dict[str, Any]]
    sentiment_breakdown: SentimentBreakdown
    sentiment_by_period: Dict[str, float]
    word_frequencies: Dict[str, int]
    model_used: str
    execution_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment_global": round(self.sentiment_global, 2),
            "probability_hausse": round(self.probability_hausse, 2),
            "news_analyzed": self.news_analyzed,
            "top_headlines": self.top_headlines[:5],
            "sentiment_breakdown": self.sentiment_breakdown.to_dict(),
            "sentiment_by_period": {
                k: round(v, 2) for k, v in self.sentiment_by_period.items()
            },
            "model_used": self.model_used,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "warnings": self.warnings
        }

# ==============================================================================
# NEWS FETCHERS
# ==============================================================================

class NewsAPIFetcher:
    """Fetches news from NewsAPI.org."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(
        self,
        query: str,
        from_date: datetime,
        to_date: datetime,
        max_articles: int = 100
    ) -> List[NewsArticle]:
        """Fetch articles from NewsAPI."""
        if not self.api_key:
            logger.warning("NewsAPI key not configured")
            return []

        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": min(max_articles, 100),
            "apiKey": self.api_key
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get("articles", []):
                try:
                    pub_date = datetime.fromisoformat(
                        item["publishedAt"].replace("Z", "+00:00")
                    )
                    articles.append(NewsArticle(
                        title=item.get("title", ""),
                        description=item.get("description"),
                        source=item.get("source", {}).get("name", "Unknown"),
                        url=item.get("url", ""),
                        published_at=pub_date
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []


class FinnhubNewsFetcher:
    """Fetches news from Finnhub API."""

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime
    ) -> List[NewsArticle]:
        """Fetch company news from Finnhub."""
        if not self.api_key:
            logger.warning("Finnhub API key not configured")
            return []

        params = {
            "symbol": ticker,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "token": self.api_key
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data:
                try:
                    pub_date = datetime.fromtimestamp(item["datetime"])
                    articles.append(NewsArticle(
                        title=item.get("headline", ""),
                        description=item.get("summary"),
                        source=item.get("source", "Unknown"),
                        url=item.get("url", ""),
                        published_at=pub_date
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse Finnhub article: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Finnhub fetch failed: {e}")
            return []


class YahooFinanceNewsFetcher:
    """Fetches news from Yahoo Finance (no API key required)."""

    def fetch(self, ticker: str) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance via yfinance."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            news = stock.news

            articles = []
            for item in news or []:
                try:
                    pub_date = datetime.fromtimestamp(
                        item.get("providerPublishTime", 0)
                    )
                    articles.append(NewsArticle(
                        title=item.get("title", ""),
                        description=None,
                        source=item.get("publisher", "Yahoo Finance"),
                        url=item.get("link", ""),
                        published_at=pub_date
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse Yahoo news: {e}")
                    continue

            return articles

        except Exception as e:
            logger.error(f"Yahoo Finance news fetch failed: {e}")
            return []

# ==============================================================================
# SENTIMENT ANALYZERS
# ==============================================================================

class VADERAnalyzer:
    """VADER sentiment analyzer (rule-based)."""

    def __init__(self):
        if not VADER_AVAILABLE:
            raise ImportError("vaderSentiment package not installed")
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of text.

        Returns:
            Tuple of (score [-1, 1], label)
        """
        if not text:
            return 0.0, "neutral"

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return compound, label


class TextBlobAnalyzer:
    """TextBlob sentiment analyzer."""

    def __init__(self):
        if not TEXTBLOB_AVAILABLE:
            raise ImportError("textblob package not installed")

    def analyze(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment using TextBlob."""
        if not text:
            return 0.0, "neutral"

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return polarity, label


class FinBERTAnalyzer:
    """FinBERT transformer-based sentiment analyzer."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch packages not installed")

        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _load_model(self):
        """Lazy load the model."""
        if not self._loaded:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.eval()
            self._loaded = True

    def analyze(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment using FinBERT."""
        if not text:
            return 0.0, "neutral"

        self._load_model()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        # FinBERT labels: positive, negative, neutral
        probs = probabilities[0].numpy()
        labels = ["positive", "negative", "neutral"]

        # Calculate score: positive - negative
        score = float(probs[0] - probs[1])
        label = labels[int(np.argmax(probs))]

        return score, label

    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, str]]:
        """Analyze multiple texts in batch for efficiency."""
        if not texts:
            return []

        self._load_model()

        results = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)

            for probs in probabilities.numpy():
                score = float(probs[0] - probs[1])
                label = ["positive", "negative", "neutral"][int(np.argmax(probs))]
                results.append((score, label))

        return results

# ==============================================================================
# MAIN SENTIMENT ANALYZER
# ==============================================================================

class NewsSentimentAnalyzer:
    """
    Main sentiment analysis orchestrator.

    Combines news fetching and sentiment analysis.
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        """Initialize analyzer with configuration."""
        self.config = config or DEFAULT_CONFIG.sentiment

        # Initialize fetchers
        self.newsapi_fetcher = NewsAPIFetcher(self.config.newsapi_key)
        self.finnhub_fetcher = FinnhubNewsFetcher(self.config.finnhub_key)
        self.yahoo_fetcher = YahooFinanceNewsFetcher()

        # Initialize sentiment analyzer based on config
        self.sentiment_analyzer = self._create_analyzer()

    def _create_analyzer(self):
        """Create sentiment analyzer based on configuration."""
        model = self.config.model.lower()

        if model == "finbert" and TRANSFORMERS_AVAILABLE:
            return FinBERTAnalyzer(self.config.finbert_model)
        elif model == "vader" and VADER_AVAILABLE:
            return VADERAnalyzer()
        elif TEXTBLOB_AVAILABLE:
            return TextBlobAnalyzer()
        elif VADER_AVAILABLE:
            logger.warning("Falling back to VADER analyzer")
            return VADERAnalyzer()
        else:
            raise ImportError(
                "No sentiment analysis library available. "
                "Install vaderSentiment, textblob, or transformers."
            )

    def analyze(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> SentimentResults:
        """
        Run complete sentiment analysis for a ticker.

        Args:
            ticker: Stock ticker symbol
            company_name: Optional company name for broader search
            lookback_days: Days of news to analyze

        Returns:
            SentimentResults with aggregated sentiment
        """
        start_time = datetime.now()
        warnings_list = []

        lookback = lookback_days or self.config.lookback_days
        to_date = datetime.now()
        from_date = to_date - timedelta(days=lookback)

        # Fetch news from multiple sources
        articles = self._fetch_all_news(
            ticker, company_name, from_date, to_date
        )

        if len(articles) < self.config.min_articles_required:
            warnings_list.append(RISK_WARNINGS["limited_news"])
            logger.warning(f"Only {len(articles)} articles found for {ticker}")

        # Analyze sentiment for each article
        articles = self._analyze_articles(articles)

        # Calculate aggregated metrics
        results = self._aggregate_results(
            articles, ticker, warnings_list, start_time
        )

        return results

    def _fetch_all_news(
        self,
        ticker: str,
        company_name: Optional[str],
        from_date: datetime,
        to_date: datetime
    ) -> List[NewsArticle]:
        """Fetch news from all available sources."""
        all_articles = []

        # Yahoo Finance (always available)
        yahoo_articles = self.yahoo_fetcher.fetch(ticker)
        all_articles.extend(yahoo_articles)
        logger.info(f"Yahoo Finance: {len(yahoo_articles)} articles")

        # Finnhub (if API key available)
        if self.config.finnhub_key:
            time.sleep(self.config.api_delay_seconds)
            finnhub_articles = self.finnhub_fetcher.fetch(
                ticker, from_date, to_date
            )
            all_articles.extend(finnhub_articles)
            logger.info(f"Finnhub: {len(finnhub_articles)} articles")

        # NewsAPI (if API key available)
        if self.config.newsapi_key:
            time.sleep(self.config.api_delay_seconds)
            query = f"{ticker}"
            if company_name:
                query += f" OR {company_name}"

            newsapi_articles = self.newsapi_fetcher.fetch(
                query, from_date, to_date, self.config.max_articles
            )
            all_articles.extend(newsapi_articles)
            logger.info(f"NewsAPI: {len(newsapi_articles)} articles")

        # Deduplicate by title similarity
        unique_articles = self._deduplicate_articles(all_articles)
        logger.info(f"Total unique articles: {len(unique_articles)}")

        # Limit to max_articles
        return unique_articles[:self.config.max_articles]

    def _deduplicate_articles(
        self,
        articles: List[NewsArticle]
    ) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        seen_titles = set()
        unique = []

        for article in articles:
            # Normalize title for comparison
            normalized = re.sub(r'[^\w\s]', '', article.title.lower())
            normalized = ' '.join(normalized.split()[:10])

            if normalized not in seen_titles and len(article.title) > 10:
                seen_titles.add(normalized)
                unique.append(article)

        return unique

    def _analyze_articles(
        self,
        articles: List[NewsArticle]
    ) -> List[NewsArticle]:
        """Analyze sentiment for all articles."""
        if not articles:
            return []

        # Use batch processing for FinBERT
        if isinstance(self.sentiment_analyzer, FinBERTAnalyzer):
            texts = [
                f"{a.title}. {a.description or ''}"
                for a in articles
            ]
            results = self.sentiment_analyzer.analyze_batch(texts)

            for article, (score, label) in zip(articles, results):
                article.sentiment_score = score
                article.sentiment_label = label
        else:
            # Sequential processing for VADER/TextBlob
            for article in articles:
                text = f"{article.title}. {article.description or ''}"
                score, label = self.sentiment_analyzer.analyze(text)
                article.sentiment_score = score
                article.sentiment_label = label

        return articles

    def _aggregate_results(
        self,
        articles: List[NewsArticle],
        ticker: str,
        warnings_list: List[str],
        start_time: datetime
    ) -> SentimentResults:
        """Aggregate sentiment results from all articles."""
        now = datetime.now()

        if not articles:
            return SentimentResults(
                sentiment_global=0.0,
                probability_hausse=0.5,
                news_analyzed=0,
                top_headlines=[],
                sentiment_breakdown=SentimentBreakdown(0, 0, 0, 0),
                sentiment_by_period={},
                word_frequencies={},
                model_used=self.config.model,
                warnings=warnings_list + ["No articles found"]
            )

        # Categorize by time period
        articles_24h = [
            a for a in articles
            if (now - a.published_at).total_seconds() < 86400
        ]
        articles_7d = [
            a for a in articles
            if (now - a.published_at).days < 7
        ]
        articles_30d = articles  # All articles

        # Calculate period sentiments
        sentiment_24h = self._mean_sentiment(articles_24h)
        sentiment_7d = self._mean_sentiment(articles_7d)
        sentiment_30d = self._mean_sentiment(articles_30d)

        # Weighted global sentiment
        sentiment_global = (
            self.config.weight_24h * sentiment_24h +
            self.config.weight_7d * sentiment_7d +
            self.config.weight_30d * sentiment_30d
        )

        # Calculate probability (sigmoid transformation)
        probability_hausse = self._sigmoid(sentiment_global * 2)

        # Sentiment breakdown
        positive_count = sum(1 for a in articles if a.sentiment_label == "positive")
        negative_count = sum(1 for a in articles if a.sentiment_label == "negative")
        neutral_count = len(articles) - positive_count - negative_count

        breakdown = SentimentBreakdown(
            positive_count=positive_count,
            neutral_count=neutral_count,
            negative_count=negative_count,
            total_articles=len(articles)
        )

        # Top headlines (sorted by absolute sentiment)
        sorted_articles = sorted(
            articles,
            key=lambda a: abs(a.sentiment_score),
            reverse=True
        )
        top_headlines = [a.to_dict() for a in sorted_articles[:5]]

        # Word frequencies
        word_freq = self._extract_word_frequencies(articles)

        execution_time = (datetime.now() - start_time).total_seconds()

        return SentimentResults(
            sentiment_global=sentiment_global,
            probability_hausse=probability_hausse,
            news_analyzed=len(articles),
            top_headlines=top_headlines,
            sentiment_breakdown=breakdown,
            sentiment_by_period={
                "24h": sentiment_24h,
                "7d": sentiment_7d,
                "30d": sentiment_30d
            },
            word_frequencies=word_freq,
            model_used=self.config.model,
            execution_time_seconds=execution_time,
            warnings=warnings_list
        )

    def _mean_sentiment(self, articles: List[NewsArticle]) -> float:
        """Calculate mean sentiment from articles."""
        if not articles:
            return 0.0
        return sum(a.sentiment_score for a in articles) / len(articles)

    def _sigmoid(self, x: float) -> float:
        """Sigmoid function to map sentiment to probability."""
        return 1 / (1 + np.exp(-x))

    def _extract_word_frequencies(
        self,
        articles: List[NewsArticle],
        top_n: int = 20
    ) -> Dict[str, int]:
        """Extract most frequent words from headlines."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'it', 'its', 'stock',
            'stocks', 'share', 'shares', 'market', 'markets'
        }

        words = []
        for article in articles:
            title_words = re.findall(r'\b[a-zA-Z]{3,}\b', article.title.lower())
            words.extend([w for w in title_words if w not in stopwords])

        counter = Counter(words)
        return dict(counter.most_common(top_n))


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def analyze_news_sentiment(
    ticker: str,
    company_name: Optional[str] = None,
    lookback_days: int = 30,
    config: Optional[SentimentConfig] = None
) -> SentimentResults:
    """
    Convenience function to analyze sentiment for a ticker.

    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name for search
        lookback_days: Days of news to analyze
        config: Optional SentimentConfig

    Returns:
        SentimentResults with sentiment analysis
    """
    analyzer = NewsSentimentAnalyzer(config)
    return analyzer.analyze(ticker, company_name, lookback_days)
