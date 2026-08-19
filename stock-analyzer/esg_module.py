"""
Module ESG Amélioré - Collecte Multi-Sources & Détection Avancée du Greenwashing
Version 4.0
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from functools import wraps
from collections import defaultdict
import time
import re
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ CONFIGURATION ============

@dataclass
class ESGConfig:
    """Configuration centralisée."""
    finnhub_key: str = ""
    fmp_key: str = ""
    newsapi_key: str = ""
    cache_ttl: int = 3600
    max_retries: int = 3

CONFIG = ESGConfig()


# ============ CACHE ============

class ESGCache:
    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._default_ttl = default_ttl

    def _make_key(self, func_name: str, *args, **kwargs) -> str:
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        expiry = time.time() + (ttl or self._default_ttl)
        self._cache[key] = (value, expiry)

    def clear(self):
        self._cache.clear()

_cache = ESGCache()


def cached(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = _cache._make_key(func.__name__, *args, **kwargs)
            result = _cache.get(key)
            if result is not None:
                return result
            result = func(*args, **kwargs)
            _cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, ConnectionError) as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed: {e}")
                        return None
                    time.sleep(base_delay * (2 ** attempt))
            return None
        return wrapper
    return decorator


# ============ 1. YFINANCE ESG ============

@cached(ttl=3600)
def get_yfinance_esg(ticker: str) -> Dict[str, Any]:
    """Récupère les données ESG via yfinance (Sustainalytics)."""
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        sustainability = tkr.sustainability

        if sustainability is None or sustainability.empty:
            return {'source': 'yfinance', 'available': False, 'ticker': ticker}

        data = sustainability.to_dict()
        values = data.get('Value', {})

        result = {
            'source': 'yfinance',
            'available': True,
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'total_esg_score': values.get('totalEsg'),
            'environment_score': values.get('environmentScore'),
            'social_score': values.get('socialScore'),
            'governance_score': values.get('governanceScore'),
            'controversy_level': values.get('highestControversy'),
            'esg_performance': values.get('esgPerformance'),
            'peer_group': values.get('peerGroup'),
            'peer_count': values.get('peerCount'),
            'percentile': values.get('percentile'),
            'controversial_sectors': {
                'coal': values.get('coal', False),
                'tobacco': values.get('tobacco', False),
                'controversial_weapons': values.get('controversialWeapons', False),
                'firearms': values.get('smallArms', False),
                'gambling': values.get('gambling', False),
                'nuclear': values.get('nuclear', False),
                'palm_oil': values.get('palmOil', False),
                'pesticides': values.get('pesticides', False),
                'adult_entertainment': values.get('adult', False),
                'military_contract': values.get('militaryContract', False),
            },
        }

        if result['total_esg_score']:
            score = result['total_esg_score']
            if score <= 10:
                result['risk_level'] = 'Négligeable'
                result['risk_category'] = 1
            elif score <= 20:
                result['risk_level'] = 'Faible'
                result['risk_category'] = 2
            elif score <= 30:
                result['risk_level'] = 'Moyen'
                result['risk_category'] = 3
            elif score <= 40:
                result['risk_level'] = 'Élevé'
                result['risk_category'] = 4
            else:
                result['risk_level'] = 'Sévère'
                result['risk_category'] = 5

        return result
    except Exception as e:
        logger.error(f"yfinance ESG error: {e}")
        return {'source': 'yfinance', 'available': False, 'ticker': ticker, 'error': str(e)}


# ============ 2. FINNHUB ESG ============

@cached(ttl=3600)
@retry_with_backoff(max_retries=2)
def get_finnhub_esg(ticker: str, api_key: str = None) -> Dict[str, Any]:
    """Récupère les données ESG via Finnhub."""
    api_key = api_key or CONFIG.finnhub_key
    if not api_key:
        return {'source': 'finnhub', 'available': False, 'error': 'API key required'}

    url = "https://finnhub.io/api/v1/stock/esg"
    params = {'symbol': ticker, 'token': api_key}
    response = requests.get(url, params=params, timeout=10)

    if response.status_code == 200:
        data = response.json()
        if data and data.get('totalESGScore'):
            return {
                'source': 'finnhub',
                'available': True,
                'ticker': ticker,
                'total_esg_score': data.get('totalESGScore'),
                'environment_score': data.get('environmentScore'),
                'social_score': data.get('socialScore'),
                'governance_score': data.get('governanceScore'),
            }
    return {'source': 'finnhub', 'available': False, 'ticker': ticker}


# ============ 3. FMP ESG ============

@cached(ttl=3600)
@retry_with_backoff(max_retries=2)
def get_fmp_esg(ticker: str, api_key: str = None) -> Dict[str, Any]:
    """Récupère les ESG via Financial Modeling Prep."""
    api_key = api_key or CONFIG.fmp_key
    if not api_key:
        return {'source': 'fmp', 'available': False, 'error': 'API key required'}

    url = "https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data"
    params = {'symbol': ticker, 'apikey': api_key}
    response = requests.get(url, params=params, timeout=10)

    if response.status_code == 200:
        data = response.json()
        if data and len(data) > 0:
            latest = data[0]
            return {
                'source': 'fmp',
                'available': True,
                'ticker': ticker,
                'environment_score': latest.get('environmentalScore'),
                'social_score': latest.get('socialScore'),
                'governance_score': latest.get('governanceScore'),
                'esg_score': latest.get('ESGScore'),
                'historical_data': data[:10],
            }
    return {'source': 'fmp', 'available': False, 'ticker': ticker}


# ============ 4. EPA ECHO VIOLATIONS ============

@cached(ttl=7200)
@retry_with_backoff(max_retries=3)
def get_epa_violations(company_name: str) -> Dict[str, Any]:
    """Récupère les violations environnementales via EPA ECHO."""
    try:
        clean_name = re.sub(r'\s+(Inc\.?|Corp\.?|Ltd\.?|LLC|SA|SE|PLC)$', '', company_name, flags=re.IGNORECASE)

        url = "https://echodata.epa.gov/echo/dfr_rest_services.get_facilities"
        params = {'output': 'JSON', 'p_fn': clean_name}
        response = requests.get(url, params=params, timeout=20)

        if response.status_code == 200:
            data = response.json()
            facilities = data.get('Results', {}).get('Facilities', [])

            violations_summary = []
            total_penalties = 0
            total_violations = 0
            non_compliant = 0

            violations_by_law = {'CAA': 0, 'CWA': 0, 'RCRA': 0}

            for facility in facilities[:50]:
                penalties = float(facility.get('TotalPenalties', 0) or 0)
                quarters_nc = int(facility.get('QtrsNoncompliance', 0) or 0)

                total_penalties += penalties
                total_violations += quarters_nc
                if quarters_nc > 0:
                    non_compliant += 1

                for law in ['CAA', 'CWA', 'RCRA']:
                    status = facility.get(f'{law}ComplianceStatus', '')
                    if status and 'Violation' in status:
                        violations_by_law[law] += 1

                violations_summary.append({
                    'facility_name': facility.get('FacName'),
                    'city': facility.get('FacCity'),
                    'state': facility.get('FacState'),
                    'caa_status': facility.get('CAAComplianceStatus'),
                    'cwa_status': facility.get('CWAComplianceStatus'),
                    'rcra_status': facility.get('RCRAComplianceStatus'),
                    'quarters_noncompliance': quarters_nc,
                    'inspections_5yr': facility.get('Inspection5yrCnt', 0),
                    'penalties': penalties,
                })

            compliance_score = (1 - non_compliant / max(1, len(facilities))) * 100 if facilities else None

            severity = 'NONE'
            if total_penalties > 1000000 or total_violations > 20:
                severity = 'CRITICAL'
            elif total_penalties > 500000 or total_violations > 10:
                severity = 'HIGH'
            elif total_penalties > 100000 or total_violations > 5:
                severity = 'MEDIUM'
            elif total_penalties > 0 or total_violations > 0:
                severity = 'LOW'

            return {
                'source': 'epa_echo',
                'available': True,
                'company_name': company_name,
                'facilities_found': len(facilities),
                'facilities': violations_summary,
                'total_penalties': total_penalties,
                'total_violations_quarters': total_violations,
                'non_compliant_facilities': non_compliant,
                'compliance_score': compliance_score,
                'violations_by_law': violations_by_law,
                'has_violations': total_violations > 0 or total_penalties > 0,
                'severity': severity,
            }
        return {'source': 'epa_echo', 'available': False}
    except Exception as e:
        logger.error(f"EPA ECHO error: {e}")
        return {'source': 'epa_echo', 'available': False, 'error': str(e)}


# ============ 5. SEC EDGAR ============

@cached(ttl=86400)
def get_sec_esg_disclosures(ticker: str) -> Dict[str, Any]:
    """Extrait les informations ESG des rapports SEC."""
    try:
        # Recherche du CIK
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=10&output=atom"
        headers = {'User-Agent': 'ESGAnalyzer/4.0 (contact@example.com)'}

        response = requests.get(search_url, headers=headers, timeout=10)

        if response.status_code == 200:
            # Parser pour trouver le CIK
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(response.content)
                # Chercher les filings
                entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')

                filings = []
                for entry in entries[:5]:
                    title = entry.find('{http://www.w3.org/2005/Atom}title')
                    updated = entry.find('{http://www.w3.org/2005/Atom}updated')
                    if title is not None:
                        filings.append({
                            'title': title.text,
                            'date': updated.text if updated is not None else None,
                        })

                return {
                    'source': 'sec_edgar',
                    'available': True,
                    'ticker': ticker,
                    'recent_filings': filings,
                    'has_10k': len(filings) > 0,
                }
            except ET.ParseError:
                pass

        return {'source': 'sec_edgar', 'available': False, 'ticker': ticker}
    except Exception as e:
        logger.error(f"SEC EDGAR error: {e}")
        return {'source': 'sec_edgar', 'available': False, 'error': str(e)}


# ============ 6. NEWS SENTIMENT ============

@cached(ttl=1800)
def get_esg_news_sentiment(company_name: str, ticker: str = None, api_key: str = None) -> Dict[str, Any]:
    """Analyse le sentiment des actualités ESG."""
    api_key = api_key or CONFIG.newsapi_key

    result = {
        'source': 'news_analysis',
        'available': False,
        'company_name': company_name,
        'articles': [],
        'sentiment_score': None,
        'controversy_count': 0,
    }

    if not api_key:
        return result

    try:
        query = f'"{company_name}" AND (ESG OR sustainability OR environmental OR climate)'
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,
            'apiKey': api_key,
        }
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            positive_kw = ['sustainable', 'renewable', 'green', 'clean', 'progress', 'leader']
            negative_kw = ['scandal', 'violation', 'fine', 'lawsuit', 'pollution', 'greenwashing']

            analyzed = []
            scores = []

            for article in articles[:15]:
                content = ((article.get('title', '') or '') + ' ' + (article.get('description', '') or '')).lower()

                pos = sum(1 for kw in positive_kw if kw in content)
                neg = sum(1 for kw in negative_kw if kw in content)

                if pos > neg:
                    sentiment, score = 'positive', 0.5
                elif neg > pos:
                    sentiment, score = 'negative', -0.5
                    result['controversy_count'] += 1
                else:
                    sentiment, score = 'neutral', 0

                scores.append(score)
                analyzed.append({
                    'title': article.get('title'),
                    'source': article.get('source', {}).get('name'),
                    'date': article.get('publishedAt'),
                    'sentiment': sentiment,
                })

            result['available'] = True
            result['articles'] = analyzed
            if scores:
                result['sentiment_score'] = round(sum(scores) / len(scores), 3)

        return result
    except Exception as e:
        logger.error(f"News sentiment error: {e}")
        return result


# ============ 7. SBTI STATUS ============

@cached(ttl=86400)
def get_sbti_status(company_name: str) -> Dict[str, Any]:
    """Vérifie le statut SBTi de l'entreprise."""
    clean_name = re.sub(r'\s+(Inc\.?|Corp\.?|Ltd\.?|LLC)$', '', company_name, flags=re.IGNORECASE).lower()

    return {
        'source': 'sbti',
        'available': True,
        'company_name': company_name,
        'verification_url': f"https://sciencebasedtargets.org/companies-taking-action?search={clean_name.replace(' ', '+')}",
        'note': 'Vérification manuelle recommandée sur sciencebasedtargets.org',
    }


# ============ 8. GREENWASHING DETECTOR ============

class GreenwashingDetector:
    """Détecteur de greenwashing linguistique."""

    BUZZWORDS = {
        'high_risk': ['carbon neutral', 'net zero', '100% sustainable', 'eco-friendly', 'green', 'clean', 'natural'],
        'medium_risk': ['sustainable', 'renewable', 'recycled', 'organic', 'biodegradable', 'conscious'],
        'vague_claims': ['committed to', 'working towards', 'striving for', 'aiming to', 'exploring'],
    }

    CREDIBLE = ['third-party verified', 'audited', 'certified', 'ISO 14001', 'B Corp', 'Science Based Targets', 'CDP', 'GRI']

    @classmethod
    def analyze_text(cls, text: str) -> Dict[str, Any]:
        if not text:
            return {'score': 0, 'flags': []}

        text_lower = text.lower()

        high_count = sum(1 for bw in cls.BUZZWORDS['high_risk'] if bw in text_lower)
        medium_count = sum(1 for bw in cls.BUZZWORDS['medium_risk'] if bw in text_lower)
        vague_count = sum(1 for bw in cls.BUZZWORDS['vague_claims'] if bw in text_lower)
        credible_count = sum(1 for ind in cls.CREDIBLE if ind.lower() in text_lower)

        risk_score = high_count * 15 + medium_count * 8 + vague_count * 12 - credible_count * 20
        risk_score = max(0, min(100, risk_score))

        flags = []
        if high_count > 3:
            flags.append({'type': 'EXCESSIVE_BUZZWORDS', 'severity': 'HIGH', 'count': high_count})
        if vague_count > 2:
            flags.append({'type': 'VAGUE_COMMITMENTS', 'severity': 'MEDIUM', 'count': vague_count})
        if credible_count == 0 and (high_count + medium_count) > 5:
            flags.append({'type': 'NO_VERIFICATION', 'severity': 'HIGH'})

        return {
            'score': risk_score,
            'flags': flags,
            'buzzword_count': {'high': high_count, 'medium': medium_count, 'vague': vague_count},
            'credibility_indicators': credible_count,
        }


# ============ 9. SECTOR BENCHMARK ============

def get_sector_esg_benchmark(sector: str) -> Dict[str, Any]:
    """Récupère les benchmarks ESG du secteur."""
    BENCHMARKS = {
        'Technology': {'avg_esg': 18, 'low': 10, 'high': 28},
        'Energy': {'avg_esg': 35, 'low': 20, 'high': 50},
        'Financial Services': {'avg_esg': 22, 'low': 12, 'high': 32},
        'Healthcare': {'avg_esg': 20, 'low': 12, 'high': 30},
        'Consumer Cyclical': {'avg_esg': 24, 'low': 14, 'high': 35},
        'Industrials': {'avg_esg': 26, 'low': 15, 'high': 38},
        'Basic Materials': {'avg_esg': 32, 'low': 18, 'high': 45},
        'Utilities': {'avg_esg': 28, 'low': 15, 'high': 42},
    }

    for sec_name, data in BENCHMARKS.items():
        if sec_name.lower() in (sector or '').lower():
            return {'source': 'sector_benchmark', 'available': True, 'sector': sec_name, 'benchmark': data}

    return {'source': 'sector_benchmark', 'available': True, 'sector': sector, 'benchmark': {'avg_esg': 25, 'low': 15, 'high': 40}}


# ============ 10. COMPREHENSIVE ESG ============

def get_comprehensive_esg(ticker: str, company_name: str = None, info: dict = None) -> Dict[str, Any]:
    """Agrège les données ESG de toutes les sources."""
    results = {
        'ticker': ticker,
        'company_name': company_name or ticker,
        'timestamp': datetime.now().isoformat(),
        'sources': {},
        'composite_score': None,
        'normalized_scores': {},
        'data_quality': 'low',
    }

    sector = info.get('sector', 'Unknown') if info else 'Unknown'
    results['sector'] = sector

    # Collecte des sources
    results['sources']['yfinance'] = get_yfinance_esg(ticker)

    if CONFIG.finnhub_key:
        results['sources']['finnhub'] = get_finnhub_esg(ticker)

    if CONFIG.fmp_key:
        results['sources']['fmp'] = get_fmp_esg(ticker)

    if company_name:
        results['sources']['epa_echo'] = get_epa_violations(company_name)
        results['sources']['sbti'] = get_sbti_status(company_name)
        results['sources']['news_sentiment'] = get_esg_news_sentiment(company_name, ticker)

    results['sources']['sec_edgar'] = get_sec_esg_disclosures(ticker)
    results['sector_benchmark'] = get_sector_esg_benchmark(sector)

    # Calcul du score composite
    yf_data = results['sources'].get('yfinance', {})
    scores, weights = [], []

    if yf_data.get('available') and yf_data.get('total_esg_score'):
        yf_score = max(0, 100 - (yf_data['total_esg_score'] * 2))
        scores.append(yf_score)
        weights.append(0.40)
        results['normalized_scores']['yfinance'] = yf_score

    if results['sources'].get('finnhub', {}).get('available'):
        fh_score = results['sources']['finnhub'].get('total_esg_score')
        if fh_score:
            scores.append(fh_score)
            weights.append(0.20)
            results['normalized_scores']['finnhub'] = fh_score

    if results['sources'].get('fmp', {}).get('available'):
        fmp_score = results['sources']['fmp'].get('esg_score')
        if fmp_score:
            scores.append(fmp_score)
            weights.append(0.20)
            results['normalized_scores']['fmp'] = fmp_score

    epa = results['sources'].get('epa_echo', {})
    if epa.get('available') and epa.get('compliance_score') is not None:
        scores.append(epa['compliance_score'])
        weights.append(0.20)
        results['normalized_scores']['epa_compliance'] = epa['compliance_score']

    if scores:
        total_weight = sum(weights)
        results['composite_score'] = round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 1)

        if len(scores) >= 4:
            results['data_quality'] = 'high'
        elif len(scores) >= 3:
            results['data_quality'] = 'medium'
        elif len(scores) >= 2:
            results['data_quality'] = 'fair'

        # Cohérence entre sources
        if len(scores) >= 2:
            std_dev = np.std(scores)
            results['source_consistency'] = {
                'std_deviation': round(std_dev, 2),
                'is_consistent': std_dev < 15,
            }

    # Rating global
    if results['composite_score']:
        score = results['composite_score']
        if score >= 75:
            results['overall_rating'] = 'Excellent'
        elif score >= 60:
            results['overall_rating'] = 'Bon'
        elif score >= 45:
            results['overall_rating'] = 'Moyen'
        elif score >= 30:
            results['overall_rating'] = 'Faible'
        else:
            results['overall_rating'] = 'Critique'

    return results


# ============ 11. GREENWASHING ANALYSIS ============

def analyze_greenwashing_risk(ticker: str, company_name: str, info: dict = None) -> Dict[str, Any]:
    """Analyse complète du risque de greenwashing."""
    analysis = {
        'ticker': ticker,
        'company_name': company_name,
        'timestamp': datetime.now().isoformat(),
        'risk_score': 0,
        'risk_level': 'Unknown',
        'risk_color': '#888888',
        'flags': [],
        'positive_points': [],
        'recommendations': [],
        'detailed_scores': {},
    }

    esg_data = get_comprehensive_esg(ticker, company_name, info)
    analysis['esg_data'] = esg_data

    yf_data = esg_data.get('sources', {}).get('yfinance', {})
    epa_data = esg_data.get('sources', {}).get('epa_echo', {})
    news_data = esg_data.get('sources', {}).get('news_sentiment', {})

    analysis['epa_data'] = epa_data
    risk_points = 0

    # === FLAG 1: Écart score/violations ===
    composite = esg_data.get('composite_score') or 0
    if composite > 50 and epa_data.get('has_violations'):
        risk_points += 35
        analysis['flags'].append({
            'type': 'SCORE_VIOLATION_GAP',
            'severity': 'HIGH',
            'title': 'Écart Score/Violations',
            'description': f"Score ESG {composite}/100 malgré violations EPA",
            'detail': f"Pénalités: ${epa_data.get('total_penalties', 0):,.0f}",
        })

    # === FLAG 2: Controverses ===
    controversy = yf_data.get('controversy_level', 0) or 0
    if controversy >= 4:
        risk_points += 25
        analysis['flags'].append({
            'type': 'HIGH_CONTROVERSY',
            'severity': 'HIGH',
            'title': 'Controverses majeures',
            'description': f"Niveau: {controversy}/5",
        })
    elif controversy >= 3:
        risk_points += 15
        analysis['flags'].append({
            'type': 'MODERATE_CONTROVERSY',
            'severity': 'MEDIUM',
            'title': 'Controverses modérées',
            'description': f"Niveau: {controversy}/5",
        })

    # === FLAG 3: Secteurs controversés ===
    if yf_data.get('available'):
        sectors = yf_data.get('controversial_sectors', {})
        weights = {
            'controversial_weapons': ('Armes controversées', 25),
            'coal': ('Charbon', 20),
            'tobacco': ('Tabac', 15),
            'firearms': ('Armes', 12),
            'palm_oil': ('Huile de palme', 10),
        }
        exposed = []
        for key, (name, weight) in weights.items():
            if sectors.get(key):
                exposed.append(name)
                risk_points += weight

        if exposed:
            analysis['flags'].append({
                'type': 'CONTROVERSIAL_SECTORS',
                'severity': 'HIGH' if len(exposed) >= 3 else 'MEDIUM',
                'title': 'Secteurs controversés',
                'description': ', '.join(exposed),
            })

    # === FLAG 4: Pénalités EPA ===
    penalties = epa_data.get('total_penalties', 0) or 0
    if penalties > 1000000:
        risk_points += 35
        analysis['flags'].append({
            'type': 'SEVERE_PENALTIES',
            'severity': 'HIGH',
            'title': 'Pénalités EPA sévères',
            'description': f"${penalties:,.0f}",
        })
    elif penalties > 500000:
        risk_points += 25
        analysis['flags'].append({
            'type': 'HIGH_PENALTIES',
            'severity': 'HIGH',
            'title': 'Pénalités EPA élevées',
            'description': f"${penalties:,.0f}",
        })
    elif penalties > 100000:
        risk_points += 15
        analysis['flags'].append({
            'type': 'MODERATE_PENALTIES',
            'severity': 'MEDIUM',
            'title': 'Pénalités EPA modérées',
            'description': f"${penalties:,.0f}",
        })

    # === FLAG 5: Non-conformité chronique ===
    violations_q = epa_data.get('total_violations_quarters', 0) or 0
    if violations_q > 12:
        risk_points += 20
        analysis['flags'].append({
            'type': 'CHRONIC_NONCOMPLIANCE',
            'severity': 'HIGH',
            'title': 'Non-conformité chronique',
            'description': f"{violations_q} trimestres non-conformes",
        })

    # === FLAG 6: Couverture médiatique négative ===
    if news_data.get('available') and news_data.get('controversy_count', 0) > 5:
        risk_points += 10
        analysis['flags'].append({
            'type': 'NEGATIVE_PRESS',
            'severity': 'MEDIUM',
            'title': 'Presse négative',
            'description': f"{news_data['controversy_count']} articles négatifs",
        })

    # === POINTS POSITIFS ===
    if yf_data.get('available'):
        total_esg = yf_data.get('total_esg_score')
        if total_esg and total_esg < 15:
            risk_points -= 5
            analysis['positive_points'].append({
                'title': 'Risque ESG très faible',
                'description': f"Score: {total_esg}/100",
            })
        if controversy == 0:
            risk_points -= 5
            analysis['positive_points'].append({
                'title': 'Aucune controverse',
                'description': 'Aucun incident ESG majeur',
            })
        if yf_data.get('percentile') and yf_data['percentile'] < 25:
            risk_points -= 5
            analysis['positive_points'].append({
                'title': 'Leader sectoriel',
                'description': f"Top {yf_data['percentile']}%",
            })

    if epa_data.get('available') and not epa_data.get('has_violations'):
        risk_points -= 5
        analysis['positive_points'].append({
            'title': 'Conformité EPA',
            'description': 'Aucune violation récente',
        })

    if news_data.get('available') and (news_data.get('sentiment_score') or 0) > 0.3:
        risk_points -= 5
        analysis['positive_points'].append({
            'title': 'Presse positive',
            'description': f"Sentiment: {news_data['sentiment_score']:.2f}",
        })

    # === SCORE FINAL ===
    analysis['risk_score'] = max(0, min(100, risk_points))

    if analysis['risk_score'] >= 70:
        analysis['risk_level'] = 'CRITIQUE'
        analysis['risk_color'] = '#e74c3c'
    elif analysis['risk_score'] >= 50:
        analysis['risk_level'] = 'ÉLEVÉ'
        analysis['risk_color'] = '#e67e22'
    elif analysis['risk_score'] >= 30:
        analysis['risk_level'] = 'MODÉRÉ'
        analysis['risk_color'] = '#f1c40f'
    elif analysis['risk_score'] >= 10:
        analysis['risk_level'] = 'FAIBLE'
        analysis['risk_color'] = '#2ecc71'
    else:
        analysis['risk_level'] = 'MINIMAL'
        analysis['risk_color'] = '#27ae60'

    # === RECOMMANDATIONS ===
    if analysis['risk_score'] >= 70:
        analysis['recommendations'] = [
            "⚠️ Risque élevé - Investigation approfondie requise",
            "Analyser les rapports de durabilité vs données réelles",
            "Vérifier l'historique des controverses",
            "Comparer avec les leaders ESG du secteur",
        ]
    elif analysis['risk_score'] >= 50:
        analysis['recommendations'] = [
            "Analyser les rapports de durabilité",
            "Comparer les émissions GHG sur 5 ans",
            "Vérifier les certifications (ISO 14001, B Corp...)",
        ]
    elif analysis['risk_score'] >= 30:
        analysis['recommendations'] = [
            "Suivre les scores ESG trimestriellement",
            "Vérifier les engagements sur sciencebasedtargets.org",
        ]
    else:
        analysis['recommendations'] = [
            "Profil ESG satisfaisant - Maintenir une veille",
        ]

    if epa_data.get('has_violations'):
        analysis['recommendations'].append(f"📋 Examiner les violations sur echo.epa.gov")

    # === SCORES DÉTAILLÉS ===
    if yf_data.get('available'):
        env = yf_data.get('environment_score', 50) or 50
        soc = yf_data.get('social_score', 50) or 50
        gov = yf_data.get('governance_score', 50) or 50
        analysis['detailed_scores'] = {
            'environment': max(0, 100 - env * 2),
            'social': max(0, 100 - soc * 2),
            'governance': max(0, 100 - gov * 2),
            'controversy': max(0, 100 - controversy * 20),
        }

    return analysis


# ============ 12. SIMPLE DISPLAY ============

def get_simple_esg_display(ticker: str) -> Dict[str, Any]:
    """Données ESG pour affichage simple."""
    yf_data = get_yfinance_esg(ticker)

    if not yf_data.get('available'):
        return {'available': False}

    env = yf_data.get('environment_score', 50) or 50
    soc = yf_data.get('social_score', 50) or 50
    gov = yf_data.get('governance_score', 50) or 50
    sectors = yf_data.get('controversial_sectors', {})

    return {
        'available': True,
        'total_score': yf_data.get('total_esg_score'),
        'environment': max(0, 100 - env * 2),
        'social': max(0, 100 - soc * 2),
        'governance': max(0, 100 - gov * 2),
        'controversy_level': yf_data.get('controversy_level', 0) or 0,
        'risk_level': yf_data.get('risk_level', 'N/A'),
        'peer_group': yf_data.get('peer_group'),
        'percentile': yf_data.get('percentile'),
        'coal': sectors.get('coal', False),
        'tobacco': sectors.get('tobacco', False),
        'controversial_weapons': sectors.get('controversial_weapons', False),
        'nuclear': sectors.get('nuclear', False),
        'palm_oil': sectors.get('palm_oil', False),
    }


# ============ CONFIG HELPERS ============

def configure_api_keys(finnhub: str = None, fmp: str = None, newsapi: str = None):
    """Configure les clés API."""
    if finnhub:
        CONFIG.finnhub_key = finnhub
    if fmp:
        CONFIG.fmp_key = fmp
    if newsapi:
        CONFIG.newsapi_key = newsapi
    logger.info("API keys configured")


def clear_cache():
    """Vide le cache."""
    _cache.clear()
    logger.info("Cache cleared")


# ============ TEST ============

if __name__ == "__main__":
    print("=" * 50)
    print("Module ESG Amélioré v4.0")
    print("=" * 50)
    print("\nFonctions disponibles:")
    print("  - get_yfinance_esg(ticker)")
    print("  - get_finnhub_esg(ticker)")
    print("  - get_fmp_esg(ticker)")
    print("  - get_epa_violations(company_name)")
    print("  - get_sec_esg_disclosures(ticker)")
    print("  - get_sbti_status(company_name)")
    print("  - get_esg_news_sentiment(company_name)")
    print("  - get_comprehensive_esg(ticker, company_name)")
    print("  - analyze_greenwashing_risk(ticker, company_name)")
    print("  - get_simple_esg_display(ticker)")
    print("  - GreenwashingDetector.analyze_text(text)")
    print("\n✓ Module chargé avec succès")
