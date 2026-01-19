"""
Module ESG Pro - Analyse de niveau institutionnel
Sources: GDELT, E-PRTR, OpenSanctions, SEC EDGAR, UK EA, Wikidata, OpenCorporates, LobbyFacts
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
from collections import defaultdict
import time
import re
import logging
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ CACHE & UTILS ============

class ESGCache:
    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl

    def get(self, key: str):
        if key in self._cache:
            val, exp = self._cache[key]
            if time.time() < exp:
                return val
            del self._cache[key]
        return None

    def set(self, key: str, val: Any, ttl: int = None):
        self._cache[key] = (val, time.time() + (ttl or self._ttl))

    def clear(self):
        self._cache.clear()

_cache = ESGCache()


def cached(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = hashlib.md5(f"{func.__name__}:{args}:{kwargs}".encode()).hexdigest()
            result = _cache.get(key)
            if result: return result
            result = func(*args, **kwargs)
            _cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator


def safe_request(url: str, params: dict = None, headers: dict = None, timeout: int = 15) -> Optional[requests.Response]:
    """Requête HTTP sécurisée avec retry."""
    default_headers = {'User-Agent': 'ESGAnalyzer/5.0 (Research Purpose)'}
    if headers:
        default_headers.update(headers)

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=default_headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:  # Rate limit
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt+1}): {e}")
            time.sleep(1)
    return None


# ============ 1. GDELT - SCANDALES & POLÉMIQUES MONDIALES ============

@cached(ttl=1800)
def get_gdelt_controversies(company_name: str, days: int = 90) -> Dict[str, Any]:
    """
    GDELT Project - Monitoring médiatique mondial.
    Détecte scandales, polémiques, controverses ESG.
    """
    result = {
        'source': 'gdelt',
        'available': False,
        'company': company_name,
        'controversies': [],
        'sentiment_avg': None,
        'article_count': 0,
        'top_themes': [],
    }

    try:
        # GDELT DOC 2.0 API
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Recherche avec termes ESG négatifs
        query_terms = [
            f'"{company_name}" scandal',
            f'"{company_name}" controversy',
            f'"{company_name}" lawsuit',
            f'"{company_name}" pollution',
            f'"{company_name}" violation',
            f'"{company_name}" fine',
            f'"{company_name}" fraud',
            f'"{company_name}" corruption',
            f'"{company_name}" discrimination',
            f'"{company_name}" safety',
        ]

        all_articles = []

        for query in query_terms[:5]:  # Limiter pour éviter rate limit
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params = {
                'query': query,
                'mode': 'artlist',
                'maxrecords': 50,
                'format': 'json',
                'startdatetime': start_date.strftime('%Y%m%d%H%M%S'),
                'enddatetime': end_date.strftime('%Y%m%d%H%M%S'),
            }

            resp = safe_request(url, params=params, timeout=20)
            if resp:
                try:
                    data = resp.json()
                    articles = data.get('articles', [])
                    all_articles.extend(articles)
                except:
                    pass

            time.sleep(0.5)  # Rate limiting

        if all_articles:
            # Dédupliquer par URL
            seen_urls = set()
            unique_articles = []
            for art in all_articles:
                url = art.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(art)

            # Analyser les articles
            controversies = []
            sentiments = []
            themes = defaultdict(int)

            for art in unique_articles[:100]:
                title = art.get('title', '')
                tone = art.get('tone', 0)
                domain = art.get('domain', '')
                date = art.get('seendate', '')

                sentiments.append(tone)

                # Catégoriser la controverse
                category = 'OTHER'
                title_lower = title.lower()
                if any(w in title_lower for w in ['pollution', 'environment', 'climate', 'emission', 'spill']):
                    category = 'ENVIRONMENTAL'
                elif any(w in title_lower for w in ['worker', 'employee', 'discrimination', 'harassment', 'safety', 'labor']):
                    category = 'SOCIAL'
                elif any(w in title_lower for w in ['fraud', 'corruption', 'bribe', 'governance', 'board', 'executive']):
                    category = 'GOVERNANCE'
                elif any(w in title_lower for w in ['lawsuit', 'sue', 'court', 'legal', 'fine', 'penalty']):
                    category = 'LEGAL'

                themes[category] += 1

                controversies.append({
                    'title': title,
                    'source': domain,
                    'date': date[:10] if date else None,
                    'tone': round(tone, 2),
                    'category': category,
                    'url': art.get('url'),
                    'language': art.get('language', 'en'),
                })

            # Trier par tone (plus négatif = plus grave)
            controversies.sort(key=lambda x: x['tone'])

            result['available'] = True
            result['controversies'] = controversies[:50]
            result['article_count'] = len(unique_articles)
            result['sentiment_avg'] = round(np.mean(sentiments), 2) if sentiments else None
            result['top_themes'] = sorted(themes.items(), key=lambda x: -x[1])

            # Score de risque médiatique (0-100)
            negative_count = sum(1 for s in sentiments if s < -2)
            result['media_risk_score'] = min(100, int((negative_count / max(1, len(sentiments))) * 100 + len(unique_articles) / 2))

    except Exception as e:
        logger.error(f"GDELT error: {e}")
        result['error'] = str(e)

    return result


# ============ 2. E-PRTR - VIOLATIONS ENVIRONNEMENTALES EUROPE ============

@cached(ttl=7200)
def get_eprtr_violations(company_name: str) -> Dict[str, Any]:
    """
    European Pollutant Release and Transfer Register.
    Données sur les rejets polluants en Europe.
    """
    result = {
        'source': 'eprtr',
        'available': False,
        'company': company_name,
        'facilities': [],
        'total_releases': {},
        'countries': [],
    }

    try:
        # API E-PRTR via EEA
        clean_name = re.sub(r'\s+(Inc\.?|Corp\.?|Ltd\.?|SA|SE|PLC|AG|GmbH)$', '', company_name, flags=re.I)

        url = "https://prtr.eea.europa.eu/api/facilities"
        params = {
            'facilityName': clean_name,
            'format': 'json',
        }

        resp = safe_request(url, params=params, timeout=20)

        if resp:
            data = resp.json()
            facilities = data if isinstance(data, list) else data.get('facilities', [])

            if facilities:
                facility_list = []
                countries = set()
                total_releases = defaultdict(float)

                for fac in facilities[:30]:
                    countries.add(fac.get('countryCode', 'Unknown'))

                    # Récupérer les rejets pour cette installation
                    releases = fac.get('releases', [])
                    for rel in releases:
                        pollutant = rel.get('pollutantName', 'Unknown')
                        quantity = rel.get('quantity', 0) or 0
                        total_releases[pollutant] += quantity

                    facility_list.append({
                        'name': fac.get('facilityName'),
                        'country': fac.get('countryCode'),
                        'city': fac.get('city'),
                        'activity': fac.get('mainActivityName'),
                        'releases_count': len(releases),
                        'lat': fac.get('latitude'),
                        'lon': fac.get('longitude'),
                    })

                result['available'] = True
                result['facilities'] = facility_list
                result['total_releases'] = dict(sorted(total_releases.items(), key=lambda x: -x[1])[:10])
                result['countries'] = list(countries)
                result['facilities_count'] = len(facilities)

    except Exception as e:
        logger.error(f"E-PRTR error: {e}")
        result['error'] = str(e)

    return result


# ============ 3. OPENSANCTIONS - SANCTIONS & CONTROVERSES ============

@cached(ttl=7200)
def get_opensanctions_data(company_name: str) -> Dict[str, Any]:
    """
    OpenSanctions - Base de données mondiale des sanctions,
    personnes politiquement exposées (PEP), et entités à risque.
    """
    result = {
        'source': 'opensanctions',
        'available': False,
        'company': company_name,
        'matches': [],
        'sanctions': [],
        'risk_level': 'UNKNOWN',
        'datasets': [],
    }

    try:
        # OpenSanctions API (gratuit pour usage non-commercial)
        url = "https://api.opensanctions.org/match/default"

        payload = {
            'queries': {
                'company': {
                    'schema': 'Company',
                    'properties': {
                        'name': [company_name],
                    }
                }
            }
        }

        headers = {'Content-Type': 'application/json'}

        resp = requests.post(url, json=payload, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            responses = data.get('responses', {})
            company_results = responses.get('company', {}).get('results', [])

            if company_results:
                matches = []
                sanctions = []
                datasets = set()

                for match in company_results[:20]:
                    score = match.get('score', 0)
                    properties = match.get('properties', {})

                    # Extraire les datasets (sources des sanctions)
                    match_datasets = match.get('datasets', [])
                    datasets.update(match_datasets)

                    match_info = {
                        'name': properties.get('name', ['Unknown'])[0] if properties.get('name') else 'Unknown',
                        'score': round(score * 100, 1),
                        'country': properties.get('country', ['Unknown'])[0] if properties.get('country') else 'Unknown',
                        'datasets': match_datasets,
                        'topics': match.get('topics', []),
                        'caption': match.get('caption'),
                    }

                    matches.append(match_info)

                    # Si score élevé, c'est une vraie sanction
                    if score > 0.7:
                        sanctions.append({
                            'entity': match_info['name'],
                            'reason': ', '.join(match.get('topics', [])),
                            'sources': match_datasets,
                        })

                result['available'] = True
                result['matches'] = matches
                result['sanctions'] = sanctions
                result['datasets'] = list(datasets)

                # Évaluer le niveau de risque
                if sanctions:
                    result['risk_level'] = 'HIGH'
                elif any(m['score'] > 50 for m in matches):
                    result['risk_level'] = 'MEDIUM'
                else:
                    result['risk_level'] = 'LOW'

    except Exception as e:
        logger.error(f"OpenSanctions error: {e}")
        result['error'] = str(e)

    return result


# ============ 4. SEC EDGAR - GOUVERNANCE & LOBBYING US ============

@cached(ttl=86400)
def get_sec_governance(ticker: str) -> Dict[str, Any]:
    """
    SEC EDGAR - Données de gouvernance, rémunération executives,
    proxy statements, et Form 8-K (événements matériels).
    """
    result = {
        'source': 'sec_edgar',
        'available': False,
        'ticker': ticker,
        'company_name': None,
        'cik': None,
        'filings': [],
        'governance_issues': [],
        'executive_compensation': None,
        'insider_trades': [],
    }

    try:
        # Trouver le CIK
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=&dateb=&owner=include&count=10&output=json"
        headers = {'User-Agent': 'ESGAnalyzer/5.0 research@example.com'}

        resp = safe_request(search_url, headers=headers)
        if not resp:
            # Essayer avec le ticker directement
            search_url = f"https://data.sec.gov/submissions/CIK{ticker.upper().zfill(10)}.json"
            resp = safe_request(search_url, headers=headers)

        if resp:
            data = resp.json()

            # Extraire les infos de base
            cik = data.get('cik', '')
            result['cik'] = cik
            result['company_name'] = data.get('name')
            result['sic'] = data.get('sic')
            result['sic_description'] = data.get('sicDescription')

            # Récupérer les filings récents
            filings = data.get('filings', {}).get('recent', {})
            forms = filings.get('form', [])
            dates = filings.get('filingDate', [])
            descriptions = filings.get('primaryDocument', [])

            relevant_forms = ['10-K', '10-Q', 'DEF 14A', '8-K', 'SC 13D', 'SC 13G', '4', 'S-1']

            filing_list = []
            governance_issues = []

            for i, form in enumerate(forms[:100]):
                if form in relevant_forms:
                    filing_info = {
                        'form': form,
                        'date': dates[i] if i < len(dates) else None,
                        'description': descriptions[i] if i < len(descriptions) else None,
                    }
                    filing_list.append(filing_info)

                    # Détecter les 8-K problématiques (événements matériels)
                    if form == '8-K':
                        desc = (descriptions[i] if i < len(descriptions) else '').lower()
                        if any(w in desc for w in ['departure', 'termination', 'investigation', 'restatement', 'delisting']):
                            governance_issues.append({
                                'type': 'MATERIAL_EVENT',
                                'date': dates[i] if i < len(dates) else None,
                                'description': descriptions[i] if i < len(descriptions) else None,
                            })

            result['available'] = True
            result['filings'] = filing_list[:30]
            result['governance_issues'] = governance_issues
            result['filing_count'] = len(filing_list)

            # Vérifier les insider trades (Form 4)
            form4_count = sum(1 for f in forms if f == '4')
            result['insider_trade_count'] = form4_count

    except Exception as e:
        logger.error(f"SEC EDGAR error: {e}")
        result['error'] = str(e)

    return result


# ============ 5. UK ENVIRONMENT AGENCY - VIOLATIONS UK ============

@cached(ttl=7200)
def get_uk_environment_violations(company_name: str) -> Dict[str, Any]:
    """
    UK Environment Agency - Violations environnementales au Royaume-Uni.
    Permits, compliance, incidents.
    """
    result = {
        'source': 'uk_environment_agency',
        'available': False,
        'company': company_name,
        'permits': [],
        'compliance_events': [],
        'pollution_incidents': [],
    }

    try:
        # API UK Environment Agency
        clean_name = re.sub(r'\s+(Ltd\.?|PLC|Limited)$', '', company_name, flags=re.I)

        # Recherche des permits
        url = "https://environment.data.gov.uk/public-register/permits/search"
        params = {'name': clean_name, 'format': 'json'}

        resp = safe_request(url, params=params)

        if resp:
            data = resp.json()
            permits = data.get('items', [])

            permit_list = []
            for permit in permits[:20]:
                permit_list.append({
                    'permit_number': permit.get('permitNumber'),
                    'site_name': permit.get('siteName'),
                    'status': permit.get('status'),
                    'type': permit.get('permitType'),
                    'effective_date': permit.get('effectiveDate'),
                })

            result['permits'] = permit_list
            result['available'] = len(permit_list) > 0

        # Recherche des incidents de pollution
        incident_url = "https://environment.data.gov.uk/public-register/pollution-incidents/search"
        resp2 = safe_request(incident_url, params={'operator': clean_name, 'format': 'json'})

        if resp2:
            incidents = resp2.json().get('items', [])
            result['pollution_incidents'] = [{
                'date': inc.get('incidentDate'),
                'type': inc.get('incidentType'),
                'severity': inc.get('severity'),
                'location': inc.get('location'),
            } for inc in incidents[:10]]

    except Exception as e:
        logger.error(f"UK EA error: {e}")
        result['error'] = str(e)

    return result


# ============ 6. WIKIDATA - CONTROVERSES HISTORIQUES ============

@cached(ttl=86400)
def get_wikidata_controversies(company_name: str) -> Dict[str, Any]:
    """
    Wikidata/Wikipedia - Controverses historiques et informations structurées.
    """
    result = {
        'source': 'wikidata',
        'available': False,
        'company': company_name,
        'wikidata_id': None,
        'controversies': [],
        'legal_cases': [],
        'industry': None,
        'founded': None,
        'headquarters': None,
    }

    try:
        # Recherche Wikidata
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'search': company_name,
            'language': 'en',
            'type': 'item',
            'format': 'json',
        }

        resp = safe_request(search_url, params=params)

        if resp:
            data = resp.json()
            results = data.get('search', [])

            if results:
                entity_id = results[0].get('id')
                result['wikidata_id'] = entity_id

                # Récupérer les détails de l'entité
                entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                entity_resp = safe_request(entity_url)

                if entity_resp:
                    entity_data = entity_resp.json()
                    entity = entity_data.get('entities', {}).get(entity_id, {})
                    claims = entity.get('claims', {})

                    # Extraire les infos pertinentes
                    # P31 = instance of, P17 = country, P571 = founded, P452 = industry

                    # Controverses et procès (P793 = significant event)
                    if 'P793' in claims:
                        for claim in claims['P793']:
                            event_id = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')
                            if event_id:
                                result['controversies'].append({'event_id': event_id})

                    # Procès (P1344 = participant in)
                    if 'P1344' in claims:
                        for claim in claims['P1344']:
                            case_id = claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id')
                            if case_id:
                                result['legal_cases'].append({'case_id': case_id})

                    result['available'] = True

        # Récupérer les controverses depuis Wikipedia
        wiki_url = "https://en.wikipedia.org/w/api.php"
        wiki_params = {
            'action': 'query',
            'titles': company_name,
            'prop': 'extracts|categories',
            'exintro': True,
            'explaintext': True,
            'format': 'json',
        }

        wiki_resp = safe_request(wiki_url, params=wiki_params)

        if wiki_resp:
            wiki_data = wiki_resp.json()
            pages = wiki_data.get('query', {}).get('pages', {})

            for page_id, page in pages.items():
                if page_id != '-1':
                    # Chercher les catégories de controverse
                    categories = page.get('categories', [])
                    controversy_cats = [c for c in categories if 'controversy' in c.get('title', '').lower()
                                        or 'scandal' in c.get('title', '').lower()
                                        or 'lawsuit' in c.get('title', '').lower()]

                    if controversy_cats:
                        result['wikipedia_controversy_categories'] = [c['title'] for c in controversy_cats]

                    result['available'] = True

    except Exception as e:
        logger.error(f"Wikidata error: {e}")
        result['error'] = str(e)

    return result


# ============ 7. OPENCORPORATES - STRUCTURE ENTREPRISE ============

@cached(ttl=86400)
def get_opencorporates_data(company_name: str) -> Dict[str, Any]:
    """
    OpenCorporates - Structure juridique, filiales, dirigeants.
    """
    result = {
        'source': 'opencorporates',
        'available': False,
        'company': company_name,
        'companies': [],
        'officers': [],
        'jurisdictions': [],
        'inactive_subsidiaries': 0,
    }

    try:
        # API OpenCorporates (version gratuite limitée)
        url = "https://api.opencorporates.com/v0.4/companies/search"
        params = {
            'q': company_name,
            'format': 'json',
        }

        resp = safe_request(url, params=params)

        if resp:
            data = resp.json()
            companies = data.get('results', {}).get('companies', [])

            company_list = []
            jurisdictions = set()
            inactive = 0

            for comp in companies[:20]:
                c = comp.get('company', {})
                jurisdiction = c.get('jurisdiction_code', 'Unknown')
                jurisdictions.add(jurisdiction)

                status = c.get('current_status', '')
                if status and status.lower() in ['dissolved', 'inactive', 'struck off']:
                    inactive += 1

                company_list.append({
                    'name': c.get('name'),
                    'jurisdiction': jurisdiction,
                    'company_number': c.get('company_number'),
                    'status': status,
                    'incorporation_date': c.get('incorporation_date'),
                    'company_type': c.get('company_type'),
                    'registered_address': c.get('registered_address_in_full'),
                })

            result['available'] = len(company_list) > 0
            result['companies'] = company_list
            result['jurisdictions'] = list(jurisdictions)
            result['inactive_subsidiaries'] = inactive
            result['total_entities'] = len(companies)

    except Exception as e:
        logger.error(f"OpenCorporates error: {e}")
        result['error'] = str(e)

    return result


# ============ 8. LOBBYFACTS EU - LOBBYING EUROPÉEN ============

@cached(ttl=86400)
def get_eu_lobbying(company_name: str) -> Dict[str, Any]:
    """
    LobbyFacts / EU Transparency Register - Lobbying européen.
    """
    result = {
        'source': 'lobbyfacts_eu',
        'available': False,
        'company': company_name,
        'registrations': [],
        'lobbying_costs': None,
        'lobbyists_count': None,
        'eu_meetings': [],
    }

    try:
        # EU Transparency Register API
        clean_name = re.sub(r'\s+(Inc\.?|Corp\.?|Ltd\.?)$', '', company_name, flags=re.I)

        url = "https://data.europa.eu/api/hub/search/search"
        params = {
            'q': f'"{clean_name}" lobbying',
            'limit': 10,
            'format': 'json',
        }

        resp = safe_request(url, params=params)

        if resp:
            data = resp.json()
            results = data.get('result', {}).get('results', [])

            if results:
                result['available'] = True
                result['eu_datasets'] = [{
                    'title': r.get('title', {}).get('en', 'Unknown'),
                    'description': r.get('description', {}).get('en', '')[:200],
                } for r in results[:5]]

        # Alternative: IntegrityWatch.eu
        integrity_url = f"https://www.integritywatch.eu/api/search?q={clean_name}"
        # Note: Cette API peut ne pas être publique, fallback

    except Exception as e:
        logger.error(f"EU Lobbying error: {e}")
        result['error'] = str(e)

    return result


# ============ 9. EPA ECHO - VIOLATIONS US (AMÉLIORÉ) ============

@cached(ttl=7200)
def get_epa_violations_detailed(company_name: str) -> Dict[str, Any]:
    """
    EPA ECHO - Violations environnementales US détaillées.
    Inclut historique des amendes et inspections.
    """
    result = {
        'source': 'epa_echo',
        'available': False,
        'company': company_name,
        'facilities': [],
        'total_penalties': 0,
        'total_inspections': 0,
        'violations_by_law': {},
        'violation_history': [],
        'compliance_score': None,
        'severity': 'NONE',
    }

    try:
        clean_name = re.sub(r'\s+(Inc\.?|Corp\.?|Ltd\.?|LLC)$', '', company_name, flags=re.I)

        url = "https://echodata.epa.gov/echo/dfr_rest_services.get_facilities"
        params = {'output': 'JSON', 'p_fn': clean_name}

        resp = safe_request(url, params=params, timeout=25)

        if resp:
            data = resp.json()
            facilities = data.get('Results', {}).get('Facilities', [])

            if facilities:
                facility_list = []
                total_penalties = 0
                total_inspections = 0
                violations_by_law = {'CAA': 0, 'CWA': 0, 'RCRA': 0, 'SDWA': 0, 'TSCA': 0}
                violation_history = []
                non_compliant = 0

                for fac in facilities[:50]:
                    penalties = float(fac.get('TotalPenalties', 0) or 0)
                    quarters_nc = int(fac.get('QtrsNoncompliance', 0) or 0)
                    inspections = int(fac.get('Inspection5yrCnt', 0) or 0)

                    total_penalties += penalties
                    total_inspections += inspections

                    if quarters_nc > 0:
                        non_compliant += 1

                    # Violations par loi
                    for law in ['CAA', 'CWA', 'RCRA']:
                        status = fac.get(f'{law}ComplianceStatus', '')
                        if status and 'Violation' in status:
                            violations_by_law[law] += 1

                    # Historique des pénalités
                    if penalties > 0:
                        violation_history.append({
                            'facility': fac.get('FacName'),
                            'state': fac.get('FacState'),
                            'penalty': penalties,
                            'quarters_nc': quarters_nc,
                        })

                    facility_list.append({
                        'facility_name': fac.get('FacName'),
                        'facility_id': fac.get('RegistryID'),
                        'city': fac.get('FacCity'),
                        'state': fac.get('FacState'),
                        'caa_status': fac.get('CAAComplianceStatus'),
                        'cwa_status': fac.get('CWAComplianceStatus'),
                        'rcra_status': fac.get('RCRAComplianceStatus'),
                        'quarters_noncompliance': quarters_nc,
                        'inspections_5yr': inspections,
                        'penalties': penalties,
                        'naics': fac.get('NAICSCodes'),
                    })

                # Score de conformité
                compliance_score = (1 - non_compliant / max(1, len(facilities))) * 100

                # Sévérité
                if total_penalties > 5000000:
                    severity = 'CRITICAL'
                elif total_penalties > 1000000:
                    severity = 'HIGH'
                elif total_penalties > 100000:
                    severity = 'MEDIUM'
                elif total_penalties > 0:
                    severity = 'LOW'
                else:
                    severity = 'NONE'

                result['available'] = True
                result['facilities'] = facility_list
                result['facilities_count'] = len(facilities)
                result['total_penalties'] = total_penalties
                result['total_inspections'] = total_inspections
                result['violations_by_law'] = violations_by_law
                result['violation_history'] = sorted(violation_history, key=lambda x: -x['penalty'])[:10]
                result['compliance_score'] = round(compliance_score, 1)
                result['severity'] = severity
                result['non_compliant_facilities'] = non_compliant

    except Exception as e:
        logger.error(f"EPA ECHO error: {e}")
        result['error'] = str(e)

    return result


# ============ 10. YFINANCE ESG (BASE) ============

@cached(ttl=3600)
def get_yfinance_esg(ticker: str) -> Dict[str, Any]:
    """Yahoo Finance - Scores ESG Sustainalytics."""
    try:
        import yfinance as yf
        tkr = yf.Ticker(ticker)
        sustainability = tkr.sustainability

        if sustainability is None or sustainability.empty:
            return {'source': 'yfinance', 'available': False, 'ticker': ticker}

        data = sustainability.to_dict()
        values = data.get('Value', {})

        return {
            'source': 'yfinance',
            'available': True,
            'ticker': ticker,
            'total_esg_score': values.get('totalEsg'),
            'environment_score': values.get('environmentScore'),
            'social_score': values.get('socialScore'),
            'governance_score': values.get('governanceScore'),
            'controversy_level': values.get('highestControversy'),
            'peer_group': values.get('peerGroup'),
            'percentile': values.get('percentile'),
            'controversial_sectors': {
                'coal': values.get('coal', False),
                'tobacco': values.get('tobacco', False),
                'weapons': values.get('controversialWeapons', False),
                'nuclear': values.get('nuclear', False),
                'palm_oil': values.get('palmOil', False),
            },
        }
    except Exception as e:
        return {'source': 'yfinance', 'available': False, 'error': str(e)}


# ============ AGRÉGATEUR PRO ============

def get_comprehensive_esg_pro(ticker: str, company_name: str, info: dict = None) -> Dict[str, Any]:
    """
    Agrégation de TOUTES les sources pour analyse de niveau institutionnel.
    """
    result = {
        'ticker': ticker,
        'company_name': company_name,
        'timestamp': datetime.now().isoformat(),
        'sources': {},
        'summary': {},
        'risk_scores': {},
        'total_risk_score': 0,
    }

    # 1. Yahoo Finance ESG
    result['sources']['yfinance'] = get_yfinance_esg(ticker)

    # 2. GDELT - Scandales médiatiques
    result['sources']['gdelt'] = get_gdelt_controversies(company_name)

    # 3. EPA ECHO - Violations US
    result['sources']['epa_echo'] = get_epa_violations_detailed(company_name)

    # 4. E-PRTR - Violations EU
    result['sources']['eprtr'] = get_eprtr_violations(company_name)

    # 5. OpenSanctions - Sanctions
    result['sources']['opensanctions'] = get_opensanctions_data(company_name)

    # 6. SEC EDGAR - Gouvernance
    result['sources']['sec_edgar'] = get_sec_governance(ticker)

    # 7. UK Environment Agency
    result['sources']['uk_ea'] = get_uk_environment_violations(company_name)

    # 8. Wikidata - Controverses historiques
    result['sources']['wikidata'] = get_wikidata_controversies(company_name)

    # 9. OpenCorporates - Structure
    result['sources']['opencorporates'] = get_opencorporates_data(company_name)

    # 10. EU Lobbying
    result['sources']['eu_lobbying'] = get_eu_lobbying(company_name)

    # === CALCUL DES SCORES DE RISQUE ===

    # Risque médiatique (GDELT)
    gdelt = result['sources']['gdelt']
    if gdelt.get('available'):
        result['risk_scores']['media'] = gdelt.get('media_risk_score', 0)

    # Risque environnemental (EPA + E-PRTR)
    env_risk = 0
    epa = result['sources']['epa_echo']
    if epa.get('available'):
        penalties = epa.get('total_penalties', 0)
        if penalties > 1000000: env_risk += 40
        elif penalties > 100000: env_risk += 20
        elif penalties > 10000: env_risk += 10

        if epa.get('severity') == 'CRITICAL': env_risk += 30
        elif epa.get('severity') == 'HIGH': env_risk += 20

    eprtr = result['sources']['eprtr']
    if eprtr.get('available') and eprtr.get('facilities'):
        env_risk += min(20, len(eprtr['facilities']) * 2)

    result['risk_scores']['environmental'] = min(100, env_risk)

    # Risque sanctions
    sanctions = result['sources']['opensanctions']
    if sanctions.get('available'):
        if sanctions.get('risk_level') == 'HIGH':
            result['risk_scores']['sanctions'] = 80
        elif sanctions.get('risk_level') == 'MEDIUM':
            result['risk_scores']['sanctions'] = 40
        else:
            result['risk_scores']['sanctions'] = 10
    else:
        result['risk_scores']['sanctions'] = 0

    # Risque gouvernance
    gov_risk = 0
    sec = result['sources']['sec_edgar']
    if sec.get('available'):
        gov_issues = len(sec.get('governance_issues', []))
        gov_risk += gov_issues * 15

    yf = result['sources']['yfinance']
    if yf.get('available'):
        controversy = yf.get('controversy_level', 0) or 0
        gov_risk += controversy * 10

    result['risk_scores']['governance'] = min(100, gov_risk)

    # Risque social (from GDELT themes)
    social_risk = 0
    if gdelt.get('available'):
        themes = dict(gdelt.get('top_themes', []))
        social_risk = themes.get('SOCIAL', 0) * 5
    result['risk_scores']['social'] = min(100, social_risk)

    # Score total pondéré
    weights = {
        'media': 0.15,
        'environmental': 0.30,
        'sanctions': 0.20,
        'governance': 0.20,
        'social': 0.15,
    }

    total = sum(result['risk_scores'].get(k, 0) * w for k, w in weights.items())
    result['total_risk_score'] = round(total, 1)

    # Niveau de risque global
    if result['total_risk_score'] >= 70:
        result['risk_level'] = 'CRITICAL'
        result['risk_color'] = '#e74c3c'
    elif result['total_risk_score'] >= 50:
        result['risk_level'] = 'HIGH'
        result['risk_color'] = '#e67e22'
    elif result['total_risk_score'] >= 30:
        result['risk_level'] = 'MEDIUM'
        result['risk_color'] = '#f1c40f'
    elif result['total_risk_score'] >= 10:
        result['risk_level'] = 'LOW'
        result['risk_color'] = '#2ecc71'
    else:
        result['risk_level'] = 'MINIMAL'
        result['risk_color'] = '#27ae60'

    # Résumé
    result['summary'] = {
        'scandals_found': len(gdelt.get('controversies', [])) if gdelt.get('available') else 0,
        'total_fines_usd': epa.get('total_penalties', 0) if epa.get('available') else 0,
        'sanctions_matches': len(sanctions.get('matches', [])) if sanctions.get('available') else 0,
        'governance_issues': len(sec.get('governance_issues', [])) if sec.get('available') else 0,
        'eu_facilities': len(eprtr.get('facilities', [])) if eprtr.get('available') else 0,
        'jurisdictions': len(result['sources'].get('opencorporates', {}).get('jurisdictions', [])),
    }

    return result


# ============ ANALYSE GREENWASHING PRO ============

def analyze_greenwashing_pro(ticker: str, company_name: str, info: dict = None) -> Dict[str, Any]:
    """Analyse de greenwashing de niveau institutionnel."""

    # Récupérer toutes les données
    esg_data = get_comprehensive_esg_pro(ticker, company_name, info)

    analysis = {
        'ticker': ticker,
        'company_name': company_name,
        'timestamp': datetime.now().isoformat(),
        'esg_data': esg_data,
        'risk_score': esg_data['total_risk_score'],
        'risk_level': esg_data['risk_level'],
        'risk_color': esg_data['risk_color'],
        'flags': [],
        'positive_points': [],
        'recommendations': [],
        'detailed_scores': esg_data['risk_scores'],
    }

    # Générer les flags basés sur les données

    # Flag: Scandales médiatiques
    gdelt = esg_data['sources'].get('gdelt', {})
    if gdelt.get('available'):
        scandal_count = len(gdelt.get('controversies', []))
        if scandal_count > 20:
            analysis['flags'].append({
                'type': 'MEDIA_SCANDALS',
                'severity': 'HIGH',
                'title': f'{scandal_count} scandales médiatiques détectés',
                'description': f"Sentiment moyen: {gdelt.get('sentiment_avg', 'N/A')}",
                'source': 'GDELT',
            })
        elif scandal_count > 5:
            analysis['flags'].append({
                'type': 'MEDIA_CONTROVERSIES',
                'severity': 'MEDIUM',
                'title': f'{scandal_count} controverses médiatiques',
                'description': 'Couverture médiatique négative détectée',
                'source': 'GDELT',
            })

    # Flag: Amendes EPA
    epa = esg_data['sources'].get('epa_echo', {})
    if epa.get('available'):
        penalties = epa.get('total_penalties', 0)
        if penalties > 1000000:
            analysis['flags'].append({
                'type': 'MAJOR_FINES',
                'severity': 'HIGH',
                'title': f'${penalties:,.0f} d\'amendes environnementales (US)',
                'description': f"Sévérité: {epa.get('severity')} | {epa.get('non_compliant_facilities', 0)} installations non-conformes",
                'source': 'EPA ECHO',
            })
        elif penalties > 100000:
            analysis['flags'].append({
                'type': 'SIGNIFICANT_FINES',
                'severity': 'MEDIUM',
                'title': f'${penalties:,.0f} d\'amendes EPA',
                'description': f"{epa.get('facilities_count', 0)} installations surveillées",
                'source': 'EPA ECHO',
            })

    # Flag: Sanctions
    sanctions = esg_data['sources'].get('opensanctions', {})
    if sanctions.get('available') and sanctions.get('sanctions'):
        analysis['flags'].append({
            'type': 'SANCTIONS',
            'severity': 'HIGH',
            'title': 'Entité sous sanctions internationales',
            'description': f"{len(sanctions['sanctions'])} correspondance(s) trouvée(s)",
            'source': 'OpenSanctions',
        })

    # Flag: Problèmes de gouvernance
    sec = esg_data['sources'].get('sec_edgar', {})
    if sec.get('available') and sec.get('governance_issues'):
        analysis['flags'].append({
            'type': 'GOVERNANCE_ISSUES',
            'severity': 'MEDIUM',
            'title': f"{len(sec['governance_issues'])} événements matériels (8-K)",
            'description': 'Changements de direction, investigations, ou restatements',
            'source': 'SEC EDGAR',
        })

    # Flag: Violations EU
    eprtr = esg_data['sources'].get('eprtr', {})
    if eprtr.get('available') and eprtr.get('facilities'):
        analysis['flags'].append({
            'type': 'EU_POLLUTION',
            'severity': 'MEDIUM',
            'title': f"{len(eprtr['facilities'])} installations polluantes en Europe",
            'description': f"Pays: {', '.join(eprtr.get('countries', []))}",
            'source': 'E-PRTR',
        })

    # Flag: Structure opaque
    oc = esg_data['sources'].get('opencorporates', {})
    if oc.get('available'):
        jurisdictions = oc.get('jurisdictions', [])
        tax_havens = ['ky', 'vg', 'bm', 'je', 'gg', 'im', 'lu', 'ie', 'nl', 'ch']
        tax_haven_count = sum(1 for j in jurisdictions if j.lower() in tax_havens)
        if tax_haven_count > 2:
            analysis['flags'].append({
                'type': 'TAX_STRUCTURE',
                'severity': 'MEDIUM',
                'title': f'Présence dans {tax_haven_count} paradis fiscaux',
                'description': f"Juridictions: {', '.join(jurisdictions)}",
                'source': 'OpenCorporates',
            })

    # Points positifs
    if not epa.get('has_violations', True):
        analysis['positive_points'].append({
            'title': 'Conformité EPA',
            'description': 'Aucune violation environnementale aux USA',
        })

    if not sanctions.get('sanctions'):
        analysis['positive_points'].append({
            'title': 'Aucune sanction',
            'description': 'Non listé dans les bases de sanctions internationales',
        })

    yf = esg_data['sources'].get('yfinance', {})
    if yf.get('available'):
        if (yf.get('total_esg_score') or 100) < 20:
            analysis['positive_points'].append({
                'title': 'Score ESG excellent',
                'description': f"Top performers selon Sustainalytics",
            })
        if yf.get('controversy_level', 5) == 0:
            analysis['positive_points'].append({
                'title': 'Aucune controverse majeure',
                'description': 'Niveau de controverse: 0/5',
            })

    # Recommandations
    if analysis['risk_score'] >= 50:
        analysis['recommendations'] = [
            "⚠️ Due diligence approfondie requise avant investissement",
            "Examiner les rapports de durabilité vs données EPA/E-PRTR",
            "Vérifier les procès en cours via PACER (US) ou tribunaux EU",
            "Analyser la structure juridique pour risques fiscaux",
            "Consulter les rapports d'ONG (Greenpeace, Amnesty, etc.)",
        ]
    elif analysis['risk_score'] >= 30:
        analysis['recommendations'] = [
            "Surveiller les actualités ESG régulièrement",
            "Vérifier les engagements climat sur SBTi",
            "Comparer avec les pairs du secteur",
        ]
    else:
        analysis['recommendations'] = [
            "Profil ESG satisfaisant",
            "Maintenir une veille sur les controverses",
        ]

    return analysis


# ============ SYSTÈME DE NOTATION AFFINÉ ============

# Seuils de notation alphabétique (score de risque -> note)
# Plus le score de risque est bas, meilleure est la note
GRADE_THRESHOLDS = [
    (0, 5, 'A+', '#1a5f2a', 'Excellence ESG - Leader du secteur'),
    (5, 15, 'A', '#27ae60', 'Excellent - Très faible risque ESG'),
    (15, 25, 'A-', '#2ecc71', 'Très bon - Risque ESG minimal'),
    (25, 35, 'B+', '#58d68d', 'Bon - Risque ESG faible'),
    (35, 45, 'B', '#82e0aa', 'Satisfaisant - Risque ESG modéré-faible'),
    (45, 55, 'B-', '#f1c40f', 'Acceptable - Risque ESG modéré'),
    (55, 65, 'C+', '#f39c12', 'Moyen - Risque ESG modéré-élevé'),
    (65, 75, 'C', '#e67e22', 'Insuffisant - Risque ESG élevé'),
    (75, 85, 'C-', '#e74c3c', 'Faible - Risque ESG très élevé'),
    (85, 95, 'D', '#c0392b', 'Très faible - Risque ESG critique'),
    (95, 100, 'F', '#7b241c', 'Échec - Risque ESG extrême'),
]

# Pondérations détaillées par catégorie
ESG_WEIGHTS_DETAILED = {
    'environmental': {
        'weight': 0.35,
        'subcategories': {
            'emissions': 0.30,
            'resource_use': 0.25,
            'pollution': 0.25,
            'biodiversity': 0.20,
        }
    },
    'social': {
        'weight': 0.30,
        'subcategories': {
            'labor_practices': 0.30,
            'health_safety': 0.25,
            'community': 0.25,
            'human_rights': 0.20,
        }
    },
    'governance': {
        'weight': 0.25,
        'subcategories': {
            'board_structure': 0.30,
            'ethics': 0.30,
            'transparency': 0.25,
            'shareholder_rights': 0.15,
        }
    },
    'controversy': {
        'weight': 0.10,
        'subcategories': {
            'media_scandals': 0.40,
            'legal_issues': 0.35,
            'sanctions': 0.25,
        }
    }
}


def score_to_grade(score: float) -> Dict[str, Any]:
    """
    Convertit un score de risque (0-100) en note alphabétique.

    Args:
        score: Score de risque (0 = excellent, 100 = catastrophique)

    Returns:
        Dict avec grade, color, description
    """
    for min_val, max_val, grade, color, description in GRADE_THRESHOLDS:
        if min_val <= score < max_val:
            return {
                'grade': grade,
                'color': color,
                'description': description,
                'score': score,
            }

    # Par défaut si score >= 100
    return {
        'grade': 'F',
        'color': '#7b241c',
        'description': 'Risque ESG extrême',
        'score': min(100, score),
    }


def calculate_detailed_esg_scores(esg_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcule des scores ESG détaillés avec sous-catégories.

    Args:
        esg_data: Données ESG brutes

    Returns:
        Dict avec scores détaillés par catégorie et sous-catégorie
    """
    detailed = {
        'categories': {},
        'overall': {},
        'grades': {},
    }

    risk_scores = esg_data.get('risk_scores', {})
    sources = esg_data.get('sources', {})

    # === ENVIRONNEMENT ===
    env_score = risk_scores.get('environmental', 0)
    epa = sources.get('epa_echo', {})
    eprtr = sources.get('eprtr', {})

    env_subcategories = {
        'emissions': min(100, (eprtr.get('total_emissions', 0) / 10000) if eprtr.get('available') else env_score * 0.3),
        'pollution': min(100, env_score * 0.8 if epa.get('severity') in ['HIGH', 'CRITICAL'] else env_score * 0.5),
        'compliance': 100 - (eprtr.get('compliance_score', 100) if eprtr.get('available') else 100 - env_score),
        'fines': min(100, (epa.get('total_penalties', 0) / 50000) if epa.get('available') else 0),
    }

    detailed['categories']['environmental'] = {
        'score': env_score,
        'grade': score_to_grade(env_score),
        'subcategories': env_subcategories,
        'weight': ESG_WEIGHTS_DETAILED['environmental']['weight'],
    }

    # === SOCIAL ===
    social_score = risk_scores.get('social', 0)
    gdelt = sources.get('gdelt', {})

    # Analyser les thèmes GDELT pour les sous-catégories sociales
    themes = dict(gdelt.get('top_themes', [])) if gdelt.get('available') else {}

    social_subcategories = {
        'labor_practices': min(100, themes.get('LABOR', 0) * 10 + social_score * 0.3),
        'health_safety': min(100, themes.get('HEALTH', 0) * 10 + social_score * 0.2),
        'community': min(100, themes.get('PROTEST', 0) * 10 + social_score * 0.2),
        'human_rights': min(100, themes.get('HUMAN_RIGHTS', 0) * 10 + social_score * 0.3),
    }

    detailed['categories']['social'] = {
        'score': social_score,
        'grade': score_to_grade(social_score),
        'subcategories': social_subcategories,
        'weight': ESG_WEIGHTS_DETAILED['social']['weight'],
    }

    # === GOUVERNANCE ===
    gov_score = risk_scores.get('governance', 0)
    sec = sources.get('sec_edgar', {})
    yf_esg = sources.get('yfinance', {})

    gov_issues = sec.get('governance_issues', []) if sec.get('available') else []
    controversy = yf_esg.get('controversy_level', 0) or 0

    gov_subcategories = {
        'board_structure': min(100, len([i for i in gov_issues if 'board' in str(i).lower()]) * 20 + gov_score * 0.3),
        'ethics': min(100, controversy * 15 + gov_score * 0.3),
        'transparency': min(100, len([i for i in gov_issues if 'disclosure' in str(i).lower()]) * 15 + gov_score * 0.2),
        'executive_compensation': min(100, len([i for i in gov_issues if 'compensation' in str(i).lower()]) * 20 + gov_score * 0.2),
    }

    detailed['categories']['governance'] = {
        'score': gov_score,
        'grade': score_to_grade(gov_score),
        'subcategories': gov_subcategories,
        'weight': ESG_WEIGHTS_DETAILED['governance']['weight'],
    }

    # === CONTROVERSES ===
    media_score = risk_scores.get('media', 0)
    sanctions_score = risk_scores.get('sanctions', 0)

    controversy_subcategories = {
        'media_scandals': media_score,
        'legal_issues': min(100, (epa.get('total_penalties', 0) / 100000) * 10 if epa.get('available') else 0),
        'sanctions': sanctions_score,
    }

    controversy_combined = (media_score * 0.4 + sanctions_score * 0.35 +
                           controversy_subcategories['legal_issues'] * 0.25)

    detailed['categories']['controversy'] = {
        'score': controversy_combined,
        'grade': score_to_grade(controversy_combined),
        'subcategories': controversy_subcategories,
        'weight': ESG_WEIGHTS_DETAILED['controversy']['weight'],
    }

    # === SCORE GLOBAL AFFINÉ ===
    total_weighted = sum(
        cat_data['score'] * cat_data['weight']
        for cat_data in detailed['categories'].values()
    )

    detailed['overall'] = {
        'score': round(total_weighted, 1),
        'grade': score_to_grade(total_weighted),
    }

    # Générer les grades pour chaque catégorie
    detailed['grades'] = {
        cat: score_to_grade(data['score'])
        for cat, data in detailed['categories'].items()
    }

    return detailed


def get_esg_comparison_benchmark(sector: str = None) -> Dict[str, float]:
    """
    Retourne les scores moyens du secteur pour comparaison.

    Args:
        sector: Nom du secteur (optionnel)

    Returns:
        Dict avec scores moyens par catégorie
    """
    # Benchmarks par défaut (moyennes du marché)
    default_benchmarks = {
        'environmental': 45,
        'social': 40,
        'governance': 35,
        'controversy': 25,
        'overall': 38,
    }

    # Benchmarks par secteur (valeurs approximatives)
    sector_benchmarks = {
        'Technology': {'environmental': 35, 'social': 45, 'governance': 40, 'controversy': 30, 'overall': 38},
        'Energy': {'environmental': 65, 'social': 50, 'governance': 45, 'controversy': 55, 'overall': 54},
        'Healthcare': {'environmental': 30, 'social': 55, 'governance': 40, 'controversy': 40, 'overall': 42},
        'Financial': {'environmental': 25, 'social': 40, 'governance': 50, 'controversy': 45, 'overall': 40},
        'Consumer': {'environmental': 45, 'social': 50, 'governance': 35, 'controversy': 35, 'overall': 42},
        'Industrial': {'environmental': 55, 'social': 45, 'governance': 40, 'controversy': 40, 'overall': 45},
        'Materials': {'environmental': 60, 'social': 45, 'governance': 40, 'controversy': 45, 'overall': 48},
        'Utilities': {'environmental': 55, 'social': 35, 'governance': 35, 'controversy': 30, 'overall': 39},
        'Real Estate': {'environmental': 40, 'social': 35, 'governance': 40, 'controversy': 25, 'overall': 35},
        'Communication': {'environmental': 30, 'social': 50, 'governance': 45, 'controversy': 40, 'overall': 42},
    }

    if sector:
        for key in sector_benchmarks:
            if key.lower() in sector.lower():
                return sector_benchmarks[key]

    return default_benchmarks


def generate_esg_recommendations_detailed(detailed_scores: Dict[str, Any], threshold: float = 50) -> List[Dict[str, Any]]:
    """
    Génère des recommandations détaillées basées sur les scores.

    Args:
        detailed_scores: Scores détaillés par catégorie
        threshold: Seuil au-dessus duquel une recommandation est émise

    Returns:
        Liste de recommandations avec priorité
    """
    recommendations = []

    for category, data in detailed_scores.get('categories', {}).items():
        score = data.get('score', 0)
        subcategories = data.get('subcategories', {})

        if score >= threshold:
            # Trouver les sous-catégories problématiques
            problematic = [
                (sub, sub_score) for sub, sub_score in subcategories.items()
                if sub_score >= threshold
            ]

            category_names = {
                'environmental': 'Environnement',
                'social': 'Social',
                'governance': 'Gouvernance',
                'controversy': 'Controverses',
            }

            priority = 'HIGH' if score >= 70 else ('MEDIUM' if score >= 50 else 'LOW')

            rec = {
                'category': category_names.get(category, category),
                'priority': priority,
                'score': score,
                'grade': data.get('grade', {}).get('grade', 'N/A'),
                'issues': problematic,
                'recommendation': '',
            }

            # Générer la recommandation spécifique
            if category == 'environmental':
                if 'emissions' in dict(problematic):
                    rec['recommendation'] = "Réduire les émissions de GES via des énergies renouvelables"
                elif 'pollution' in dict(problematic):
                    rec['recommendation'] = "Améliorer les processus de traitement des déchets"
                else:
                    rec['recommendation'] = "Renforcer les politiques environnementales"

            elif category == 'social':
                if 'labor_practices' in dict(problematic):
                    rec['recommendation'] = "Améliorer les conditions de travail et la rémunération"
                elif 'health_safety' in dict(problematic):
                    rec['recommendation'] = "Renforcer les protocoles de santé et sécurité"
                else:
                    rec['recommendation'] = "Développer les initiatives sociales"

            elif category == 'governance':
                if 'ethics' in dict(problematic):
                    rec['recommendation'] = "Renforcer le code éthique et la formation"
                elif 'transparency' in dict(problematic):
                    rec['recommendation'] = "Améliorer la transparence des rapports"
                else:
                    rec['recommendation'] = "Moderniser la structure de gouvernance"

            elif category == 'controversy':
                rec['recommendation'] = "Améliorer la communication de crise et les relations publiques"

            recommendations.append(rec)

    # Trier par priorité
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))

    return recommendations


# ============ AFFICHAGE SIMPLE ============

def get_simple_esg_display(ticker: str) -> Dict[str, Any]:
    """Version simplifiée pour affichage rapide."""
    yf = get_yfinance_esg(ticker)

    if not yf.get('available'):
        return {'available': False}

    env = yf.get('environment_score', 50) or 50
    soc = yf.get('social_score', 50) or 50
    gov = yf.get('governance_score', 50) or 50

    return {
        'available': True,
        'total_score': yf.get('total_esg_score'),
        'environment': max(0, 100 - env * 2),
        'social': max(0, 100 - soc * 2),
        'governance': max(0, 100 - gov * 2),
        'controversy_level': yf.get('controversy_level', 0) or 0,
        'peer_group': yf.get('peer_group'),
        **yf.get('controversial_sectors', {}),
    }


# ============ ANALYSE ALIGNEMENT ACCORD DE PARIS ============

# Trajectoires SBTi par secteur (% réduction annuelle requise pour 1.5°C)
SBTI_SECTOR_TRAJECTORIES = {
    'Energy': {'annual_reduction': 4.2, 'target_2030': 45, 'target_2050': 90, 'fossil_exit': True},
    'Utilities': {'annual_reduction': 4.0, 'target_2030': 40, 'target_2050': 95, 'fossil_exit': True},
    'Materials': {'annual_reduction': 3.0, 'target_2030': 30, 'target_2050': 85, 'fossil_exit': False},
    'Industrials': {'annual_reduction': 3.0, 'target_2030': 30, 'target_2050': 85, 'fossil_exit': False},
    'Consumer Discretionary': {'annual_reduction': 2.5, 'target_2030': 25, 'target_2050': 80, 'fossil_exit': False},
    'Consumer Staples': {'annual_reduction': 2.5, 'target_2030': 25, 'target_2050': 80, 'fossil_exit': False},
    'Health Care': {'annual_reduction': 2.0, 'target_2030': 20, 'target_2050': 75, 'fossil_exit': False},
    'Financials': {'annual_reduction': 4.0, 'target_2030': 40, 'target_2050': 100, 'fossil_exit': True},  # Alignement portefeuille
    'Information Technology': {'annual_reduction': 2.5, 'target_2030': 25, 'target_2050': 80, 'fossil_exit': False},
    'Communication Services': {'annual_reduction': 2.5, 'target_2030': 25, 'target_2050': 80, 'fossil_exit': False},
    'Real Estate': {'annual_reduction': 3.5, 'target_2030': 35, 'target_2050': 90, 'fossil_exit': False},
    'default': {'annual_reduction': 3.0, 'target_2030': 30, 'target_2050': 85, 'fossil_exit': False},
}

# Poids du Scope 3 par secteur (% des émissions totales typiques)
SCOPE3_WEIGHT_BY_SECTOR = {
    'Energy': 85,
    'Financials': 95,
    'Consumer Discretionary': 90,
    'Consumer Staples': 85,
    'Information Technology': 80,
    'Industrials': 75,
    'Materials': 70,
    'Health Care': 75,
    'Communication Services': 80,
    'Utilities': 40,
    'Real Estate': 60,
    'default': 75,
}


def get_carbon_data_from_info(info: dict) -> Dict[str, Any]:
    """
    Extrait les données carbone disponibles depuis les infos Yahoo Finance.

    Note: Yahoo Finance ne fournit pas toujours les données carbone détaillées.
    Cette fonction tente d'extraire ce qui est disponible.
    """
    carbon_data = {
        'available': False,
        'scope1': None,
        'scope2': None,
        'scope3': None,
        'total_emissions': None,
        'carbon_intensity': None,
        'year': None,
        'source': 'estimated',
    }

    if not info:
        return carbon_data

    # Tentative d'extraction des données carbone si disponibles
    # Yahoo Finance peut avoir des données ESG dans certains cas

    return carbon_data


@cached(ttl=3600)
def estimate_carbon_footprint(ticker: str, info: dict, sector: str) -> Dict[str, Any]:
    """
    Estime l'empreinte carbone basée sur les données financières et le secteur.

    Méthodologie :
    - Utilise l'intensité carbone moyenne du secteur
    - Ajuste selon la taille (revenus) de l'entreprise
    - Applique les ratios Scope 1/2/3 sectoriels
    """
    result = {
        'available': True,
        'estimated': True,
        'methodology': 'Estimation par proxy sectoriel',
        'confidence': 'LOW',
        'scope1': None,
        'scope2': None,
        'scope3': None,
        'total_emissions': None,
        'carbon_intensity': None,  # tCO2e / M€ revenue
        'unit': 'tCO2e',
        'currency': 'EUR',
        'year': datetime.now().year - 1,
    }

    if not info:
        result['available'] = False
        return result

    # Obtenir les revenus
    revenue = info.get('totalRevenue') or info.get('revenue') or 0
    if revenue == 0:
        result['available'] = False
        result['error'] = 'Revenus non disponibles'
        return result

    # Convertir en millions EUR (approximation)
    revenue_meur = revenue / 1_000_000

    # Intensité carbone moyenne par secteur (tCO2e / M€)
    sector_intensity = {
        'Energy': 850,
        'Utilities': 700,
        'Materials': 450,
        'Industrials': 150,
        'Consumer Discretionary': 80,
        'Consumer Staples': 120,
        'Health Care': 40,
        'Financials': 15,
        'Information Technology': 25,
        'Communication Services': 30,
        'Real Estate': 100,
        'default': 100,
    }

    intensity = sector_intensity.get(sector, sector_intensity['default'])

    # Calculer les émissions estimées
    total_emissions = revenue_meur * intensity

    # Répartition Scope 1/2/3 par secteur
    scope3_percent = SCOPE3_WEIGHT_BY_SECTOR.get(sector, 75) / 100
    scope1_percent = (1 - scope3_percent) * 0.6  # 60% du non-Scope3
    scope2_percent = (1 - scope3_percent) * 0.4  # 40% du non-Scope3

    result['scope1'] = round(total_emissions * scope1_percent)
    result['scope2'] = round(total_emissions * scope2_percent)
    result['scope3'] = round(total_emissions * scope3_percent)
    result['total_emissions'] = round(total_emissions)
    result['carbon_intensity'] = round(intensity, 1)
    result['revenue_meur'] = round(revenue_meur, 1)

    return result


def analyze_paris_alignment(ticker: str, company_name: str, info: dict, esg_data: dict = None) -> Dict[str, Any]:
    """
    Analyse l'alignement de l'entreprise avec l'Accord de Paris.

    Retourne :
    - Score d'alignement (0-100)
    - Gap de crédibilité climatique (%)
    - Température implicite (°C)
    - Détails par Scope
    - Recommandations
    """
    result = {
        'ticker': ticker,
        'company_name': company_name,
        'timestamp': datetime.now().isoformat(),
        'data_available': False,
        'alignment_score': None,
        'credibility_gap': None,
        'implied_temperature': None,
        'paris_status': None,
        'sector': None,
        'trajectory': {},
        'emissions': {},
        'commitments': {},
        'greenwashing_alerts': [],
        'recommendations': [],
        'methodology_notes': [],
    }

    if not info:
        result['error'] = 'Données entreprise non disponibles'
        return result

    # Identifier le secteur
    sector = info.get('sector', 'default')
    result['sector'] = sector

    # Obtenir la trajectoire SBTi requise
    trajectory = SBTI_SECTOR_TRAJECTORIES.get(sector, SBTI_SECTOR_TRAJECTORIES['default'])
    result['trajectory'] = {
        'required_annual_reduction': trajectory['annual_reduction'],
        'target_2030': trajectory['target_2030'],
        'target_2050': trajectory['target_2050'],
        'fossil_exit_required': trajectory['fossil_exit'],
        'scope3_weight_typical': SCOPE3_WEIGHT_BY_SECTOR.get(sector, 75),
    }

    # Estimer les émissions
    emissions = estimate_carbon_footprint(ticker, info, sector)
    result['emissions'] = emissions

    if emissions.get('available'):
        result['data_available'] = True

    # Analyser les engagements (basé sur les données ESG si disponibles)
    commitments = analyze_climate_commitments(info, esg_data)
    result['commitments'] = commitments

    # Calculer le score d'alignement
    alignment = calculate_alignment_score(emissions, commitments, trajectory, sector)
    result['alignment_score'] = alignment['score']
    result['paris_status'] = alignment['status']
    result['implied_temperature'] = alignment['temperature']

    # Calculer le gap de crédibilité
    credibility = calculate_credibility_gap(emissions, commitments, trajectory)
    result['credibility_gap'] = credibility['gap']
    result['credibility_level'] = credibility['level']

    # Détecter le greenwashing
    greenwashing = detect_climate_greenwashing(emissions, commitments, credibility, info)
    result['greenwashing_alerts'] = greenwashing['alerts']
    result['greenwashing_score'] = greenwashing['score']

    # Générer les recommandations
    result['recommendations'] = generate_climate_recommendations(
        alignment, credibility, greenwashing, sector, trajectory
    )

    # Notes méthodologiques
    if emissions.get('estimated'):
        result['methodology_notes'].append(
            "⚠️ Émissions estimées par proxy sectoriel - données réelles non disponibles"
        )

    result['methodology_notes'].append(
        f"Trajectoire SBTi {sector}: -{trajectory['annual_reduction']}%/an pour alignement 1.5°C"
    )

    return result


def analyze_climate_commitments(info: dict, esg_data: dict = None) -> Dict[str, Any]:
    """
    Analyse les engagements climatiques déclarés par l'entreprise.
    """
    commitments = {
        'net_zero_target': None,
        'net_zero_year': None,
        'sbti_status': 'unknown',  # validated, committed, none
        'scope_coverage': [],
        'interim_targets': [],
        'compensation_reliance': 'unknown',
        'transition_plan': False,
        'capex_green_ratio': None,
    }

    # Si on a des données ESG détaillées
    if esg_data:
        sources = esg_data.get('sources', {})
        yf_esg = sources.get('yfinance', {})

        # Vérifier le score environnement
        env_score = yf_esg.get('environment_score')
        if env_score:
            # Score bas = bon (Sustainalytics)
            if env_score < 20:
                commitments['sbti_status'] = 'likely_committed'
            elif env_score > 40:
                commitments['sbti_status'] = 'likely_none'

    # Vérifier la présence de mentions dans les données
    company_name = info.get('longName', '').lower() if info else ''

    # Estimation basée sur le secteur et la taille
    market_cap = info.get('marketCap', 0) if info else 0

    # Les grandes entreprises sont plus susceptibles d'avoir des engagements
    if market_cap > 50_000_000_000:  # > 50B
        commitments['net_zero_year'] = 2050  # Estimation conservatrice
        commitments['scope_coverage'] = ['Scope 1', 'Scope 2']
        commitments['transition_plan'] = True
    elif market_cap > 10_000_000_000:  # > 10B
        commitments['net_zero_year'] = 2050
        commitments['scope_coverage'] = ['Scope 1', 'Scope 2']

    return commitments


def calculate_alignment_score(emissions: dict, commitments: dict, trajectory: dict, sector: str) -> Dict[str, Any]:
    """
    Calcule le score d'alignement avec l'Accord de Paris.

    Score 80-100 : Aligné 1.5°C
    Score 60-79 : Aligné 2°C
    Score 40-59 : Non aligné
    Score 0-39 : Critique
    """
    result = {
        'score': 50,  # Score par défaut (incertain)
        'status': 'Données insuffisantes',
        'temperature': 2.5,  # Température implicite par défaut
        'details': {},
    }

    base_score = 50

    # Ajustements basés sur les engagements
    if commitments.get('sbti_status') == 'likely_committed':
        base_score += 15
    elif commitments.get('sbti_status') == 'likely_none':
        base_score -= 10

    if commitments.get('net_zero_year'):
        year = commitments['net_zero_year']
        if year <= 2040:
            base_score += 20
        elif year <= 2050:
            base_score += 10
        else:
            base_score -= 5

    # Couverture des Scopes
    scopes = commitments.get('scope_coverage', [])
    if 'Scope 3' in scopes:
        base_score += 15
    elif len(scopes) >= 2:
        base_score += 5

    # Plan de transition
    if commitments.get('transition_plan'):
        base_score += 10

    # Ajustement sectoriel (secteurs difficiles à décarboner)
    difficult_sectors = ['Energy', 'Utilities', 'Materials', 'Industrials']
    if sector in difficult_sectors:
        # Bonus si score correct dans secteur difficile
        if base_score > 50:
            base_score += 5

    # Plafonner le score
    score = max(0, min(100, base_score))

    # Déterminer le statut
    if score >= 80:
        result['status'] = 'Aligné 1.5°C'
        result['temperature'] = 1.5
    elif score >= 60:
        result['status'] = 'Aligné 2°C'
        result['temperature'] = 2.0
    elif score >= 40:
        result['status'] = 'Non aligné'
        result['temperature'] = 2.7
    else:
        result['status'] = 'Critique'
        result['temperature'] = 3.5

    result['score'] = score

    return result


def calculate_credibility_gap(emissions: dict, commitments: dict, trajectory: dict) -> Dict[str, Any]:
    """
    Calcule le gap de crédibilité entre engagements et trajectoire réelle.

    Gap > 15% : Alerte greenwashing
    Gap > 30% : Greenwashing avéré
    """
    result = {
        'gap': None,
        'level': 'unknown',
        'details': {},
    }

    # Sans données d'émissions historiques, on ne peut pas calculer la trajectoire réelle
    # On utilise donc une estimation basée sur les engagements

    if not commitments.get('net_zero_year'):
        result['gap'] = 40  # Pas d'engagement = gap élevé
        result['level'] = 'HIGH'
        result['details']['reason'] = 'Absence d\'objectif Net Zero déclaré'
        return result

    # Estimation du gap basée sur la couverture des Scopes
    scopes = commitments.get('scope_coverage', [])

    if 'Scope 3' in scopes:
        gap = 10  # Faible gap si Scope 3 inclus
    elif 'Scope 1' in scopes and 'Scope 2' in scopes:
        gap = 25  # Gap modéré si Scope 1+2 seulement
    else:
        gap = 50  # Gap élevé si couverture limitée

    # Ajustement si pas de plan de transition
    if not commitments.get('transition_plan'):
        gap += 15

    result['gap'] = min(100, gap)

    if gap <= 15:
        result['level'] = 'LOW'
    elif gap <= 30:
        result['level'] = 'MEDIUM'
    else:
        result['level'] = 'HIGH'

    return result


def detect_climate_greenwashing(emissions: dict, commitments: dict, credibility: dict, info: dict) -> Dict[str, Any]:
    """
    Détecte les signes de greenwashing climatique.
    """
    alerts = []
    score = 0  # 0 = pas de greenwashing, 100 = greenwashing avéré

    # 1. Gap de crédibilité élevé
    gap = credibility.get('gap', 0)
    if gap > 30:
        alerts.append({
            'type': 'CREDIBILITY_GAP',
            'severity': 'HIGH',
            'title': f'Gap de crédibilité de {gap}%',
            'description': 'Écart significatif entre les engagements déclarés et la trajectoire probable'
        })
        score += 30
    elif gap > 15:
        alerts.append({
            'type': 'CREDIBILITY_GAP',
            'severity': 'MEDIUM',
            'title': f'Gap de crédibilité modéré ({gap}%)',
            'description': 'Les engagements semblent optimistes par rapport aux actions'
        })
        score += 15

    # 2. Scope 3 ignoré
    scopes = commitments.get('scope_coverage', [])
    if 'Scope 3' not in scopes:
        sector = info.get('sector', 'default') if info else 'default'
        scope3_weight = SCOPE3_WEIGHT_BY_SECTOR.get(sector, 75)

        if scope3_weight >= 80:
            alerts.append({
                'type': 'SCOPE3_IGNORED',
                'severity': 'HIGH',
                'title': f'Scope 3 non couvert ({scope3_weight}% des émissions typiques)',
                'description': 'Les émissions de la chaîne de valeur sont ignorées dans les engagements'
            })
            score += 25
        elif scope3_weight >= 60:
            alerts.append({
                'type': 'SCOPE3_IGNORED',
                'severity': 'MEDIUM',
                'title': f'Scope 3 non couvert ({scope3_weight}% des émissions typiques)',
                'description': 'Une part significative des émissions n\'est pas adressée'
            })
            score += 15

    # 3. Objectif Net Zero lointain sans jalons intermédiaires
    net_zero_year = commitments.get('net_zero_year')
    interim_targets = commitments.get('interim_targets', [])

    if net_zero_year and net_zero_year > 2045 and not interim_targets:
        alerts.append({
            'type': 'DISTANT_TARGET',
            'severity': 'MEDIUM',
            'title': f'Objectif Net Zero en {net_zero_year} sans jalons',
            'description': 'Objectif lointain sans engagements intermédiaires vérifiables'
        })
        score += 15

    # 4. Pas d'engagement SBTi pour les grandes entreprises
    market_cap = info.get('marketCap', 0) if info else 0
    sbti_status = commitments.get('sbti_status', 'unknown')

    if market_cap > 10_000_000_000 and sbti_status in ['unknown', 'likely_none']:
        alerts.append({
            'type': 'NO_SBTI',
            'severity': 'MEDIUM',
            'title': 'Absence d\'engagement SBTi',
            'description': 'Grande entreprise sans validation Science Based Targets'
        })
        score += 10

    # 5. Secteur à risque sans plan de sortie fossile
    sector = info.get('sector', '') if info else ''
    trajectory = SBTI_SECTOR_TRAJECTORIES.get(sector, {})

    if trajectory.get('fossil_exit') and sector in ['Energy', 'Utilities']:
        alerts.append({
            'type': 'FOSSIL_EXPOSURE',
            'severity': 'HIGH',
            'title': 'Secteur nécessitant sortie des fossiles',
            'description': 'L\'alignement 1.5°C requiert un plan de sortie des énergies fossiles'
        })
        score += 20

    result = {
        'alerts': alerts,
        'score': min(100, score),
        'level': 'NONE' if score < 20 else ('LOW' if score < 40 else ('MEDIUM' if score < 60 else 'HIGH')),
    }

    return result


def generate_climate_recommendations(alignment: dict, credibility: dict, greenwashing: dict,
                                     sector: str, trajectory: dict) -> List[Dict[str, Any]]:
    """
    Génère des recommandations pour améliorer l'alignement climatique.
    """
    recommendations = []

    # Basé sur le score d'alignement
    score = alignment.get('score', 50)

    if score < 40:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Stratégie',
            'action': 'Définir une stratégie climat avec objectifs Science Based Targets',
            'impact': 'Fondamental pour la crédibilité et l\'accès aux capitaux verts'
        })

    if score < 60:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Objectifs',
            'action': f'S\'engager sur une réduction de {trajectory.get("annual_reduction", 3)}%/an minimum',
            'impact': 'Nécessaire pour l\'alignement 2°C'
        })

    # Basé sur le gap de crédibilité
    gap = credibility.get('gap', 0)

    if gap > 20:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Transparence',
            'action': 'Publier un plan de transition détaillé avec jalons intermédiaires',
            'impact': 'Réduire le gap de crédibilité et renforcer la confiance'
        })

    # Basé sur les alertes greenwashing
    for alert in greenwashing.get('alerts', []):
        if alert['type'] == 'SCOPE3_IGNORED':
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Scope 3',
                'action': 'Intégrer le Scope 3 dans les objectifs de réduction',
                'impact': 'Couvrir l\'essentiel des émissions réelles'
            })
            break

    # Recommandations sectorielles
    if sector == 'Financials':
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Finance verte',
            'action': 'Aligner le portefeuille d\'investissements (méthodologie PCAF)',
            'impact': 'Réduire les émissions financées'
        })
    elif sector == 'Energy':
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Transition énergétique',
            'action': 'Planifier la sortie des énergies fossiles d\'ici 2040',
            'impact': 'Condition sine qua non pour l\'alignement 1.5°C'
        })

    # Recommandation générale si peu d'alertes
    if len(recommendations) < 2:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Communication',
            'action': 'Renforcer la transparence des données carbone (CDP, GRI)',
            'impact': 'Améliorer la notation ESG et l\'attractivité pour les investisseurs'
        })

    return recommendations


def get_temperature_color(temp: float) -> str:
    """Retourne une couleur selon la température implicite."""
    if temp <= 1.5:
        return '#27ae60'  # Vert
    elif temp <= 2.0:
        return '#82e0aa'  # Vert clair
    elif temp <= 2.5:
        return '#f1c40f'  # Jaune
    elif temp <= 3.0:
        return '#e67e22'  # Orange
    else:
        return '#e74c3c'  # Rouge


def get_alignment_color(score: int) -> str:
    """Retourne une couleur selon le score d'alignement."""
    if score >= 80:
        return '#27ae60'
    elif score >= 60:
        return '#82e0aa'
    elif score >= 40:
        return '#f1c40f'
    else:
        return '#e74c3c'


# ============ TEST ============

if __name__ == "__main__":
    print("=" * 60)
    print("Module ESG Pro v5.0 - Niveau Institutionnel")
    print("=" * 60)
    print("\nSources disponibles:")
    print("  1. GDELT - Scandales & polémiques mondiales")
    print("  2. EPA ECHO - Violations environnementales US")
    print("  3. E-PRTR - Violations environnementales EU")
    print("  4. OpenSanctions - Sanctions internationales")
    print("  5. SEC EDGAR - Gouvernance & filings US")
    print("  6. UK Environment Agency - Violations UK")
    print("  7. Wikidata - Controverses historiques")
    print("  8. OpenCorporates - Structure juridique")
    print("  9. LobbyFacts EU - Lobbying européen")
    print(" 10. Yahoo Finance - Scores Sustainalytics")
    print("\n✓ Module chargé avec succès")
