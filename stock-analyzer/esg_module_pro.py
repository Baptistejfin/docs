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
