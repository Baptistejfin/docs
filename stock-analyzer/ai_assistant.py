"""
AI Assistant Module - ALADDIN-Like Financial Analysis Assistant
Intégration avec Mistral (Ollama local ou API Cloud)

Ce module fournit un assistant IA conversationnel pour l'analyse fondamentale.
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib


# ============ CONFIGURATION ============

class AIProvider(Enum):
    """Fournisseurs d'IA supportés."""
    OLLAMA = "ollama"
    MISTRAL_API = "mistral_api"
    OPENAI = "openai"


@dataclass
class ChatMessage:
    """Représente un message dans la conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class CompanyContext:
    """Contexte de l'entreprise actuellement analysée."""
    name: str = ""
    ticker: str = ""
    sector: str = ""
    industry: str = ""
    country: str = ""
    current_price: float = 0.0
    market_cap: float = 0.0
    financial_data: Dict = field(default_factory=dict)
    ratios: Dict = field(default_factory=dict)
    dcf_results: Dict = field(default_factory=dict)
    black_swan_risks: Dict = field(default_factory=dict)
    esg_data: Dict = field(default_factory=dict)


@dataclass
class SessionHistory:
    """Historique des entreprises analysées dans la session."""
    companies: List[Dict] = field(default_factory=list)

    def add_company(self, name: str, ticker: str):
        """Ajoute une entreprise à l'historique."""
        entry = {
            "name": name,
            "ticker": ticker,
            "analyzed_at": datetime.now().isoformat()
        }
        # Éviter les doublons consécutifs
        if not self.companies or self.companies[-1]["ticker"] != ticker:
            self.companies.append(entry)

    def get_recent(self, n: int = 5) -> List[Dict]:
        """Retourne les n dernières entreprises analysées."""
        return self.companies[-n:]


# ============ ACTIONS RAPIDES ============

QUICK_ACTIONS = [
    {
        "id": "summary",
        "label": "📊 Résumé rapide",
        "icon": "📊",
        "prompt": "Fais un résumé rapide des points clés de cette entreprise en 5 bullet points maximum. Sois concis et va à l'essentiel.",
        "category": "analysis"
    },
    {
        "id": "strengths_weaknesses",
        "label": "💪 Forces/Faiblesses",
        "icon": "💪",
        "prompt": "Liste les 3 principales forces et les 3 principales faiblesses de cette entreprise basées sur les données financières disponibles.",
        "category": "analysis"
    },
    {
        "id": "valuation",
        "label": "💰 Valorisation",
        "icon": "💰",
        "prompt": "L'action est-elle correctement valorisée ? Analyse le PER, compare aux concurrents du secteur, et donne ton avis sur la cherté ou la décote du titre.",
        "category": "valuation"
    },
    {
        "id": "risks",
        "label": "⚠️ Risques",
        "icon": "⚠️",
        "prompt": "Quels sont les principaux risques à surveiller pour cette entreprise ? Identifie les red flags potentiels dans les données financières.",
        "category": "risk"
    },
    {
        "id": "investment_decision",
        "label": "🎯 Décision",
        "icon": "🎯",
        "prompt": "Sur une échelle de 1 à 10, quel serait ton score d'investissement pour cette entreprise ? Justifie ta réponse avec les points positifs et négatifs clés.",
        "category": "decision"
    },
    {
        "id": "catalysts",
        "label": "🚀 Catalyseurs",
        "icon": "🚀",
        "prompt": "Quels pourraient être les catalyseurs positifs et négatifs à court terme (6-12 mois) pour cette entreprise ?",
        "category": "analysis"
    }
]

REPORT_TYPES = [
    {
        "id": "summary",
        "label": "📄 Rapport Synthétique",
        "description": "Résumé d'une page avec les points essentiels",
        "prompt": """Génère un rapport synthétique d'analyse pour cette entreprise.

Structure du rapport :
1. **Présentation** (2-3 lignes)
2. **Points Clés** (5 bullet points)
3. **Valorisation** (PER, comparaison secteur)
4. **Forces** (3 points)
5. **Faiblesses/Risques** (3 points)
6. **Conclusion** (recommandation en 2 lignes)

Sois concis et professionnel."""
    },
    {
        "id": "detailed",
        "label": "📑 Rapport Détaillé",
        "description": "Analyse complète et approfondie",
        "prompt": """Génère un rapport d'analyse détaillé et complet pour cette entreprise.

Structure du rapport :
1. **Résumé Exécutif**
2. **Présentation de l'Entreprise**
   - Activité et positionnement
   - Secteur et concurrence
3. **Analyse Financière**
   - Revenus et croissance
   - Rentabilité (marges, ROE, ROA)
   - Bilan (dette, liquidité)
   - Cash flows
4. **Valorisation**
   - Multiples (PER, P/B, EV/EBITDA)
   - DCF si disponible
   - Comparaison sectorielle
5. **Analyse SWOT**
6. **Risques Identifiés**
7. **Catalyseurs Potentiels**
8. **Conclusion et Recommandation**

Sois exhaustif et utilise toutes les données disponibles."""
    },
    {
        "id": "investment_thesis",
        "label": "📈 Thèse d'Investissement",
        "description": "Bull case vs Bear case",
        "prompt": """Rédige une thèse d'investissement complète pour cette entreprise.

Structure :
1. **Résumé de la Thèse** (2-3 lignes)

2. **BULL CASE** (Scénario optimiste)
   - 3-4 arguments en faveur de l'investissement
   - Catalyseurs positifs potentiels
   - Prix cible haussier estimé

3. **BEAR CASE** (Scénario pessimiste)
   - 3-4 risques majeurs
   - Scénarios négatifs possibles
   - Prix cible baissier estimé

4. **BASE CASE** (Scénario central)
   - Hypothèses réalistes
   - Prix cible médian

5. **Recommandation Finale**
   - Verdict : Acheter / Conserver / Vendre
   - Niveau de conviction (1-10)
   - Horizon d'investissement recommandé

Sois objectif et équilibré dans ton analyse."""
    },
    {
        "id": "comparison",
        "label": "⚖️ Comparaison",
        "description": "Compare avec les entreprises précédemment analysées",
        "prompt": """Compare l'entreprise actuelle avec les autres entreprises analysées dans cette session.

Pour chaque entreprise, compare :
- Valorisation (PER, multiples)
- Croissance
- Rentabilité
- Solidité financière
- Risques

Conclus sur laquelle représente la meilleure opportunité d'investissement et pourquoi."""
    }
]


# ============ CLASSE PRINCIPALE ============

class MistralAssistant:
    """
    Assistant IA pour l'analyse financière.
    Supporte Ollama (local) et Mistral API (cloud).
    """

    def __init__(
        self,
        provider: AIProvider = AIProvider.OLLAMA,
        model: str = "mistral:latest",
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialise l'assistant.

        Args:
            provider: Fournisseur d'IA (OLLAMA ou MISTRAL_API)
            model: Nom du modèle à utiliser
            api_key: Clé API (requis pour MISTRAL_API)
            base_url: URL de base pour Ollama
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        # État de la conversation
        self.conversation_history: List[ChatMessage] = []
        self.current_context: CompanyContext = CompanyContext()
        self.session_history: SessionHistory = SessionHistory()

        # Configuration
        self.max_history_length = 20
        self.temperature = 0.7
        self.max_tokens = 2048

        # État de connexion
        self._is_available: Optional[bool] = None

    def check_availability(self) -> bool:
        """Vérifie si l'IA est disponible."""
        if self.provider == AIProvider.OLLAMA:
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                self._is_available = response.status_code == 200
            except:
                self._is_available = False
        elif self.provider == AIProvider.MISTRAL_API:
            self._is_available = bool(self.api_key)
        else:
            self._is_available = False

        return self._is_available

    @property
    def is_available(self) -> bool:
        """Retourne True si l'IA est disponible."""
        if self._is_available is None:
            self.check_availability()
        return self._is_available

    def set_context(self, company_data: Dict[str, Any]):
        """
        Met à jour le contexte avec les données de l'entreprise.

        Args:
            company_data: Dictionnaire contenant les données de l'entreprise
        """
        old_ticker = self.current_context.ticker

        # Mettre à jour le contexte
        self.current_context = CompanyContext(
            name=company_data.get("name", ""),
            ticker=company_data.get("ticker", ""),
            sector=company_data.get("sector", ""),
            industry=company_data.get("industry", ""),
            country=company_data.get("country", ""),
            current_price=company_data.get("current_price", 0.0),
            market_cap=company_data.get("market_cap", 0.0),
            financial_data=company_data.get("financial_data", {}),
            ratios=company_data.get("ratios", {}),
            dcf_results=company_data.get("dcf_results", {}),
            black_swan_risks=company_data.get("black_swan_risks", {}),
            esg_data=company_data.get("esg_data", {})
        )

        # Ajouter à l'historique de session
        if self.current_context.name and self.current_context.ticker:
            self.session_history.add_company(
                self.current_context.name,
                self.current_context.ticker
            )

        # Notifier le changement d'entreprise si différent
        if old_ticker and old_ticker != self.current_context.ticker:
            notification = f"[Système] L'utilisateur analyse maintenant {self.current_context.name} ({self.current_context.ticker})."
            self.conversation_history.append(ChatMessage(
                role="system",
                content=notification,
                metadata={"type": "context_change"}
            ))

    def _build_system_prompt(self) -> str:
        """Construit le prompt système avec le contexte complet."""

        # Informations de base
        company_info = f"""
ENTREPRISE ACTUELLEMENT ANALYSÉE :
- Nom : {self.current_context.name or 'Non définie'}
- Ticker : {self.current_context.ticker or 'N/A'}
- Secteur : {self.current_context.sector or 'N/A'}
- Industrie : {self.current_context.industry or 'N/A'}
- Pays : {self.current_context.country or 'N/A'}
- Prix actuel : ${self.current_context.current_price:.2f}
- Capitalisation : ${self.current_context.market_cap/1e9:.2f}B
"""

        # Ratios financiers
        ratios = self.current_context.ratios
        if ratios:
            ratios_info = """
RATIOS FINANCIERS :
"""
            for key, value in ratios.items():
                if value is not None:
                    ratios_info += f"- {key}: {value}\n"
        else:
            ratios_info = "\nRATIOS FINANCIERS : Non disponibles\n"

        # Données financières
        fin_data = self.current_context.financial_data
        if fin_data:
            financial_info = """
DONNÉES FINANCIÈRES :
"""
            for key, value in fin_data.items():
                if value is not None:
                    financial_info += f"- {key}: {value}\n"
        else:
            financial_info = "\nDONNÉES FINANCIÈRES : Non disponibles\n"

        # DCF
        dcf = self.current_context.dcf_results
        if dcf:
            dcf_info = f"""
VALORISATION DCF :
- Valeur intrinsèque : ${dcf.get('value_per_share', 'N/A')}
- Prix actuel : ${self.current_context.current_price:.2f}
- Potentiel : {dcf.get('upside', 'N/A')}%
"""
        else:
            dcf_info = "\nVALORISATION DCF : Non calculée\n"

        # Risques Cygne Noir
        black_swan = self.current_context.black_swan_risks
        if black_swan:
            bs_info = f"""
ANALYSE CYGNE NOIR :
- Score de risque global : {black_swan.get('global_risk_score', 'N/A')}/10
- Niveau : {black_swan.get('risk_level', 'N/A')}
- Principaux risques : {', '.join([r.get('name_fr', '') for r in black_swan.get('top_risks', [])[:3]])}
"""
        else:
            bs_info = "\nANALYSE CYGNE NOIR : Non disponible\n"

        # Historique de session
        recent_companies = self.session_history.get_recent(5)
        if recent_companies:
            history_info = """
ENTREPRISES ANALYSÉES DANS CETTE SESSION :
"""
            for comp in recent_companies:
                history_info += f"- {comp['name']} ({comp['ticker']})\n"
        else:
            history_info = ""

        # Prompt système complet
        system_prompt = f"""Tu es un assistant d'analyse financière expert, inspiré du système ALADDIN de BlackRock.
Tu assistes l'utilisateur dans l'analyse fondamentale des entreprises cotées en bourse.

{company_info}
{ratios_info}
{financial_info}
{dcf_info}
{bs_info}
{history_info}

RÈGLES IMPORTANTES :
1. Réponds TOUJOURS en français
2. Sois précis, professionnel et factuel
3. Cite des chiffres concrets quand ils sont disponibles
4. Structure tes réponses avec du markdown (titres, listes, gras)
5. Si tu ne disposes pas d'une information, dis-le clairement plutôt que d'inventer
6. Adapte le niveau de détail à la question posée
7. Pour les recommandations d'investissement, rappelle que tu ne donnes pas de conseil financier personnalisé
8. Tu peux comparer l'entreprise actuelle avec les précédentes si l'utilisateur le demande

STYLE DE RÉPONSE :
- Utilise des bullet points pour les listes
- Mets en **gras** les éléments importants
- Sois concis mais complet
- Termine par une conclusion ou un point d'action quand pertinent
"""
        return system_prompt

    def _prepare_messages(self, include_system: bool = True) -> List[Dict]:
        """Prépare les messages pour l'API."""
        messages = []

        if include_system:
            messages.append({
                "role": "system",
                "content": self._build_system_prompt()
            })

        # Ajouter l'historique de conversation (limité)
        history = self.conversation_history[-self.max_history_length:]
        for msg in history:
            if msg.role != "system":  # Éviter les doublons système
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return messages

    def chat_ollama(self, user_message: str, stream: bool = True) -> Generator[str, None, None]:
        """Envoie un message via Ollama et retourne la réponse en streaming."""

        # Ajouter le message utilisateur à l'historique
        self.conversation_history.append(ChatMessage(
            role="user",
            content=user_message
        ))

        messages = self._prepare_messages()

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": stream,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                stream=stream,
                timeout=120
            )

            if response.status_code != 200:
                error_msg = f"Erreur Ollama: {response.status_code}"
                yield error_msg
                return

            full_response = ""

            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            full_response += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            continue
            else:
                data = response.json()
                full_response = data.get("message", {}).get("content", "")
                yield full_response

            # Sauvegarder la réponse complète
            self.conversation_history.append(ChatMessage(
                role="assistant",
                content=full_response
            ))

        except requests.exceptions.Timeout:
            yield "⏱️ La requête a expiré. Veuillez réessayer."
        except requests.exceptions.ConnectionError:
            yield "❌ Impossible de se connecter à Ollama. Vérifiez qu'Ollama est lancé."
        except Exception as e:
            yield f"❌ Erreur: {str(e)}"

    def chat_mistral_api(self, user_message: str, stream: bool = True) -> Generator[str, None, None]:
        """Envoie un message via l'API Mistral Cloud."""

        if not self.api_key:
            yield "❌ Clé API Mistral non configurée."
            return

        # Ajouter le message utilisateur à l'historique
        self.conversation_history.append(ChatMessage(
            role="user",
            content=user_message
        ))

        messages = self._prepare_messages()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": stream
                },
                stream=stream,
                timeout=120
            )

            if response.status_code != 200:
                error_msg = f"Erreur API Mistral: {response.status_code}"
                yield error_msg
                return

            full_response = ""

            if stream:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                full_response += chunk
                                yield chunk
                            except json.JSONDecodeError:
                                continue
            else:
                data = response.json()
                full_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                yield full_response

            # Sauvegarder la réponse complète
            self.conversation_history.append(ChatMessage(
                role="assistant",
                content=full_response
            ))

        except requests.exceptions.Timeout:
            yield "⏱️ La requête a expiré. Veuillez réessayer."
        except Exception as e:
            yield f"❌ Erreur: {str(e)}"

    def chat(self, user_message: str, stream: bool = True) -> Generator[str, None, None]:
        """
        Envoie un message et retourne la réponse.
        Sélectionne automatiquement le bon provider.
        """
        if self.provider == AIProvider.OLLAMA:
            yield from self.chat_ollama(user_message, stream)
        elif self.provider == AIProvider.MISTRAL_API:
            yield from self.chat_mistral_api(user_message, stream)
        else:
            yield "❌ Provider non supporté."

    def chat_sync(self, user_message: str) -> str:
        """Version synchrone de chat (pour la génération de rapports)."""
        return "".join(self.chat(user_message, stream=False))

    def execute_quick_action(self, action_id: str) -> Generator[str, None, None]:
        """Exécute une action rapide prédéfinie."""
        action = next((a for a in QUICK_ACTIONS if a["id"] == action_id), None)

        if not action:
            yield f"❌ Action '{action_id}' non trouvée."
            return

        yield from self.chat(action["prompt"])

    def generate_report(self, report_type: str = "summary") -> Generator[str, None, None]:
        """Génère un rapport d'analyse."""
        report = next((r for r in REPORT_TYPES if r["id"] == report_type), None)

        if not report:
            yield f"❌ Type de rapport '{report_type}' non trouvé."
            return

        yield from self.chat(report["prompt"])

    def get_welcome_message(self) -> str:
        """Retourne le message de bienvenue."""
        if self.current_context.name:
            return f"""👋 Bonjour ! Je suis votre assistant d'analyse financière.

Je vois que vous analysez **{self.current_context.name}** ({self.current_context.ticker}).

Comment puis-je vous aider ?
- Posez-moi une question sur l'entreprise
- Utilisez les boutons d'action rapide ci-dessous
- Demandez-moi de générer un rapport d'analyse

💡 *Exemples de questions :*
- "Le PER est-il élevé pour le secteur ?"
- "Quels sont les principaux risques ?"
- "L'entreprise est-elle trop endettée ?"
"""
        else:
            return """👋 Bonjour ! Je suis votre assistant d'analyse financière.

Sélectionnez d'abord une entreprise à analyser, puis revenez me poser vos questions !

Je pourrai alors vous aider à :
- Analyser les ratios financiers
- Évaluer la valorisation
- Identifier les risques
- Générer des rapports d'investissement
"""

    def clear_history(self):
        """Réinitialise l'historique de conversation."""
        self.conversation_history = []

    def get_conversation_summary(self) -> str:
        """Retourne un résumé de la conversation actuelle."""
        if not self.conversation_history:
            return "Aucune conversation en cours."

        user_messages = [m for m in self.conversation_history if m.role == "user"]
        return f"{len(user_messages)} messages échangés dans cette session."

    def export_conversation(self) -> Dict:
        """Exporte la conversation au format JSON."""
        return {
            "current_company": asdict(self.current_context),
            "session_history": asdict(self.session_history),
            "conversation": [asdict(m) for m in self.conversation_history],
            "exported_at": datetime.now().isoformat()
        }


# ============ FONCTIONS UTILITAIRES POUR STREAMLIT ============

def get_assistant_css() -> str:
    """Retourne le CSS pour l'interface de chat."""
    return """
<style>
/* ===== BOUTON FLOTTANT ===== */
.floating-chat-btn {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    transition: all 0.3s ease;
}

.floating-chat-btn:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.floating-chat-btn.has-notification::after {
    content: '';
    position: absolute;
    top: 5px;
    right: 5px;
    width: 12px;
    height: 12px;
    background: #ff4757;
    border-radius: 50%;
    border: 2px solid white;
}

/* ===== PANNEAU DE CHAT ===== */
.chat-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 400px;
    height: 100vh;
    background: #1a1a2e;
    box-shadow: -5px 0 30px rgba(0, 0, 0, 0.3);
    z-index: 10000;
    display: flex;
    flex-direction: column;
    transform: translateX(100%);
    transition: transform 0.3s ease;
}

.chat-panel.open {
    transform: translateX(0);
}

/* Header */
.chat-header {
    padding: 15px 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.chat-header h3 {
    margin: 0;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.chat-header-buttons {
    display: flex;
    gap: 10px;
}

.chat-header-btn {
    background: rgba(255, 255, 255, 0.2);
    border: none;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s;
}

.chat-header-btn:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* Zone de messages */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

/* Bulles de message */
.chat-message {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.5;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-message.user {
    align-self: flex-end;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 4px;
}

.chat-message.assistant {
    align-self: flex-start;
    background: #2d2d44;
    color: #e0e0e0;
    border-bottom-left-radius: 4px;
}

.chat-message.system {
    align-self: center;
    background: rgba(102, 126, 234, 0.1);
    color: #888;
    font-size: 12px;
    padding: 8px 12px;
    border-radius: 10px;
}

/* Indicateur de frappe */
.typing-indicator {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    background: #2d2d44;
    border-radius: 18px;
    width: fit-content;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    background: #667eea;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* Zone de saisie */
.chat-input-container {
    padding: 15px 20px;
    background: #16162a;
    border-top: 1px solid #2d2d44;
}

.chat-input-wrapper {
    display: flex;
    gap: 10px;
    background: #2d2d44;
    border-radius: 25px;
    padding: 5px 5px 5px 15px;
}

.chat-input {
    flex: 1;
    background: transparent;
    border: none;
    color: white;
    font-size: 14px;
    outline: none;
}

.chat-input::placeholder {
    color: #666;
}

.chat-send-btn {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s;
}

.chat-send-btn:hover {
    transform: scale(1.05);
}

/* Actions rapides */
.chat-quick-actions {
    padding: 10px 20px;
    background: #16162a;
    border-top: 1px solid #2d2d44;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.quick-action-btn {
    padding: 8px 12px;
    background: #2d2d44;
    border: 1px solid #3d3d5c;
    border-radius: 20px;
    color: #e0e0e0;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
}

.quick-action-btn:hover {
    background: #3d3d5c;
    border-color: #667eea;
}

/* Responsive */
@media (max-width: 768px) {
    .chat-panel {
        width: 100%;
    }
}

/* Markdown dans les messages */
.chat-message h1, .chat-message h2, .chat-message h3 {
    margin: 10px 0 5px 0;
    font-size: 14px;
}

.chat-message ul, .chat-message ol {
    margin: 5px 0;
    padding-left: 20px;
}

.chat-message li {
    margin: 3px 0;
}

.chat-message strong {
    color: #667eea;
}

.chat-message code {
    background: rgba(102, 126, 234, 0.2);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
}

/* Scrollbar */
.chat-messages::-webkit-scrollbar {
    width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
    background: #1a1a2e;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: #3d3d5c;
    border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: #4d4d6c;
}
</style>
"""


def format_financial_context(info: Dict, data: Dict = None) -> Dict:
    """
    Formate les données financières pour le contexte de l'assistant.

    Args:
        info: Dictionnaire info de yfinance
        data: Données supplémentaires (DCF, Black Swan, etc.)

    Returns:
        Dictionnaire formaté pour set_context()
    """
    # Ratios principaux
    ratios = {}
    ratio_keys = [
        ('trailingPE', 'PER'),
        ('forwardPE', 'PER Forward'),
        ('priceToBook', 'P/B'),
        ('priceToSalesTrailing12Months', 'P/S'),
        ('enterpriseToEbitda', 'EV/EBITDA'),
        ('returnOnEquity', 'ROE'),
        ('returnOnAssets', 'ROA'),
        ('profitMargins', 'Marge Nette'),
        ('operatingMargins', 'Marge Opérationnelle'),
        ('grossMargins', 'Marge Brute'),
        ('debtToEquity', 'Dette/Equity'),
        ('currentRatio', 'Current Ratio'),
        ('quickRatio', 'Quick Ratio'),
        ('dividendYield', 'Rendement Dividende'),
        ('payoutRatio', 'Payout Ratio'),
        ('beta', 'Beta'),
    ]

    for key, label in ratio_keys:
        value = info.get(key)
        if value is not None:
            if 'Margin' in label or 'ROE' in label or 'ROA' in label or 'Rendement' in label or 'Payout' in label:
                ratios[label] = f"{value * 100:.2f}%"
            elif isinstance(value, float):
                ratios[label] = f"{value:.2f}"
            else:
                ratios[label] = str(value)

    # Données financières
    financial_data = {}
    fin_keys = [
        ('totalRevenue', 'Chiffre d\'affaires'),
        ('revenueGrowth', 'Croissance CA'),
        ('netIncomeToCommon', 'Résultat Net'),
        ('ebitda', 'EBITDA'),
        ('freeCashflow', 'Free Cash Flow'),
        ('operatingCashflow', 'Cash Flow Opérationnel'),
        ('totalDebt', 'Dette Totale'),
        ('totalCash', 'Trésorerie'),
        ('totalAssets', 'Actifs Totaux'),
        ('totalStockholderEquity', 'Capitaux Propres'),
    ]

    for key, label in fin_keys:
        value = info.get(key)
        if value is not None:
            if abs(value) >= 1e12:
                financial_data[label] = f"${value/1e12:.2f}T"
            elif abs(value) >= 1e9:
                financial_data[label] = f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                financial_data[label] = f"${value/1e6:.2f}M"
            elif 'Growth' in label:
                financial_data[label] = f"{value * 100:.1f}%"
            else:
                financial_data[label] = f"${value:,.0f}"

    context = {
        "name": info.get("shortName", ""),
        "ticker": info.get("symbol", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "country": info.get("country", ""),
        "current_price": info.get("regularMarketPrice", 0) or 0,
        "market_cap": info.get("marketCap", 0) or 0,
        "ratios": ratios,
        "financial_data": financial_data,
    }

    # Ajouter les données supplémentaires si disponibles
    if data:
        if "dcf_results" in data:
            context["dcf_results"] = data["dcf_results"]
        if "black_swan_risks" in data:
            context["black_swan_risks"] = data["black_swan_risks"]
        if "esg_data" in data:
            context["esg_data"] = data["esg_data"]

    return context


# ============ CLASSE STREAMLIT CHAT COMPONENT ============

class StreamlitChatUI:
    """
    Composant de chat pour Streamlit.
    Gère l'interface utilisateur et l'état de session.
    """

    def __init__(self, assistant: MistralAssistant):
        """
        Initialise le composant.

        Args:
            assistant: Instance de MistralAssistant
        """
        self.assistant = assistant

    @staticmethod
    def init_session_state():
        """Initialise les variables de session Streamlit."""
        import streamlit as st

        if "chat_open" not in st.session_state:
            st.session_state.chat_open = False

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        if "chat_input" not in st.session_state:
            st.session_state.chat_input = ""

        if "assistant" not in st.session_state:
            st.session_state.assistant = None

        if "waiting_response" not in st.session_state:
            st.session_state.waiting_response = False

    @staticmethod
    def toggle_chat():
        """Bascule l'état d'ouverture du chat."""
        import streamlit as st
        st.session_state.chat_open = not st.session_state.chat_open

    @staticmethod
    def close_chat():
        """Ferme le panneau de chat."""
        import streamlit as st
        st.session_state.chat_open = False

    @staticmethod
    def clear_chat():
        """Efface l'historique de conversation."""
        import streamlit as st
        st.session_state.chat_messages = []
        if st.session_state.assistant:
            st.session_state.assistant.clear_history()


# ============ TESTS ============

if __name__ == "__main__":
    # Test de base
    print("=== Test du module AI Assistant ===\n")

    # Créer un assistant
    assistant = MistralAssistant(
        provider=AIProvider.OLLAMA,
        model="mistral:latest"
    )

    # Vérifier la disponibilité
    print(f"Ollama disponible: {assistant.check_availability()}")

    # Définir un contexte de test
    test_context = {
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "current_price": 195.50,
        "market_cap": 3000000000000,
        "ratios": {
            "PER": "29.5",
            "ROE": "147.25%",
            "Marge Nette": "25.31%"
        },
        "financial_data": {
            "Chiffre d'affaires": "$383.29B",
            "Résultat Net": "$96.99B",
            "Free Cash Flow": "$99.58B"
        }
    }

    assistant.set_context(test_context)

    # Afficher le message de bienvenue
    print("\n--- Message de bienvenue ---")
    print(assistant.get_welcome_message())

    # Test d'une conversation (si Ollama est disponible)
    if assistant.is_available:
        print("\n--- Test de conversation ---")
        print("User: Quel est le PER d'Apple ?\n")
        print("Assistant: ", end="")
        for chunk in assistant.chat("Quel est le PER d'Apple ?"):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        print("\n⚠️ Ollama n'est pas disponible. Lancez Ollama pour tester la conversation.")

    print("\n=== Test terminé ===")
