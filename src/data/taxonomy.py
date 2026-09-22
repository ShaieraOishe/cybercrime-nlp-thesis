"""Unified 7-Class Cybercrime Taxonomy & Domain Heuristics.

==============================================================================
THESIS CONTEXT:
One of the core contributions of this research is harmonizing disparate,
noisy complaint sources (CFPB, CLAIR, UCI SMS Spam, Davidson, Scam Survivors,
and technical cyber threat feeds) into a structured 7-class cybercrime typology.

This module formalizes:
1. The canonical 7 target classes.
2. Formal domain descriptions aligned with legal and criminological standards.
3. Keyword heuristics used for preliminary mapping and data auditing.
==============================================================================
"""

from typing import Dict, List, Optional
import re

# Canonical 7 Cybercrime Classes evaluated in the comparative thesis
UNIFIED_CLASSES = [
    "online_financial_fraud",      # CLAIR advance-fee + CFPB wire/crypto scams
    "phishing_smishing",           # UCI SMS spam + spoofed banking links
    "cyber_harassment_blackmail",  # Davidson & Waseem hate speech / targeted threats
    "cyber_threat_intelligence",   # Behzadan's CyberTweets + technical incident telemetry
    "ecommerce_fraud",             # BBB Scam Tracker + CFPB merchant non-delivery
    "sextortion",                  # Scam Survivors + TExtPhish webcam extortion lures
    "identity_threat"              # CFPB identity theft + synthetic ID compromises
]

# Academic and legal definitions for each category
CLASS_DESCRIPTIONS: Dict[str, str] = {
    "online_financial_fraud": (
        "Unauthorized electronic wire transfers, advance-fee 419 email schemes, "
        "cryptocurrency investment scams, and deceptive banking transactions."
    ),
    "phishing_smishing": (
        "Deceptive SMS communications (smishing) and phishing emails designed to trick "
        "victims into visiting credential-harvesting web portals or paying bogus fees."
    ),
    "cyber_harassment_blackmail": (
        "Targeted digital stalking, swatting threats, abusive brigading, doxxing personal "
        "home addresses, and malicious defamation intended to intimidate victims."
    ),
    "cyber_threat_intelligence": (
        "Technical telemetry regarding zero-day vulnerability exploits (CVEs), botnet "
        "amplification DDoS attacks, ransomware C2 infrastructure, and malware signatures."
    ),
    "ecommerce_fraud": (
        "Consumer complaints involving sham online storefronts, counterfeit luxury items, "
        "non-delivery of purchased merchandise, and unauthorized recurring card billing."
    ),
    "sextortion": (
        "Digital extortion where adversaries claim to possess compromising webcam video "
        "or intimate imagery, demanding ransom payments in Bitcoin or gift cards."
    ),
    "identity_threat": (
        "Theft of personal credentials (SSN, national ID), synthetic identity creation, "
        "unauthorized credit lines opened without consent, and data breach exposures."
    )
}

# Regex pattern rules used to map arbitrary text or unstructured descriptions
CATEGORY_KEYWORD_RULES: Dict[str, List[str]] = {
    "online_financial_fraud": [
        r"\bwire fraud\b", r"\bcrypto(?:currency)? scam\b", r"\bunauthorized (?:wire|transfer|transaction)\b",
        r"\binvestment fraud\b", r"\badvance fee\b", r"\b419 scheme\b", r"\bescrow\b"
    ],
    "phishing_smishing": [
        r"\bphishing\b", r"\bsmishing\b", r"\bspoofed (?:email|link|site)\b",
        r"\bpackage delivery\b", r"\bcredential harvest\b", r"\bverify your (?:card|password)\b"
    ],
    "cyber_harassment_blackmail": [
        r"\bharassment\b", r"\bstalking\b", r"\bdoxx(?:ed|ing)\b", r"\bthreat(?:en|ening)?\b",
        r"\bdefamat(?:ory|ion)\b", r"\bswatting\b", r"\bhate speech\b"
    ],
    "cyber_threat_intelligence": [
        r"\bcve-\d{4}-\d+\b", r"\bzero-day\b", r"\bddos\b", r"\bransomware c2\b",
        r"\bmalware signature\b", r"\bthreat actor\b", r"\bexfiltration\b"
    ],
    "ecommerce_fraud": [
        r"\bonline purchase\b", r"\bmerchant dispute\b", r"\bnever delivered\b",
        r"\bcounterfeit\b", r"\bfake store\b", r"\bdeceptive shopping\b"
    ],
    "sextortion": [
        r"\bsextortion\b", r"\bwebcam (?:video|recording)\b", r"\bintimate (?:video|photo)\b",
        r"\bblackmail\b", r"\brecorded you\b", r"\bdual-screen\b"
    ],
    "identity_threat": [
        r"\bidentity theft\b", r"\bstolen ssn\b", r"\bopened without (?:my )?consent\b",
        r"\bsynthetic identity\b", r"\bcompromised credentials\b", r"\bdata breach\b"
    ]
}

def map_text_to_taxonomy(text: str, default: str = "online_financial_fraud") -> str:
    """Classifies an arbitrary narrative into a taxonomy category using keyword rules.

    Args:
        text (str): The narrative text or incident summary.
        default (str): Fallback category if no specific keywords match.

    Returns:
        str: The mapped category string.
    """
    if not text:
        return default
        
    text_lower = text.lower()

    for category, patterns in CATEGORY_KEYWORD_RULES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return category

    return default
