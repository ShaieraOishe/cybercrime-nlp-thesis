"""Text Preprocessing & Privacy-Preserving PII Sanitization Module.

==============================================================================
THESIS CONTEXT:
In automated cybercrime complaint triage, complaints written by victims
frequently contain sensitive Personally Identifiable Information (PII) such as
real names, email addresses, phone numbers, credit card digits, and crypto
wallets. 

This module performs privacy-preserving text sanitization:
1. Replaces sensitive PII with standardized generic semantic tokens (e.g. <EMAIL>, <PHONE>).
2. Prevents machine learning and transformer models from overfitting to specific victim data.
3. Normalizes Indicators of Compromise (IOCs) like malicious URLs and IP addresses.
4. Complies with legal and ethical data protection standards (e.g. GDPR, HIPAA, CCPA).
==============================================================================
"""

import re
from typing import List, Optional

class CybercrimeTextCleaner:
    """Sanitizes raw cybercrime complaint narratives by masking sensitive PII

    and normalizing technical noise.
    """

    # --------------------------------------------------------------------------
    # COMPILED REGEX PATTERNS FOR PII & TECHNICAL INDICATORS (IOCs)
    # --------------------------------------------------------------------------
    
    # Matches standard email addresses (e.g., victim@example.com -> <EMAIL>)
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    )
    
    # Matches international and domestic phone number formats (e.g., +1-555-123-4567 -> <PHONE>)
    PHONE_PATTERN = re.compile(
        r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    )
    
    # Matches web hyperlinks (e.g., http://phishing-site.com -> <URL>)
    URL_PATTERN = re.compile(
        r'https?://\S+|www\.\S+'
    )
    
    # Matches IPv4 addresses (e.g., 192.168.1.1 -> <IP_ADDRESS>)
    IP_PATTERN = re.compile(
        r'\b\d{1,3}(?:\.\d{1,3}){3}\b'
    )
    
    # Matches 13 to 16 digit credit card numbers with optional spaces/hyphens (-> <CARD_NUMBER>)
    CARD_PATTERN = re.compile(
        r'\b(?:\d[ -]*?){13,16}\b'
    )
    
    # Matches Bitcoin (Legacy/SegWit) and Ethereum public wallet addresses (-> <CRYPTO_WALLET>)
    CRYPTO_WALLET_PATTERN = re.compile(
        r'\b(0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b'
    )
    
    # Collapses multiple consecutive spaces/tabs/newlines into a single clean space
    EXTRA_WHITESPACE_PATTERN = re.compile(r'\s+')

    def __init__(self, mask_pii: bool = True, preserve_urls_domains: bool = False):
        """Initializes the cleaner.

        Args:
            mask_pii (bool): If True, replaces victim emails, phones, cards, and wallets with tokens.
            preserve_urls_domains (bool): If True, retains raw URLs; if False, masks with <URL>.
        """
        self.mask_pii = mask_pii
        self.preserve_urls_domains = preserve_urls_domains

    def clean(self, text: Optional[str]) -> str:
        """Sanitizes and normalizes a single text string.

        Args:
            text (str): Raw unstructured complaint narrative.

        Returns:
            str: Cleaned, anonymized text string.
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text.strip()

        # Step 1: Mask personally identifiable information (PII)
        if self.mask_pii:
            cleaned = self.EMAIL_PATTERN.sub("<EMAIL>", cleaned)
            cleaned = self.PHONE_PATTERN.sub("<PHONE>", cleaned)
            cleaned = self.CARD_PATTERN.sub("<CARD_NUMBER>", cleaned)
            cleaned = self.CRYPTO_WALLET_PATTERN.sub("<CRYPTO_WALLET>", cleaned)

        # Step 2: Mask technical URLs and IP addresses unless explicitly preserved
        if not self.preserve_urls_domains:
            cleaned = self.URL_PATTERN.sub("<URL>", cleaned)
            cleaned = self.IP_PATTERN.sub("<IP_ADDRESS>", cleaned)

        # Step 3: Collapse whitespace into single spaces and strip leading/trailing spaces
        cleaned = self.EXTRA_WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Sanitizes a list of complaint narratives in batch.

        Args:
            texts (List[str]): List of raw texts.

        Returns:
            List[str]: List of sanitized texts.
        """
        return [self.clean(t) for t in texts]
