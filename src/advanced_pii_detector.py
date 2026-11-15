"""
PII detection for the RAG guardrails.

One synchronous detection path: Microsoft Presidio (pattern recognizers with
checksum validation, backed by the spaCy ``en_core_web_sm`` model) plus
project-specific recognizers (API keys, self-disclosed names, and SSN/phone
formats).

There is no fallback detector. If Presidio or the spaCy model cannot be loaded,
construction raises and the service refuses to start.

Public figures named in the paper (authors, institutions) are not PII, so
NER-based PERSON/ORG/GPE entities are deliberately not requested.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Presidio logs the analyzed text at DEBUG level; never let raw PII reach the logs.
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)

SPACY_MODEL = "en_core_web_sm"

# Presidio entity types requested from the analyzer.
PRESIDIO_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"]

# Existing placeholder style used in `pii_masked_input`.
MASK_LABELS: Dict[str, str] = {
    "EMAIL_ADDRESS": "[EMAIL_MASKED]",
    "PHONE_NUMBER": "[PHONE_MASKED]",
    "US_SSN": "[SSN_MASKED]",
    "CREDIT_CARD": "[CREDIT_CARD_MASKED]",
    "IP_ADDRESS": "[IP_ADDRESS_MASKED]",
    "API_KEY": "[API_KEY_MASKED]",
    "NAME": "[NAME_MASKED]",
}

# Context words are case-insensitive; the disclosed name must be capitalised.
_NAME_PATTERN = re.compile(
    r"(?i:my name is|i am|i'm|call me|full name)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

# Format-based recognizers. Presidio validates values and rejects well-known
# placeholder numbers (e.g. 123-45-6789, 555-01xx), which red-team inputs use,
# so separator-delimited SSN and phone formats are also matched by shape.
_FORMAT_PATTERNS = {
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE_NUMBER": re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
}

_API_KEY_REGEX = r"\b(?:sk-|pk_|api[_-]?key[_-]?|token[_-]?)[A-Za-z0-9]{20,}\b"


@dataclass(frozen=True)
class PIIEntity:
    """A detected PII span. ``text`` is raw PII: never log it or return it publicly."""
    type: str
    start: int
    end: int
    score: float


class PIIDetector:
    """Detects and masks PII. Synchronous and safe to call from any thread."""

    def __init__(self, score_threshold: float = 0.5):
        self.score_threshold = score_threshold
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
        }).create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        self._analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="API_KEY",
            patterns=[Pattern("api_key", _API_KEY_REGEX, 0.9)],
        ))

    def detect(self, text: str) -> List[PIIEntity]:
        """Return non-overlapping PII entities sorted by position."""
        found = [
            PIIEntity(r.entity_type, r.start, r.end, r.score)
            for r in self._analyzer.analyze(
                text=text,
                language="en",
                entities=PRESIDIO_ENTITIES + ["API_KEY"],
                score_threshold=self.score_threshold,
            )
        ]
        for entity_type, pattern in _FORMAT_PATTERNS.items():
            found.extend(PIIEntity(entity_type, m.start(), m.end(), 0.85) for m in pattern.finditer(text))
        for match in _NAME_PATTERN.finditer(text):
            found.append(PIIEntity("NAME", match.start(1), match.end(1), 0.85))
        return self._drop_overlaps(found)

    def mask(self, text: str, entities: List[PIIEntity]) -> str:
        """Replace each entity span with its placeholder."""
        masked = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            masked = masked[:entity.start] + MASK_LABELS[entity.type] + masked[entity.end:]
        return masked

    @staticmethod
    def _drop_overlaps(entities: List[PIIEntity]) -> List[PIIEntity]:
        """Keep the highest-scoring (then longest) entity of any overlapping group."""
        kept: List[PIIEntity] = []
        for entity in sorted(entities, key=lambda e: (-e.score, -(e.end - e.start))):
            if all(entity.end <= k.start or entity.start >= k.end for k in kept):
                kept.append(entity)
        return sorted(kept, key=lambda e: e.start)
