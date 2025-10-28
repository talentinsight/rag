"""
Advanced PII Detection Module
Combines multiple approaches for robust PII detection
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Optional imports - will fallback gracefully if not available
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

try:
    # First check if torch is available
    # Only then import transformers to avoid warnings
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # PyTorch not available, skip transformers entirely
    TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PIIDetectionMethod(Enum):
    REGEX = "regex"
    PRESIDIO = "presidio"
    TRANSFORMERS = "transformers"
    SPACY = "spacy"
    HYBRID = "hybrid"


@dataclass
class PIIDetectionResult:
    """Result from PII detection"""
    has_pii: bool
    confidence: float
    detected_entities: List[Dict[str, Any]]
    method_used: str
    processing_time_ms: float
    metadata: Dict[str, Any]


class AdvancedPIIDetector:
    """
    Advanced PII Detection with multiple fallback methods
    """
    
    def __init__(self, 
                 preferred_method: PIIDetectionMethod = PIIDetectionMethod.HYBRID,
                 confidence_threshold: float = 0.7,
                 enable_caching: bool = True):
        """
        Initialize Advanced PII Detector
        
        Args:
            preferred_method: Primary detection method to use
            confidence_threshold: Minimum confidence for PII detection
            enable_caching: Whether to cache results for performance
        """
        self.preferred_method = preferred_method
        self.confidence_threshold = confidence_threshold
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        
        # Initialize available detectors
        self._init_detectors()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info(f"Initialized AdvancedPIIDetector with method: {preferred_method}")
    
    def _init_detectors(self):
        """Initialize all available detection methods"""
        self.available_methods = []
        
        # 1. Presidio (Microsoft's PII detection)
        if PRESIDIO_AVAILABLE:
            try:
                self.presidio_analyzer = AnalyzerEngine()
                self.presidio_anonymizer = AnonymizerEngine()
                self.available_methods.append(PIIDetectionMethod.PRESIDIO)
                logger.info("✅ Presidio PII detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Presidio: {e}")
        
        # 2. Transformers (Hugging Face models)
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use Microsoft's DeBERTa model for PII detection
                self.transformer_detector = pipeline(
                    "ner",
                    model="microsoft/deberta-v3-base",
                    tokenizer="microsoft/deberta-v3-base",
                    aggregation_strategy="simple"
                )
                self.available_methods.append(PIIDetectionMethod.TRANSFORMERS)
                logger.info("✅ Transformers PII detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Transformers: {e}")
        
        # 3. spaCy (with custom rules)
        if SPACY_AVAILABLE:
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
                self.available_methods.append(PIIDetectionMethod.SPACY)
                logger.info("✅ spaCy PII detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize spaCy: {e}")
        
        # 4. Regex (always available as fallback)
        self.available_methods.append(PIIDetectionMethod.REGEX)
        self._init_regex_patterns()
        logger.info("✅ Regex PII detector initialized")
        
        logger.info(f"Available detection methods: {[m.value for m in self.available_methods]}")
    
    def _init_regex_patterns(self):
        """Initialize comprehensive and dynamic PII patterns"""
        self.pii_patterns = {
            # Core PII patterns (enhanced and more robust)
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            
            # Financial patterns (comprehensive)
            "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "bank_account": r"\b[0-9]{8,17}\b",
            "routing_number": r"\b[0-9]{9}\b",
            
            # Identity patterns (enhanced)
            "name": r"(?i)\b(?:my name is|i am|i'm|call me|name:|full name|first name|last name)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            "passport": r"\b[A-Z]{1,2}[0-9]{6,9}\b",
            "driver_license": r"\b[A-Z]{1,2}[0-9]{6,8}\b",
            "national_id": r"\b[A-Z]{2}[0-9]{6,12}\b",
            
            # Technical identifiers (expanded)
            "api_key": r"\b(?:sk-|pk_|api[_-]?key[_-]?|token[_-]?|secret[_-]?)[A-Za-z0-9]{15,}\b",
            "jwt_token": r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
            "aws_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
            "github_token": r"\bgh[ps]_[A-Za-z0-9]{36}\b",
            
            # Network identifiers (comprehensive)
            "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "ipv6": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
            "mac_address": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
            "url_with_auth": r"\b(?:https?://)[^@\s]+:[^@\s]+@[^\s]+\b",
            
            # Medical/Health (new category)
            "medical_record": r"\b(?:MRN|MR#|Medical Record)\s*:?\s*[A-Z0-9]{6,12}\b",
            "insurance_id": r"\b[A-Z]{2,3}[0-9]{6,12}\b",
            
            # Address patterns (enhanced)
            "address": r"(?i)\b(?:address|addr|location)\s*:?\s*[0-9]+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd)\b",
            "zip_code": r"\b[0-9]{5}(?:-[0-9]{4})?\b",
            
            # Date of birth patterns
            "date_of_birth": r"(?i)\b(?:dob|date of birth|born on|birthday)\s*:?\s*[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}\b",
            
            # Biometric identifiers
            "fingerprint_id": r"\b[A-Z0-9]{8,16}(?:FP|PRINT)\b",
            
            # Custom dynamic patterns (context-aware)
            "personal_identifier": r"(?i)\b(?:my|personal|private)\s+(?:id|identifier|number|code)\s*:?\s*[A-Za-z0-9]{6,}\b"
        }
        
        # Dynamic pattern enhancement based on context
        self._enhance_patterns_dynamically()
    
    def _enhance_patterns_dynamically(self):
        """Dynamically enhance PII patterns based on runtime context"""
        # Add region-specific patterns dynamically
        import locale
        try:
            current_locale = locale.getlocale()[0]
            if current_locale and 'US' in current_locale:
                # US-specific enhancements
                self.pii_patterns["us_tax_id"] = r"\b[0-9]{2}-[0-9]{7}\b"
                self.pii_patterns["us_phone_ext"] = r"\b[0-9]{3}-[0-9]{3}-[0-9]{4}\s*(?:ext|x)\s*[0-9]{1,5}\b"
            elif current_locale and 'GB' in current_locale:
                # UK-specific enhancements
                self.pii_patterns["uk_nhs"] = r"\b[0-9]{3}\s*[0-9]{3}\s*[0-9]{4}\b"
                self.pii_patterns["uk_postcode"] = r"\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b"
        except:
            pass  # Ignore locale detection errors
        
        # Add time-based dynamic patterns
        from datetime import datetime
        current_year = datetime.now().year
        
        # Dynamic year-based patterns for IDs that include years
        self.pii_patterns["year_based_id"] = rf"\b(?:{current_year-100}|{current_year-50}|{current_year})[0-9]{{6,10}}\b"
        
        # Context-aware enhancement based on common PII indicators
        self._add_contextual_patterns()
    
    def _add_contextual_patterns(self):
        """Add contextual PII patterns based on common disclosure patterns"""
        # Dynamic patterns for common PII disclosure contexts
        contextual_patterns = {
            "credential_disclosure": r"(?i)\b(?:username|user|login|password|pass|pwd)\s*[:=]\s*[A-Za-z0-9@._-]{3,}\b",
            "contact_disclosure": r"(?i)\b(?:contact|reach me|call me|email me)\s*[:@]\s*[A-Za-z0-9@._-]+\b",
            "emergency_contact": r"(?i)\b(?:emergency contact|in case of emergency|ice)\s*[:]\s*[A-Za-z0-9\s@._-]+\b",
            "family_member": r"(?i)\b(?:my (?:mother|father|son|daughter|wife|husband|spouse))\s+(?:is|name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            "work_info": r"(?i)\b(?:work at|employed by|company|employer)\s*[:]\s*([A-Za-z\s&.,]+)\b",
            "location_sharing": r"(?i)\b(?:live at|address is|located at|my location)\s*[:]\s*([A-Za-z0-9\s,.-]+)\b"
        }
        
        # Add contextual patterns to main patterns
        self.pii_patterns.update(contextual_patterns)
        
        # Add dynamic number sequence detection
        self.pii_patterns["suspicious_number_sequence"] = r"\b[0-9]{6,16}\b"  # Catch various ID formats
        
        logger.info(f"✅ Enhanced PII patterns: {len(self.pii_patterns)} total patterns loaded")
    
    async def detect_pii(self, text: str) -> PIIDetectionResult:
        """
        Detect PII using the configured method with fallbacks
        
        Args:
            text: Input text to analyze
            
        Returns:
            PIIDetectionResult with detection details
        """
        import time
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching and text in self.cache:
            cached_result = self.cache[text]
            cached_result.metadata["from_cache"] = True
            return cached_result
        
        # Try preferred method first
        result = None
        methods_tried = []
        
        if self.preferred_method == PIIDetectionMethod.HYBRID:
            result = await self._detect_hybrid(text)
            methods_tried.append("hybrid")
        elif self.preferred_method in self.available_methods:
            result = await self._detect_with_method(text, self.preferred_method)
            methods_tried.append(self.preferred_method.value)
        
        # Fallback to other methods if needed
        if not result or result.confidence < self.confidence_threshold:
            for method in self.available_methods:
                if method != self.preferred_method:
                    fallback_result = await self._detect_with_method(text, method)
                    methods_tried.append(method.value)
                    
                    if fallback_result and fallback_result.confidence >= self.confidence_threshold:
                        result = fallback_result
                        break
        
        # Final fallback to regex if nothing else worked
        if not result:
            result = await self._detect_regex(text)
            methods_tried.append("regex_fallback")
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time
        result.metadata["methods_tried"] = methods_tried
        
        # Cache result
        if self.enable_caching:
            self.cache[text] = result
        
        return result
    
    async def _detect_hybrid(self, text: str) -> PIIDetectionResult:
        """
        Hybrid detection using multiple methods and consensus
        """
        results = []
        
        # Run multiple detectors in parallel
        tasks = []
        for method in self.available_methods:
            if method != PIIDetectionMethod.HYBRID:
                task = self._detect_with_method(text, method)
                tasks.append(task)
        
        # Wait for all results
        if tasks:
            method_results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in method_results if isinstance(r, PIIDetectionResult)]
        
        # Combine results using consensus
        return self._combine_results(results, "hybrid")
    
    async def _detect_with_method(self, text: str, method: PIIDetectionMethod) -> PIIDetectionResult:
        """Detect PII with specific method"""
        try:
            if method == PIIDetectionMethod.PRESIDIO and PIIDetectionMethod.PRESIDIO in self.available_methods:
                return await self._detect_presidio(text)
            elif method == PIIDetectionMethod.TRANSFORMERS and PIIDetectionMethod.TRANSFORMERS in self.available_methods:
                return await self._detect_transformers(text)
            elif method == PIIDetectionMethod.SPACY and PIIDetectionMethod.SPACY in self.available_methods:
                return await self._detect_spacy(text)
            else:
                return await self._detect_regex(text)
        except Exception as e:
            logger.error(f"Error in {method.value} detection: {e}")
            return PIIDetectionResult(
                has_pii=False,
                confidence=0.0,
                detected_entities=[],
                method_used=f"{method.value}_error",
                processing_time_ms=0.0,
                metadata={"error": str(e)}
            )
    
    async def _detect_presidio(self, text: str) -> PIIDetectionResult:
        """Detect PII using Microsoft Presidio"""
        def _presidio_analyze():
            results = self.presidio_analyzer.analyze(text=text, language='en')
            return results
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        presidio_results = await loop.run_in_executor(self.executor, _presidio_analyze)
        
        entities = []
        max_confidence = 0.0
        
        for result in presidio_results:
            entity = {
                "type": result.entity_type,
                "text": text[result.start:result.end],
                "start": result.start,
                "end": result.end,
                "confidence": result.score
            }
            entities.append(entity)
            max_confidence = max(max_confidence, result.score)
        
        return PIIDetectionResult(
            has_pii=len(entities) > 0,
            confidence=max_confidence,
            detected_entities=entities,
            method_used="presidio",
            processing_time_ms=0.0,
            metadata={"presidio_version": "latest"}
        )
    
    async def _detect_transformers(self, text: str) -> PIIDetectionResult:
        """Detect PII using Transformers NER"""
        def _transformer_analyze():
            return self.transformer_detector(text)
        
        loop = asyncio.get_event_loop()
        ner_results = await loop.run_in_executor(self.executor, _transformer_analyze)
        
        entities = []
        max_confidence = 0.0
        
        # Filter for PII-related entities
        pii_labels = ['PERSON', 'ORG', 'GPE', 'PHONE', 'EMAIL', 'SSN', 'CREDIT_CARD']
        
        for result in ner_results:
            if result['entity_group'] in pii_labels:
                entity = {
                    "type": result['entity_group'],
                    "text": result['word'],
                    "start": result['start'],
                    "end": result['end'],
                    "confidence": result['score']
                }
                entities.append(entity)
                max_confidence = max(max_confidence, result['score'])
        
        return PIIDetectionResult(
            has_pii=len(entities) > 0,
            confidence=max_confidence,
            detected_entities=entities,
            method_used="transformers",
            processing_time_ms=0.0,
            metadata={"model": "microsoft/deberta-v3-base"}
        )
    
    async def _detect_spacy(self, text: str) -> PIIDetectionResult:
        """Detect PII using spaCy NER"""
        def _spacy_analyze():
            doc = self.spacy_nlp(text)
            return doc.ents
        
        loop = asyncio.get_event_loop()
        spacy_entities = await loop.run_in_executor(self.executor, _spacy_analyze)
        
        entities = []
        max_confidence = 0.0
        
        # Filter for PII-related entities
        pii_labels = ['PERSON', 'ORG', 'GPE', 'PHONE_NUMBER', 'EMAIL']
        
        for ent in spacy_entities:
            if ent.label_ in pii_labels:
                entity = {
                    "type": ent.label_,
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.8  # spaCy doesn't provide confidence scores
                }
                entities.append(entity)
                max_confidence = max(max_confidence, 0.8)
        
        return PIIDetectionResult(
            has_pii=len(entities) > 0,
            confidence=max_confidence,
            detected_entities=entities,
            method_used="spacy",
            processing_time_ms=0.0,
            metadata={"model": "en_core_web_sm"}
        )
    
    async def _detect_regex(self, text: str) -> PIIDetectionResult:
        """Detect PII using enhanced regex patterns"""
        entities = []
        max_confidence = 0.0
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Context-aware filtering (from your existing logic)
                if self._is_likely_false_positive(text, match, pii_type):
                    continue
                
                entity = {
                    "type": pii_type,
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9  # High confidence for regex matches
                }
                entities.append(entity)
                max_confidence = max(max_confidence, 0.9)
        
        return PIIDetectionResult(
            has_pii=len(entities) > 0,
            confidence=max_confidence,
            detected_entities=entities,
            method_used="regex",
            processing_time_ms=0.0,
            metadata={"patterns_used": len(self.pii_patterns)}
        )
    
    def _is_likely_false_positive(self, text: str, match, pii_type: str) -> bool:
        """Enhanced false positive detection from your existing logic"""
        text_lower = text.lower()
        
        # Test data indicators
        if any(indicator in text_lower for indicator in ['test', 'example', 'sample', 'demo', 'mock']):
            return True
        
        # Emotional context for credit cards
        if pii_type == "credit_card":
            if re.search(r"\b(?:hate|angry|mad|upset)\b", text, re.IGNORECASE):
                return True
        
        # API key context check
        if pii_type == "api_key":
            if not re.search(r"\b(?:sk-|pk_|api|key|token)", text, re.IGNORECASE):
                return True
        
        return False
    
    def _combine_results(self, results: List[PIIDetectionResult], method_name: str) -> PIIDetectionResult:
        """Combine multiple detection results using consensus"""
        if not results:
            return PIIDetectionResult(
                has_pii=False,
                confidence=0.0,
                detected_entities=[],
                method_used=method_name,
                processing_time_ms=0.0,
                metadata={"no_results": True}
            )
        
        # Simple consensus: if majority says PII, then PII
        pii_votes = sum(1 for r in results if r.has_pii)
        has_pii = pii_votes > len(results) / 2
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Combine all entities
        all_entities = []
        for result in results:
            all_entities.extend(result.detected_entities)
        
        # Remove duplicates based on text and position
        unique_entities = []
        seen = set()
        for entity in all_entities:
            key = (entity['text'], entity['start'], entity['end'])
            if key not in seen:
                unique_entities.append(entity)
                seen.add(key)
        
        return PIIDetectionResult(
            has_pii=has_pii,
            confidence=avg_confidence,
            detected_entities=unique_entities,
            method_used=method_name,
            processing_time_ms=0.0,
            metadata={
                "consensus_votes": f"{pii_votes}/{len(results)}",
                "methods_combined": [r.method_used for r in results]
            }
        )
    
    def get_available_methods(self) -> List[str]:
        """Get list of available detection methods"""
        return [method.value for method in self.available_methods]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics"""
        return {
            "available_methods": self.get_available_methods(),
            "preferred_method": self.preferred_method.value,
            "confidence_threshold": self.confidence_threshold,
            "cache_size": len(self.cache) if self.cache else 0,
            "presidio_available": PRESIDIO_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "spacy_available": SPACY_AVAILABLE
        }


# Example usage and testing
async def main():
    """Test the Advanced PII Detector"""
    detector = AdvancedPIIDetector(
        preferred_method=PIIDetectionMethod.HYBRID,
        confidence_threshold=0.7
    )
    
    # Test cases
    test_cases = [
        "My name is John Doe and my email is john@example.com",
        "Call me at 555-123-4567 for more information",
        "My SSN is 123-45-6789",
        "I hate you so much",  # Should not be flagged as PII
        "The API key is sk-1234567890abcdef1234567890abcdef",
        "This is just a normal sentence without any PII"
    ]
    
    print("🔍 Testing Advanced PII Detection")
    print("=" * 50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {text}")
        result = await detector.detect_pii(text)
        
        print(f"  PII Detected: {result.has_pii}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Method: {result.method_used}")
        print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
        
        if result.detected_entities:
            print(f"  Entities Found:")
            for entity in result.detected_entities:
                print(f"    - {entity['type']}: '{entity['text']}' (conf: {entity['confidence']:.2f})")
    
    print(f"\n📊 Detector Stats:")
    stats = detector.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
