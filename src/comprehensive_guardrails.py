"""
Comprehensive Guardrails System for RAG API
Implements all major guardrail categories with professional-grade protection
"""

import re
import json
import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import advanced PII detector
try:
    from advanced_pii_detector import AdvancedPIIDetector, PIIDetectionMethod
    ADVANCED_PII_AVAILABLE = True
except ImportError as e:
    ADVANCED_PII_AVAILABLE = False
    logger.warning(f"Advanced PII detector not available: {e}")

@dataclass
class GuardrailResult:
    """Result of a guardrail check"""
    passed: bool
    score: float
    reason: str
    category: str
    severity: str = "medium"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ComprehensiveGuardrails:
    """
    Comprehensive guardrails system implementing all major protection categories
    """
    
    def __init__(self):
        self.request_counts = defaultdict(list)
        self.response_times = defaultdict(list)
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "categories": defaultdict(int),
            "severity_counts": defaultdict(int),
            "average_latency_ms": 0.0
        }
        
        # Load patterns and configurations
        self._load_patterns()
        
        # Initialize advanced PII detector if available
        self.advanced_pii_detector = None
        if ADVANCED_PII_AVAILABLE:
            try:
                self.advanced_pii_detector = AdvancedPIIDetector(
                    preferred_method=PIIDetectionMethod.HYBRID,
                    confidence_threshold=0.7
                )
                logger.info("✅ Advanced PII Detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Advanced PII Detector: {e}")
                self.advanced_pii_detector = None
    
    def _load_patterns(self):
        """Load all detection patterns"""
        
        # PII Patterns - Improved with reduced false positives
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            # More specific credit card pattern with Luhn algorithm context
            "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            # More specific API key pattern - common prefixes and formats
            "api_key": r"\b(?:sk-|pk_|api[_-]?key[_-]?|token[_-]?)[A-Za-z0-9]{20,}\b",
            # Enhanced name pattern with more contexts
            "name": r"(?i)\b(?:my name is|i am|i'm|call me|name:|full name)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        }
        
        # Adult Content Patterns
        self.adult_content_patterns = [
            r"\b(sex|sexual|explicit|porn|nude|naked)\b",
            r"\b(orgasm|masturbat|intercourse)\b",
            r"\b(genitals?|penis|vagina|breast)\b",
            r"\b(erotic|seductive|intimate)\b"
        ]
        
        # Profanity Patterns (comprehensive)
        self.profanity_patterns = [
            r"\b(fuck|shit|damn|bitch|ass|crap)\b",
            r"\b(bastard|asshole|dickhead|motherfucker)\b",
            r"\b(whore|slut|cunt|pussy)\b",
            r"\b(piss|tits|cock|dick)\b"
        ]
        
        # Self-Harm Patterns
        self.self_harm_patterns = [
            r"\b(kill myself|suicide|end my life|hurt myself)\b",
            r"\b(cut myself|self harm|self-harm|overdose)\b",
            r"\b(want to die|don't want to live|no point in living)\b",
            r"\b(razor|blade|pills to die|jump off)\b"
        ]
        
        # Bias Detection Keywords
        self.bias_keywords = {
            "gender": ["he should", "she should", "men are", "women are", "typical woman", "typical man"],
            "racial": ["they people", "those people", "typical of them", "their kind"],
            "age": ["too old", "too young", "millennials are", "boomers are"],
            "religious": ["those believers", "typical religious", "atheists are"]
        }
        
        # Prompt Injection Patterns
        self.injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"forget\s+(all\s+)?previous\s+instructions?",
            r"show\s+me\s+your\s+system\s+prompt",
            r"act\s+as\s+if\s+you\s+are",
            r"pretend\s+you\s+are",
            r"system\s*:",
            r"admin\s*:",
            r"override\s+safety",
            r"disable\s+guardrails?"
        ]
        
        # Data Leakage Patterns
        self.data_leakage_patterns = [
            r"show\s+me\s+(all|the)\s+(data|database|files)",
            r"dump\s+(database|data|table)",
            r"select\s+\*\s+from",
            r"export\s+(all|data|users)",
            r"backup\s+(file|data)",
            r"configuration\s+(file|data)",
            r"environment\s+variables?"
        ]

    def check_pii_detection(self, text: str) -> GuardrailResult:
        """Enhanced PII detection with ML/AI fallback"""
        # Try advanced detector first if available
        if self.advanced_pii_detector:
            try:
                # Run async detection in sync context
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                advanced_result = loop.run_until_complete(
                    self.advanced_pii_detector.detect_pii(text)
                )
                
                return GuardrailResult(
                    passed=not advanced_result.has_pii,
                    score=advanced_result.confidence,
                    reason=f"Advanced PII detection: {'PII found' if advanced_result.has_pii else 'No PII detected'} "
                           f"(method: {advanced_result.method_used}, "
                           f"entities: {len(advanced_result.detected_entities)})",
                    category="pii_detection",
                    severity="high" if advanced_result.has_pii else "low",
                    metadata={
                        "method": advanced_result.method_used,
                        "confidence": advanced_result.confidence,
                        "entities": advanced_result.detected_entities,
                        "processing_time_ms": advanced_result.processing_time_ms,
                        "advanced_detector": True
                    }
                )
            except Exception as e:
                logger.warning(f"Advanced PII detection failed, falling back to regex: {e}")
        
        # Fallback to original regex-based detection
        return self._check_pii_detection_regex(text)
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text while preserving context for RAG processing
        
        Args:
            text: Input text that may contain PII
            
        Returns:
            str: Text with PII masked using placeholders
        """
        masked_text = text
        
        # Use advanced PII detector if available
        if self.advanced_pii_detector:
            try:
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                advanced_result = loop.run_until_complete(
                    self.advanced_pii_detector.detect_pii(text)
                )
                
                if advanced_result.passed:  # No PII detected
                    return text
                
                # Mask detected PII entities
                entities = advanced_result.metadata.get("entities", [])
                for entity in entities:
                    entity_text = entity.get("text", "")
                    entity_type = entity.get("type", "PII")
                    if entity_text:
                        masked_text = masked_text.replace(entity_text, f"[{entity_type}]")
                
                return masked_text
                
            except Exception as e:
                logger.warning(f"Advanced PII masking failed, falling back to regex: {e}")
        
        # Fallback to regex-based masking
        return self._mask_pii_regex(text)
    
    def _mask_pii_regex(self, text: str) -> str:
        """
        Regex-based PII masking fallback
        
        Args:
            text: Input text
            
        Returns:
            str: Text with PII masked
        """
        masked_text = text
        
        # Mask different PII types
        masked_text = re.sub(self.pii_patterns["email"], "[EMAIL]", masked_text)
        masked_text = re.sub(self.pii_patterns["phone"], "[PHONE]", masked_text)
        masked_text = re.sub(self.pii_patterns["credit_card"], "[CREDIT_CARD]", masked_text)
        masked_text = re.sub(self.pii_patterns["ssn"], "[SSN]", masked_text)
        masked_text = re.sub(self.pii_patterns["api_key"], "[API_KEY]", masked_text)
        
        # Mask names (more carefully to avoid false positives)
        name_match = re.search(self.pii_patterns["name"], masked_text)
        if name_match:
            full_name = name_match.group(1) if name_match.groups() else name_match.group(0)
            masked_text = masked_text.replace(full_name, "[PERSON]")
        
        return masked_text
    
    def _check_pii_detection_regex(self, text: str) -> GuardrailResult:
        """Original regex-based PII detection as fallback"""
        pii_found = []
        detected_types = []
        
        # Common false positive patterns to exclude
        false_positive_patterns = [
            r"\b(test|example|sample|demo)\b",  # Test data indicators
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b.*\b(test|example|sample)\b",  # Test credit cards
            r"\b[A-Za-z0-9]{32,}\b.*\b(hash|id|uuid|guid)\b"  # Technical identifiers
        ]
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Filter out false positives for specific types
                filtered_matches = []
                for match in matches:
                    match_str = str(match) if isinstance(match, tuple) else match
                    
                    # Check if this match is likely a false positive
                    is_false_positive = False
                    for fp_pattern in false_positive_patterns:
                        if re.search(fp_pattern, text, re.IGNORECASE):
                            # Additional context check around the match
                            match_context = text.lower()
                            if any(indicator in match_context for indicator in ['test', 'example', 'sample', 'demo', 'mock']):
                                is_false_positive = True
                                break
                    
                    # Special handling for API keys - require common prefixes
                    if pii_type == "api_key" and not re.search(r"\b(?:sk-|pk_|api|key|token)", text, re.IGNORECASE):
                        is_false_positive = True
                    
                    # Special handling for credit cards - basic Luhn check context
                    if pii_type == "credit_card":
                        # If it's in a context that suggests it's not a real CC
                        if re.search(r"\b(?:hate|angry|mad|upset)\b", text, re.IGNORECASE):
                            is_false_positive = True
                    
                    if not is_false_positive:
                        filtered_matches.append(match)
                
                if filtered_matches:
                    pii_found.extend(filtered_matches)
                    detected_types.append(pii_type)
        
        if pii_found:
            return GuardrailResult(
                passed=False,
                score=1.0,
                reason=f"PII detected (regex): {len(pii_found)} instances of {', '.join(detected_types)}",
                category="pii_detection",
                severity="high",
                metadata={
                    "detected_types": detected_types, 
                    "count": len(pii_found),
                    "method": "regex_fallback",
                    "advanced_detector": False
                }
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No PII detected (regex)",
            category="pii_detection",
            severity="low",
            metadata={"method": "regex_fallback", "advanced_detector": False}
        )
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text and return the masked version
        
        Args:
            text (str): Input text to mask
            
        Returns:
            str: Text with PII masked
        """
        masked_text = text
        
        # Define masking patterns for each PII type
        pii_masks = {
            "email": "[EMAIL_MASKED]",
            "phone": "[PHONE_MASKED]", 
            "credit_card": "[CREDIT_CARD_MASKED]",
            "ssn": "[SSN_MASKED]",
            "api_key": "[API_KEY_MASKED]",
            "name": "[NAME_MASKED]"
        }
        
        # Apply masking for each PII type
        for pii_type, pattern in self.pii_patterns.items():
            mask = pii_masks.get(pii_type, "[PII_MASKED]")
            if pii_type == "name":
                # Special handling for name pattern (replace only the captured name part)
                masked_text = re.sub(pattern, lambda m: m.group(0).replace(m.group(1), mask), masked_text, flags=re.IGNORECASE)
            else:
                masked_text = re.sub(pattern, mask, masked_text, flags=re.IGNORECASE)
        
        return masked_text

    def check_adult_content(self, text: str) -> GuardrailResult:
        """Adult content detection"""
        matches = []
        for pattern in self.adult_content_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            severity = "high" if len(matches) > 2 else "medium"
            return GuardrailResult(
                passed=False,
                score=min(len(matches) * 0.3, 1.0),
                reason=f"Adult content detected: {len(matches)} indicators",
                category="adult_content",
                severity=severity,
                metadata={"indicators": len(matches)}
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No adult content detected",
            category="adult_content",
            severity="low"
        )

    def check_profanity_filter(self, text: str) -> GuardrailResult:
        """Advanced profanity detection"""
        matches = []
        for pattern in self.profanity_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            matches.extend(found)
        
        if matches:
            severity = "high" if len(matches) > 3 else "medium"
            return GuardrailResult(
                passed=False,
                score=min(len(matches) * 0.25, 1.0),
                reason=f"Profanity detected: {len(matches)} instances",
                category="profanity_filter",
                severity=severity,
                metadata={"count": len(matches)}
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No profanity detected",
            category="profanity_filter",
            severity="low"
        )

    def check_self_harm_detection(self, text: str) -> GuardrailResult:
        """Self-harm and suicide prevention"""
        matches = []
        for pattern in self.self_harm_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            return GuardrailResult(
                passed=False,
                score=1.0,
                reason=f"Self-harm indicators detected: {len(matches)} patterns",
                category="self_harm_detection",
                severity="critical",
                metadata={
                    "crisis_intervention_needed": True,
                    "indicators": len(matches),
                    "support_message": "If you're experiencing thoughts of self-harm, please reach out to a mental health professional or crisis helpline."
                }
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No self-harm indicators detected",
            category="self_harm_detection",
            severity="low"
        )

    def check_bias_detection(self, text: str) -> GuardrailResult:
        """Bias and fairness detection"""
        bias_found = []
        bias_types = []
        
        for bias_type, keywords in self.bias_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    bias_found.append(keyword)
                    if bias_type not in bias_types:
                        bias_types.append(bias_type)
        
        if bias_found:
            return GuardrailResult(
                passed=False,
                score=min(len(bias_found) * 0.3, 1.0),
                reason=f"Potential bias detected: {', '.join(bias_types)}",
                category="bias_detection",
                severity="medium",
                metadata={"bias_types": bias_types, "indicators": bias_found}
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No bias indicators detected",
            category="bias_detection",
            severity="low"
        )

    def check_data_leakage_prevention(self, text: str) -> GuardrailResult:
        """Data leakage prevention"""
        matches = []
        for pattern in self.data_leakage_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            return GuardrailResult(
                passed=False,
                score=1.0,
                reason=f"Data leakage attempt detected: {len(matches)} patterns",
                category="data_leakage_prevention",
                severity="high",
                metadata={"patterns": len(matches)}
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No data leakage attempts detected",
            category="data_leakage_prevention",
            severity="low"
        )

    def check_input_sanitation(self, text: str) -> GuardrailResult:
        """Input sanitation and prompt injection detection"""
        matches = []
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            return GuardrailResult(
                passed=False,
                score=0.9,
                reason=f"Prompt injection detected: {len(matches)} patterns",
                category="input_sanitation",
                severity="high",
                metadata={"injection_patterns": len(matches)}
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="No prompt injection detected",
            category="input_sanitation",
            severity="low"
        )

    def check_rate_limits(self, client_id: str, max_requests: int = 10, window_minutes: int = 1) -> GuardrailResult:
        """Rate limiting and cost control"""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id] 
            if req_time > window_start
        ]
        
        current_count = len(self.request_counts[client_id])
        
        if current_count >= max_requests:
            return GuardrailResult(
                passed=False,
                score=1.0,
                reason=f"Rate limit exceeded: {current_count}/{max_requests} requests",
                category="rate_limits",
                severity="medium",
                metadata={"current_count": current_count, "limit": max_requests}
            )
        
        self.request_counts[client_id].append(now)
        return GuardrailResult(
            passed=True,
            score=current_count / max_requests,
            reason=f"Rate limit OK: {current_count + 1}/{max_requests}",
            category="rate_limits",
            severity="low"
        )

    def check_latency_performance(self, start_time: datetime, max_latency_ms: int = 5000) -> GuardrailResult:
        """Latency and performance monitoring"""
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        if processing_time_ms > max_latency_ms:
            return GuardrailResult(
                passed=False,
                score=processing_time_ms / max_latency_ms,
                reason=f"Latency exceeded: {processing_time_ms:.0f}ms > {max_latency_ms}ms",
                category="latency_performance",
                severity="medium",
                metadata={"processing_time_ms": processing_time_ms, "limit_ms": max_latency_ms}
            )
        
        return GuardrailResult(
            passed=True,
            score=processing_time_ms / max_latency_ms,
            reason=f"Latency OK: {processing_time_ms:.0f}ms",
            category="latency_performance",
            severity="low",
            metadata={"processing_time_ms": processing_time_ms}
        )

    def check_schema_validation(self, response_data: Dict[str, Any]) -> GuardrailResult:
        """Schema and output structure validation"""
        required_fields = ["answer", "question", "timestamp"]
        missing_fields = []
        
        for field in required_fields:
            if field not in response_data:
                missing_fields.append(field)
        
        if missing_fields:
            return GuardrailResult(
                passed=False,
                score=len(missing_fields) / len(required_fields),
                reason=f"Schema validation failed: missing {', '.join(missing_fields)}",
                category="schema_validation",
                severity="medium",
                metadata={"missing_fields": missing_fields}
            )
        
        return GuardrailResult(
            passed=True,
            score=0.0,
            reason="Schema validation passed",
            category="schema_validation",
            severity="low"
        )

    def check_all_input_guardrails(self, text: str, client_id: str = "default") -> Tuple[bool, List[GuardrailResult]]:
        """Run all input-related guardrail checks"""
        start_time = datetime.now()
        results = []
        
        # Critical checks first
        results.append(self.check_rate_limits(client_id))
        results.append(self.check_self_harm_detection(text))
        results.append(self.check_pii_detection(text))
        results.append(self.check_data_leakage_prevention(text))
        results.append(self.check_input_sanitation(text))
        
        # Content quality checks
        results.append(self.check_adult_content(text))
        results.append(self.check_profanity_filter(text))
        results.append(self.check_bias_detection(text))
        
        # Performance check
        results.append(self.check_latency_performance(start_time, max_latency_ms=1000))
        
        # Update stats
        self._update_stats(results)
        
        # Determine if all passed
        all_passed = all(r.passed for r in results)
        return all_passed, results

    def check_input_guardrails_with_pii_filtering(self, text: str, client_id: str = "default") -> Tuple[bool, List[GuardrailResult]]:
        """Run guardrails with PII filtering only (PRODUCTION VERSION)"""
        start_time = datetime.now()
        results = []
        
        # Critical checks that can block requests
        results.append(self.check_rate_limits(client_id))
        results.append(self.check_pii_detection(text))  # ONLY PII blocks requests
        results.append(self.check_data_leakage_prevention(text))
        results.append(self.check_input_sanitation(text))
        
        # Detection-only checks (don't block, just log)
        adult_result = self.check_adult_content(text)
        adult_result.passed = True  # Force pass - just detect, don't block
        results.append(adult_result)
        
        profanity_result = self.check_profanity_filter(text)
        profanity_result.passed = True  # Force pass - just detect, don't block
        results.append(profanity_result)
        
        bias_result = self.check_bias_detection(text)
        bias_result.passed = True  # Force pass - just detect, don't block
        results.append(bias_result)
        
        # Performance check
        results.append(self.check_latency_performance(start_time, max_latency_ms=1000))
        
        # Update stats
        self._update_stats(results)
        
        # Only PII and critical checks can fail the request
        blocking_results = [r for r in results if r.category in ["rate_limits", "pii_detection", "data_leakage_prevention", "input_sanitation", "latency_performance"]]
        all_passed = all(r.passed for r in blocking_results)
        
        return all_passed, results

    def check_output_guardrails(self, response_data: Dict[str, Any], start_time: datetime) -> Tuple[bool, List[GuardrailResult]]:
        """Run output-related guardrail checks"""
        results = []
        
        # Schema validation
        results.append(self.check_schema_validation(response_data))
        
        # Final latency check
        results.append(self.check_latency_performance(start_time, max_latency_ms=10000))
        
        # Content checks on response if available
        if "answer" in response_data:
            answer_text = response_data["answer"]
            results.append(self.check_adult_content(answer_text))
            results.append(self.check_profanity_filter(answer_text))
            results.append(self.check_pii_detection(answer_text))
        
        # Update stats
        self._update_stats(results)
        
        all_passed = all(r.passed for r in results)
        return all_passed, results

    def _update_stats(self, results: List[GuardrailResult]):
        """Update internal statistics"""
        self.stats["total_requests"] += 1
        
        for result in results:
            self.stats["categories"][result.category] += 1
            self.stats["severity_counts"][result.severity] += 1
            
            if not result.passed:
                self.stats["blocked_requests"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total = self.stats["total_requests"]
        blocked = self.stats["blocked_requests"]
        
        return {
            **self.stats,
            "success_rate": (total - blocked) / total if total > 0 else 1.0,
            "block_rate": blocked / total if total > 0 else 0.0,
            "categories_summary": dict(self.stats["categories"]),
            "severity_summary": dict(self.stats["severity_counts"])
        }

    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "categories": defaultdict(int),
            "severity_counts": defaultdict(int),
            "average_latency_ms": 0.0
        }
        self.request_counts.clear()
        self.response_times.clear()
