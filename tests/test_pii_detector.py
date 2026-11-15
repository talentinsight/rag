"""PII detector tests. All PII is synthetic."""
import pytest

from src.advanced_pii_detector import PIIDetector


@pytest.fixture(scope="module")
def detector():
    return PIIDetector()


def masked(detector, text):
    return detector.mask(text, detector.detect(text))


@pytest.mark.parametrize("text,leaked,label", [
    ("My email is jane.doe@acme-mail.net", "jane.doe@acme-mail.net", "[EMAIL_MASKED]"),
    ("Contact test@example.com please", "test@example.com", "[EMAIL_MASKED]"),
    ("My SSN is 123-45-6789", "123-45-6789", "[SSN_MASKED]"),
    ("Call me at 415-555-0132", "415-555-0132", "[PHONE_MASKED]"),
    ("Call (415) 555-0132 now", "(415) 555-0132", "[PHONE_MASKED]"),
    ("My card is 4111 1111 1111 1111", "4111 1111 1111 1111", "[CREDIT_CARD_MASKED]"),
    ("key sk-abcdefghijklmnopqrstuvwxyz123456", "sk-abcdefghijklmnopqrstuvwxyz123456", "[API_KEY_MASKED]"),
    ("Server 192.168.1.20 failed", "192.168.1.20", "[IP_ADDRESS_MASKED]"),
    ("My name is Jane Doe, explain attention", "Jane Doe", "[NAME_MASKED]"),
])
def test_pii_is_detected_and_masked(detector, text, leaked, label):
    assert detector.detect(text)
    out = masked(detector, text)
    assert leaked not in out and label in out


@pytest.mark.parametrize("text", [
    "What is the attention mechanism?",
    "What did Vaswani et al. at Google Brain propose?",  # public authors/institutions are not PII
    "I am interested in attention heads",
    "The base model has 65 million parameters, 6 layers and d_model=512",
    "Section 3.2.1 reports 28.4 BLEU on WMT 2014 (arXiv:1706.03762)",
])
def test_paper_questions_are_not_pii(detector, text):
    assert detector.detect(text) == []
    assert masked(detector, text) == text


def test_multiple_entities_are_all_masked(detector):
    out = masked(detector, "SSN 123-45-6789, mail jane.doe@acme-mail.net")
    assert out == "SSN [SSN_MASKED], mail [EMAIL_MASKED]"
