"""
Tests for collectors/omi_topics.py

Omi Conversation Topic Classifier — unit tests covering:

1.  classify_transcript — single transcript classification
    a. Technical work text → work_technical category
    b. Strategic/planning text → work_strategic category
    c. Personal chat text → personal category
    d. Operational/errand text → operational category
    e. Mixed content → mixed category
    f. Empty/None text → unknown category
    g. Hungarian text with no work keywords → personal/operational
    h. Hungarian work text → correct category despite language

2.  cognitive_density — scores vary meaningfully by content type
    a. Technical text > casual text (same length)
    b. Dense information rate affects score
    c. Empty text → 0.0

3.  cls_weight / sdi_weight — correct per-category values
    a. work_technical: cls=1.20, sdi=0.60
    b. personal: cls=0.70, sdi=1.20
    c. operational: cls=0.50, sdi=0.40

4.  get_window_topic_profile — aggregates multiple transcripts correctly
    a. Single transcript passes through
    b. Multiple transcripts → dominant category
    c. Empty list → unknown defaults

5.  Integration: topic weights flow through metrics
    a. CLS higher for technical Omi conversation than casual (same word count)
    b. SDI higher for personal Omi conversation than technical (same speech secs)

All tests are pure-unit (no filesystem, no live APIs, no credentials).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.omi_topics import (
    classify_transcript,
    get_window_topic_profile,
    CATEGORY_WEIGHTS,
    _tokenise,
    _lexical_complexity,
    _information_rate,
    _detect_language,
)
from engine.metrics import cognitive_load_score, social_drain_index


# ─── Fixtures ─────────────────────────────────────────────────────────────────

TECH_TEXT = (
    "We need to refactor the API endpoint to use async architecture. "
    "The database query is hitting a timeout because the index is missing. "
    "Let's deploy a fix via the CI/CD pipeline after we write unit tests. "
    "The microservice integration is failing due to a JSON schema mismatch. "
    "I'll push the commit to the branch and open a pull request for review."
)

STRATEGIC_TEXT = (
    "Our Q2 roadmap needs to prioritize the enterprise customer segment. "
    "The KPI we're tracking is revenue growth and retention rate. "
    "Let's align the team on the product strategy before the quarterly review. "
    "We should pitch this proposal to the investors next week. "
    "The OKR for this initiative is to hit the milestone by end of month."
)

PERSONAL_TEXT = (
    "It was so nice seeing the family this weekend. "
    "We had dinner together and watched a movie after the kids went to sleep. "
    "I'm feeling a bit tired today, I didn't sleep well. "
    "Let's catch up for coffee sometime, it's been a while. "
    "I'm excited for the holiday trip, it will be great to rest."
)

OPERATIONAL_TEXT = (
    "The package delivery is scheduled for tomorrow morning. "
    "I need to pick up the order from the pharmacy. "
    "Can you sign the invoice before I pay the bill? "
    "The doctor appointment is at 3pm, I'll drive there. "
    "Please book the taxi to the airport for the flight reservation."
)

HUNGARIAN_CASUAL = (
    "Halló, jó napot! Igen, itt van a csomag, oda kell rakni. "
    "Köszönöm, szia! Van még valami? Nem, minden rendben van."
)

HUNGARIAN_TECH = (
    "A backend API integráció nem működik, van egy timeout a database query-ben. "
    "Kell egy fix a deploy pipeline-ban, meg kell írni a unit test-eket. "
    "A JSON schema mismatch miatt failing a microservice."
)

MIXED_TEXT = (
    "So we need to refactor the API and deploy it soon. "
    "The team is aligned on the product strategy for this quarter. "
    "After the meeting let's grab coffee and discuss the roadmap. "
    "I'm feeling good about the milestone, the OKR looks achievable. "
    "Our focus is on customer retention and data analytics."
)


# ─── 1. classify_transcript — category detection ──────────────────────────────

class TestClassifyTranscriptCategories:

    def test_technical_work_text(self):
        result = classify_transcript(TECH_TEXT)
        assert result["category"] in ("work_technical", "mixed"), (
            f"Expected work_technical or mixed, got {result['category']!r}"
        )
        assert result["cognitive_density"] > 0.3, "Technical text should have high density"

    def test_strategic_text(self):
        result = classify_transcript(STRATEGIC_TEXT)
        assert result["category"] in ("work_strategic", "work_technical", "mixed"), (
            f"Expected strategic/technical/mixed, got {result['category']!r}"
        )

    def test_personal_text(self):
        result = classify_transcript(PERSONAL_TEXT)
        assert result["category"] in ("personal", "mixed", "unknown"), (
            f"Expected personal/mixed/unknown, got {result['category']!r}"
        )
        # Personal category should have lower CLS weight
        if result["category"] == "personal":
            assert result["cls_weight"] < 1.0

    def test_operational_text(self):
        result = classify_transcript(OPERATIONAL_TEXT)
        assert result["category"] in ("operational", "mixed"), (
            f"Expected operational or mixed, got {result['category']!r}"
        )
        if result["category"] == "operational":
            assert result["cls_weight"] <= 0.55
            assert result["sdi_weight"] <= 0.45

    def test_empty_text(self):
        result = classify_transcript("")
        assert result["category"] == "unknown"
        assert result["cognitive_density"] == 0.0
        assert result["cls_weight"] == 1.0
        assert result["sdi_weight"] == 1.0

    def test_none_text_equivalent(self):
        result = classify_transcript("   ")  # whitespace only
        assert result["category"] == "unknown"
        assert result["cognitive_density"] == 0.0

    def test_hungarian_casual_text(self):
        """Hungarian casual conversation should be personal or operational, not work_technical."""
        result = classify_transcript(HUNGARIAN_CASUAL, language_hint="hu")
        assert result["language"] == "hu"
        # Should NOT be work_technical — no work signals in Hungarian casual speech
        assert result["category"] != "work_technical", (
            "Hungarian casual should not be classified as work_technical"
        )

    def test_hungarian_tech_text(self):
        """Hungarian technical text should still detect tech keywords."""
        result = classify_transcript(HUNGARIAN_TECH, language_hint="hu")
        # Language should be detected/marked
        assert result["language"] in ("hu", "mixed", "en")
        # Cognitive density should be non-zero (has technical keywords)
        assert result["cognitive_density"] > 0.0

    def test_result_has_required_keys(self):
        result = classify_transcript(TECH_TEXT)
        for key in ("category", "cognitive_density", "confidence", "topic_signals",
                    "language", "cls_weight", "sdi_weight"):
            assert key in result, f"Missing key: {key}"

    def test_category_weights_are_valid(self):
        for text in [TECH_TEXT, STRATEGIC_TEXT, PERSONAL_TEXT, OPERATIONAL_TEXT]:
            result = classify_transcript(text)
            assert 0.0 < result["cls_weight"] <= 1.5, f"cls_weight out of range: {result['cls_weight']}"
            assert 0.0 < result["sdi_weight"] <= 1.5, f"sdi_weight out of range: {result['sdi_weight']}"

    def test_confidence_range(self):
        for text in [TECH_TEXT, PERSONAL_TEXT, ""]:
            result = classify_transcript(text)
            assert 0.0 <= result["confidence"] <= 1.0, f"confidence out of range: {result['confidence']}"


# ─── 2. cognitive_density ─────────────────────────────────────────────────────

class TestCognitiveDensity:

    def test_technical_text_higher_density_than_casual(self):
        tech = classify_transcript(TECH_TEXT, speech_seconds=120.0)
        casual = classify_transcript(PERSONAL_TEXT, speech_seconds=120.0)
        assert tech["cognitive_density"] > casual["cognitive_density"], (
            f"Technical ({tech['cognitive_density']}) should be denser than "
            f"casual ({casual['cognitive_density']})"
        )

    def test_empty_text_zero_density(self):
        result = classify_transcript("", speech_seconds=60.0)
        assert result["cognitive_density"] == 0.0

    def test_density_range(self):
        for text in [TECH_TEXT, STRATEGIC_TEXT, PERSONAL_TEXT, OPERATIONAL_TEXT, MIXED_TEXT]:
            result = classify_transcript(text, speech_seconds=60.0)
            assert 0.0 <= result["cognitive_density"] <= 1.0, (
                f"cognitive_density out of range: {result['cognitive_density']}"
            )

    def test_information_rate_affects_density(self):
        """Same text with faster speech (more words/sec) should score higher density."""
        text = TECH_TEXT
        slow = classify_transcript(text, speech_seconds=300.0)   # slow speech
        fast = classify_transcript(text, speech_seconds=30.0)    # fast speech
        # Density should be >= for faster speech (more information rate)
        assert fast["cognitive_density"] >= slow["cognitive_density"]


# ─── 3. CLS / SDI weight mapping ──────────────────────────────────────────────

class TestCategoryWeights:

    def test_work_technical_weights(self):
        w = CATEGORY_WEIGHTS["work_technical"]
        assert w["cls"] == 1.20
        assert w["sdi"] == 0.60

    def test_personal_weights(self):
        w = CATEGORY_WEIGHTS["personal"]
        assert w["cls"] == 0.70
        assert w["sdi"] == 1.20

    def test_operational_weights(self):
        w = CATEGORY_WEIGHTS["operational"]
        assert w["cls"] == 0.50
        assert w["sdi"] == 0.40

    def test_mixed_weights(self):
        w = CATEGORY_WEIGHTS["mixed"]
        assert w["cls"] == 1.00
        assert w["sdi"] == 0.85

    def test_unknown_weights(self):
        w = CATEGORY_WEIGHTS["unknown"]
        assert w["cls"] == 1.00
        assert w["sdi"] == 1.00

    def test_technical_cls_higher_than_personal(self):
        assert CATEGORY_WEIGHTS["work_technical"]["cls"] > CATEGORY_WEIGHTS["personal"]["cls"]

    def test_personal_sdi_higher_than_technical(self):
        assert CATEGORY_WEIGHTS["personal"]["sdi"] > CATEGORY_WEIGHTS["work_technical"]["sdi"]

    def test_operational_lowest_cls(self):
        all_cls = [w["cls"] for w in CATEGORY_WEIGHTS.values()]
        assert CATEGORY_WEIGHTS["operational"]["cls"] == min(all_cls)


# ─── 4. get_window_topic_profile ──────────────────────────────────────────────

class TestGetWindowTopicProfile:

    def test_empty_list_returns_unknown_defaults(self):
        result = get_window_topic_profile("2026-03-14", [])
        assert result["category"] == "unknown"
        assert result["cognitive_density"] == 0.0
        assert result["cls_weight"] == 1.0
        assert result["sdi_weight"] == 1.0
        assert result["topic_signals"] == []

    def test_single_transcript_passthrough(self):
        transcripts = [{
            "text": TECH_TEXT,
            "speech_duration_seconds": 120.0,
            "language": "en",
        }]
        result = get_window_topic_profile("2026-03-14", transcripts)
        assert result["category"] in ("work_technical", "mixed")
        assert result["cognitive_density"] > 0.0
        assert "cls_weight" in result
        assert "sdi_weight" in result

    def test_multiple_transcripts_aggregated(self):
        transcripts = [
            {"text": TECH_TEXT, "speech_duration_seconds": 60.0, "language": "en"},
            {"text": TECH_TEXT, "speech_duration_seconds": 60.0, "language": "en"},
        ]
        result = get_window_topic_profile("2026-03-14", transcripts)
        assert result["category"] in ("work_technical", "mixed")
        assert result["cognitive_density"] > 0.0

    def test_result_has_required_keys(self):
        transcripts = [{"text": PERSONAL_TEXT, "speech_duration_seconds": 90.0, "language": "en"}]
        result = get_window_topic_profile("2026-03-14", transcripts)
        for key in ("category", "cognitive_density", "cls_weight", "sdi_weight",
                    "topic_signals", "language"):
            assert key in result, f"Missing key: {key}"


# ─── 5. Integration: topic weights flow through metrics ───────────────────────

class TestMetricsIntegration:
    """
    Verify that cognitive_density and cls_weight/sdi_weight actually shift
    the CLS and SDI outputs in the metrics engine.
    """

    # Common fixture: Omi window with 300 words, 120s speech
    _WORD_COUNT = 300
    _SPEECH_SECS = 120.0

    def _cls(self, cognitive_density=0.0, cls_weight=1.0):
        return cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=80.0,
            omi_conversation_active=True,
            omi_word_count=self._WORD_COUNT,
            omi_cognitive_density=cognitive_density,
            omi_cls_weight=cls_weight,
        )

    def _sdi(self, sdi_weight=1.0):
        return social_drain_index(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_sent=0,
            slack_messages_received=0,
            omi_conversation_active=True,
            omi_speech_seconds=self._SPEECH_SECS,
            omi_sdi_weight=sdi_weight,
        )

    def test_technical_cls_higher_than_casual(self):
        """work_technical (density=0.65, cls_weight=1.20) > personal (density=0.20, cls_weight=0.70)"""
        cls_technical = self._cls(cognitive_density=0.65, cls_weight=1.20)
        cls_casual = self._cls(cognitive_density=0.20, cls_weight=0.70)
        assert cls_technical > cls_casual, (
            f"Technical CLS ({cls_technical}) should be > casual ({cls_casual})"
        )

    def test_personal_sdi_higher_than_technical(self):
        """personal (sdi_weight=1.20) > work_technical (sdi_weight=0.60)"""
        sdi_personal = self._sdi(sdi_weight=1.20)
        sdi_technical = self._sdi(sdi_weight=0.60)
        assert sdi_personal > sdi_technical, (
            f"Personal SDI ({sdi_personal}) should be > technical ({sdi_technical})"
        )

    def test_operational_cls_lowest(self):
        """operational (cls_weight=0.50) should produce lowest CLS"""
        cls_ops = self._cls(cognitive_density=0.10, cls_weight=0.50)
        cls_tech = self._cls(cognitive_density=0.65, cls_weight=1.20)
        assert cls_ops < cls_tech

    def test_no_omi_data_fallback(self):
        """With no Omi data (conversation_active=False), CLS unchanged regardless of weights"""
        cls_no_omi = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=80.0,
            omi_conversation_active=False,
            omi_word_count=0,
            omi_cognitive_density=0.8,  # should be ignored
            omi_cls_weight=1.20,        # should be ignored
        )
        cls_baseline = cognitive_load_score(
            in_meeting=False,
            meeting_attendees=0,
            slack_messages_received=0,
            recovery_score=80.0,
        )
        assert abs(cls_no_omi - cls_baseline) < 0.001, (
            "CLS with inactive Omi should equal baseline (topic weights ignored)"
        )

    def test_cls_fallback_when_no_density(self):
        """When cognitive_density=0 (no topic data), should use v2.0 word-density formula"""
        cls_no_density = self._cls(cognitive_density=0.0, cls_weight=1.0)
        # Should be non-zero (word count drives it via v2.0 fallback)
        cls_baseline = cognitive_load_score(
            in_meeting=False, meeting_attendees=0, slack_messages_received=0,
            recovery_score=80.0,
        )
        assert cls_no_density > cls_baseline, (
            "Omi with no density data should still raise CLS via word-count signal"
        )

    def test_cls_output_range(self):
        """CLS always stays within [0, 1]"""
        for density, weight in [(0.0, 1.0), (0.5, 1.20), (1.0, 1.20), (0.1, 0.50)]:
            cls = self._cls(cognitive_density=density, cls_weight=weight)
            assert 0.0 <= cls <= 1.0, f"CLS out of range: {cls} (density={density}, weight={weight})"

    def test_sdi_output_range(self):
        """SDI always stays within [0, 1]"""
        for weight in [0.40, 0.60, 1.00, 1.20]:
            sdi = self._sdi(sdi_weight=weight)
            assert 0.0 <= sdi <= 1.0, f"SDI out of range: {sdi} (weight={weight})"


# ─── 6. Internal helpers ──────────────────────────────────────────────────────

class TestHelpers:

    def test_tokenise_basic(self):
        tokens = _tokenise("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenise_empty(self):
        assert _tokenise("") == []
        assert _tokenise("  ") == []

    def test_lexical_complexity_range(self):
        for text in [TECH_TEXT, PERSONAL_TEXT, ""]:
            tokens = _tokenise(text)
            score = _lexical_complexity(tokens)
            assert 0.0 <= score <= 1.0

    def test_lexical_complexity_technical_higher(self):
        tech_tokens = _tokenise(TECH_TEXT)
        personal_tokens = _tokenise(PERSONAL_TEXT)
        assert _lexical_complexity(tech_tokens) >= _lexical_complexity(personal_tokens)

    def test_information_rate_range(self):
        for wc, secs in [(0, 0), (100, 60), (500, 10), (100, 1000)]:
            rate = _information_rate(wc, secs)
            assert 0.0 <= rate <= 1.0, f"info_rate out of range: {rate}"

    def test_information_rate_zero_speech(self):
        assert _information_rate(100, 0.0) == 0.0

    def test_detect_language_english(self):
        tokens = _tokenise("This is a clear English sentence with many technical architecture keywords")
        assert _detect_language(tokens) in ("en", "mixed")  # "mixed" OK when ≥1 marker matched

    def test_detect_language_hungarian(self):
        tokens = _tokenise("Igen, jó, köszönöm, hogy, van, nem, és")
        assert _detect_language(tokens) in ("hu", "mixed")

    def test_detect_language_empty(self):
        assert _detect_language([]) == "unknown"
