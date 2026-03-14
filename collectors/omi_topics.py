"""
Presence Tracker — Omi Conversation Topic Classifier

Classifies Omi transcript text into conversation categories and computes a
cognitive density score.  Zero external dependencies — pure Python keyword
matching over transcript tokens.

## Why this matters

The Omi collector already extracts word_count, speech_seconds, and speech_ratio
from transcripts.  But ALL conversations are treated identically: a dense
technical architecture discussion and a casual "good morning" both register as
`conversation_active=True` and produce the same CLS/SDI delta for the same
word count.

This is wrong.  A 400-word technical briefing on distributed systems is far more
cognitively demanding than a 400-word social exchange about weekend plans.  Topic
classification lets the metrics distinguish:

  - Technical/work discussion → high CLS contribution (active problem-solving)
  - Strategic/planning conversation → high CLS, moderate SDI
  - Personal/social chat → moderate SDI, lower CLS
  - Operational/admin exchange → minimal CLS, minimal SDI
  - Mixed → blended contribution

## Output

classify_transcript(text) → dict:
{
    "category": str,           # "work_technical" | "work_strategic" | "personal" |
                               #  "operational" | "mixed" | "unknown"
    "cognitive_density": float, # 0.0 (light) → 1.0 (dense/demanding)
    "confidence": float,        # 0.0 → 1.0 (how clearly one category dominates)
    "topic_signals": list[str], # top signals that triggered the classification
    "language": str,            # "en" | "hu" | "mixed" | "unknown"
}

## Cognitive Density

cognitive_density is a 0-1 score capturing how cognitively demanding the text is,
independent of category.  It combines:

1. **Lexical complexity**: average word length, proportion of long words (>6 chars)
   — longer words correlate with technical/domain-specific vocabulary
2. **Concept density**: count of domain keywords per 100 words
   — dense keyword usage = active problem-solving domain
3. **Information rate**: words per second of speech (from word_count + speech_secs)
   — fast dense speech = high engagement
4. **Category multiplier**: work_technical and work_strategic get a density boost
   because the same word density is cognitively heavier in a domain context

cognitive_density feeds into the CLS formula as a weight on the Omi contribution:
  omi_component = 0.5 * cognitive_density_boost + 0.5 * word_density
This replaces the flat 0.5 baseline with a content-aware score.

## Category Multipliers for CLS/SDI

category_cls_weight: how much this conversation type contributes to CLS
category_sdi_weight: how much this conversation type contributes to SDI

Category             CLS weight   SDI weight
work_technical       1.20         0.60   (high cognitive, lower social)
work_strategic       1.10         0.90   (high cognitive, high social)
personal             0.70         1.20   (lower cognitive, higher social)
operational          0.50         0.40   (minimal — delivery, admin, errands)
mixed                1.00         0.85   (neutral baseline)
unknown              1.00         1.00   (no info — conservative neutral)

## Language Detection

A simple Hungarian keyword check identifies Hungarian transcripts.
Hungarian conversations at home (personal/operational) should not inflate
David's work-cognitive-load metrics.  Hungarian text with no work keywords
is downgraded to personal/operational by default.

## Usage

    from collectors.omi_topics import classify_transcript, classify_day

    # Single transcript
    result = classify_transcript(text, speech_seconds=120.0)

    # All transcripts for a day (returns aggregated daily topic profile)
    profile = classify_day("2026-03-12")

"""

import re
import sys
from pathlib import Path
from typing import Optional

# ─── Topic keyword banks ──────────────────────────────────────────────────────
# Each bank is a list of lowercase tokens.  Match is case-insensitive substring
# after tokenising the text into words.  Scoring is additive: each matched
# keyword adds 1 to the category score, normalised by total keywords found.

_WORK_TECHNICAL = [
    # Software & systems
    "api", "backend", "frontend", "database", "schema", "endpoint", "webhook",
    "deploy", "deployment", "pipeline", "integration", "microservice", "architecture",
    "code", "coding", "bug", "fix", "refactor", "function", "class", "module",
    "library", "framework", "docker", "kubernetes", "server", "cloud", "aws",
    "github", "git", "commit", "branch", "pull request", "merge", "test",
    "testing", "unit test", "ci/cd", "devops", "infrastructure", "scaling",
    # AI / ML
    "model", "llm", "agent", "embedding", "vector", "inference", "training",
    "dataset", "prompt", "token", "neural", "machine learning", "fine-tuning",
    "workflow", "temporal", "activity", "orchestration", "automation",
    # Data
    "query", "sql", "json", "csv", "parse", "transform", "etl", "analytics",
    "dashboard", "metric", "signal", "data", "store", "cache", "index",
    # General technical
    "algorithm", "logic", "implementation", "debug", "error", "exception",
    "performance", "latency", "throughput", "timeout", "retry", "async",
    "sync", "thread", "process", "memory", "cpu", "runtime",
]

_WORK_STRATEGIC = [
    # Planning & goals
    "strategy", "strategic", "roadmap", "initiative", "priority", "milestone",
    "goal", "objective", "okr", "kpi", "plan", "planning", "quarterly",
    "annual", "sprint", "backlog", "timeline", "deadline", "launch",
    # Business
    "revenue", "growth", "market", "customer", "client", "deal", "contract",
    "proposal", "pitch", "investor", "fundraise", "budget", "cost", "roi",
    "product", "feature", "mvp", "feedback", "user", "retention", "churn",
    # Team & people
    "hire", "hiring", "team", "report", "review", "performance", "feedback",
    "decision", "alignment", "stakeholder", "meeting", "agenda", "action item",
    "follow up", "next steps", "blocker", "risk", "dependency",
    # Communication
    "email", "slack", "message", "update", "status", "report", "summary",
    "presentation", "demo", "proposal",
]

_PERSONAL = [
    # Social
    "family", "friend", "dinner", "lunch", "breakfast", "coffee", "weekend",
    "vacation", "holiday", "trip", "travel", "movie", "book", "music",
    "sport", "gym", "run", "walk", "sleep", "tired", "rest",
    # Feelings & casual
    "feel", "feeling", "happy", "sad", "excited", "worried", "stress",
    "fun", "great", "awesome", "nice", "good", "bad", "okay",
    # Home
    "home", "house", "kids", "child", "baby", "school", "partner", "wife",
    "husband", "mom", "dad", "parent", "birthday", "anniversary",
]

_OPERATIONAL = [
    # Logistics
    "delivery", "package", "order", "pick up", "pickup", "drop off", "dropoff",
    "schedule", "appointment", "doctor", "dentist", "pharmacy", "grocery",
    "shopping", "buy", "purchase", "pay", "invoice", "receipt",
    # Administrative
    "sign", "signature", "form", "document", "contract", "paperwork",
    "admin", "registration", "renewal", "subscription", "cancel",
    # Transport
    "car", "taxi", "uber", "train", "bus", "flight", "airport", "ticket",
    "reservation", "booking",
]

# Hungarian language indicators (very common Hungarian words)
_HUNGARIAN_MARKERS = [
    "hogy", "van", "nem", "igen", "köszönöm", "kérem", "szia", "hello",
    "igen", "ők", "én", "te", "mi", "ti", "az", "ez", "egy", "de",
    "most", "akkor", "amikor", "ahol", "amit", "aki", "ami", "jó",
    "jól", "itt", "ott", "már", "még", "csak", "is", "és", "vagy",
    "mert", "ha", "de", "meg", "fel", "le", "be", "ki", "el",
    "halló", "hát", "persze", "rendben", "oke", "okay", "tessék",
]

# ─── Category weights for CLS and SDI ────────────────────────────────────────

CATEGORY_WEIGHTS = {
    "work_technical": {"cls": 1.20, "sdi": 0.60},
    "work_strategic": {"cls": 1.10, "sdi": 0.90},
    "personal":       {"cls": 0.70, "sdi": 1.20},
    "operational":    {"cls": 0.50, "sdi": 0.40},
    "mixed":          {"cls": 1.00, "sdi": 0.85},
    "unknown":        {"cls": 1.00, "sdi": 1.00},
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens, stripping punctuation."""
    if not text:
        return []
    # Lowercase, strip punctuation except apostrophes and hyphens within words
    cleaned = re.sub(r"[^\w\s'-]", " ", text.lower())
    return [t for t in cleaned.split() if len(t) >= 2]


def _score_bank(tokens: list[str], bank: list[str]) -> tuple[int, list[str]]:
    """
    Score a token list against a keyword bank.

    Returns (score, matched_keywords).
    Score = number of unique keywords from bank found in tokens.
    Multi-word phrases in bank are checked against the original joined text.
    """
    token_set = set(tokens)
    joined = " ".join(tokens)
    matched = []

    for keyword in bank:
        if " " in keyword:
            # Multi-word phrase — check in joined string
            if keyword in joined:
                matched.append(keyword)
        elif keyword in token_set:
            matched.append(keyword)

    return len(matched), matched


def _detect_language(tokens: list[str]) -> str:
    """
    Detect language from tokens.

    Returns "hu" (Hungarian), "en" (English), "mixed", or "unknown".
    Uses a simple marker-word presence check: if ≥3 Hungarian markers are
    found in the first 100 tokens, classify as Hungarian.
    """
    if not tokens:
        return "unknown"

    sample = tokens[:100]
    sample_set = set(sample)
    hu_count = sum(1 for m in _HUNGARIAN_MARKERS if m in sample_set)

    if hu_count >= 3:
        return "hu"
    elif hu_count >= 1:
        return "mixed"
    else:
        return "en"


def _lexical_complexity(tokens: list[str]) -> float:
    """
    Compute lexical complexity from token list.

    Returns 0.0–1.0.  Based on:
    - Mean word length (normalised, longer = more complex)
    - Proportion of long words (>6 chars) — technical vocabulary indicator
    """
    if not tokens:
        return 0.0
    mean_len = sum(len(t) for t in tokens) / len(tokens)
    long_word_ratio = sum(1 for t in tokens if len(t) > 6) / len(tokens)
    # Normalise mean length: 3 chars = 0, 8+ chars = 1
    mean_len_norm = min(1.0, max(0.0, (mean_len - 3) / 5))
    return round(0.5 * mean_len_norm + 0.5 * long_word_ratio, 4)


def _information_rate(word_count: int, speech_seconds: float) -> float:
    """
    Words per second of speech, normalised to 0–1.

    Fast dense speech (≥3 words/sec) scores 1.0.
    Slow/sparse speech (≤0.5 words/sec) scores 0.0.
    """
    if speech_seconds <= 0 or word_count <= 0:
        return 0.0
    wps = word_count / speech_seconds
    # Normalise: 0.5 wps = 0.0, 3.0 wps = 1.0
    return round(min(1.0, max(0.0, (wps - 0.5) / 2.5)), 4)


# ─── Core classifier ──────────────────────────────────────────────────────────

def classify_transcript(
    text: str,
    speech_seconds: float = 0.0,
    language_hint: str = "",
) -> dict:
    """
    Classify a single transcript text into a topic category.

    Args:
        text: Full transcript text string.
        speech_seconds: Speech duration (for information rate calculation).
        language_hint: Language tag from Omi transcript (e.g. "en", "hu").
                       Used to supplement the internal language detection.

    Returns dict:
    {
        "category": str,            # work_technical | work_strategic | personal |
                                    #  operational | mixed | unknown
        "cognitive_density": float,  # 0.0 (light) → 1.0 (dense/demanding)
        "confidence": float,         # 0.0 → 1.0
        "topic_signals": list[str],  # top matched keywords (max 5)
        "language": str,             # en | hu | mixed | unknown
        "cls_weight": float,         # CLS multiplier for this conversation type
        "sdi_weight": float,         # SDI multiplier for this conversation type
    }
    """
    if not text or not text.strip():
        return {
            "category": "unknown",
            "cognitive_density": 0.0,
            "confidence": 0.0,
            "topic_signals": [],
            "language": language_hint or "unknown",
            "cls_weight": 1.0,
            "sdi_weight": 1.0,
        }

    tokens = _tokenise(text)
    word_count = len(tokens)

    # Language detection
    detected_lang = _detect_language(tokens)
    # Merge with hint: if Omi says "hu" and we agree or are unsure, use "hu"
    if language_hint == "hu" or detected_lang == "hu":
        language = "hu"
    elif detected_lang == "mixed":
        language = "mixed"
    else:
        language = "en"

    # Score each category
    tech_score, tech_signals = _score_bank(tokens, _WORK_TECHNICAL)
    strat_score, strat_signals = _score_bank(tokens, _WORK_STRATEGIC)
    pers_score, pers_signals = _score_bank(tokens, _PERSONAL)
    ops_score, ops_signals = _score_bank(tokens, _OPERATIONAL)

    total_score = tech_score + strat_score + pers_score + ops_score

    # Determine category
    scores = {
        "work_technical": tech_score,
        "work_strategic": strat_score,
        "personal": pers_score,
        "operational": ops_score,
    }
    best_cat = max(scores, key=lambda k: scores[k])
    best_score = scores[best_cat]

    # Collect top signals for reporting
    all_signals = {
        "work_technical": tech_signals,
        "work_strategic": strat_signals,
        "personal": pers_signals,
        "operational": ops_signals,
    }
    top_signals = all_signals[best_cat][:5]

    # Hungarian language modifier:
    # If text is primarily Hungarian with no significant work keywords,
    # default to personal/operational.  Work keyword density is much lower
    # in Hungarian casual conversation — avoid inflating work metrics.
    if language == "hu" and best_score <= 2:
        # Hungarian text with minimal work signals → personal/operational
        if ops_score >= pers_score:
            best_cat = "operational"
            top_signals = ops_signals[:5]
        else:
            best_cat = "personal"
            top_signals = pers_signals[:5]
        best_score = scores[best_cat]

    # Determine confidence (how dominant is the top category?)
    if total_score == 0:
        category = "unknown"
        confidence = 0.0
    elif best_score == 0:
        category = "unknown"
        confidence = 0.0
    else:
        # Confidence = top score / total (how much of signal is in this category)
        confidence = round(best_score / total_score, 4) if total_score > 0 else 0.0

        # Mixed: if top two categories have similar scores
        scores_sorted = sorted(scores.values(), reverse=True)
        if len(scores_sorted) >= 2 and scores_sorted[0] > 0:
            second_ratio = scores_sorted[1] / scores_sorted[0]
            if second_ratio >= 0.60:
                # Two categories nearly tied → mixed
                category = "mixed"
                confidence = round(1.0 - second_ratio, 4)
                top_signals = (tech_signals + strat_signals)[:3]
            else:
                category = best_cat
        else:
            category = best_cat

    # ── Cognitive density ──────────────────────────────────────────────────
    # Combines:
    # 1. Lexical complexity (word length profile) — 0.35 weight
    # 2. Keyword density (matched keywords / word_count) — 0.35 weight
    # 3. Information rate (words/sec) — 0.20 weight
    # 4. Category bonus — 0.10 weight

    lexical = _lexical_complexity(tokens)

    kw_density = min(1.0, total_score / max(1, word_count / 20)) if word_count > 0 else 0.0

    info_rate = _information_rate(word_count, speech_seconds)

    category_bonus = {
        "work_technical": 0.80,
        "work_strategic": 0.65,
        "mixed": 0.50,
        "personal": 0.25,
        "operational": 0.15,
        "unknown": 0.30,
    }.get(category, 0.30)

    cognitive_density = round(
        0.35 * lexical +
        0.35 * kw_density +
        0.20 * info_rate +
        0.10 * category_bonus,
        4,
    )

    weights = CATEGORY_WEIGHTS.get(category, CATEGORY_WEIGHTS["unknown"])

    return {
        "category": category,
        "cognitive_density": cognitive_density,
        "confidence": confidence,
        "topic_signals": top_signals,
        "language": language,
        "cls_weight": weights["cls"],
        "sdi_weight": weights["sdi"],
    }


def classify_day(date_str: str) -> dict:
    """
    Classify all Omi transcripts for a day and return an aggregated daily profile.

    Reads from ~/omi/transcripts/YYYY-MM-DD/*.json directly.

    Returns:
    {
        "date": str,
        "transcript_count": int,
        "category_distribution": {category: count},
        "dominant_category": str,
        "mean_cognitive_density": float,
        "mean_cls_weight": float,
        "mean_sdi_weight": float,
        "languages": {language: count},
        "all_signals": list[str],       # combined unique topic signals
        "per_transcript": list[dict],   # classification per transcript
    }

    Returns an empty-ish dict when no transcripts exist for the date.
    """
    from pathlib import Path
    import json

    transcript_dir = Path.home() / "omi" / "transcripts" / date_str
    if not transcript_dir.exists():
        return {
            "date": date_str,
            "transcript_count": 0,
            "category_distribution": {},
            "dominant_category": "unknown",
            "mean_cognitive_density": 0.0,
            "mean_cls_weight": 1.0,
            "mean_sdi_weight": 1.0,
            "languages": {},
            "all_signals": [],
            "per_transcript": [],
        }

    per_transcript = []
    transcript_files = sorted(transcript_dir.glob("*.json"))

    for tf in transcript_files:
        try:
            with tf.open() as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        text = record.get("text", "") or ""
        speech_secs = float(record.get("speech_duration_seconds") or 0.0)
        lang_hint = record.get("language", "")
        ts = record.get("timestamp", "")

        result = classify_transcript(text, speech_seconds=speech_secs, language_hint=lang_hint)
        result["timestamp"] = ts
        result["filename"] = tf.name
        per_transcript.append(result)

    if not per_transcript:
        return {
            "date": date_str,
            "transcript_count": 0,
            "category_distribution": {},
            "dominant_category": "unknown",
            "mean_cognitive_density": 0.0,
            "mean_cls_weight": 1.0,
            "mean_sdi_weight": 1.0,
            "languages": {},
            "all_signals": [],
            "per_transcript": [],
        }

    # Aggregate
    from collections import Counter, defaultdict

    cat_counts: Counter = Counter(r["category"] for r in per_transcript)
    lang_counts: Counter = Counter(r["language"] for r in per_transcript)
    all_signals = list({
        sig
        for r in per_transcript
        for sig in r["topic_signals"]
    })

    mean_density = sum(r["cognitive_density"] for r in per_transcript) / len(per_transcript)
    mean_cls_w = sum(r["cls_weight"] for r in per_transcript) / len(per_transcript)
    mean_sdi_w = sum(r["sdi_weight"] for r in per_transcript) / len(per_transcript)
    dominant = cat_counts.most_common(1)[0][0]

    return {
        "date": date_str,
        "transcript_count": len(per_transcript),
        "category_distribution": dict(cat_counts),
        "dominant_category": dominant,
        "mean_cognitive_density": round(mean_density, 4),
        "mean_cls_weight": round(mean_cls_w, 4),
        "mean_sdi_weight": round(mean_sdi_w, 4),
        "languages": dict(lang_counts),
        "all_signals": sorted(all_signals)[:20],
        "per_transcript": per_transcript,
    }


def get_window_topic_profile(
    date_str: str,
    window_transcripts: list[dict],
) -> dict:
    """
    Compute a topic profile for a single 15-minute window's transcripts.

    Args:
        date_str: "YYYY-MM-DD" (for context)
        window_transcripts: list of raw Omi transcript records for this window
            (each has "text", "speech_duration_seconds", "language")

    Returns a topic profile dict with category, cognitive_density, cls_weight,
    sdi_weight, and topic_signals — suitable for embedding in the window's
    omi signals dict.
    """
    if not window_transcripts:
        return {
            "category": "unknown",
            "cognitive_density": 0.0,
            "cls_weight": 1.0,
            "sdi_weight": 1.0,
            "topic_signals": [],
            "language": "unknown",
        }

    results = []
    for t in window_transcripts:
        text = t.get("text", "") or ""
        speech_secs = float(t.get("speech_duration_seconds") or 0.0)
        lang_hint = t.get("language", "")
        results.append(classify_transcript(text, speech_seconds=speech_secs, language_hint=lang_hint))

    if len(results) == 1:
        r = results[0]
        return {
            "category": r["category"],
            "cognitive_density": r["cognitive_density"],
            "cls_weight": r["cls_weight"],
            "sdi_weight": r["sdi_weight"],
            "topic_signals": r["topic_signals"],
            "language": r["language"],
        }

    # Multiple transcripts in window — aggregate
    from collections import Counter
    cats = Counter(r["category"] for r in results)
    dominant = cats.most_common(1)[0][0]
    mean_density = sum(r["cognitive_density"] for r in results) / len(results)
    mean_cls = sum(r["cls_weight"] for r in results) / len(results)
    mean_sdi = sum(r["sdi_weight"] for r in results) / len(results)
    all_sigs = list({sig for r in results for sig in r["topic_signals"]})[:5]
    langs = Counter(r["language"] for r in results)
    dominant_lang = langs.most_common(1)[0][0]

    return {
        "category": dominant,
        "cognitive_density": round(mean_density, 4),
        "cls_weight": round(mean_cls, 4),
        "sdi_weight": round(mean_sdi, 4),
        "topic_signals": sorted(all_sigs),
        "language": dominant_lang,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Classify Omi transcript topics for a date"
    )
    parser.add_argument("date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-transcript detail")
    args = parser.parse_args()

    profile = classify_day(args.date)

    if args.json:
        print(json.dumps(profile, indent=2, default=str))
        sys.exit(0)

    if profile["transcript_count"] == 0:
        print(f"No Omi transcripts found for {args.date}")
        sys.exit(0)

    print(f"\nOmi Topic Profile — {args.date}")
    print(f"  Transcripts:        {profile['transcript_count']}")
    print(f"  Dominant category:  {profile['dominant_category']}")
    print(f"  Mean cog. density:  {profile['mean_cognitive_density']:.2f}")
    print(f"  CLS weight:         {profile['mean_cls_weight']:.2f}×")
    print(f"  SDI weight:         {profile['mean_sdi_weight']:.2f}×")
    print(f"  Languages:          {profile['languages']}")
    print(f"  Category dist:      {profile['category_distribution']}")
    if profile['all_signals']:
        print(f"  Top signals:        {', '.join(profile['all_signals'][:10])}")

    if args.verbose and profile['per_transcript']:
        print(f"\n  Per-transcript breakdown:")
        for r in profile['per_transcript']:
            ts = r.get('timestamp', '')[:16]
            cat = r['category']
            dens = r['cognitive_density']
            lang = r['language']
            sigs = ', '.join(r['topic_signals'][:3]) or '—'
            print(f"    {ts}  [{lang}] {cat:16s}  density={dens:.2f}  signals=[{sigs}]")
    print()
