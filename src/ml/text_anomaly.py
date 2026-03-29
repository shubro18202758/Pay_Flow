"""
PayFlow — Text Anomaly Detection for Beneficiary Fields
=========================================================
Parses unstructured text fields (beneficiary names, merchant descriptors,
remittance info) to detect subtle manipulation patterns:

  1. Character Substitution — Cyrillic/homoglyph lookalikes (e.g., "а" vs "a")
  2. Spelling Perturbation — Levenshtein-based near-match to known entity names
  3. Pattern Anomalies     — Unusual character distributions, excessive digits,
                            special characters in name fields
  4. Script Mixing         — Mixed Unicode scripts within a single field
  5. Known Phishing Tokens — Common social engineering lure phrases

AML Context:
  Beneficiary name manipulation is a key indicator of mule account operation
  and phishing-based fraud. SWIFT MT103 field :59 (Beneficiary) and NEFT
  remittance fields are commonly abused. Indian FIU-IND STR guidelines
  specifically flag "inconsistent beneficiary details" as a red indicator.

Algorithm:
  Pure CPU text analysis — no ML model needed. Character-level feature
  extraction using Unicode category analysis + edit distance computation
  via Wagner-Fischer DP (O(nm) where n,m ≤ 50 chars for names).

Output Features (per text field):
  - homoglyph_count       : count of visually-similar non-ASCII substitutions
  - mixed_script_flag     : 1 if multiple Unicode scripts detected
  - digit_ratio           : fraction of characters that are digits
  - special_char_ratio    : fraction of non-alphanumeric, non-space characters
  - entropy               : Shannon entropy of character distribution
  - min_edit_distance     : minimum Levenshtein distance to known entity pool
  - suspicious_token_flag : 1 if contains known phishing lure patterns
  - name_length_anomaly   : z-score of name length vs population
  - text_anomaly_score    : composite risk score (0.0 – 1.0)
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import NamedTuple


# ── Output Feature Vector ──────────────────────────────────────────────────

class TextAnomalyFeatures(NamedTuple):
    """9 text anomaly features per field, packed for numpy vectorization."""
    homoglyph_count: int
    mixed_script_flag: int
    digit_ratio: float
    special_char_ratio: float
    entropy: float
    min_edit_distance: int
    suspicious_token_flag: int
    name_length_anomaly: float
    text_anomaly_score: float


# ── Homoglyph Detection ────────────────────────────────────────────────────

# Maps visually confusable Unicode characters → their ASCII lookalike.
# Covers the most common Cyrillic/Greek/mathematical substitutions
# used in beneficiary name fraud.

_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0430": "a",  # Cyrillic а → Latin a
    "\u0435": "e",  # Cyrillic е → Latin e
    "\u043e": "o",  # Cyrillic о → Latin o
    "\u0440": "p",  # Cyrillic р → Latin p
    "\u0441": "c",  # Cyrillic с → Latin c
    "\u0443": "y",  # Cyrillic у → Latin y
    "\u0445": "x",  # Cyrillic х → Latin x
    "\u0456": "i",  # Cyrillic і → Latin i
    "\u0458": "j",  # Cyrillic ј → Latin j
    "\u04bb": "h",  # Cyrillic һ → Latin h
    "\u0391": "A",  # Greek Α → Latin A
    "\u0392": "B",  # Greek Β → Latin B
    "\u0395": "E",  # Greek Ε → Latin E
    "\u0397": "H",  # Greek Η → Latin H
    "\u0399": "I",  # Greek Ι → Latin I
    "\u039a": "K",  # Greek Κ → Latin K
    "\u039c": "M",  # Greek Μ → Latin M
    "\u039d": "N",  # Greek Ν → Latin N
    "\u039f": "O",  # Greek Ο → Latin O
    "\u03a1": "P",  # Greek Ρ → Latin P
    "\u03a4": "T",  # Greek Τ → Latin T
    "\u03a5": "Y",  # Greek Υ → Latin Y
    "\u03a7": "X",  # Greek Χ → Latin X
    "\u03b1": "a",  # Greek α → Latin a  (lowercase)
    "\u03bf": "o",  # Greek ο → Latin o
    "\u2010": "-",  # Hyphen
    "\u2011": "-",  # Non-breaking hyphen
    "\u2012": "-",  # Figure dash
    "\u2013": "-",  # En dash
    "\u2014": "-",  # Em dash
    "\uff10": "0",  # Fullwidth 0
    "\uff11": "1",  # Fullwidth 1
}


def _count_homoglyphs(text: str) -> int:
    """Count visually confusable non-ASCII characters that mimic ASCII."""
    return sum(1 for ch in text if ch in _HOMOGLYPH_MAP)


# ── Script Detection ───────────────────────────────────────────────────────

def _detect_scripts(text: str) -> set[str]:
    """
    Return the set of Unicode script names present in the text.
    Ignores Common/Inherited scripts (numbers, punctuation, spaces).
    """
    scripts: set[str] = set()
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):  # Letters only
            try:
                name = unicodedata.name(ch, "")
                if name:
                    # Extract script from Unicode name (first word is usually script)
                    script = name.split()[0]
                    scripts.add(script)
            except ValueError:
                pass
    return scripts


# ── Shannon Entropy ─────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy. Higher entropy = more random."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# ── Levenshtein Distance ───────────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """
    Wagner-Fischer algorithm. O(nm) time and O(min(n,m)) space.
    Used for fuzzy matching beneficiary names against known entities.
    """
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if not s2:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            curr_row.append(min(
                prev_row[j + 1] + 1,
                curr_row[j] + 1,
                prev_row[j] + (0 if c1 == c2 else 1),
            ))
        prev_row = curr_row

    return prev_row[-1]


# ── Suspicious Patterns ────────────────────────────────────────────────────

# Regex patterns for social engineering lure phrases commonly found in
# manipulated beneficiary/remittance fields.
_SUSPICIOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(urgent|verify|confirm|suspend|block|freeze)\b", re.IGNORECASE),
    re.compile(r"\b(otp|pin|password|cvv)\b", re.IGNORECASE),
    re.compile(r"\b(refund|cashback|prize|reward|lottery|won)\b", re.IGNORECASE),
    re.compile(r"\b(kyc\s*update|link\s*expire|account\s*close)\b", re.IGNORECASE),
    re.compile(r"\b(rbi|sebi|income\s*tax|police|court\s*order)\b", re.IGNORECASE),
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b\d{10}\b"),  # 10-digit mobile numbers embedded in names
]


def _has_suspicious_tokens(text: str) -> bool:
    """Check if text contains any social engineering lure patterns."""
    return any(pat.search(text) for pat in _SUSPICIOUS_PATTERNS)


# ── Known Entity Pool ──────────────────────────────────────────────────────

# Reference names for edit-distance comparison. In production this would be
# loaded from the customer master, but for the prototype we use major Indian
# bank entity names and common legitimate merchant descriptors.
_KNOWN_ENTITIES: list[str] = [
    "union bank of india", "state bank of india", "hdfc bank", "icici bank",
    "axis bank", "kotak mahindra", "punjab national bank", "bank of baroda",
    "canara bank", "indian overseas bank", "central bank of india",
    "reserve bank of india", "npci", "national payments corporation",
    "income tax department", "goods and services tax",
]


def _min_edit_to_known(text: str) -> int:
    """
    Minimum Levenshtein distance from text to any known entity.
    Low distance (1-2) with non-exact match = possible typosquatting.
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return 99

    best = 99
    for entity in _KNOWN_ENTITIES:
        # Quick length filter: skip if length difference > 5
        if abs(len(text_lower) - len(entity)) > 5:
            continue
        d = _levenshtein(text_lower, entity)
        if d < best:
            best = d
            if d == 0:
                break  # Exact match, no need to check more
    return best


# ── Main Analyzer ──────────────────────────────────────────────────────────

class TextAnomalyAnalyzer:
    """
    Stateful text anomaly detector with population-level name length tracking.

    Accumulates name length statistics for z-score computation. All other
    features are stateless per-event computations.
    """

    def __init__(self) -> None:
        self._name_count: int = 0
        self._name_len_sum: float = 0.0
        self._name_len_sq_sum: float = 0.0

    def analyze(self, text: str) -> TextAnomalyFeatures:
        """
        Extract text anomaly features from a single beneficiary/name field.

        Returns a 9-element NamedTuple ready for numpy array conversion.
        """
        if not text or not text.strip():
            return TextAnomalyFeatures(
                homoglyph_count=0, mixed_script_flag=0,
                digit_ratio=0.0, special_char_ratio=0.0,
                entropy=0.0, min_edit_distance=99,
                suspicious_token_flag=0, name_length_anomaly=0.0,
                text_anomaly_score=0.0,
            )

        text = text.strip()
        n = len(text)

        # ── Individual features ─────────────────────────────────────────
        homoglyphs = _count_homoglyphs(text)

        scripts = _detect_scripts(text)
        mixed_script = 1 if len(scripts) > 1 else 0

        digit_count = sum(1 for ch in text if ch.isdigit())
        digit_ratio = digit_count / n

        special_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        special_ratio = special_count / n

        entropy = _shannon_entropy(text)

        min_edit = _min_edit_to_known(text)

        suspicious = 1 if _has_suspicious_tokens(text) else 0

        # Name length z-score (population-level)
        self._name_count += 1
        self._name_len_sum += n
        self._name_len_sq_sum += n * n

        name_z = 0.0
        if self._name_count > 5:
            mean_len = self._name_len_sum / self._name_count
            var_len = (self._name_len_sq_sum / self._name_count) - mean_len ** 2
            std_len = math.sqrt(var_len) if var_len > 0 else 1.0
            name_z = (n - mean_len) / std_len if std_len > 0.01 else 0.0

        # ── Composite anomaly score ─────────────────────────────────────
        # Each signal contributes to a weighted composite:
        score = 0.0
        score += 0.25 * min(homoglyphs / 3.0, 1.0)        # 3+ homoglyphs = max
        score += 0.15 * mixed_script                        # binary
        score += 0.10 * min(digit_ratio / 0.3, 1.0)        # >30% digits = max
        score += 0.10 * min(special_ratio / 0.2, 1.0)      # >20% specials = max
        score += 0.15 * (1.0 if 1 <= min_edit <= 3 else 0.0)  # near-miss = suspicious
        score += 0.15 * suspicious                           # phishing tokens
        score += 0.10 * min(abs(name_z) / 4.0, 1.0)        # extreme lengths

        return TextAnomalyFeatures(
            homoglyph_count=homoglyphs,
            mixed_script_flag=mixed_script,
            digit_ratio=round(digit_ratio, 4),
            special_char_ratio=round(special_ratio, 4),
            entropy=round(entropy, 4),
            min_edit_distance=min_edit,
            suspicious_token_flag=suspicious,
            name_length_anomaly=round(name_z, 4),
            text_anomaly_score=round(score, 4),
        )

    def analyzed_count(self) -> int:
        return self._name_count
