"""Pre-filter for blocked topics and prompt-extraction attempts.

Deterministic first layer of defense. The LLM's system prompt is the second
layer; together they implement defense in depth.

Design philosophy: NARROW. Catch the obvious cases to (a) save LLM calls on
clearly-out-of-scope queries and (b) provide a deterministic guarantee on the
assignment's rubric requirements. False positives (legitimate questions
incorrectly blocked) are worse UX than false negatives — those are caught by
the system prompt, which already passed manual adversarial testing.

All patterns use word boundaries (\\b) to prevent matching legitimate finance
terms that contain blocked substrings (catalyst, category, dogma, underdog,
Taylor rule, Swift programming language, etc.).
"""

import re
from typing import Optional

from prompts import (
    REFUSAL_PETS,
    REFUSAL_ZODIAC,
    REFUSAL_SWIFT,
    REFUSAL_PROMPT_LEAK,
)


# Each entry: (compiled pattern, refusal string).
# Patterns are case-insensitive. Order matters only for performance, not correctness.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ---- Pets: cats, dogs, common variants ----
    # Word boundaries prevent "catalyst", "category", "dogma", "underdog", etc.
    (
        re.compile(
            r"\b(cats?|kittens?|kitty|feline|felines|"
            r"dogs?|puppy|puppies|canine|canines|doggy|doggos?)\b",
            re.IGNORECASE,
        ),
        REFUSAL_PETS,
    ),

    # ---- Zodiac / astrology / sign names ----
    # "Cancer" intentionally excluded as a standalone match (medical / general
    # usage too common). The system prompt catches zodiac-context "cancer".
    (
        re.compile(
            r"\b(zodiac|horoscopes?|astrology|astrological|"
            r"aries|taurus|gemini|leo|virgo|libra|scorpio|sagittarius|"
            r"capricorn|aquarius|pisces)\b",
            re.IGNORECASE,
        ),
        REFUSAL_ZODIAC,
    ),

    # ---- Taylor Swift ----
    # Match only the full name and known fan terms. "Taylor" alone is
    # legitimate (Taylor rule). "Swift" alone is legitimate (Swift Industries,
    # Swift programming language, Swift settlement).
    (
        re.compile(
            r"\b(taylor\s+swift|t\.?\s*swift|swifties?|eras\s+tour)\b",
            re.IGNORECASE,
        ),
        REFUSAL_SWIFT,
    ),

    # ---- Direct prompt extraction ----
    (
        re.compile(
            r"\b(system\s*prompt|your\s+(prompt|instructions|system\s+message|"
            r"configuration|directive|rules|guidelines))\b|"
            r"\b(reveal|show|print|disclose|share|tell\s+me)\s+"
            r"(me\s+)?(your|the)\s+(prompt|instructions|system|rules|config|configuration)\b",
            re.IGNORECASE,
        ),
        REFUSAL_PROMPT_LEAK,
    ),

    # ---- Classic injection patterns ----
    (
        re.compile(
            r"\b(ignore|disregard|forget)\s+"
            r"(all\s+|the\s+|your\s+|any\s+)?"
            r"(prior|previous|above|earlier|preceding)\s+"
            r"(instructions?|prompts?|messages?|rules?|directives?)\b|"
            r"\b(developer\s+mode|admin\s+mode|test\s+mode|debug\s+mode|"
            r"jailbreak|DAN\s+mode|do\s+anything\s+now)\b",
            re.IGNORECASE,
        ),
        REFUSAL_PROMPT_LEAK,
    ),
]


def precheck(message: str) -> Optional[str]:
    """
    Check a user message against the pre-filter ruleset.

    Args:
        message: raw user input.

    Returns:
        A canned refusal string if the message matches a blocked pattern.
        None if the message passes (caller should proceed to the LLM).
    """
    for pattern, refusal in _PATTERNS:
        if pattern.search(message):
            return refusal
    return None
