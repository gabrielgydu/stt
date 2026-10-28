"""
Custom spellings, applied to a transcript after the provider returns, so every
provider spells names the same way.

    [{"from": ["Sao Paulo", "Saopaulo"], "to": "São Paulo"}]

A `from` form matches regardless of case and only as whole words: it must not
be preceded or followed by a letter or digit, which, unlike regex's ASCII
`\\b` in other languages, also holds for accented words ("Paraná"). Spaces
inside a form match any run of whitespace. The longest form wins where two
overlap, `to` goes in verbatim (it may be several words), and replaced text is
never matched again.

Forms and text are compared in Unicode NFC, so an accented form typed as
letter + combining accent (as some editors and macOS write it) matches the
precomposed letters providers return, and the other way round. The text and
`to` come back in NFC, which looks the same.
"""

import json
import os
import re
import unicodedata

# Combining accents count as part of a word, so "Parana" does not match the
# decomposed form of "Paraná" (an "a" followed by U+0301).
_WORD = r"[^\W_]|[\u0300-\u036f]"


class Spellings:
    """Rules (a list as in the JSON file) compiled once; ValueError when one is
    malformed. `apply` rewrites a string, `apply_transcript` a Transcript in
    place."""

    def __init__(self, rules):
        forms = []  # (from, to)
        for rule in _check(rules):
            for form in rule["from"]:
                forms.append((_nfc(form), _nfc(rule["to"])))
        forms.sort(key=lambda pair: len(pair[0]), reverse=True)
        self._to = [to for _, to in forms]
        # One group per form, so the match says which rule it was; the
        # alternation tries longer forms first.
        pattern = "|".join(
            "(" + r"\s+".join(re.escape(word) for word in form.split()) + ")"
            for form, _ in forms
        )
        self._regex = re.compile(f"(?<!{_WORD})(?:{pattern})(?!{_WORD})", re.IGNORECASE) if forms else None

    def apply(self, text):
        if not self._regex or not text:
            return text
        return self._regex.sub(lambda m: self._to[m.lastindex - 1], _nfc(text))

    def apply_transcript(self, transcript):
        transcript.text = self.apply(transcript.text)
        for utterance in transcript.utterances:
            utterance.text = self.apply(utterance.text)
        return transcript


def _nfc(text):
    return unicodedata.normalize("NFC", text)


def _check(rules):
    """The rules, validated; ValueError naming the first bad one."""
    if not isinstance(rules, list):
        raise ValueError('spellings must be a list of {"from": [...], "to": "..."}')
    checked = []
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"spelling #{i + 1} is not an object")
        forms = rule.get("from")
        if isinstance(forms, str):
            forms = [forms]
        to = rule.get("to")
        if (not isinstance(forms, list) or not forms
                or not all(isinstance(f, str) and f.strip() for f in forms)):
            raise ValueError(f"spelling #{i + 1}: \"from\" must be a non-empty list of strings")
        if not isinstance(to, str):
            raise ValueError(f"spelling #{i + 1}: \"to\" must be a string")
        checked.append({"from": [f.strip() for f in forms], "to": to})
    return checked


def load(file_path):
    """The rules in a JSON file, validated (ValueError, or OSError when the
    file cannot be read)."""
    with open(os.path.expanduser(file_path), encoding="utf-8") as f:
        try:
            rules = json.load(f)
        except ValueError as e:
            raise ValueError(f"{file_path} is not valid JSON: {e}") from None
    _check(rules)
    return rules
