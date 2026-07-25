from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# The verdict object every validator returns
# ---------------------------------------------------------------------------
@dataclass
class GuardrailResult:
    allowed: bool          # False = block the turn
    text: str              # possibly cleaned/redacted version of the input
    reason: str | None = None  # human-readable explanation when blocked/changed

    @classmethod
    def ok(cls, text: str) -> "GuardrailResult":
        return cls(allowed=True, text=text)

    @classmethod
    def block(cls, text: str, reason: str) -> "GuardrailResult":
        return cls(allowed=False, text=text, reason=reason)


# ---------------------------------------------------------------------------
# Base class — the contract every rule must satisfy
# ---------------------------------------------------------------------------
class Validator(ABC):
    """One safety rule. Subclass and implement `check`."""

    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        ...


# ---------------------------------------------------------------------------
# Built-in validators (no external dependencies)
# ---------------------------------------------------------------------------
class MinWordsValidator(Validator):
    """Reject inputs that are too short to be a real question."""

    def __init__(self, min_words: int = 3) -> None:
        self.min_words = min_words

    def check(self, text: str) -> GuardrailResult:
        if len(text.split()) < self.min_words:
            return GuardrailResult.block(
                text,
                f"Input too short — please ask a full question "
                f"(at least {self.min_words} words).",
            )
        return GuardrailResult.ok(text)


class MinLengthValidator(Validator):
    """Reject outputs that are suspiciously short (usually an LLM error)."""

    def __init__(self, min_chars: int = 20) -> None:
        self.min_chars = min_chars

    def check(self, text: str) -> GuardrailResult:
        if len(text.strip()) < self.min_chars:
            return GuardrailResult.block(
                text, "Response was too short to be a safe, complete answer."
            )
        return GuardrailResult.ok(text)


class PromptInjectionValidator(Validator):
    """Block classic prompt-injection / jailbreak phrases in user input."""

    BLOCKED = (
        "ignore previous",
        "ignore all previous",
        "jailbreak",
        "prompt injection",
        "forget instructions",
        "forget your instructions",
        "disregard your",
        "override your",
        "act as if",
        "bypass",
    )

    def check(self, text: str) -> GuardrailResult:
        lower = text.lower()
        for phrase in self.BLOCKED:
            if phrase in lower:
                return GuardrailResult.block(
                    text, f"Blocked by injection filter (matched: '{phrase}')."
                )
        return GuardrailResult.ok(text)


class ToxicityKeywordValidator(Validator):
    """Block obviously harmful/toxic content in either direction."""

    BLOCKED = (
        "kill yourself",
        "you are worthless",
        "hate speech",
        "racial slur",
        "religious slur",
    )

    def check(self, text: str) -> GuardrailResult:
        lower = text.lower()
        for phrase in self.BLOCKED:
            if phrase in lower:
                return GuardrailResult.block(
                    text, "Blocked by the toxicity filter."
                )
        return GuardrailResult.ok(text)


class PIIRedactionValidator(Validator):
    """Redact personal data instead of blocking — cleans the text and passes.

    This is the one validator that *transforms* rather than *rejects*: it
    returns allowed=True but with a scrubbed `text`. It shows the pipeline can
    both filter AND rewrite.
    """

    PATTERNS = {
        "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
        "PHONE": re.compile(r"\b(?:\+?\d[\s-]?){9,13}\d\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    def check(self, text: str) -> GuardrailResult:
        cleaned = text
        found: list[str] = []
        for label, pattern in self.PATTERNS.items():
            if pattern.search(cleaned):
                found.append(label)
                cleaned = pattern.sub(f"[REDACTED_{label}]", cleaned)
        if found:
            return GuardrailResult(
                allowed=True,
                text=cleaned,
                reason=f"Redacted PII: {', '.join(found)}.",
            )
        return GuardrailResult.ok(text)


# ---------------------------------------------------------------------------
# Optional "famous library" validator — Guardrails AI (auto-detected)
# ---------------------------------------------------------------------------
class GuardrailsAIValidator(Validator):
    """Wraps Guardrails AI (https://github.com/guardrails-ai) if installed.

    We keep the SAME `check()` interface, so the engine cannot tell the
    difference between this and a built-in rule. If the library is missing,
    `available()` returns False and the engine simply skips it.
    """

    def __init__(self) -> None:
        self._guard = None
        try:
            from guardrails import Guard  # type: ignore

            # A minimal Guard. Teams typically attach hub validators here,
            # e.g. Guard().use(ToxicLanguage, on_fail="exception").
            self._guard = Guard()
        except Exception:
            self._guard = None

    def available(self) -> bool:
        return self._guard is not None

    def check(self, text: str) -> GuardrailResult:
        if self._guard is None:
            return GuardrailResult.ok(text)
        try:
            self._guard.validate(text)
            return GuardrailResult.ok(text)
        except Exception as exc:
            return GuardrailResult.block(text, f"Blocked by Guardrails AI: {exc}")


# ---------------------------------------------------------------------------
# The engine — runs a pipeline of validators
# ---------------------------------------------------------------------------
class GuardrailEngine:
    """Runs input validators before the LLM and output validators after it.

    A validator can BLOCK (stop the turn) or TRANSFORM (rewrite the text and
    keep going). The engine threads the (possibly rewritten) text through the
    whole pipeline and short-circuits on the first block.
    """

    def __init__(
        self,
        input_validators: list[Validator],
        output_validators: list[Validator],
    ) -> None:
        self.input_validators = input_validators
        self.output_validators = output_validators

    def _run(self, validators: list[Validator], text: str) -> GuardrailResult:
        current = text
        for validator in validators:
            result = validator.check(current)
            if not result.allowed:
                return result           # blocked — stop immediately
            current = result.text       # carry forward any transformation
        return GuardrailResult.ok(current)

    def check_input(self, text: str) -> GuardrailResult:
        return self._run(self.input_validators, text)

    def check_output(self, text: str) -> GuardrailResult:
        return self._run(self.output_validators, text)

    # -- Factory: assemble the standard pipeline -------------------------------
    @classmethod
    def default(cls, use_guardrails_ai: bool = True) -> "GuardrailEngine":
        input_validators: list[Validator] = [
            MinWordsValidator(min_words=3),
            PromptInjectionValidator(),
            PIIRedactionValidator(),
            ToxicityKeywordValidator(),
        ]
        output_validators: list[Validator] = [
            MinLengthValidator(min_chars=20),
            ToxicityKeywordValidator(),
        ]

        if use_guardrails_ai:
            famous = GuardrailsAIValidator()
            if famous.available():
                print("[guardrails] Guardrails AI detected — enabled.")
                input_validators.append(famous)
                output_validators.append(famous)

        return cls(input_validators, output_validators)
