"""Grounded generation provider with strict factual attribution and safety reminders."""

import re
from typing import Protocol

from chiller_rag.models import Citation, RetrievedPassage


class LlmProvider(Protocol):
    """Protocol for grounded answer generation backends."""

    def generate_answer(
        self,
        question: str,
        passages: list[RetrievedPassage],
    ) -> tuple[str, bool, list[Citation]]: ...


SAFETY_KEYWORDS = {
    "pressure",
    "refrigerant",
    "electrical",
    "voltage",
    "lockout",
    "hazard",
    "leak",
    "toxic",
    "flammable",
    "oil",
    "disconnect",
    "overheat",
    "emergency",
}

SAFETY_REMINDER = (
    "\n\n[MANDATORY SAFETY NOTICE]: Always consult the physical manufacturer documentation "
    "and ensure service procedures are executed only by certified HVAC/R technicians "
    "following formal Lockout/Tagout (LOTO) protocols."
)


class GroundedRuleLlmProvider:
    """Deterministic, local grounded generation engine enforcing strict factual attribution.

    Extracts direct technical evidence from retrieved passages, synthesizes a concise answer,
    generates verified citations with verbatim quotes, and injects safety notices for hazardous procedures.
    """

    def __init__(self, min_token_match: int = 1) -> None:
        self.min_token_match = min_token_match

    def generate_answer(
        self,
        question: str,
        passages: list[RetrievedPassage],
    ) -> tuple[str, bool, list[Citation]]:
        if not passages:
            return (
                "The indexed manuals do not contain sufficient evidence to answer this technical question.",
                True,
                [],
            )

        q_words = set(re.findall(r"\w+", question.lower()))
        # Remove common stop words
        stop_words = {
            "what",
            "is",
            "the",
            "for",
            "in",
            "of",
            "and",
            "how",
            "to",
            "a",
            "an",
            "on",
            "are",
            "do",
        }
        key_q_words = q_words - stop_words

        matching_passages: list[tuple[RetrievedPassage, str, list[str]]] = []

        for p in passages:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p.text) if s.strip()]
            for sent in sentences:
                sent_words = set(re.findall(r"\w+", sent.lower()))
                overlap = key_q_words.intersection(sent_words)
                if len(overlap) >= self.min_token_match:
                    matching_passages.append((p, sent, list(overlap)))

        if not matching_passages:
            return (
                "The indexed chiller documentation does not provide verifiable information matching the query terms.",
                True,
                [],
            )

        # Sort by overlap length and passage score
        matching_passages.sort(key=lambda x: (len(x[2]), x[0].score), reverse=True)

        selected_answers: list[str] = []
        citations: list[Citation] = []
        seen_chunks: set[str] = set()

        for p, sent, _ in matching_passages[:3]:
            selected_answers.append(sent)
            if p.chunk_id not in seen_chunks:
                seen_chunks.add(p.chunk_id)
                citations.append(
                    Citation(
                        document_id=p.document_id,
                        filename=p.filename,
                        page_number=p.page_number,
                        chunk_id=p.chunk_id,
                        quote=sent[:180],
                    )
                )

        answer_text = "According to the chiller manual documentation: " + " ".join(selected_answers)

        # Check for safety-critical terms
        is_safety_critical = any(
            kw in question.lower() or kw in answer_text.lower() for kw in SAFETY_KEYWORDS
        )
        if is_safety_critical:
            answer_text += SAFETY_REMINDER

        return answer_text, False, citations


class FakeLlmProvider:
    """Configurable fake LLM provider for isolated test suites."""

    def __init__(
        self,
        canned_answer: str = "Test answer grounded in documentation.",
        insufficient: bool = False,
    ) -> None:
        self.canned_answer = canned_answer
        self.insufficient = insufficient

    def generate_answer(
        self,
        question: str,
        passages: list[RetrievedPassage],
    ) -> tuple[str, bool, list[Citation]]:
        if self.insufficient or not passages:
            return ("No evidence found in documentation.", True, [])

        citations = [
            Citation(
                document_id=passages[0].document_id,
                filename=passages[0].filename,
                page_number=passages[0].page_number,
                chunk_id=passages[0].chunk_id,
                quote=passages[0].text[:100],
            )
        ]
        return (self.canned_answer, False, citations)
