"""
Dialogue parsing and preprocessing utilities.

Handles extraction of context/response pairs from various conversation formats.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class Speaker(str, Enum):
    """Speaker roles in a conversation."""
    HUMAN = "human"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    speaker: Speaker
    content: str

    def __str__(self) -> str:
        return f"{self.speaker.value}: {self.content}"


@dataclass
class ParsedDialogue:
    """A parsed dialogue with context and response separated."""
    context: str
    response: str
    num_turns: int
    is_multi_turn: bool

    @property
    def full_text(self) -> str:
        """Get the complete dialogue text."""
        return f"{self.context}{self.response}"


class DialogueParser:
    """Parser for various dialogue formats.

    Supports:
    - Anthropic format: "Human: ... Assistant: ..."
    - OpenAI format: {"role": "user/assistant", "content": "..."}
    - Custom formats via regex patterns
    """

    # Pattern for Anthropic-style dialogues
    ANTHROPIC_PATTERN = re.compile(
        r"(?:^|\n\n)(Human|Assistant):\s*",
        re.MULTILINE
    )

    # Common role mappings
    ROLE_MAPPING = {
        "human": Speaker.HUMAN,
        "user": Speaker.HUMAN,
        "assistant": Speaker.ASSISTANT,
        "bot": Speaker.ASSISTANT,
        "system": Speaker.SYSTEM,
    }

    @classmethod
    def parse_anthropic_format(cls, text: str) -> list[ConversationTurn]:
        """Parse Anthropic-style dialogue format.

        Args:
            text: Text in format "Human: ... Assistant: ..."

        Returns:
            List of ConversationTurn objects
        """
        turns: list[ConversationTurn] = []

        # Split by speaker markers
        parts = cls.ANTHROPIC_PATTERN.split(text)

        # First part might be empty or system prompt
        i = 0
        if parts and not parts[0].strip():
            i = 1

        # Process speaker-content pairs
        while i < len(parts) - 1:
            speaker_str = parts[i].lower()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""

            speaker = cls.ROLE_MAPPING.get(speaker_str, Speaker.HUMAN)
            if content:
                turns.append(ConversationTurn(speaker=speaker, content=content))

            i += 2

        return turns

    @classmethod
    def parse_openai_format(cls, messages: list[dict]) -> list[ConversationTurn]:
        """Parse OpenAI-style message format.

        Args:
            messages: List of {"role": str, "content": str} dicts

        Returns:
            List of ConversationTurn objects
        """
        turns: list[ConversationTurn] = []

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "").strip()

            speaker = cls.ROLE_MAPPING.get(role, Speaker.HUMAN)
            if content:
                turns.append(ConversationTurn(speaker=speaker, content=content))

        return turns

    @classmethod
    def split_context_response(
        cls,
        turns: list[ConversationTurn],
    ) -> tuple[str, str]:
        """Split turns into context and final response.

        The context includes all turns except the last assistant turn.
        The response is the final assistant turn.

        Args:
            turns: List of conversation turns

        Returns:
            Tuple of (context_text, response_text)
        """
        if not turns:
            return "", ""

        # Find the last assistant turn
        last_assistant_idx = -1
        for i in range(len(turns) - 1, -1, -1):
            if turns[i].speaker == Speaker.ASSISTANT:
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            # No assistant turn, treat all as context
            context = cls._format_turns(turns)
            return context, ""

        # Everything before last assistant is context
        context_turns = turns[:last_assistant_idx]
        response = turns[last_assistant_idx].content

        # Format context
        context = cls._format_turns(context_turns)

        # Add the response prefix
        if context:
            context += "\n\nAssistant:"

        return context, response

    @classmethod
    def _format_turns(cls, turns: list[ConversationTurn]) -> str:
        """Format turns back to Anthropic-style text."""
        if not turns:
            return ""

        parts = []
        for turn in turns:
            role = "Human" if turn.speaker == Speaker.HUMAN else "Assistant"
            parts.append(f"{role}: {turn.content}")

        return "\n\n".join(parts)

    @classmethod
    def parse_and_split(
        cls,
        text: str,
        format_type: str = "anthropic",
    ) -> ParsedDialogue:
        """Parse text and split into context/response.

        Args:
            text: Dialogue text
            format_type: Format type ("anthropic" or "openai")

        Returns:
            ParsedDialogue with context and response
        """
        if format_type == "anthropic":
            turns = cls.parse_anthropic_format(text)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        context, response = cls.split_context_response(turns)

        return ParsedDialogue(
            context=context,
            response=response,
            num_turns=len(turns),
            is_multi_turn=len(turns) > 2,
        )


def count_turns(text: str) -> int:
    """Count the number of dialogue turns in Anthropic-format text.

    Args:
        text: Dialogue text

    Returns:
        Number of turns
    """
    turns = DialogueParser.parse_anthropic_format(text)
    return len(turns)


def is_single_turn(text: str) -> bool:
    """Check if dialogue is single-turn (one human, one assistant).

    Args:
        text: Dialogue text

    Returns:
        True if single-turn dialogue
    """
    return count_turns(text) <= 2


def extract_last_response(text: str) -> str:
    """Extract the last assistant response from a dialogue.

    Args:
        text: Dialogue text

    Returns:
        The last assistant response text
    """
    turns = DialogueParser.parse_anthropic_format(text)

    for turn in reversed(turns):
        if turn.speaker == Speaker.ASSISTANT:
            return turn.content

    return ""
