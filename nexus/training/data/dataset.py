"""
RLHF Dataset implementations for reward model training.

Supports multiple data formats:
- Anthropic HH-RLHF format (chosen/rejected pairs)
- OpenAI comparison format
- Generic preference pairs
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from nexus.training.data.preprocessing import (
    DialogueParser,
    ParsedDialogue,
    is_single_turn,
)


class DatasetFormat(str, Enum):
    """Supported dataset formats."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GENERIC = "generic"


@dataclass
class PreferencePair:
    """A preference pair for reward model training."""
    chosen: str
    rejected: str
    chosen_parsed: ParsedDialogue | None = None
    rejected_parsed: ParsedDialogue | None = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RLHFDataset(Dataset):
    """Dataset for RLHF reward model training.

    Loads preference pairs (chosen/rejected) and prepares them for training.
    Supports filtering by turn count and various preprocessing options.

    Attributes:
        data: List of PreferencePair objects
        format_type: Dataset format
        single_turn_only: Whether to filter to single-turn dialogues
    """

    def __init__(
        self,
        data_path: str | Path,
        format_type: DatasetFormat = DatasetFormat.ANTHROPIC,
        single_turn_only: bool = True,
        max_samples: int | None = None,
        parse_dialogues: bool = True,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to data file (JSONL format)
            format_type: Format of the data
            single_turn_only: Filter to single-turn dialogues only
            max_samples: Maximum number of samples to load
            parse_dialogues: Whether to parse dialogues into context/response
        """
        self.data_path = Path(data_path)
        self.format_type = format_type
        self.single_turn_only = single_turn_only
        self.parse_dialogues = parse_dialogues

        self.data: list[PreferencePair] = []
        self._load_data(max_samples)

    def _load_data(self, max_samples: int | None) -> None:
        """Load data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())
                    pair = self._parse_item(item)

                    if pair and self._should_include(pair):
                        self.data.append(pair)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
                except Exception:
                    continue  # Skip problematic samples

    def _parse_item(self, item: dict[str, Any]) -> PreferencePair | None:
        """Parse a single data item into a PreferencePair."""
        if self.format_type == DatasetFormat.ANTHROPIC:
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
        elif self.format_type == DatasetFormat.OPENAI:
            chosen = item.get("preferred", item.get("chosen", ""))
            rejected = item.get("rejected", item.get("dispreferred", ""))
        else:
            chosen = item.get("chosen", item.get("positive", ""))
            rejected = item.get("rejected", item.get("negative", ""))

        if not chosen or not rejected:
            return None

        # Parse dialogues if requested
        chosen_parsed = None
        rejected_parsed = None

        if self.parse_dialogues:
            try:
                chosen_parsed = DialogueParser.parse_and_split(
                    chosen, "anthropic" if self.format_type == DatasetFormat.ANTHROPIC else "anthropic"
                )
                rejected_parsed = DialogueParser.parse_and_split(
                    rejected, "anthropic" if self.format_type == DatasetFormat.ANTHROPIC else "anthropic"
                )
            except Exception:
                pass  # Keep as unparsed

        return PreferencePair(
            chosen=chosen,
            rejected=rejected,
            chosen_parsed=chosen_parsed,
            rejected_parsed=rejected_parsed,
            metadata=item.get("metadata", {}),
        )

    def _should_include(self, pair: PreferencePair) -> bool:
        """Check if pair should be included based on filters."""
        if self.single_turn_only:
            if pair.chosen_parsed and pair.chosen_parsed.is_multi_turn:
                return False
            if pair.rejected_parsed and pair.rejected_parsed.is_multi_turn:
                return False
            # Fallback to text-based check
            if not pair.chosen_parsed:
                if not is_single_turn(pair.chosen) or not is_single_turn(pair.rejected):
                    return False
        return True

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.data[idx]

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.data),
            "format": self.format_type.value,
            "single_turn_only": self.single_turn_only,
        }

        if self.data and self.data[0].chosen_parsed:
            multi_turn = sum(1 for p in self.data if p.chosen_parsed and p.chosen_parsed.is_multi_turn)
            stats["multi_turn_count"] = multi_turn
            stats["single_turn_count"] = len(self.data) - multi_turn

        return stats


class StreamingRLHFDataset(IterableDataset):
    """Streaming version of RLHF dataset for large files.

    Reads data lazily without loading everything into memory.
    """

    def __init__(
        self,
        data_path: str | Path,
        format_type: DatasetFormat = DatasetFormat.ANTHROPIC,
        single_turn_only: bool = True,
        buffer_size: int = 1000,
    ):
        """Initialize streaming dataset.

        Args:
            data_path: Path to data file
            format_type: Data format
            single_turn_only: Filter to single-turn only
            buffer_size: Size of shuffle buffer
        """
        self.data_path = Path(data_path)
        self.format_type = format_type
        self.single_turn_only = single_turn_only
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[PreferencePair]:
        """Iterate over the dataset."""
        with open(self.data_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    pair = self._parse_item(item)

                    if pair and self._should_include(pair):
                        yield pair
                except Exception:
                    continue

    def _parse_item(self, item: dict) -> PreferencePair | None:
        """Parse item (same as RLHFDataset)."""
        if self.format_type == DatasetFormat.ANTHROPIC:
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
        else:
            chosen = item.get("chosen", item.get("preferred", ""))
            rejected = item.get("rejected", item.get("dispreferred", ""))

        if not chosen or not rejected:
            return None

        return PreferencePair(chosen=chosen, rejected=rejected)

    def _should_include(self, pair: PreferencePair) -> bool:
        """Filter check."""
        if self.single_turn_only:
            return is_single_turn(pair.chosen) and is_single_turn(pair.rejected)
        return True
