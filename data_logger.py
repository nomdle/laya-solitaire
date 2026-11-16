"""
data_logger.py - Fine-tuning Data Collector & Episode Quality Filter for Laya Solitaire Agent.

Collects turn-by-turn game context, available options, and chosen target actions,
saving only verified, progressive, or winning episode data in JSONL format for fine-tuning.
Stalled or looping episodes are automatically discarded.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple


class DataLogger:
    """
    Logs decision steps into a JSONL dataset for Laya fine-tuning with episode-level filtering.
    Format:
    {
      "context": "...",
      "options": ["...", "..."],
      "target": "..."
    }
    """

    def __init__(
        self,
        filepath: str = "dataset/laya_dataset.jsonl",
        filter_beneficial: bool = False,
        buffer_size: int = 1
    ):
        self.filepath = filepath
        self.filter_beneficial = filter_beneficial
        self.buffer_size = buffer_size

        # In-memory buffers
        self._current_episode: List[Dict[str, Any]] = []
        self._committed_buffer: List[Dict[str, Any]] = []

        # Telemetry stats
        self.stats = {
            "episodes_started": 0,
            "episodes_committed": 0,
            "episodes_discarded": 0,
            "discard_reasons": {}
        }

        # Ensure parent directory exists
        dir_name = os.path.dirname(self.filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def start_episode(self):
        """Initializes a new episode buffer."""
        self._current_episode.clear()
        self.stats["episodes_started"] += 1

    def log_step(
        self,
        context: str,
        options: List[str],
        target: str,
        is_beneficial: bool = True,
        save_immediate: bool = False
    ) -> bool:
        """
        Record a single decision step into the current episode buffer.
        Returns True if step was buffered, False if rejected.
        """
        if not options or target not in options:
            return False

        if self.filter_beneficial and not is_beneficial:
            return False

        entry = {
            "context": context.strip(),
            "options": list(options),
            "target": target.strip()
        }

        self._current_episode.append(entry)

        if save_immediate:
            self._committed_buffer.append(entry)
            self.flush()

        return True

    def commit_episode(
        self,
        won: bool = False,
        foundation_progress: int = 0,
        min_progress_required: int = 1
    ) -> Tuple[bool, int]:
        """
        Evaluates the current episode and commits its turns to the dataset
        only if it achieved meaningful progress or won.
        Returns (committed: bool, turns_saved: int).
        """
        num_turns = len(self._current_episode)
        if num_turns == 0:
            return False, 0

        # Check quality threshold: winning or made positive foundation/tableau progress
        is_worthy = won or (foundation_progress >= min_progress_required)

        if is_worthy:
            self._committed_buffer.extend(self._current_episode)
            self.flush()
            self.stats["episodes_committed"] += 1
            turns_saved = num_turns
            self._current_episode.clear()
            return True, turns_saved
        else:
            return self.discard_episode(reason="insufficient_progress")

    def discard_episode(self, reason: str = "loop_or_stall") -> Tuple[bool, int]:
        """
        Discards the current episode buffer without saving anything to the file.
        Prevents infinite loops or stalled actions from contaminating the training dataset.
        """
        num_dropped = len(self._current_episode)
        self._current_episode.clear()
        self.stats["episodes_discarded"] += 1
        self.stats["discard_reasons"][reason] = self.stats["discard_reasons"].get(reason, 0) + 1
        return False, num_dropped

    def flush(self):
        """Flushes committed buffer to JSONL file."""
        if not self._committed_buffer:
            return

        with open(self.filepath, "a", encoding="utf-8") as f:
            for item in self._committed_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._committed_buffer.clear()

    def count_records(self) -> int:
        """Count total lines stored in the JSONL file."""
        if not os.path.exists(self.filepath):
            return 0
        with open(self.filepath, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def get_sample_records(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the first few sample records from the JSONL file."""
        samples = []
        if not os.path.exists(self.filepath):
            return samples
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    if len(samples) >= limit:
                        break
        return samples
