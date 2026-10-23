"""Data provenance and event accounting; no legacy implementation imports."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Episode:
    episode_id: str
    source_episode_id: str
    split: Literal['train', 'calibration', 'test']
    origin: Literal['offline', 'real_evaluation', 'synthetic']


def validate_dataset(episodes: list[Episode]) -> None:
    seen, groups = set(), {}
    for ep in episodes:
        if not ep.episode_id or not ep.source_episode_id:
            raise ValueError('Explicit episode and original-session IDs required')
        if ep.split not in {'train', 'calibration', 'test'}:
            raise ValueError('Unknown split')
        if ep.origin != 'offline':
            raise ValueError('Training dataset may contain only offline episodes')
        if ep.episode_id in seen:
            raise ValueError('Duplicate episode')
        seen.add(ep.episode_id)
        previous = groups.setdefault(ep.source_episode_id, ep.split)
        if previous != ep.split:
            raise ValueError('Original session crosses dataset splits')


@dataclass
class MilestoneLedger:
    """One ledger per episode; skill changes never reset global progress."""
    achieved: set[str] = field(default_factory=set)

    def observe(self, milestone: str, *, confirmed: bool) -> float:
        if not milestone:
            raise ValueError('Milestone ID required')
        if not confirmed or milestone in self.achieved:
            return 0.0
        self.achieved.add(milestone)
        return 1.0
