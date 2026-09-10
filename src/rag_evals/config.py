"""Run configuration.

One YAML file per experiment run. The config is the *complete* description of a
row in the results table: two runs with the same config must produce the same
row, and any difference between two rows must be visible as a difference between
their configs. That property is what lets a delta be attributed to a cause.

:meth:`RunConfig.diff` exists to enforce the discipline rule from SPEC section 4
mechanically. ``rag-evals run`` prints the diff against the declared baseline and
warns when more than one variable moved, because a table where three things
changed at once proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .retrieve import RetrievalConfig


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"          # fixed | recursive | section
    target_tokens: int = 512
    overlap_tokens: int = 0


@dataclass
class GenerationConfig:
    enabled: bool = True
    model: str = "claude-opus-5"
    max_context_chars: int = 24000


@dataclass
class RunConfig:
    run_id: str
    description: str = ""
    baseline: str = ""               # run_id this run is a delta against
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        if not path.exists() and not path.is_absolute():
            candidate = paths.CONFIGS / path.name
            if candidate.exists():
                path = candidate
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            run_id=raw.get("run_id") or path.stem,
            description=raw.get("description", ""),
            baseline=raw.get("baseline", ""),
            chunking=ChunkingConfig(**(raw.get("chunking") or {})),
            retrieval=RetrievalConfig(**(raw.get("retrieval") or {})),
            generation=GenerationConfig(**(raw.get("generation") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "description": self.description,
            "baseline": self.baseline,
            "chunking": asdict(self.chunking),
            "retrieval": asdict(self.retrieval),
            "generation": asdict(self.generation),
        }

    def flat(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for section in ("chunking", "retrieval", "generation"):
            for k, v in asdict(getattr(self, section)).items():
                out[f"{section}.{k}"] = v
        return out

    def diff(self, other: "RunConfig") -> dict[str, tuple[Any, Any]]:
        """Fields that differ between two configs, ``{field: (theirs, mine)}``."""
        a, b = other.flat(), self.flat()
        return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


# Fields that do not change what is being measured, only how it is spelled.
_COSMETIC = {"generation.model", "retrieval.embed_model", "retrieval.rerank_model"}


def isolation_warning(cfg: RunConfig, baseline: RunConfig) -> str | None:
    """Warn when a run moved more than one variable against its baseline."""
    changed = {k for k in cfg.diff(baseline) if k not in _COSMETIC}
    # rerank_depth only has meaning when rerank is on; toggling them together
    # is one conceptual change, not two.
    if changed >= {"retrieval.rerank", "retrieval.rerank_depth"}:
        changed.discard("retrieval.rerank_depth")
    if len(changed) <= 1:
        return None
    return (
        f"run {cfg.run_id!r} changes {len(changed)} variables against baseline "
        f"{baseline.run_id!r}: {', '.join(sorted(changed))}. The delta cannot be "
        "attributed to any one of them."
    )
