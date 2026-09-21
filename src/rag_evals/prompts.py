"""Versioned system prompts, and the audit that keeps them honest.

A prompt is part of the measurement instrument. Changing the judge prompt
changes what "grounded" means; changing the generator prompt changes what is
being graded. Neither belongs inline in a ``.py`` file, where an edit is
invisible to the results table. So every system prompt lives in
``prompts/<name>/<version>.md`` with front matter, and every run record writes
down the version *and the content hash* it ran with.

Three rules, each enforced mechanically rather than by convention:

1. **A version is immutable once a result references it.** ``rag-evals prompts
   audit`` compares the hash in every committed run record against the file on
   disk, and ``tests/test_prompts.py`` pins the hash of every published
   version. To change a prompt, add ``v(N+1)``. Do not edit ``vN``.
2. **A prompt bump is a variable.** ``prompts.judge`` and friends are part of
   :class:`~rag_evals.config.RunConfig`, so
   :func:`~rag_evals.config.isolation_warning` counts a version change like
   any other knob. Bumping the judge and the generator in one run proves
   nothing about either.
3. **Rollback is a config edit, and it is free.** The LLM cache key includes
   the system prompt text, so re-running a config pinned to a version that
   already has results costs $0 and reproduces the earlier row exactly.

File format::

    ---
    name: judge
    version: v2
    status: active        # draft | active | retired
    created: 2026-09-18
    owner: who is accountable for this text
    purpose: one line
    changes: what moved against the previous version, and why
    variables: [abstain_marker]
    ---
    <system prompt text; {abstain_marker} is substituted at load time>
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import paths

PROMPT_NAMES = ("generate", "judge", "rewrite")
STATUSES = ("draft", "active", "retired")

_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)
_VERSION = re.compile(r"v(\d+)")


class PromptError(ValueError):
    """A prompt file is missing, malformed, or used in a way its metadata forbids."""


@dataclass(frozen=True)
class PromptsConfig:
    """The ``prompts:`` section of a run config: one pinned version per prompt.

    Defaults are explicit versions rather than "latest" on purpose. A config
    that silently tracked the newest prompt would change what it measures every
    time someone added a file, and a rollback would have nothing to roll back
    to.
    """

    generate: str = "v1"
    judge: str = "v1"
    rewrite: str = "v1"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str                 # the body as stored, placeholders unrendered
    meta: dict[str, Any]
    path: Path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def status(self) -> str:
        return str(self.meta.get("status", "draft"))

    @property
    def variables(self) -> list[str]:
        v = self.meta.get("variables") or []
        return [str(x) for x in v]

    def render(self, **variables: Any) -> str:
        """Substitute declared placeholders. Refuses to hand back a half-rendered prompt."""
        out = self.text
        for k, v in variables.items():
            out = out.replace("{" + k + "}", str(v))
        missing = [v for v in self.variables if "{" + v + "}" in out]
        if missing:
            raise PromptError(
                f"{self.name}/{self.version} declares variables {missing} that were not supplied"
            )
        return out

    def record(self) -> dict[str, Any]:
        """What a run record stores: enough to detect drift and to roll back."""
        try:
            rel = self.path.resolve().relative_to(paths.ROOT.resolve()).as_posix()
        except ValueError:
            rel = f"prompts/{self.name}/{self.version}.md"
        return {"version": self.version, "sha256": self.sha256, "path": rel}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _version_key(version: str) -> tuple[int, Any]:
    m = _VERSION.fullmatch(version)
    return (0, int(m.group(1))) if m else (1, version)


def parse(path: str | Path) -> Prompt:
    path = Path(path)
    if not path.exists():
        raise PromptError(f"prompt file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    m = _FRONT_MATTER.match(raw)
    if not m:
        raise PromptError(f"{path}: missing YAML front matter (--- ... ---)")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise PromptError(f"{path}: bad front matter: {exc}") from None
    if not isinstance(meta, dict):
        raise PromptError(f"{path}: front matter must be a mapping")
    body = raw[m.end():].strip("\r\n")

    name = str(meta.get("name", ""))
    version = str(meta.get("version", ""))
    if name != path.parent.name:
        raise PromptError(f"{path}: front matter name {name!r} does not match directory {path.parent.name!r}")
    if version != path.stem:
        raise PromptError(f"{path}: front matter version {version!r} does not match filename {path.stem!r}")
    if meta.get("status", "draft") not in STATUSES:
        raise PromptError(f"{path}: status must be one of {STATUSES}, got {meta.get('status')!r}")
    for key in ("purpose", "changes", "created", "owner"):
        if not meta.get(key):
            raise PromptError(f"{path}: front matter is missing {key!r}")
    if not body:
        raise PromptError(f"{path}: empty prompt body")
    return Prompt(name=name, version=version, text=body, meta=meta, path=path)


def prompts_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root is not None else paths.PROMPTS


def load(name: str, version: str, root: str | Path | None = None) -> Prompt:
    if name not in PROMPT_NAMES:
        raise PromptError(f"unknown prompt {name!r}; expected one of {PROMPT_NAMES}")
    return parse(prompts_dir(root) / name / f"{version}.md")


def versions(name: str, root: str | Path | None = None) -> list[Prompt]:
    """Every version of one prompt, oldest first."""
    d = prompts_dir(root) / name
    files = sorted(d.glob("v*.md"), key=lambda p: _version_key(p.stem)) if d.exists() else []
    return [parse(p) for p in files]


def all_prompts(root: str | Path | None = None) -> list[Prompt]:
    out: list[Prompt] = []
    for name in PROMPT_NAMES:
        out.extend(versions(name, root))
    return out


@dataclass(frozen=True)
class PromptSet:
    """The three prompts a run actually uses, resolved from its config."""

    generate: Prompt
    judge: Prompt
    rewrite: Prompt

    @classmethod
    def from_config(cls, cfg: PromptsConfig, root: str | Path | None = None) -> "PromptSet":
        return cls(
            generate=load("generate", cfg.generate, root),
            judge=load("judge", cfg.judge, root),
            rewrite=load("rewrite", cfg.rewrite, root),
        )

    def __iter__(self):
        yield from (self.generate, self.judge, self.rewrite)

    def record(self) -> dict[str, dict[str, Any]]:
        return {p.name: p.record() for p in self}

    def warnings(self) -> list[str]:
        """Draft and retired versions may be run, but never quietly."""
        out = []
        for p in self:
            if p.status == "draft":
                out.append(
                    f"prompt {p.name}/{p.version} is a draft. Promote it to active before "
                    "committing a result that depends on it."
                )
            elif p.status == "retired":
                out.append(
                    f"prompt {p.name}/{p.version} is retired. Fine for a rollback or a "
                    "reproduction; not for a new finding."
                )
        return out


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def _run_records(results_dir: Path) -> list[Path]:
    """Run records only. Judge-validation reports live alongside them but are
    derived from a run, not runs themselves, and carry no prompt block."""
    if not results_dir.exists():
        return []
    return sorted(p for p in results_dir.glob("*.json") if not p.stem.startswith("judge-validation"))


def _read_result_prompts(path: Path) -> tuple[str, dict[str, Any] | None, bool]:
    """(run_id, prompts block or None, stub_llm flag) from one run record."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("run_id", path.stem)), data.get("prompts"), bool(data.get("stub_llm"))


def usage_by_results(results_dir: str | Path | None = None) -> dict[tuple[str, str], list[str]]:
    """Which runs used which prompt version, from the committed run records."""
    d = Path(results_dir) if results_dir is not None else paths.RESULTS
    used: dict[tuple[str, str], list[str]] = {}
    for p in _run_records(d):
        run_id, block, _stub = _read_result_prompts(p)
        if not block:
            continue
        for name, rec in block.items():
            used.setdefault((name, str(rec.get("version"))), []).append(run_id)
    return used


def audit(
    prompts_root: str | Path | None = None,
    results_dir: str | Path | None = None,
    configs_dir: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Check that the prompt files, the run records, and the configs agree.

    Returns ``(problems, notes)``. A problem is something that makes a committed
    number untrustworthy: a run record whose prompt hash no longer matches the
    file on disk (someone edited a version in place), a record that references
    a version that no longer exists, a result that depends on a draft, or a
    config pinned to a missing version. Notes are informational: pre-versioning
    run records, which carry no prompt block and are assumed to have run on v1
    because v1 is a verbatim extraction of the text that was inline at the time.
    """
    problems: list[str] = []
    notes: list[str] = []

    # 1. every prompt file parses and its metadata is coherent
    on_disk: dict[tuple[str, str], Prompt] = {}
    for name in PROMPT_NAMES:
        d = prompts_dir(prompts_root) / name
        if not d.exists():
            problems.append(f"prompts/{name}/ is missing")
            continue
        for f in sorted(d.glob("*.md")):
            try:
                p = parse(f)
            except PromptError as exc:
                problems.append(str(exc))
                continue
            on_disk[(p.name, p.version)] = p
        if not any(k[0] == name for k in on_disk):
            problems.append(f"prompts/{name}/ has no versions")

    # 2. every run record's recorded hash matches the file on disk
    rd = Path(results_dir) if results_dir is not None else paths.RESULTS
    for f in _run_records(rd):
        try:
            run_id, block, stub = _read_result_prompts(f)
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"{f.name}: unreadable run record ({exc})")
            continue
        if block is None:
            notes.append(f"{run_id}: pre-versioning record, no prompts block; assumed v1")
            continue
        for name, rec in block.items():
            key = (name, str(rec.get("version")))
            p = on_disk.get(key)
            if p is None:
                problems.append(f"{run_id}: references {name}/{key[1]}, which is not on disk")
                continue
            if rec.get("sha256") != p.sha256:
                problems.append(
                    f"{run_id}: {name}/{p.version} was edited after this run. Recorded "
                    f"{str(rec.get('sha256'))[:12]}, on disk {p.sha256[:12]}. Restore the file "
                    "and put the new text in a new version."
                )
            if p.status == "draft" and not stub:
                problems.append(f"{run_id}: depends on draft prompt {name}/{p.version}; promote it or drop the result")

    # 3. every config pins versions that exist
    cd = Path(configs_dir) if configs_dir is not None else paths.CONFIGS
    for f in sorted(cd.glob("*.yaml")) if cd.exists() else []:
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        pinned = raw.get("prompts") or {}
        for name in PROMPT_NAMES:
            version = str(pinned.get(name, PromptsConfig().__dict__[name]))
            if (name, version) not in on_disk:
                problems.append(f"{f.name}: pins {name}/{version}, which is not on disk")

    return problems, notes
