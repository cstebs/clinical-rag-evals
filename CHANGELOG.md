# Changelog

All notable changes to this harness. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each release has a
**Prompts** heading in addition to the usual ones, because a prompt is part of
the measurement instrument and a change to one is a change to what the numbers
mean. Versioned prompt files live in `prompts/`; see `prompts/README.md` for
the change-control and rollback procedure.

## [Unreleased]

## [0.2.0] - 2026-09-18

### Added
- `prompts/` registry. Every system prompt is a versioned file with front
  matter (`name`, `version`, `status`, `created`, `owner`, `purpose`,
  `changes`, `variables`). `src/rag_evals/prompts.py` loads, renders, hashes
  and audits them.
- `prompts:` section in every run config, pinning one explicit version per
  prompt. There is deliberately no "latest". Prompt versions participate in
  `RunConfig.diff` and the isolation warning, so a version bump is one
  experimental variable.
- Every run record now carries a `prompts` block: version, SHA-256 of the
  stored text, and path, for each of the three prompts. A row can no longer
  drift from the text that produced it.
- `rag-evals prompts list | show | diff | audit`. `audit` exits non-zero if any
  committed run record's prompt hash no longer matches the file on disk, if a
  result depends on a draft version, or if a config pins a version that does
  not exist.
- `tests/test_prompts.py`: pins the hash of every published prompt version
  (an in-place edit fails the suite), covers loading and rendering, the
  isolation rule applied to prompt bumps, the shipped configs, and the audit.
- Documented rollback procedure (`prompts/README.md`). Because the LLM cache
  key already included the system prompt text, rolling a config back to a
  previously run prompt version reproduces the earlier row with zero API
  calls.
- This changelog.

### Changed
- `generate.py`, `judge.py` and `retrieve.py` load their system prompts from
  `prompts/` instead of module-level string constants. `build_call`,
  `generate_batch`, `judge_batch`, `QueryRewriter` and `Retriever` accept the
  prompt to use; with none given they fall back to v1, which is what every
  pre-versioning result ran on.
- `rag-evals run` prints the pinned prompt versions and short hashes at the
  start of every run, and warns when a pinned version is a draft or retired.

### Prompts
- `generate/v1`, `judge/v1`, `rewrite/v1`: extracted verbatim from the inline
  constants. No text changed. All fourteen existing result rows, the 40 hand
  labels, and the judge-validation numbers remain valid against these files.
  Pre-0.2.0 run records carry no `prompts` block and are reported by the audit
  as "assumed v1" rather than backfilled.

## [0.1.0] - 2026-09-10

Initial public release: corpus ingestion for 124 FDA guidance documents,
three chunkers, BM25 / dense / hybrid retrieval with optional cross-encoder
reranking and LLM query rewriting, an 85-question gold set with four question
types, an LLM judge with stored reasoning, judge validation against hand
labels with Cohen's kappa, a 14-configuration experiment matrix, per-run cost
accounting, and the results table in the README.

### Fixed, before release (recorded here so the history is not lost)
- The section-aware chunker's heading detector reported 98% coverage while
  emitting spurious chunks from addresses, footnote markers, dates, numbered
  list items and table-of-contents lines. Fixing it moved three of five
  published findings; the README documents the correction and the test suite
  now checks the detector against each class of junk that fooled it.
- `QueryRewriter` swallowed API failures and fell back to the raw query, which
  produced a row labelled `+rewrite` in which no rewriting had happened. The
  handler was narrowed so only a malformed rewrite response falls back and any
  API failure aborts the run. Per-run cost accounting was the tripwire.
- `REWRITE_SCHEMA` used `minItems` / `maxItems`, which structured outputs
  reject. Removed; the prompt asks for 2 to 4 alternates and the code slices.
- Requests from an identity-linked API key were rejected without a workspace
  header. `llm.py` sends `anthropic-workspace-id` when
  `ANTHROPIC_WORKSPACE_ID` is set.
