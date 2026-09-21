# Prompt registry

Every system prompt the harness sends to a model lives here, one directory per
prompt and one file per version:

```
prompts/
  generate/v1.md    <- answer generator
  judge/v1.md       <- LLM judge (the measurement instrument)
  rewrite/v1.md     <- query rewriter
```

The prompts are not inline in the source because a prompt is part of the
measurement instrument. An edit to the judge prompt changes what "grounded"
means for every later row of the results table, and an edit made in a `.py`
file is invisible to that table. Here it is a versioned file, its hash is
written into every run record, and an in-place edit fails the audit.

## The file format

```markdown
---
name: judge
version: v2
status: draft             # draft | active | retired
created: 2026-09-18
owner: Colin Stebelski
purpose: one line, what this prompt is for
changes: what moved against the previous version, and why
variables: [abstain_marker]
---
<the system prompt text; {abstain_marker} is substituted at load time>
```

`name` must match the directory, `version` must match the filename, and every
metadata field is required. The loader rejects a file that breaks any of that.

## What a run records

Every `results/<run_id>.json` written since 0.2.0 carries:

```json
"prompts": {
  "generate": {"version": "v1", "sha256": "704fc3dd...", "path": "prompts/generate/v1.md"},
  "judge":    {"version": "v1", "sha256": "a3989c07...", "path": "prompts/judge/v1.md"},
  "rewrite":  {"version": "v1", "sha256": "31aa3576...", "path": "prompts/rewrite/v1.md"}
}
```

The hash is of the stored text, before variable substitution. Run records from
before 0.2.0 have no `prompts` block; they ran on the text that is now `v1`,
which was extracted verbatim, and `rag-evals prompts audit` reports them as
"pre-versioning, assumed v1" rather than silently backfilling a claim.

## Rules

1. **A version is immutable once any result references it.** To change a
   prompt, add the next version. Never edit a published file in place. This is
   enforced twice: `rag-evals prompts audit` compares every run record's hash to
   the file on disk, and `tests/test_prompts.py` pins the hash of every
   non-draft version.
2. **A version bump is one experimental variable.** `prompts.generate`,
   `prompts.judge` and `prompts.rewrite` are fields of the run config, so the
   isolation rule (SPEC section 4, `rag-evals run` warns on violation) treats a
   prompt change like a chunking or retrieval change. Bumping the judge and the
   generator in the same run attributes the delta to neither.
3. **Every config pins explicit versions.** There is no "latest". A config
   that tracked the newest file would change what it measures every time a
   version was added, and a rollback would have nothing to roll back to.
4. **Draft versions may be run, not published.** A draft can be executed to
   produce numbers to look at. A committed result that depends on a draft fails
   the audit, so promote to `active` first, or drop the result.

## Change control: how a prompt change ships

1. Copy the current version to `v(N+1).md`. Set `status: draft`, fill in
   `changes` with what moved and why, and edit the text.
2. Add a config that pins the new version and names the config with the old
   version as its `baseline`, so exactly one variable moves:

   ```yaml
   run_id: 015-judge-v2
   description: Judge prompt v2: tighter "material qualifier" rule for correctness.
   baseline: 008-retrieval-hybrid
   # chunking / retrieval / generation identical to 008
   prompts:
     generate: v1
     judge: v2
     rewrite: v1
   ```

3. `rag-evals run configs/015-judge-v2.yaml`. The run prints the pinned
   versions and hashes, warns that `judge/v2` is a draft, and records both in
   the result.
4. For a judge change specifically, re-run `rag-evals validate-judge` against
   the new run. A judge prompt that moves the numbers but not the agreement
   with hand labels has not improved anything; it has only changed the meaning
   of the score.
5. Compare in `rag-evals report`. If the change earns its place, set the new
   version to `active`, the old one to `retired`, add the new version's hash to
   the frozen table in `tests/test_prompts.py`, and record the change in
   `CHANGELOG.md` under "Prompts". If it does not, leave the file in place as a
   draft with the finding noted in `changes`, or delete it. A disproved idea
   with its result attached is worth keeping.
6. `rag-evals prompts audit` must be clean before the commit.

## Rollback

A rollback is a config edit, and it costs nothing.

1. In the affected config, set `prompts.<name>` back to the previous version
   (or `git revert` the commit that bumped it).
2. Re-run the config. The LLM cache key includes the full system prompt text,
   so every call for a version that already has results is a cache hit: the
   re-run makes zero API calls, costs $0, and reproduces the earlier row
   exactly. `usage.cache_hits` in the new record shows it.
3. Set the rolled-back version's `status` to `retired` if it should not be
   used for new findings. Retired files stay on disk forever, because old run
   records point at them by hash and the table is only reproducible while the
   text they ran on still exists.
4. Record the rollback and the reason in `CHANGELOG.md`.

To see what changed between two versions before deciding:

```bash
rag-evals prompts list                  # every version, status, hash, and which runs used it
rag-evals prompts diff judge v1 v2      # unified diff of the text, with the v2 "changes" note
rag-evals prompts show judge v1         # metadata and full text
rag-evals prompts audit                 # exits 1 if any record, file, or config disagree
```
