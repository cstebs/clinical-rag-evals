"""Versioned prompts: immutability, loading, config discipline, and the audit."""

import json
from pathlib import Path

import pytest

from rag_evals import paths
from rag_evals import prompts as pm
from rag_evals.config import RunConfig, isolation_warning
from rag_evals.generate import ABSTAIN_MARKER
from rag_evals.generate import build_call as build_generate_call
from rag_evals.generate import system_prompt as generate_system
from rag_evals.judge import system_prompt as judge_system
from rag_evals.prompts import PromptsConfig
from rag_evals.retrieve import RetrievalConfig, rewrite_system_prompt

# ---------------------------------------------------------------------------
# immutability
# ---------------------------------------------------------------------------

# SHA-256 of the stored text of every published (non-draft) prompt version.
# A published version is immutable once a result references it. If a test here
# fails, do not update the hash: restore the file and put the new text in a new
# version. Add a row only when promoting a draft to active.
FROZEN = {
    ("generate", "v1"): "704fc3ddbfc7cd98c60fd999fe4d37d112b29e260a7f0e9be3abdd89f8b8ab14",
    ("judge", "v1"): "a3989c0798db40f8b47ec1e56e95bee067894c8dad58afc5617c519f2cba5cf5",
    ("rewrite", "v1"): "31aa3576544c7fd0083f9ad3db5f79c510b688b876b438333af0ff702be42bc7",
}


class TestImmutability:
    def test_published_versions_are_frozen(self):
        for (name, version), sha in FROZEN.items():
            p = pm.load(name, version)
            assert p.sha256 == sha, (
                f"{name}/{version} was edited in place. Published prompt versions are "
                "immutable; restore it and add a new version."
            )

    def test_every_non_draft_version_is_pinned(self):
        for p in pm.all_prompts():
            if p.status != "draft":
                assert (p.name, p.version) in FROZEN, (
                    f"{p.name}/{p.version} is {p.status} but its hash is not pinned in "
                    "tests/test_prompts.py FROZEN"
                )

    def test_every_prompt_name_has_an_active_version(self):
        for name in pm.PROMPT_NAMES:
            assert any(p.status == "active" for p in pm.versions(name)), name


# ---------------------------------------------------------------------------
# loading and rendering
# ---------------------------------------------------------------------------


def _write_prompt(root: Path, name: str, version: str, body: str, status: str = "active", **meta):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = {
        "name": name,
        "version": version,
        "status": status,
        "created": "2026-09-18",
        "owner": "test",
        "purpose": "test prompt",
        "changes": "test",
        **meta,
    }
    lines = ["---"] + [f"{k}: {json.dumps(v) if isinstance(v, list) else v}" for k, v in fm.items()] + ["---", "", body, ""]
    (d / f"{version}.md").write_text("\n".join(lines), encoding="utf-8")
    return d / f"{version}.md"


class TestLoading:
    def test_front_matter_and_body(self):
        p = pm.load("judge", "v1")
        assert p.name == "judge" and p.version == "v1" and p.status == "active"
        assert p.meta["owner"] and p.meta["purpose"] and p.meta["changes"]
        assert "{abstain_marker}" in p.text
        assert p.variables == ["abstain_marker"]

    def test_render_substitutes_declared_variable(self):
        text = pm.load("judge", "v1").render(abstain_marker=ABSTAIN_MARKER)
        assert ABSTAIN_MARKER in text
        assert "{abstain_marker}" not in text

    def test_render_refuses_a_half_rendered_prompt(self):
        with pytest.raises(pm.PromptError):
            pm.load("generate", "v1").render()

    def test_unknown_version_and_name(self):
        with pytest.raises(pm.PromptError):
            pm.load("judge", "v999")
        with pytest.raises(pm.PromptError):
            pm.load("nope", "v1")

    def test_record_has_version_hash_and_path(self):
        rec = pm.load("generate", "v1").record()
        assert rec["version"] == "v1"
        assert rec["sha256"] == FROZEN[("generate", "v1")]
        assert rec["path"] == "prompts/generate/v1.md"

    def test_versions_sort_numerically(self, tmp_path):
        for v in ("v10", "v2", "v1"):
            _write_prompt(tmp_path, "judge", v, f"body {v}")
        assert [p.version for p in pm.versions("judge", tmp_path)] == ["v1", "v2", "v10"]

    def test_metadata_mismatch_is_rejected(self, tmp_path):
        f = _write_prompt(tmp_path, "judge", "v1", "body")
        f.rename(f.with_name("v2.md"))
        with pytest.raises(pm.PromptError):
            pm.load("judge", "v2", tmp_path)

    def test_bad_status_is_rejected(self, tmp_path):
        _write_prompt(tmp_path, "judge", "v1", "body", status="shipped")
        with pytest.raises(pm.PromptError):
            pm.load("judge", "v1", tmp_path)

    def test_missing_front_matter_is_rejected(self, tmp_path):
        d = tmp_path / "judge"
        d.mkdir()
        (d / "v1.md").write_text("just a prompt with no front matter", encoding="utf-8")
        with pytest.raises(pm.PromptError):
            pm.load("judge", "v1", tmp_path)


class TestPipelineUsesVersionedText:
    def test_generate_call_uses_versioned_text(self):
        call = build_generate_call("Q?", [], 1000)
        assert call.system == pm.load("generate", "v1").render(abstain_marker=ABSTAIN_MARKER)

    def test_default_prompts_are_v1(self):
        assert generate_system() == pm.load("generate", "v1").render(abstain_marker=ABSTAIN_MARKER)
        assert judge_system() == pm.load("judge", "v1").render(abstain_marker=ABSTAIN_MARKER)
        assert rewrite_system_prompt() == pm.load("rewrite", "v1").render()

    def test_judge_prompt_is_recognised_by_the_offline_stub(self):
        # cli._stub decides "judge or generator" by looking for "grading" in the
        # system text. A judge version that drops the word would silently make
        # the stub answer as a generator.
        for p in pm.versions("judge"):
            assert "grading" in p.text, p.version

    def test_abstention_marker_appears_exactly_once_in_each(self):
        for name in ("generate", "judge"):
            for p in pm.versions(name):
                assert p.text.count("{abstain_marker}") == 1, f"{name}/{p.version}"


# ---------------------------------------------------------------------------
# config discipline
# ---------------------------------------------------------------------------


class TestConfigDiscipline:
    def test_prompt_bump_is_one_variable(self):
        base = RunConfig(run_id="a")
        cfg = RunConfig(run_id="b", prompts=PromptsConfig(judge="v2"))
        assert isolation_warning(cfg, base) is None
        assert cfg.diff(base) == {"prompts.judge": ("v1", "v2")}

    def test_two_prompt_bumps_warn(self):
        base = RunConfig(run_id="a")
        cfg = RunConfig(run_id="b", prompts=PromptsConfig(judge="v2", generate="v2"))
        warn = isolation_warning(cfg, base)
        assert warn and "2 variables" in warn

    def test_prompt_bump_plus_retrieval_change_warns(self):
        base = RunConfig(run_id="a")
        cfg = RunConfig(
            run_id="b",
            retrieval=RetrievalConfig(index="hybrid"),
            prompts=PromptsConfig(judge="v2"),
        )
        warn = isolation_warning(cfg, base)
        assert warn and "prompts.judge" in warn and "retrieval.index" in warn

    def test_yaml_roundtrip(self, tmp_path):
        f = tmp_path / "x.yaml"
        f.write_text("run_id: x\nprompts:\n  judge: v3\n", encoding="utf-8")
        cfg = RunConfig.load(f)
        assert cfg.prompts == PromptsConfig(generate="v1", judge="v3", rewrite="v1")
        assert cfg.to_dict()["prompts"] == {"generate": "v1", "judge": "v3", "rewrite": "v1"}
        assert cfg.flat()["prompts.judge"] == "v3"

    def test_shipped_configs_pin_published_versions(self):
        cfgs = sorted(paths.CONFIGS.glob("*.yaml"))
        assert cfgs
        for path in cfgs:
            cfg = RunConfig.load(path)
            for name in pm.PROMPT_NAMES:
                p = pm.load(name, getattr(cfg.prompts, name))
                assert p.status != "draft", f"{path.name} pins draft {name}/{p.version}"

    def test_prompt_set_resolves_and_records(self):
        ps = pm.PromptSet.from_config(PromptsConfig())
        rec = ps.record()
        assert set(rec) == set(pm.PROMPT_NAMES)
        assert rec["judge"]["sha256"] == FROZEN[("judge", "v1")]
        assert ps.warnings() == []


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def _write_result(results: Path, run_id: str, prompts_block, stub: bool = False):
    results.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "retrieval": {}}
    if prompts_block is not None:
        payload["prompts"] = prompts_block
    if stub:
        payload["stub_llm"] = True
    (results / f"{run_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _fixture_tree(tmp_path: Path):
    prompts = tmp_path / "prompts"
    for name in pm.PROMPT_NAMES:
        _write_prompt(prompts, name, "v1", f"{name} body v1")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "001.yaml").write_text("run_id: '001'\nprompts:\n  judge: v1\n", encoding="utf-8")
    results = tmp_path / "results"
    return prompts, configs, results


class TestAudit:
    def test_repo_audits_clean(self):
        problems, notes = pm.audit()
        assert problems == []
        # Every pre-versioning row is named, not silently backfilled.
        assert all("assumed v1" in n for n in notes)

    def test_consistent_tree_is_clean(self, tmp_path):
        prompts, configs, results = _fixture_tree(tmp_path)
        good = pm.PromptSet.from_config(PromptsConfig(), prompts).record()
        _write_result(results, "001", good)
        problems, notes = pm.audit(prompts, results, configs)
        assert problems == [] and notes == []

    def test_in_place_edit_after_a_run_is_caught(self, tmp_path):
        prompts, configs, results = _fixture_tree(tmp_path)
        _write_result(results, "001", pm.PromptSet.from_config(PromptsConfig(), prompts).record())
        _write_prompt(prompts, "judge", "v1", "judge body v1 but edited")   # the sin
        problems, _ = pm.audit(prompts, results, configs)
        assert any("001" in p and "judge/v1" in p and "edited after this run" in p for p in problems)

    def test_missing_version_referenced_by_result_is_caught(self, tmp_path):
        prompts, configs, results = _fixture_tree(tmp_path)
        block = pm.PromptSet.from_config(PromptsConfig(), prompts).record()
        block["judge"]["version"] = "v9"
        _write_result(results, "001", block)
        problems, _ = pm.audit(prompts, results, configs)
        assert any("judge/v9" in p and "not on disk" in p for p in problems)

    def test_result_depending_on_a_draft_is_caught_unless_stubbed(self, tmp_path):
        prompts, configs, results = _fixture_tree(tmp_path)
        _write_prompt(prompts, "judge", "v2", "judge body v2", status="draft")
        block = pm.PromptSet.from_config(PromptsConfig(judge="v2"), prompts).record()
        _write_result(results, "001", block)
        problems, _ = pm.audit(prompts, results, configs)
        assert any("draft" in p for p in problems)
        _write_result(results, "001", block, stub=True)
        problems, _ = pm.audit(prompts, results, configs)
        assert not any("draft" in p for p in problems)

    def test_config_pinning_a_missing_version_is_caught(self, tmp_path):
        prompts, configs, results = _fixture_tree(tmp_path)
        (configs / "002.yaml").write_text("run_id: '002'\nprompts:\n  generate: v7\n", encoding="utf-8")
        problems, _ = pm.audit(prompts, results, configs)
        assert any("002.yaml" in p and "generate/v7" in p for p in problems)

    def test_pre_versioning_record_is_a_note_not_a_problem(self, tmp_path):
        prompts, configs, results = _fixture_tree(tmp_path)
        _write_result(results, "000", None)
        problems, notes = pm.audit(prompts, results, configs)
        assert problems == []
        assert notes and "000" in notes[0] and "assumed v1" in notes[0]

    def test_usage_by_results(self, tmp_path):
        prompts, configs, results = _fixture_tree(tmp_path)
        _write_result(results, "001", pm.PromptSet.from_config(PromptsConfig(), prompts).record())
        _write_result(results, "000", None)
        used = pm.usage_by_results(results)
        assert used[("judge", "v1")] == ["001"]
