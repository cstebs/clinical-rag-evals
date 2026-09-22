"""The version string lives in two places; they must agree.

Also pins the shape of the ``code`` provenance block every run record carries.
"""

import tomllib
from pathlib import Path

import rag_evals
from rag_evals.run import code_provenance

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert rag_evals.__version__ == declared


def test_code_provenance_shape():
    block = code_provenance()
    assert set(block) == {"git_sha", "dirty", "version"}
    assert block["version"] == rag_evals.__version__
    # In a checkout the SHA is 40 hex chars; outside one it is None, never "".
    if block["git_sha"] is not None:
        assert len(block["git_sha"]) == 40 and int(block["git_sha"], 16) >= 0
        assert isinstance(block["dirty"], bool)
    else:
        assert block["dirty"] is None
