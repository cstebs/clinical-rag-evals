"""The results figure must regenerate from the committed run records.

The figure in the README is a build artefact of ``results/``. If this test
fails, the figure and the table can drift apart, which is the one failure mode
a generated figure is supposed to rule out.
"""

from pathlib import Path

import pytest

from rag_evals import plot, report

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_figure_renders_from_committed_results(tmp_path):
    pytest.importorskip("matplotlib")
    runs = report.load_runs(RESULTS)
    out = plot.plot_retrieval(runs, out=tmp_path / "fig" / "retrieval.png")
    assert out.exists()
    assert out.stat().st_size > 20_000  # a blank canvas is a few KB


def test_figure_refuses_empty_input(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(SystemExit):
        plot.plot_retrieval([], out=tmp_path / "retrieval.png")
