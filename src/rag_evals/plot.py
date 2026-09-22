"""Render the retrieval results as a figure.

The README table is the deliverable. This figure is the same numbers in a form
a reader can take in at a glance: one dot per run, two panels sharing the run
axis, the deliberately weak baseline drawn as a reference line with its noise
band. It is generated from the committed run records and never hand-edited,
so it cannot drift from the table any more than the table can drift from the
data.

Design notes, so the choices are not re-litigated:

- Dots, not bars. The values cluster between 0.7 and 0.95, and a bar chart
  must start at zero, which would flatten the differences the table exists to
  show. Dots carry no implied length, so the axis can be cropped honestly.
- One hue. Every run is the same kind of thing, so they share one colour;
  only the baseline is set apart, in gray. A rainbow of fourteen would say
  nothing that the run labels do not already say.
- The noise band is ±0.02 around the baseline value, the figure the README
  gives for "about one and a half questions changing their mind" at n=73.
  It applies to recall@5 directly and is drawn at the same width on nDCG@10
  as a visual reminder that small deltas there are also small.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import paths, report

FIGURE = paths.ROOT / "docs" / "figures" / "retrieval.png"

PANELS = [
    ("recall@5", "recall@5", "fraction of gold documents in the top 5 chunks"),
    ("ndcg@10", "nDCG@10", "ranking quality over the top 10, deduplicated by document"),
]

NOISE = 0.02

# Palette: one data hue plus neutrals. Text never wears the series colour.
SERIES = "#2a78d6"
BASELINE = "#9a9994"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"
BAND = "#f0efec"


def plot_retrieval(
    runs: list[dict[str, Any]],
    out: Path = FIGURE,
    baseline_id: str = "001-baseline",
) -> Path:
    """Write the retrieval figure for ``runs`` to ``out`` and return the path.

    matplotlib is imported lazily so the rest of the harness does not pay for
    it; install with ``pip install -e ".[plot]"``.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "matplotlib is not installed; run: pip install -e \".[plot]\""
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in runs if r.get("retrieval")]
    if not runs:
        raise SystemExit("no run records with retrieval metrics found")
    runs.sort(key=lambda r: r["run_id"])
    baseline = next((r for r in runs if r["run_id"] == baseline_id), None)
    n_scored = runs[0]["retrieval"].get("n_scored")

    labels = [f"{r['run_id']}   {report._label(r)}" for r in runs]
    y = list(range(len(runs)))[::-1]  # first run at the top

    fig, axes = plt.subplots(
        1, len(PANELS), figsize=(11.5, 0.42 * len(runs) + 1.9), sharey=True,
        dpi=160, facecolor=SURFACE,
    )
    for ax, (key, title, subtitle) in zip(axes, PANELS):
        values = [r["retrieval"][key] for r in runs]
        ax.set_facecolor(SURFACE)

        if baseline is not None:
            b = baseline["retrieval"][key]
            ax.axvspan(b - NOISE, b + NOISE, color=BAND, zorder=0)
            ax.axvline(b, color=INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=1, zorder=1)

        for yi, v, r in zip(y, values, runs):
            is_base = r["run_id"] == baseline_id
            ax.plot(
                [v], [yi], marker="o", markersize=8,
                color=BASELINE if is_base else SERIES,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
            )
            ax.annotate(
                f"{v:.3f}", (v, yi), xytext=(8, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8, color=INK_SECONDARY, zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2},
            )

        lo, hi = min(values), max(values)
        pad = max(0.03, (hi - lo) * 0.18)
        ax.set_xlim(lo - pad, hi + pad + 0.03)
        ax.set_ylim(-0.7, len(runs) - 0.3)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=14)
        ax.text(
            0, 1.005, subtitle, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
            va="bottom", ha="left",
        )
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", labelsize=8, colors=INK_SECONDARY, length=0)
        ax.tick_params(axis="y", labelsize=8.5, colors=INK, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, family="monospace")

    scored = f"{n_scored} scored questions, negatives excluded" if n_scored else ""
    fig.suptitle(
        "Retrieval results: one variable changed per run",
        x=0.01, y=0.985, ha="left", fontsize=13, color=INK, fontweight="bold",
    )
    base_note = (
        f"dashed line = {baseline_id} ({report._label(baseline)}); "
        f"shaded band = ±{NOISE:.2f}, about the noise floor at this sample size"
        if baseline is not None else ""
    )
    fig.text(0.01, 0.945, "; ".join(s for s in (scored, base_note) if s),
             fontsize=8.5, color=INK_SECONDARY, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.935))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
