"""The Model tab's inspect/compare panel: what it shows without being asked.

The panel replaced a three-tab viewer in commit 8f53b1a0 and inherited none of its
restraint: eleven widgets, five prose captions and a full per-marker table stood between
the operator and the picture they came to look at. These tests pin the layout down.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: restore the marker toggle
# lost with _overlay_frame, and keep the compare controls from creeping back | date: 2026-09-16]

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

LABELER = Path(__file__).resolve().parents[1] / "nems_lbhb/motor/streamlit_vid_labeler"

PARTS = ["headpost", "front", "left_ear", "right_ear", "rear"]

# rows in dlc_headfix_qc.RIGID_ROLES order: front, headpost, leftear, rightear. The
# ferret glyph anchors on the front-to-headpost span, so this cannot be all zeros.
HEAD_SHAPE = np.array([[0.0, 0.5], [0.0, -0.5], [-0.4, 0.0], [0.4, 0.0]])


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.fixture
def core(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    module = importlib.import_module("labeler_core")
    monkeypatch.setattr(module, "PREFS_FILE", tmp_path / "prefs.json")
    return module


@pytest.fixture
def inspection(core, monkeypatch):
    module = importlib.import_module("fit_inspection")
    monkeypatch.setattr(module, "inventory", lambda *a, **k: [])
    return module


def result(seed=0, has_reference=True):
    """A stand-in for dlc_headfix_qc.SpreadResult carrying only what the plots read."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(
        {"role": ["rigid"] * 4 + ["body"],
         "ratio_spread": [1.1, 1.3, 0.9, 1.0, np.nan],
         "ratio_offset": [0.2, 0.3, 0.1, 0.2, np.nan],
         "rms_spread": [0.0, 0.21, 0.24, 0.19, 0.55],
         "rms_spread_proc": [0.18, 0.21, 0.24, 0.19, 0.55],
         "frac_tracked": [0.99, 0.98, 0.97, 0.96, 0.80],
         "p_median": [0.99, 0.98, 0.97, 0.96, 0.70],
         "flag": ["ok", "warn", "ok", "ok", ""],
         "flag_track": ["ok", "ok", "ok", "ok", "warn"]},
        index=PARTS)
    scatter = {p: rng.normal(0, 0.2, size=(40, 2)) for p in PARTS}
    return SimpleNamespace(table=table, has_reference=has_reference, reference=None,
                           scatter_proc=scatter, scatter_aligned=None, scatter_hf=scatter,
                           warnings=[], scale_px=400.0, origin_part="headpost",
                           mean_shape=HEAD_SHAPE, frac_valid=0.9,
                           n_valid=90, n_total=100)


# ── the regression: a toggle for the markers ──────────────────────────────────

def test_the_dead_overlay_frame_is_gone_from_the_model_tab(core):
    # commit 8f53b1a0 stopped calling it but left 40 lines behind, which read as live
    model = importlib.import_module("tabs.model")
    assert not hasattr(model, "_overlay_frame")
    source = (LABELER / "tabs/model.py").read_text()
    assert "_overlay_frame" not in source


def test_the_frame_view_can_turn_its_markers_off(core, inspection, monkeypatch):
    """Overlaying unconditionally leaves no way to see the frame underneath.

    That is the whole point of looking at a frame when a fit looks wrong -- is the
    marker in the wrong place, or is the animal?
    """
    assert hasattr(inspection, "markers_wanted")

    state = State(inspect_markers=True)
    monkeypatch.setattr(core.st, "session_state", state)
    assert inspection.markers_wanted() is True

    state["inspect_markers"] = False
    assert inspection.markers_wanted() is False


def test_markers_default_to_on(core, inspection, monkeypatch):
    # the overlay is why the view exists; off is the deliberate act, not the default
    monkeypatch.setattr(core.st, "session_state", State())
    assert inspection.markers_wanted() is True


# ── A and B in one figure ─────────────────────────────────────────────────────

def test_one_scatter_carries_both_fits(core):
    fig = core.hfqc_compare_figure([("A", result(0)), ("B", result(1))])
    names = [t.name or "" for t in fig.data]
    assert any(n.startswith("A ") for n in names)
    assert any(n.startswith("B ") for n in names)


def points_of(fig, side=None):
    """Marker colours of the point clouds (not the quad, which also carries markers)."""
    out = set()
    for t in fig.data:
        name = t.name or ""
        if "head shape" in name or "2sd" in name:
            continue
        if side and not name.startswith(side):
            continue
        colour = getattr(t.marker, "color", None)
        if isinstance(colour, str):
            out.add(colour)
    return out


def lines_of(fig, side=None):
    """Colours of things drawn ON TOP of the clouds: the quad and the ellipses."""
    out = set()
    for t in fig.data:
        name = t.name or ""
        if "head shape" not in name and "2sd" not in name:
            continue
        if side and not name.startswith(side):
            continue
        colour = getattr(getattr(t, "line", None), "color", None)
        if isinstance(colour, str):
            out.add(colour)
    return out


def test_a_single_fit_keeps_one_colour_per_bodypart(core):
    """The default view is unchanged: colour means bodypart, outline in black.

    Nothing is being compared, so spending colour on the model would say nothing.
    """
    fig = core.hfqc_compare_figure([("A", result(0))])
    assert len(points_of(fig)) > 1                    # a colour per marker, as before
    assert lines_of(fig) == {"black"}


def test_comparing_two_fits_switches_colour_to_mean_model(core):
    """Two clouds in the same ten colours cannot be told apart; that is the whole ask."""
    fig = core.hfqc_compare_figure([("A", result(0)), ("B", result(1))])
    assert len(points_of(fig, "A")) == 1
    assert len(points_of(fig, "B")) == 1
    assert points_of(fig, "A") != points_of(fig, "B")


def test_each_models_outline_stands_out_from_its_own_points(core):
    """An outline in the same colour as the cloud it sits on is invisible.

    So the quad and the ellipses get a SECOND colour per model -- still that model's, but
    a different hue from its points.
    """
    fig = core.hfqc_compare_figure([("A", result(0)), ("B", result(1))])
    for side in ("A", "B"):
        assert len(lines_of(fig, side)) == 1
        assert lines_of(fig, side) != points_of(fig, side)
    assert lines_of(fig, "A") != lines_of(fig, "B")


def test_the_bars_keep_their_qc_flag_colours(core):
    """The bars were already right: colour there means ok/warn/fail, per marker.

    Grouping and opacity separate the two models, and the thresholds are drawn as lines,
    so the model does not need the colour channel here.
    """
    flags = set(core._HFQC_FLAG_COLOR.values())
    fig = core.hfqc_compare_bars([("A", result(0)), ("B", result(1))])
    checked = 0
    for trace in fig.data:
        name = trace.name or ""
        if not (name.endswith("tracked") or name.endswith("ratio_spread")):
            continue
        colour = getattr(trace.marker, "color", None)
        # a per-marker LIST, not one colour for the whole row -- a scalar here is the
        # regression: the row goes one flat colour and the per-marker verdict is lost
        assert isinstance(colour, (list, tuple)), f"{name} lost its per-marker colours"
        assert set(colour) <= flags, f"{name} is not using flag colours"
        checked += 1
    assert checked == 4                       # two rows x two models


def test_the_bodypart_is_still_named_on_every_trace(core):
    # colour no longer carries it, so the label and the hover must
    fig = core.hfqc_compare_figure([("A", result(0))])
    named = [t.name for t in fig.data if (t.name or "").startswith("A ")]
    assert any("left_ear" in (n or "") for n in named)


def test_a_single_fit_still_draws(core):
    fig = core.hfqc_compare_figure([("A", result(0))])
    assert len(fig.data) > 0


def test_the_bars_pair_the_two_fits_per_bodypart(core):
    fig = core.hfqc_compare_bars([("A", result(0)), ("B", result(1))])
    names = [t.name or "" for t in fig.data]
    assert any(n.startswith("A") for n in names)
    assert any(n.startswith("B") for n in names)
    assert fig.layout.barmode == "group"


def test_bars_drop_the_training_row_when_no_fit_has_a_reference(core):
    """The panel itself must go, not just its bars.

    Ratio-vs-training is undefined without labels. Leaving the subplot in place -- axes,
    title and threshold lines, no data -- reads as a row of zeros, which is a worse
    answer than no row at all.
    """
    def titles(fig):
        return " ".join(a.text or "" for a in fig.layout.annotations).lower()

    with_ref = core.hfqc_compare_bars([("A", result(0, has_reference=True))])
    without = core.hfqc_compare_bars([("A", result(0, has_reference=False))])
    assert "training" in titles(with_ref)
    assert "training" not in titles(without)
    assert len(without.data) < len(with_ref.data)


def test_an_empty_selection_does_not_raise(core):
    assert core.hfqc_compare_figure([]) is not None
    assert core.hfqc_compare_bars([]) is not None


# ── the chip summary that replaced the table ──────────────────────────────────

def test_the_chip_line_states_the_numbers_the_table_was_consulted_for(core):
    text = core.hfqc_summary_chips(result(0))
    assert "spread" in text.lower()
    assert "tracked" in text.lower()


def test_the_chip_line_survives_a_result_that_failed_to_score(core):
    assert core.hfqc_summary_chips(None) == ""
# [AGENT EDIT END]


# ── the spread view's layout ──────────────────────────────────────────────────
# [AGENT EDIT START | agent: claude | user: wingertj | reason: A/B used to render a full
# stack per side in two columns; pin it to one figure pair plus a chip line | date: 2026-09-16]

def fit(name, side_id):
    return {"id": side_id, "path": f"/x/{side_id}.h5", "label": name, "alias": "two_chimney",
            "release_id": "two_chimney-r0001", "warning": "", "camera": "recording.avi",
            "video": "/x/recording.avi", "published": False, "origins": ["sorted"],
            "roots": {}, "config": "", "provenance": "release"}


SPREAD_SCRIPT = """
import labeler_core as core, fit_inspection
core.init_session_state()
fit_inspection.render_spread(fit_inspection._TEST_FITS, 0.9)
"""


@pytest.fixture
def spread_app(core, inspection, monkeypatch):
    from streamlit.testing.v1 import AppTest
    scored = {"a": result(0), "b": result(1)}
    monkeypatch.setattr(inspection, "existing_score",
                        lambda f, cutoff: (scored[f["id"]], None))
    monkeypatch.setattr(inspection, "_TEST_FITS", [fit("A fit", "a"), fit("B fit", "b")],
                        raising=False)
    return AppTest.from_string(SPREAD_SCRIPT)


def test_the_spread_view_draws_one_figure_pair_for_both_fits(spread_app, core, monkeypatch):
    import plotly.graph_objects as go
    seen = []
    monkeypatch.setattr(core, "hfqc_compare_figure",
                        lambda pairs, **k: seen.append(("scatter", pairs)) or go.Figure())
    monkeypatch.setattr(core, "hfqc_compare_bars",
                        lambda pairs: seen.append(("bars", pairs)) or go.Figure())

    at = spread_app.run(timeout=30)
    assert not at.exception
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the plots are a strip to switch between, one built at a time | date: 2026-09-25]
    assert [kind for kind, _ in seen] == ["scatter"]
    at.radio(key="inspect_spread_plot").set_value("bars").run()
    assert not at.exception
    assert [kind for kind, _ in seen] == ["scatter", "bars"]
    # [AGENT EDIT END]
    for _kind, pairs in seen:
        assert [side for side, _r in pairs] == ["A", "B"]


def test_the_per_marker_table_is_not_in_the_default_spread_view(spread_app):
    at = spread_app.run(timeout=30)
    assert not at.exception

    assert len(at.dataframe) == 0


def test_each_fit_gets_its_numbers_as_a_chip_line(spread_app):
    at = spread_app.run(timeout=30)
    assert not at.exception

    captions = " ".join(c.value for c in at.caption).lower()
    assert "spread" in captions and "tracked" in captions
# [AGENT EDIT END]


# ── the control row ───────────────────────────────────────────────────────────

RENDER_SCRIPT = """
import labeler_core as core, fit_inspection
core.init_session_state()
fit_inspection.render("/x/session/recording.avi")
"""


@pytest.fixture
def render_app(core, inspection, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(inspection, "inventory",
                        lambda *a, **k: [fit("A fit", "a"), fit("B fit", "b")])
    monkeypatch.setattr(inspection, "fit_rows", lambda fits, cutoff: [])
    return AppTest.from_string(RENDER_SCRIPT)


def test_publishing_comes_after_the_view_not_before_it(render_app, inspection, monkeypatch):
    # you publish a fit because of what you just saw; putting the button above the
    # picture asks for the decision first
    order = []
    monkeypatch.setattr(inspection, "render_frames",
                        lambda *a, **k: order.append("view"))
    monkeypatch.setattr(inspection, "render_actions",
                        lambda *a, **k: order.append("publish"))

    at = render_app.run(timeout=30)
    assert not at.exception

    assert order == ["view", "publish"]


def test_the_panel_states_things_in_fragments_not_sentences(render_app, inspection,
                                                             monkeypatch):
    # GUI text is terse; the reasoning belongs in hover. A 100-character caption above
    # the view is exactly the kind of thing that pushed the picture off the screen.
    monkeypatch.setattr(inspection, "render_frames", lambda *a, **k: None)

    at = render_app.run(timeout=30)
    assert not at.exception

    too_long = [c.value for c in at.caption if len(c.value) > 80]
    assert too_long == []
# [AGENT EDIT END]


# ── context the scatter cannot be read without ────────────────────────────────
# [AGENT EDIT START | agent: claude | user: wingertj | reason: the combined scatter shipped
# only the point clouds; the head outline, the ferret glyph, the frame in the axis titles
# and the clamped-outlier notice were all dropped with them | date: 2026-09-18]

def test_the_compare_scatter_keeps_the_mean_head_shape_outline(core):
    """Without the quad the cloud has no anatomy: you cannot tell an ear from a nose."""
    fig = core.hfqc_compare_figure([("A", result(0))])
    names = [t.name or "" for t in fig.data]
    assert any("head shape" in n.lower() for n in names)


def test_the_ferret_glyph_is_drawn_under_the_compare_scatter(core):
    fig = core.hfqc_compare_figure([("A", result(0))])
    assert len(fig.layout.images) >= 1


def test_each_fit_draws_its_own_head_outline_distinguishably(core):
    # the two fits are of the SAME video, so a disagreement between their mean shapes is
    # itself the finding -- but one ferret glyph is enough for both
    fig = core.hfqc_compare_figure([("A", result(0)), ("B", result(1))])
    outlines = [t for t in fig.data if "head shape" in (t.name or "").lower()]
    assert len(outlines) == 2
    assert outlines[0].line.dash != outlines[1].line.dash
    assert len(fig.layout.images) == 1


def test_the_axes_say_which_frame_the_spread_is_measured_in(core):
    fig = core.hfqc_compare_figure([("A", result(0))])
    assert "head-lengths" in (fig.layout.xaxis.title.text or "")
    assert "procrustes" in (fig.layout.xaxis.title.text or "").lower()


def test_clamped_outliers_are_declared_rather_than_silently_moved(core):
    # a point pinned to the border is a lie unless the count says so
    far = result(0)
    far.scatter_proc = {p: np.full((5, 2), 99.0) for p in PARTS}
    fig = core.hfqc_compare_figure([("A", far)])
    assert any("clamp" in (a.text or "").lower() for a in fig.layout.annotations)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: every spread plot is reachable from the strip, none from the settings popover | date: 2026-09-25]
def test_every_spread_plot_is_one_click_away(spread_app, core, monkeypatch):
    monkeypatch.setattr(core, "_hfqc_style_table", lambda *a: __import__("pandas").DataFrame({"x": [1]}))
    calls = []
    monkeypatch.setattr(core, "_hfqc_population_section", lambda *a, **k: calls.append(k))
    at = spread_app.run(timeout=30)
    strip = at.radio(key="inspect_spread_plot")
    assert list(strip.options) == ["Spread", "Per marker", "Vs database", "Table"]

    strip.set_value("table").run()
    assert not at.exception and len(at.dataframe) == 2          # one table per fit
    at.radio(key="inspect_spread_plot").set_value("database").run()
    assert not at.exception and len(calls) == 2
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: warnings are short tags with the full text in hover, never full-width paragraphs | date: 2026-09-25]
@pytest.mark.parametrize("text, label, alarming", [
    ("training reference built from only 75 labeled frames across 5 folder(s) (< 100); "
     "ratios are provisional", "weak reference · 75 frames", True),
    ("only 40 valid frames (< 200); statistics are noisy", "40 valid frames · noisy", True),
    ("DAMAGED PROVENANCE (defect 2). left_chimney_PRNfb was retrained in place ...",
     "Damaged provenance", True),
    ("DEFECT 9: snapshot-60000 exists in BOTH iteration-0 ...", "Defect 9", True),
    ("DLC test error is NOT comparable to r0001/r0002 ...", "release note", False),
    ("Trained 2026-08-08 from ImageNet-pretrained resnet_50 ...", "release note", False),
])
def test_a_warning_reduces_to_a_few_word_tag(core, text, label, alarming):
    assert core.short_note(text, "release note") == (label, alarming)


def test_a_note_keeps_its_full_text_in_the_hover(core):
    from streamlit.testing.v1 import AppTest
    long = ("DAMAGED PROVENANCE (defect 2). left_chimney_PRNfb was retrained in place on "
            "2025-06-12 into the same iteration-0/train/, destroying the weights.")
    at = AppTest.from_string(f"""
import labeler_core as core
core.notes_caption([{long!r}, "only 40 valid frames (< 200); statistics are noisy",
                    "only 40 valid frames (< 200); statistics are noisy"], prefix="A")
""").run()
    assert not at.exception and not at.warning
    (caption,) = at.caption
    assert caption.value == ("A · :orange[⚠ Damaged provenance] · "
                             ":orange[⚠ 40 valid frames · noisy]")
    assert long in caption.help
# [AGENT EDIT END]
