"""The Database tab: what it still shows, what it stopped showing, what you triage on."""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: minimal Database tab -- drop
# staleness, release stamping and the QC flags; keep provenance plus the two head-marker
# statistics that say which sessions need attention | date: 2026-09-16]

import importlib
from pathlib import Path

import pandas as pd
import pytest

LABELER = Path(__file__).resolve().parents[1] / "nems_lbhb/motor/streamlit_vid_labeler"

# The columns this tab deliberately no longer carries.  Each one is a question the
# regular user cannot act on: whether the release id was read off the filename rather
# than recovered by evidence (`attribution`), whether the alias has moved on since
# (`stale`), and two QC verdicts whose thresholds live somewhere else entirely.
DROPPED = ("attribution", "stale", "flag_track", "flag_ratio", "source", "p_cut",
           "attribution_evidence", "release_path")

ROWS = [
    # siteid    video_stem         stream  release_id            repro spread p_median
    ("REI059a", "REI059a01_a_NTD", "",     "two_chimney-r0002",  1,    0.21,  0.997),
    ("REI059a", "REI059a02_p_FTC", "camB", "two_chimney-r0002",  1,    0.62,  0.880),
    ("PRN048a", "PRN048a01_a_NFC", "",     "left_chimney-r0001", 0,    0.19,  0.999),
    ("SQD020a", "SQD020a05_a_NTD", "",     None,              None,  None,   None),
]


def table():
    """A fit_status frame: two clean rows, one non-reproducible, one never attributed.

    Every dropped column is present and populated on purpose -- a test that the tab
    hides them is worthless if the fixture never offered them.
    """
    df = pd.DataFrame(ROWS, columns=["siteid", "video_stem", "stream", "release_id",
                                     "reproducible", "rms_spread_mean", "p_median"])
    df["routing_alias"] = ["two_chimney", "two_chimney", "left_chimney", "pupil"]
    df["attribution"] = ["inferred_scorer", "stamped", "inferred_scorer", "ambiguous"]
    df["stale"] = [False, False, True, True]
    df["flag_track"] = ["ok", "warn", "ok", ""]
    df["flag_ratio"] = ["ok", "fail", "ok", ""]
    df["source"] = "sweeper"
    df["p_cut"] = 0.9
    df["attribution_evidence"] = "because"
    df["release_path"] = "/auto/data/dlc/releases/x"
    df["target_mtime_utc"] = pd.Timestamp("2024-01-11 02:47:49")
    return df


@pytest.fixture
def database(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    return importlib.import_module("tabs.database")


@pytest.fixture
def core(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    return importlib.import_module("labeler_core")


@pytest.fixture
def provenance(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    return importlib.import_module("panels.provenance")


# ── the one bit that matters: can we run this model again? ─────────────────────

def test_reproducible_is_no_when_the_weights_can_no_longer_be_rerun(core):
    got = core.reproducible_flag(table())
    assert list(got) == [True, True, False, False]


def test_reproducible_is_no_rather_than_blank_when_nothing_was_attributed(core):
    # the stored column is NULL for a row no release could be pinned to, and a blank
    # cell reads as "fine" at a glance -- which is the opposite of what it means
    only_unattributed = table().iloc[[3]]
    assert list(core.reproducible_flag(only_unattributed)) == [False]


def test_an_unknown_reproducible_flag_is_not_a_yes(core):
    # a release is named, but nothing recorded whether re-running it reproduces this
    # output. gDLCFit's default for the column is NULL, so a job can write exactly this
    # row, and "we never checked" must not render as "yes, it reproduces".
    df = table().iloc[[0]].copy()
    df["reproducible"] = None
    assert list(core.reproducible_flag(df)) == [False]


def test_reproducible_needs_both_a_release_and_the_flag(core):
    df = table().iloc[[0]].copy()
    df["release_id"] = None
    assert list(core.reproducible_flag(df)) == [False]


# ── what the table shows ──────────────────────────────────────────────────────

def test_the_table_carries_only_the_minimal_column_set(database):
    assert database.COLUMNS == ["siteid", "video_stem", "stream", "routing_alias",
                                "release_id", "reproducible", "head_spread",
                                "head_conf", "target_mtime_utc"]


def test_neither_table_still_carries_a_dropped_column(database, provenance):
    for name in DROPPED:
        assert name not in database.COLUMNS
        assert name not in provenance.SESSION_COLUMNS


def test_the_session_table_agrees_with_the_database_table(provenance, database):
    # the Model tab shows one session's rows and the Database tab shows every row; two
    # column sets that disagree would make the same fit look different in two places
    assert provenance.SESSION_COLUMNS == [c for c in database.COLUMNS if c != "siteid"]


def test_there_is_no_staleness_panel_left_to_call(provenance):
    assert not hasattr(provenance, "render_staleness")


# ── triage filters ────────────────────────────────────────────────────────────

def stems(df):
    return list(df["video_stem"])


def test_no_filter_lists_everything(database, core):
    got = database.apply_filters(core.prepare_fits(table()), "", "(any)", 0.0, 1.0, False)
    assert stems(got) == ["REI059a01_a_NTD", "REI059a02_p_FTC", "PRN048a01_a_NFC",
                          "SQD020a05_a_NTD"]


def test_spread_filter_keeps_the_rows_at_or_above_it(database, core):
    got = database.apply_filters(core.prepare_fits(table()), "", "(any)", 0.5, 1.0, False)
    assert stems(got) == ["REI059a02_p_FTC"]


def test_confidence_filter_keeps_the_rows_at_or_below_it(database, core):
    got = database.apply_filters(core.prepare_fits(table()), "", "(any)", 0.0, 0.9, False)
    assert stems(got) == ["REI059a02_p_FTC"]


def test_an_active_stat_filter_drops_rows_whose_qc_was_never_computed(database, core):
    # 141 of 569 live rows predate QC entirely. Hunting for bad spread must not sweep
    # them in: an unmeasured session is not a measured-bad one, and treating it as one
    # would put a permanent block of unactionable rows at the top of every search.
    got = database.apply_filters(core.prepare_fits(table()), "", "(any)", 0.1, 1.0, False)
    assert "SQD020a05_a_NTD" not in stems(got)


def test_not_reproducible_filter_shows_exactly_what_cannot_be_rerun(database, core):
    got = database.apply_filters(core.prepare_fits(table()), "", "(any)", 0.0, 1.0, True)
    assert stems(got) == ["PRN048a01_a_NFC", "SQD020a05_a_NTD"]


def test_text_filter_matches_a_site_or_a_session(database, core):
    prepared = core.prepare_fits(table())
    assert stems(database.apply_filters(prepared, "REI059", "(any)", 0.0, 1.0, False)) == [
        "REI059a01_a_NTD", "REI059a02_p_FTC"]
    assert stems(database.apply_filters(prepared, "a05_a_ntd", "(any)", 0.0, 1.0, False)) == [
        "SQD020a05_a_NTD"]


def test_alias_filter_narrows_to_one_model(database, core):
    got = database.apply_filters(core.prepare_fits(table()), "", "left_chimney",
                                 0.0, 1.0, False)
    assert stems(got) == ["PRN048a01_a_NFC"]


# ── rendered ──────────────────────────────────────────────────────────────────

SCRIPT = """
import labeler_core as core
from tabs import database
core.init_session_state()
database.render()
"""

REGISTRY = {"aliases": {"two_chimney": {"current": None, "qc": "headfix"},
                        "left_chimney": {"current": None, "qc": "headfix"}},
            "alias_names": ["two_chimney", "left_chimney"]}


@pytest.fixture
def app(monkeypatch, database, provenance):
    from streamlit.testing.v1 import AppTest
    core = importlib.import_module("labeler_core")
    monkeypatch.setattr(core, "_registry_mtime", lambda: 0.0)
    # the Refresh button binds core.fit_status_cached.clear, which a bare lambda lacks
    fake_status = lambda *a, **k: table()
    fake_status.clear = lambda: None
    monkeypatch.setattr(core, "fit_status_cached", fake_status)
    monkeypatch.setattr(provenance.dlc_registry, "load_registry", lambda *a, **k: REGISTRY)
    return AppTest.from_string(SCRIPT)


def test_the_rendered_table_is_the_minimal_column_set(app, database):
    at = app.run(timeout=30)
    assert not at.exception

    assert list(at.dataframe[-1].value.columns) == database.COLUMNS


def test_nothing_in_the_tab_mentions_staleness_or_stamping(app):
    at = app.run(timeout=30)
    assert not at.exception

    text = " ".join([c.value for c in at.caption] + [m.value for m in at.markdown]
                    + [w.value for w in at.warning]).lower()
    assert "stale" not in text
    assert "stamp" not in text


def test_the_stale_and_unstamped_checkboxes_are_gone(app):
    at = app.run(timeout=30)
    assert not at.exception

    keys = {w.key for w in at.checkbox}
    assert "db_filter_stale" not in keys
    assert "db_filter_unstamped" not in keys


def test_the_caption_says_how_many_rows_have_no_qc_at_all(app):
    # spread and confidence are blank for a pre-QC row, and a blank cell in a triage
    # table reads as "nothing wrong here" unless the count is stated
    at = app.run(timeout=30)
    assert not at.exception

    assert any("1 without qc" in c.value.lower() for c in at.caption)


def test_the_spread_filter_reaches_the_rendered_table(app):
    at = app.run(timeout=30)
    at.number_input(key="db_filter_spread").set_value(0.5).run()
    assert not at.exception

    assert list(at.dataframe[-1].value["video_stem"]) == ["REI059a02_p_FTC"]


def test_the_confidence_filter_reaches_the_rendered_table(app):
    at = app.run(timeout=30)
    at.number_input(key="db_filter_conf").set_value(0.9).run()
    assert not at.exception

    assert list(at.dataframe[-1].value["video_stem"]) == ["REI059a02_p_FTC"]


def test_the_not_reproducible_checkbox_reaches_the_rendered_table(app):
    at = app.run(timeout=30)
    at.checkbox(key="db_filter_broken").check().run()
    assert not at.exception

    assert list(at.dataframe[-1].value["video_stem"]) == ["PRN048a01_a_NFC",
                                                          "SQD020a05_a_NTD"]
# [AGENT EDIT END]


# ── vocabulary, across the whole GUI ──────────────────────────────────────────
# [AGENT EDIT START | agent: claude | user: wingertj | reason: "stale" and "stamped" were
# app-wide vocabulary, not just this tab's; a grep-once fix would drift back | date: 2026-09-16]

BANNED = ("stale", "unstamp", "stamped")


def gui_string_literals():
    """Every string constant in the labeler that is not a docstring.

    Docstrings and comments are deliberately exempt: the reasoning for why a concept was
    dropped has to survive in the source, and none of it reaches the screen. What does
    reach the screen is a literal handed to st.* -- label, caption, help or chip.
    """
    import ast

    for path in sorted(LABELER.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    docs.add(id(first.value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docs):
                yield path.name, node.lineno, node.value


def test_no_on_screen_text_says_stale_or_stamped():
    offenders = [f"{name}:{line} {text!r}"
                 for name, line, text in gui_string_literals()
                 for word in BANNED if word in text.lower()]
    assert offenders == []
# [AGENT EDIT END]
