"""The finder tab: what it lists, and which session a click targets."""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: tests for the new session finder tab | date: 2026-09-15]

import importlib
from pathlib import Path

import pandas as pd
import pytest

LABELER = Path(__file__).resolve().parents[1] / "nems_lbhb/motor/streamlit_vid_labeler"


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

ROWS = [
    # siteid     session              runclass behavior needs        dlc    cfg    video
    ("PRN001a", "PRN001a01_a_NTD", "NTD", "active", "",           True,  True,  "/d/1.avi"),
    ("PRN001a", "PRN001a02_a_NTD", "NTD", "active", "dlc+config", False, False, "/d/2.avi"),
    ("PRN001a", "PRN001a03_p_FTC", "FTC", "passive", "config",    True,  False, "/d/3.avi"),
    ("PRN002a", "PRN002a01_a_NFC", "NFC", "active", "dlc",       False, True,  "/d/4.avi"),
    ("PRN002a", "PRN002a02_p_NFB", "NFB", "passive", "no video",  False, False, None),
]


def table():
    df = pd.DataFrame(ROWS, columns=["siteid", "session", "runclass", "behavior",
                                     "needs", "has_dlc", "has_config", "video"])
    for extra in ("routing_alias", "release_id", "pendate"):
        df[extra] = None
    df["slot"] = "recording.avi"
    df["rawid"] = range(1, len(df) + 1)
    return df


@pytest.fixture
def finder(monkeypatch):
    pytest.importorskip("streamlit")
    monkeypatch.syspath_prepend(str(LABELER))
    return importlib.import_module("tabs.session_finder")


def test_row_selection_uses_position_not_the_filtered_frames_index(finder):
    # what st.dataframe hands back is the row's POSITION in the frame it rendered, and
    # that frame is a filtered view whose pandas index still carries the original labels
    view = table()[lambda d: d["needs"] != ""]
    assert list(view.index) == [1, 2, 3, 4]

    assert finder.selected_video(view, [0]) == "/d/2.avi"
    assert finder.selected_video(view, [2]) == "/d/4.avi"


def test_nothing_is_targeted_without_a_selection_or_a_video(finder):
    view = table()
    assert finder.selected_video(view, []) is None
    assert finder.selected_video(view, None) is None
    assert finder.selected_video(view, [4]) is None          # the no-video row
    assert finder.selected_video(view, [99]) is None
    assert finder.selected_video(pd.DataFrame(), [0]) is None


SCRIPT = """
import labeler_core as core
from tabs import session_finder
core.init_session_state()
session_finder.render()
"""


@pytest.fixture
def app(monkeypatch, tmp_path, finder):
    from streamlit.testing.v1 import AppTest
    core = importlib.import_module("labeler_core")
    monkeypatch.setattr(core, "PREFS_FILE", tmp_path / "prefs.json")
    monkeypatch.setattr(core, "recorded_animals", lambda: ["Prince", "Reishi"])
    monkeypatch.setattr(core, "session_status", lambda animal: table())
    return AppTest.from_string(SCRIPT)


def test_lists_only_sessions_that_need_work(app):
    at = app.run(timeout=30)
    assert not at.exception

    listed = at.dataframe[0].value
    assert list(listed["session"]) == ["PRN001a02_a_NTD", "PRN001a03_p_FTC",
                                       "PRN002a01_a_NFC"]


def test_runclass_filter_narrows_the_list(app):
    at = app.run(timeout=30)
    at.multiselect(key="finder_runclass").select("NTD").run()
    assert not at.exception

    assert list(at.dataframe[0].value["session"]) == ["PRN001a02_a_NTD"]


def test_finished_sessions_are_available_but_not_shown_by_default(app):
    at = app.run(timeout=30)
    at.checkbox(key="finder_show_done").check().run()
    assert not at.exception

    listed = at.dataframe[0].value
    assert "PRN001a01_a_NTD" in list(listed["session"])
    assert "PRN002a02_p_NFB" not in list(listed["session"])   # nothing to do, no video


def test_scope_defaults_to_the_animal_of_the_session_already_open(finder):
    assert finder.animal_for(
        "/auto/data/daq/StinkySquid/SQD020/SQD020a05_a_NTD/recording.avi") == "StinkySquid"
    assert finder.animal_for("/auto/data/daq/Prince/PRN048") == "Prince"
    assert finder.animal_for("") is None
    assert finder.animal_for("/tmp/elsewhere/recording.avi") is None


def test_selection_is_read_off_streamlits_real_event_object(finder):
    # AppTest cannot click a dataframe row, so this is the only thing standing between a
    # shape change upstream and a table whose rows silently stop opening anything
    from streamlit.elements.lib.event_utils import AttributeDictionary

    event = AttributeDictionary({"selection": {"rows": [2], "columns": []}})
    assert finder.selection_rows(event) == [2]
    assert finder.selection_rows(
        AttributeDictionary({"selection": {"rows": [], "columns": []}})) == []
    assert finder.selection_rows(None) == []
    assert finder.selection_rows(AttributeDictionary({})) == []


def test_the_animal_scope_is_remembered_for_the_next_session(app):
    from streamlit.testing.v1 import AppTest

    at = app.run(timeout=30)
    assert at.selectbox(key="finder_animal").value == "Prince"
    at.selectbox(key="finder_animal").select("Reishi").run()

    # a new browser session has no video open, so without a memory the tab would land on
    # whichever animal happens to have been recorded most recently -- usually not this one
    fresh = AppTest.from_string(SCRIPT).run(timeout=30)
    assert not fresh.exception
    assert fresh.selectbox(key="finder_animal").value == "Reishi"


def test_a_selection_is_acted_on_once_not_re_applied_on_every_rerun(finder):
    # the selection outlives the rerun that carried it and is read again on every later
    # rerun -- including the one caused by typing a different path into the session
    # strip, which an unguarded handler would then silently revert
    assert finder.pending_open("/d/2.avi", None) == "/d/2.avi"
    assert finder.pending_open("/d/2.avi", "/d/2.avi") is None
    assert finder.pending_open("/d/4.avi", "/d/2.avi") == "/d/4.avi"
    assert finder.pending_open(None, "/d/2.avi") is None


def test_clicking_a_row_points_the_app_at_that_session(finder, monkeypatch):
    # AppTest cannot click a dataframe row, so the whole selection -> open path is
    # assembled here against the real event type and the real session state
    from streamlit.elements.lib.event_utils import AttributeDictionary
    core = importlib.import_module("labeler_core")
    state = State(arena_config_dirty=False, points={"LEFTOVER": (1, 2)})
    monkeypatch.setattr(core.st, "session_state", state)
    view = table()[lambda d: d["needs"] != ""]
    event = AttributeDictionary({"selection": {"rows": [2], "columns": []}})

    assert finder.handle_selection(view, event) is True

    assert state["video_path"] == "/d/4.avi"
    # the path box is a session-strip widget, already live by the time a tab body runs;
    # open_session asks the strip to adopt it instead of writing it here
    assert state["pending_strip_sync"] is True
    assert state["current_frame"] is None
    assert state["points"] == {}
    assert state["finder_opened"] == "/d/4.avi"

    assert finder.handle_selection(view, event) is False


def test_clicking_a_row_does_not_discard_unsaved_arena_annotations(finder, monkeypatch):
    from streamlit.elements.lib.event_utils import AttributeDictionary
    core = importlib.import_module("labeler_core")
    state = State(arena_config_dirty=True, video_path="/d/already_open.avi",
                  points={"A": (1, 2)})
    monkeypatch.setattr(core.st, "session_state", state)
    view = table()[lambda d: d["needs"] != ""]

    assert finder.handle_selection(
        view, AttributeDictionary({"selection": {"rows": [0], "columns": []}})) is False

    assert state["video_path"] == "/d/already_open.avi"
    assert state["points"] == {"A": (1, 2)}
    assert "save or discard" in state["open_session_error"]


def test_there_is_no_separate_open_button_to_hunt_for(app):
    at = app.run(timeout=30)
    assert not at.exception

    assert len(at.button) == 1                       # Rescan, and nothing else
    assert "rescan" in at.button[0].label.lower()


def test_the_table_says_that_clicking_a_row_is_what_opens_a_session(app):
    # clicking a row to navigate is invisible until you happen to try it, so the one
    # affordance this tab has must be stated on screen
    at = app.run(timeout=30)
    assert not at.exception

    assert any("click a row" in c.value.lower() for c in at.caption)


def test_every_run_consults_the_selection(app, monkeypatch, finder):
    # AppTest cannot click a dataframe row, so nothing else here would notice if render()
    # stopped reading the selection at all -- the table would draw and do nothing
    seen = []
    monkeypatch.setattr(finder, "handle_selection",
                        lambda view, event: seen.append((view, event)) or False)

    at = app.run(timeout=30)
    assert not at.exception

    assert len(seen) == 1
    rendered, event = seen[0]
    assert list(rendered["session"]) == list(at.dataframe[0].value["session"])
    assert event is not None


def test_a_refused_switch_says_why_in_the_tab(app):
    at = app.run(timeout=30)
    at.session_state["open_session_error"] = "unsaved arena config · save or discard"
    at.run(timeout=30)
    assert not at.exception

    assert any("unsaved arena config" in w.value for w in at.warning)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: a refused switch marked the
# row handled, so the same click could never be retried after saving | date: 2026-09-16]
def test_a_refused_switch_can_be_retried_once_the_config_is_saved(finder, monkeypatch):
    """Refusing must not consume the click.

    The selection does not change when the user goes and saves their arena config, so
    recording the row as handled on a REFUSAL stranded it: the only way back was to click
    a different row and then this one again.
    """
    from streamlit.elements.lib.event_utils import AttributeDictionary
    core = importlib.import_module("labeler_core")
    state = State(arena_config_dirty=True, video_path="/d/already_open.avi",
                  points={"A": (1, 2)})
    monkeypatch.setattr(core.st, "session_state", state)
    view = table()[lambda d: d["needs"] != ""]
    event = AttributeDictionary({"selection": {"rows": [0], "columns": []}})

    assert finder.handle_selection(view, event) is False
    assert state["video_path"] == "/d/already_open.avi"

    state["arena_config_dirty"] = False              # the user saves, then nothing else
    assert finder.handle_selection(view, event) is True
    assert state["video_path"] == "/d/2.avi"
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the runclass and show-finished
# filters survived reruns but not a browser restart, so every session began by re-picking
# the same two runclasses | date: 2026-09-16]
def test_the_runclass_filter_is_remembered_for_the_next_session(app):
    from streamlit.testing.v1 import AppTest

    at = app.run(timeout=30)
    at.multiselect(key="finder_runclass").select("NTD").run()

    fresh = AppTest.from_string(SCRIPT).run(timeout=30)
    assert not fresh.exception
    assert fresh.multiselect(key="finder_runclass").value == ["NTD"]


def test_showing_finished_sessions_is_remembered_for_the_next_session(app):
    from streamlit.testing.v1 import AppTest

    at = app.run(timeout=30)
    at.checkbox(key="finder_show_done").check().run()

    fresh = AppTest.from_string(SCRIPT).run(timeout=30)
    assert not fresh.exception
    assert fresh.checkbox(key="finder_show_done").value is True


def test_a_remembered_runclass_this_animal_lacks_is_hidden_not_forgotten(app):
    """Pruning is for the widget, never for the stored preference.

    A runclass the current animal has no sessions of cannot be offered -- Streamlit
    raises on a stored value absent from the options. Writing that pruned list back
    would quietly erase the runclass the moment you looked at an animal without it.
    """
    core = importlib.import_module("labeler_core")
    core.save_user_prefs({"finder_runclass": ["NTD", "BNT"]})

    at = app.run(timeout=30)
    assert not at.exception

    assert at.multiselect(key="finder_runclass").value == ["NTD"]     # BNT not offered
    assert core.load_user_prefs()["finder_runclass"] == ["NTD", "BNT"]


def test_changing_the_filter_replaces_what_was_remembered(app):
    core = importlib.import_module("labeler_core")
    core.save_user_prefs({"finder_runclass": ["NTD"]})

    at = app.run(timeout=30)
    at.multiselect(key="finder_runclass").select("NFC").run()

    assert sorted(core.load_user_prefs()["finder_runclass"]) == ["NFC", "NTD"]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: runclass options came from the
# with-video subset, so the list silently depended on what happened to have a camera; add
# the active/passive split celldb already records | date: 2026-09-16]
def test_runclass_options_cover_everything_the_animal_recorded(app):
    """Not just the runclasses that happen to have a camera file.

    Deriving the options from the video-bearing rows made the list depend on an accident
    of the rig -- fine if your runclasses all have video, invisible and confusing if they
    do not. The animal's own set is the complete answer; lab-wide there are 146.
    """
    at = app.run(timeout=30)
    assert not at.exception

    options = at.multiselect(key="finder_runclass").options
    assert any("NFB" in o for o in options)      # only on a session with no video


def test_runclass_options_say_how_many_sessions_have_video(app):
    # a runclass with nothing to work on must look empty rather than silently filter to
    # nothing when picked
    at = app.run(timeout=30)
    options = at.multiselect(key="finder_runclass").options

    assert any(o.startswith("NFB") and "0" in o for o in options)
    assert any(o.startswith("NTD") and "2" in o for o in options)


def test_the_behavior_filter_narrows_to_active_or_passive(app):
    at = app.run(timeout=30)
    at.multiselect(key="finder_behavior").select("passive").run()
    assert not at.exception

    assert list(at.dataframe[0].value["session"]) == ["PRN001a03_p_FTC"]


def test_the_behavior_filter_is_remembered_for_the_next_session(app):
    from streamlit.testing.v1 import AppTest

    at = app.run(timeout=30)
    at.multiselect(key="finder_behavior").select("active").run()

    fresh = AppTest.from_string(SCRIPT).run(timeout=30)
    assert not fresh.exception
    assert fresh.multiselect(key="finder_behavior").value == ["active"]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: tests for the finder's bulk DLC queue section | date: 2026-09-22]
def fitted_table():
    df = table()
    df["pendate"] = ["2024-01-01", "2024-01-01", "2024-01-01", "2024-03-01", "2024-03-01"]
    df.loc[0, "routing_alias"] = "left_chimney"               # PRN001a01 has a fit
    return df


def route_all(alias="left_chimney", release="r007"):
    return lambda video: {"alias": alias, "release": release, "unknown": "", "error": None}


def no_queue(notes):
    return {}


def note_for(video, alias, release):
    return (f"note:{video}", f"publish:{video}")


def test_queue_lists_only_listed_sessions_missing_dlc(finder):
    view = fitted_table()[lambda d: d["needs"] != ""]
    plan = finder.queue_candidates(view, fitted_table(), route_all(), note_for, no_queue)
    # the config-only row has a .dlc.h5 already; the no-video row has nothing to run on
    assert list(plan["session"]) == ["PRN001a02_a_NTD", "PRN002a01_a_NFC"]


def test_prior_fit_prefers_the_same_site_then_the_nearest_date(finder):
    df = fitted_table()
    assert finder.prior_fit(df, df.iloc[1]) == ("left_chimney", "same site")
    alias, where = finder.prior_fit(df, df.iloc[3])            # PRN002a: no fit at site
    assert alias == "left_chimney" and where.startswith("PRN001a")
    assert finder.prior_fit(table(), table().iloc[1]) == (None, "")


def test_only_confident_sessions_are_ticked_by_default(finder):
    view = fitted_table()[lambda d: d["needs"] != ""]
    df = fitted_table()
    agree = finder.queue_candidates(view, df, route_all(), note_for, no_queue)
    assert list(agree["queue"]) == [True, True]

    other = finder.queue_candidates(view, df, route_all("right_chimney"), note_for, no_queue)
    assert list(other["agrees"]) == [False, False]
    assert not other["queue"].any()

    unreleased = finder.queue_candidates(view, df, route_all(release=None), note_for,
                                         no_queue)
    assert not unreleased["queue"].any()


def test_sessions_already_in_the_queue_are_not_ticked_again(finder):
    view = fitted_table()[lambda d: d["needs"] != ""]
    states = lambda notes: {"publish:/d/2.avi": "running", "note:/d/2.avi": "done",
                            "note:/d/4.avi": "dead"}
    plan = finder.queue_candidates(view, fitted_table(), route_all(), note_for, states)
    assert list(plan["in_queue"]) == ["running", "dead"]
    assert list(plan["queue"]) == [False, True]


def test_the_queue_section_routes_only_once_opened(app, monkeypatch):
    core = importlib.import_module("labeler_core")
    calls = []

    def routed(video, mtime):
        calls.append(video)
        return {"alias": "left_chimney", "release": "r007", "unknown": "", "error": None}

    monkeypatch.setattr(core, "routed_session", routed)
    monkeypatch.setattr(core, "queue_states", lambda notes: {})
    at = app.run(timeout=30)
    assert not at.exception and not calls

    at.toggle(key="finder_queue_on").set_value(True).run()
    assert not at.exception
    assert sorted(calls) == ["/d/2.avi", "/d/4.avi"]
    assert at.button(key="finder_queue_go").label == "Queue 0 → temp"     # no prior fit


def test_the_queue_goes_to_temp_unless_sorted_is_chosen(app, monkeypatch):
    core = importlib.import_module("labeler_core")
    monkeypatch.setattr(core, "routed_session", lambda video, mtime: {
        "alias": "left_chimney", "release": "r007", "unknown": "", "error": None})
    monkeypatch.setattr(core, "queue_states", lambda notes: {})
    at = app.run(timeout=30)
    at.toggle(key="finder_queue_on").set_value(True).run()
    assert at.radio(key="finder_queue_dest").value == "temp"
    at.radio(key="finder_queue_dest").set_value("sorted").run()
    assert not at.exception
    assert at.button(key="finder_queue_go").label.endswith("→ sorted")


def test_temp_and_sorted_jobs_are_tracked_under_their_own_queue_notes(finder):
    core = importlib.import_module("labeler_core")
    temp = core.finder_job_notes("/d/S01/recording.avi", "left_chimney", "r007", "temp")
    sorted_ = core.finder_job_notes("/d/S01/recording.avi", "left_chimney", "r007")
    assert len(temp) == 1 and temp[0].endswith("action_fit_temp_analysis")
    assert not set(temp) & set(sorted_)
# [AGENT EDIT END]
