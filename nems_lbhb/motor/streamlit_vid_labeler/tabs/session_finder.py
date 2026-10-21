"""Find the next session to work on, and point the app at it.

Whether a session still needs DLC is a question about three places at once: celldb knows
what was recorded, ``sorted/`` knows what came back, and gDLCFit knows which model made
it.  Answering it by hand meant walking celldb's hyperlinks one site at a time.  This
lists a whole animal and hands the rest of the app a video path.

Clicking a row changes which session the page is looking at, nothing else.  The one
write lives in the "Queue DLC" section under the table: for sessions with no .dlc.h5 it
routes each video, shows the alias and release it would run beside the alias of the
nearest existing fit, and queues the ticked ones -- the Model tab's "Preview → temp"
(the default) or "Production → sorted" job, many sessions at a time.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: new labeler tab -- find the next session to work on and open it | date: 2026-09-15]

from pathlib import Path

import pandas as pd
import streamlit as st

import labeler_core as core

COLUMNS = ["needs", "siteid", "session", "runclass", "behavior", "has_dlc",
           "has_config", "routing_alias", "release_id", "pendate"]

HELP = {
    "needs": "What is missing for this session: 'dlc' has no .dlc.h5 in sorted/, "
             "'config' has no arena config, 'dlc+config' has neither. routing_alias and "
             "release_id name the model behind whatever IS there.",
    "behavior": "active = the animal was performing a task, passive = it was not. "
                "celldb records this per recording; 325 of these animals' 2280 are active.",
    "show finished": "Also list sessions with nothing outstanding. Sessions with no "
                     "video are never listed -- there is nothing to run on them.",
    "click": "Clicking a row points the whole page -- frame, DLC overlay, arena "
             "markers, fit -- at that session, exactly as typing its path into the box "
             "at the top would.",
}


def animal_for(video_path):
    """The animal whose data directory holds ``video_path``, or None."""
    parts = Path(str(video_path or "")).parts
    if "daq" in parts:
        index = parts.index("daq")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def selection_rows(event):
    """The selected row positions from ``st.dataframe``'s event object.

    The event is an ``AttributeDictionary``; nothing in the type system says so, and a
    missing key raises rather than returning None, so both hops are guarded.
    """
    selection = getattr(event, "selection", None)
    return list(getattr(selection, "rows", None) or [])


def selected_video(table, rows):
    """The video for a single-row dataframe selection, or None.

    ``st.dataframe`` reports the row's POSITION in the frame it rendered.  That frame is
    a filtered view whose pandas index still carries the unfiltered labels, so reading it
    with ``.loc`` would quietly open a different session than the one clicked.
    """
    if table is None or not len(table) or not rows:
        return None
    try:
        position = int(rows[0])
    except (TypeError, ValueError):
        return None
    if position < 0 or position >= len(table):
        return None
    video = table.iloc[position].get("video")
    return None if video is None or pd.isna(video) else str(video)


def pending_open(video, handled):
    """``video`` if this selection still needs acting on, else None.

    A dataframe selection outlives the rerun that carried it: it sits in session state
    and is read again on every later rerun of this tab.  Without this guard, typing a
    different path into the session strip reruns the page, the stale selection is read
    back, and the finder silently reverts to the row that was clicked minutes ago.
    """
    return video if video and video != handled else None


def handle_selection(view, event):
    """Point the app at whatever row the selection names.  True if the page must rerun.

    False covers three situations on purpose -- nothing selected, the selected row is
    the one already handled, and the switch was refused to protect unsaved arena
    annotations -- because all three mean the same thing here: do not rerun.  The
    refusal leaves its reason in ``open_session_error``, which :func:`render` shows.
    """
    video = pending_open(selected_video(view, selection_rows(event)),
                         st.session_state.get("finder_opened"))
    if not video:
        return False
    if not core.open_session(video):
        # refused to protect unsaved arena annotations. Deliberately NOT recorded as
        # handled: the selection does not change while the user goes and saves, so
        # consuming the click here would strand the row until they clicked another and
        # came back. The refusal's reason is in open_session_error, which render() shows.
        return False
    st.session_state.finder_opened = video
    return True


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the filters survived reruns
# but not a browser restart, so every session began by re-picking the same runclasses
# | date: 2026-09-16]
def runclass_options(table, with_video):
    """Every runclass this animal recorded, labelled with how many have a video.

    Options used to come from the video-bearing rows alone, which made the list depend
    on an accident of the rig: a runclass recorded without a camera simply was not
    offered. Offering all of them and stating the count keeps the list complete and
    makes an empty one look empty instead of filtering to nothing when picked.

    Lab-wide there are 146 runclasses, so the set stays scoped to the chosen animal.
    """
    if not len(table):
        return {}
    counts = (with_video["runclass"].value_counts().to_dict()
              if len(with_video) else {})
    codes = sorted(table["runclass"].dropna().unique().tolist())
    return {code: f"{code} · {int(counts.get(code, 0))}" for code in codes}


def remember_filters(chosen, offered, show_done, behavior=None):
    """Persist the filters without letting per-animal pruning erase a preference.

    ``offered`` is what the STORED runclass list survived pruning to for this animal.
    When the widget still reads exactly that, the user has chosen nothing this run, and
    writing it back would silently drop whichever runclasses this animal happens to lack.
    """
    if list(chosen) != list(offered):
        core.remember_pref("finder_runclass", list(chosen))
    core.remember_pref("finder_show_done", bool(show_done))
    if behavior is not None:
        core.remember_pref("finder_behavior", list(behavior))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: queue the routed model for
# a batch of missing-dlc sessions straight from the finder | date: 2026-09-22]
QUEUE_COLUMNS = ["queue", "session", "siteid", "alias", "release", "prior", "agrees",
                 "in_queue", "unknown"]

QUEUE_HELP = {
    "section": "Queues the model the geometry router picks for each session that has no "
               ".dlc.h5. Only sessions whose routed alias has a frozen release can be "
               "queued.",
    "dest": "temp = the Model tab's 'Preview → temp': output goes to the session's "
            "tmp/dlc folder to check first. sorted = 'Production → sorted': output goes "
            "straight to sorted/. A session with a checked temp preview of the same "
            "release is copied to sorted/ rather than re-run.",
    "prior": "The alias behind the closest existing fit: same site first, otherwise "
             "the nearest penetration date for this animal. 'agrees' means the router "
             "picked that same alias.",
    "queue": "Ticked by default only when the routed alias has a release, matches the "
             "prior fit, and the session is not already queued or running.",
    "in_queue": "State of this exact job in tQueue: queued, running, done or dead. "
                "Running jobs are never reset; re-queueing a done or dead one re-runs it.",
}


def prior_fit(table, row):
    """``(alias, where)`` of the fit nearest to ``row``, or ``(None, '')``.

    Nearest means same site first, then the closest penetration date for the animal --
    the "same animal, same booth, same time period" check done by eye otherwise.
    """
    if table is None or not len(table) or "routing_alias" not in table:
        return None, ""
    done = table[table["routing_alias"].notna() & (table["routing_alias"] != "")]
    if not len(done):
        return None, ""
    site = done[done["siteid"] == row.get("siteid")]
    if len(site):
        return str(site.iloc[-1]["routing_alias"]), "same site"
    dates = pd.to_datetime(done["pendate"], errors="coerce")
    when = pd.to_datetime(row.get("pendate"), errors="coerce")
    if pd.isna(when) or dates.isna().all():
        return str(done.iloc[-1]["routing_alias"]), "animal"
    gap = (dates - when).abs()
    nearest = gap.idxmin()
    return (str(done.loc[nearest, "routing_alias"]),
            f"{done.loc[nearest, 'siteid']} · {int(gap[nearest].days)} d")


def queue_candidates(view, table, route, note_for, states):
    """One row per listed session missing a .dlc.h5, with the model it would run.

    ``route(video) -> {alias, release, unknown, error}``; ``note_for(video, alias,
    release)`` names the tQueue jobs that would do this work; ``states(notes) ->
    {note: state}``.  Passed in so
    this stays a pure function of the frame.
    """
    empty = pd.DataFrame(columns=QUEUE_COLUMNS + ["video"])
    if view is None or not len(view):
        return empty
    todo = view[view["needs"].str.contains("dlc", na=False) & view["video"].notna()]
    rows = []
    for _, r in todo.iterrows():
        routed = route(r["video"]) or {}
        alias, release = routed.get("alias"), routed.get("release")
        prior, where = prior_fit(table, r)
        rows.append(dict(
            session=r["session"], siteid=r["siteid"], video=r["video"],
            alias=alias or "", release=release or "",
            prior=f"{prior} ({where})" if prior else "",
            agrees=bool(alias) and alias == prior,
            unknown=routed.get("error") or routed.get("unknown") or "",
            notes=tuple(note_for(r["video"], alias, release)) if alias and release else ()))
    if not rows:
        return empty
    out = pd.DataFrame(rows)
    known = states(tuple(n for notes in out["notes"] for n in notes)) or {}
    rank = {"running": 0, "queued": 1, "dead": 2, "done": 3}
    out["in_queue"] = [min((known[n] for n in notes if n in known),
                           key=lambda q: rank.get(q, 9), default="")
                       for notes in out["notes"]]
    out["queue"] = [bool(a and rel and ok and q not in ("queued", "running"))
                    for a, rel, ok, q in zip(out["alias"], out["release"], out["agrees"],
                                             out["in_queue"])]
    return out[QUEUE_COLUMNS + ["video"]]


def render_queue(view, table):
    """The bulk-queue section under the table. Routing runs only once it is opened."""
    missing = (int((view["needs"].str.contains("dlc", na=False)
                    & view["video"].notna()).sum()) if len(view) else 0)
    if not missing:
        return
    st.divider()
    st.toggle(f"Queue DLC · {missing} missing", key="finder_queue_on",
              help=QUEUE_HELP["section"])
    if not st.session_state.get("finder_queue_on"):
        return

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: default to previews in temp; sorted is an explicit choice | date: 2026-09-22]
    dest = st.radio("Destination", ["temp", "sorted"], key="finder_queue_dest",
                    horizontal=True, help=QUEUE_HELP["dest"])
    # [AGENT EDIT END]
    mtime = core._registry_mtime()
    with st.spinner(f"routing {missing} sessions"):
        plan = queue_candidates(
            view, table, route=lambda v: core.routed_session(v, mtime),
            note_for=lambda v, a, r: core.finder_job_notes(v, a, r, dest),
            states=core.queue_states)

    # keyed by the session list: an edit made under one filter must not land on
    # whichever row occupies that position under the next. The destination is in the
    # key too, because 'in queue' -- and so the default ticks -- depend on it.
    key = ("finder_queue_edit_" + dest + "_"
           + str(abs(hash(tuple(plan["session"])))))
    edited = st.data_editor(
        plan[QUEUE_COLUMNS], key=key, hide_index=True, use_container_width=True,
        disabled=[c for c in QUEUE_COLUMNS if c != "queue"],
        column_config={
            "queue": st.column_config.CheckboxColumn("queue", help=QUEUE_HELP["queue"]),
            "prior": st.column_config.TextColumn("prior fit", help=QUEUE_HELP["prior"]),
            "agrees": st.column_config.CheckboxColumn("agrees", help=QUEUE_HELP["prior"]),
            "in_queue": st.column_config.TextColumn("in queue", help=QUEUE_HELP["in_queue"]),
            "unknown": st.column_config.TextColumn("router caveats"),
        })
    chosen = plan[edited["queue"].astype(bool).values]
    blocked = chosen[(chosen["alias"] == "") | (chosen["release"] == "")]
    chosen = chosen.drop(blocked.index)

    go, note = st.columns([1, 3])
    with go:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button (hover lists every job's queue message); failures in a dialog | date: 2026-09-28]
        core.action_button(f"Queue {len(chosen)} → {dest}", "finder_queue_go", queue_chosen,
                           args=(chosen, dest), type="primary", disabled=not len(chosen),
                           use_container_width=True)
        # [AGENT EDIT END]
    with note:
        if len(blocked):
            st.caption(f":orange[{len(blocked)} ticked with no released model · skipped]")


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the queue step as a job for core.action_button | date: 2026-09-28]
def queue_chosen(chosen, dest):
    """Queue every row of ``chosen``; one message per job, failures as errors."""
    results = [(r.session, core.queue_routed_fit(r.video, r.alias, r.release, dest))
               for r in chosen.itertuples(index=False)]
    core.queue_states.clear()
    failed = [(s, m) for s, m in results if str(m).startswith("Error")]
    for session, message in results:
        core.say("error" if (session, message) in failed else "success",
                 f"{session}: {message}")
    core.say("error" if failed else "success",
             f"{len(results) - len(failed)} of {len(results)} queued")
# [AGENT EDIT END]
# [AGENT EDIT END]


def render():
    core.retain_widget_values(
        "finder_animal finder_runclass finder_behavior finder_show_done "
        "finder_queue_on".split())

    animals = core.recorded_animals()
    if not animals:
        st.warning("no animals · celldb unreachable")
        return
    if st.session_state.get("finder_animal") not in animals:
        # the open session first, then the last one worked on; animals[0] is only a
        # floor, and it means "most recently recorded", which is rarely the right guess
        opened = animal_for(st.session_state.get("video_path", ""))
        remembered = core.load_user_prefs().get("finder_animal")
        st.session_state.finder_animal = next(
            (a for a in (opened, remembered) if a in animals), animals[0])

    pick, runclass_col, behavior_col, done_col, rescan_col = st.columns(
        [1.2, 2.0, 1.3, 0.9, 0.7])
    with pick:
        animal = st.selectbox("Animal", animals, key="finder_animal")
    core.remember_pref("finder_animal", animal)

    try:
        table = core.session_status(animal)
    except Exception as e:                                    # noqa: BLE001
        st.warning(f"session status unavailable: {e}")
        return

    # a session with no video is never listed: there is no DLC to run on it and no frame
    # to click through to. 40 of Reishi's NFB/NFC sessions are like this -- the headphone
    # rig records no camera -- and they are not work items.
    with_video = table[table["video"].notna()] if len(table) else table
    labels = runclass_options(table, with_video)
    runclasses = list(labels)
    prefs = core.load_user_prefs()
    stored = list(prefs.get("finder_runclass") or [])
    if "finder_runclass" not in st.session_state:
        st.session_state.finder_runclass = stored
    if "finder_show_done" not in st.session_state:
        st.session_state.finder_show_done = bool(prefs.get("finder_show_done"))
    # Streamlit raises on a stored value absent from the options, so the WIDGET is pruned
    # to what this animal actually has. The stored preference deliberately is not -- see
    # remember_filters. `offered` is what the PREFERENCE looks like here, which is the
    # only thing the live selection can be compared against: by the time this runs, a
    # click has already written its new value into session state.
    st.session_state.finder_runclass = [c for c in st.session_state.finder_runclass
                                        if c in runclasses]
    offered = [c for c in stored if c in runclasses]
    if "finder_behavior" not in st.session_state:
        st.session_state.finder_behavior = list(prefs.get("finder_behavior") or [])
    behaviors = sorted(table["behavior"].dropna().unique().tolist()) if len(table) else []
    st.session_state.finder_behavior = [b for b in st.session_state.finder_behavior
                                        if b in behaviors]
    with runclass_col:
        chosen = st.multiselect("Runclass", runclasses, key="finder_runclass",
                                format_func=lambda c: labels.get(c, c),
                                placeholder="all")
    with behavior_col:
        behavior = st.multiselect("Behavior", behaviors, key="finder_behavior",
                                  placeholder="all", help=HELP["behavior"])
    with done_col:
        show_done = st.checkbox("show finished", key="finder_show_done",
                                help=HELP["show finished"])
    remember_filters(chosen, offered, show_done, behavior)
    with rescan_col:
        st.write("")
        st.button("Rescan", key="finder_rescan", on_click=core.refresh_session_status,
                  help="Re-read sorted/ and gDLCFit for this animal.")

    if st.session_state.get("open_session_error"):
        st.warning(st.session_state.open_session_error, icon="⚠️")

    view = with_video
    if chosen:
        view = view[view["runclass"].isin(chosen)]
    if behavior:
        view = view[view["behavior"].isin(behavior)]
    outstanding = int((view["needs"] != "").sum()) if len(view) else 0
    if not show_done:
        view = view[view["needs"] != ""]

    st.caption(f"{animal} · {len(with_video)} with video · {outstanding} outstanding"
               f" · {len(view)} listed", help=HELP["needs"])
    st.caption("click a row to open that session", help=HELP["click"])
    if not len(view):
        st.success("nothing outstanding in this scope", icon="✅")
        return

    event = st.dataframe(view[COLUMNS], hide_index=True, use_container_width=True,
                         height=520, key="finder_table", on_select="rerun",
                         selection_mode="single-row")
    if handle_selection(view, event):
        st.rerun()
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: bulk-queue the routed model for listed missing-dlc sessions | date: 2026-09-22]
    render_queue(view, table)
    # [AGENT EDIT END]
# [AGENT EDIT END]
