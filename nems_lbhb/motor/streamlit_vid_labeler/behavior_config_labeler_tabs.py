"""Animal Tracking Tracker (formerly DLC Arena Labeler) — tabbed layout.

A drop-in alternative to ``behavior_config_labeler_plotly.py``: same functions, same
session-state keys, same jobs, but the lifecycle is split across tabs instead of six
columns and a sidebar.  The original script is untouched, so both layouts can be run
side by side:

    streamlit run behavior_config_labeler_tabs.py     # this one
    streamlit run behavior_config_labeler_plotly.py   # the original

Layout:
    session strip (always visible)  video path · frame · arena config · chips
    tab 1  Model            route it, run it, check it
    tab 2  Update model     add frames → curate → evaluate → retrain → release
    tab 3  Find sessions    what still needs DLC, per animal; click to open one
    tab 4  Database         registry, gDLCFit, tracking QC (read-only)
    tab 5  New DLC model    isolated create form (never touches the routed model)
    tab 6  Arena markers    click-to-place annotation + save the arena config
    tab 7  Arena geometry   scale, axis, fixation, speakers
"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labeler_core as core                                   # noqa: E402
from tabs import (arena_geometry, arena_markers, database,     # noqa: E402
                  load_panel, model, new_model, session_finder, update_model)

# [AGENT EDIT START | agent: claude | user: wingertj | reason: app renamed | date: 2026-09-25]
st.set_page_config(layout="wide", page_title="Animal Tracking Tracker")
# [AGENT EDIT END]

core.init_session_state(st.query_params)
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: preserve editing and manual choices across conditional sections | date: 2026-09-09]
core.retain_widget_values()
# [AGENT EDIT END]
core.inject_css()


# ── model resolution (no UI: the routing panel in tab 1 owns the widgets) ───────
def resolve_active_model():
    """Settle on one model for the whole page, from alias or raw override.

    Reads ``routing_alias`` and resolves it through ``dlc_registry.resolve``, which is
    what distinguishes a frozen RELEASE (what production inference runs, and what writes
    its id into the output) from the alias's live training PROJECT.  The raw picker is the
    escape hatch; choosing it means fits come out with no provenance at all, which tab 1
    surfaces rather than hides.

    ``sync_routing_for_video`` has already seeded ``routing_alias`` for the current
    video by the time this runs, so the suggestion applies on the same run the video
    changes -- see the comment on that function for why the order matters.
    """
    models = core.discover_dlc_models()
    prefs = core.load_user_prefs()

    def raw_model():
        raw = st.session_state.get("dlc_model_select") or prefs.get("dlc_model_select")
        if raw not in models:
            raw = next(iter(models), None)
        if not raw:
            return None
        return {"alias": None, "release_id": None, "config": models[raw],
                "project": str(Path(models[raw]).parent), "snapshot": None,
                "source": "raw", "via": "raw", "label": raw, "train_frame_size": None}

    if st.session_state.get("use_raw_model"):
        return raw_model(), models

    alias = st.session_state.get("routing_alias")
    if not alias and not st.session_state.get("routing_declined"):
        # remembered values carry a MANUAL choice forward; they must not substitute for a
        # routing decision the router declined to make for this video (see
        # core.sync_routing_for_video)
        alias = (st.session_state.get("active_model_alias")
                 or prefs.get("dlc_model_alias"))

    # the picker's explicit non-choice is not an alias -- resolving it would report a
    # bogus "does not resolve" instead of "you have not chosen one". Read the sentinel
    # off the panel that renders it so there is one definition; when the panel failed to
    # load there is no picker, so there is nothing to skip.
    _panel = load_panel("routing")
    if alias and alias == getattr(_panel, "NO_MODEL", None):
        st.session_state.pop("unresolved_alias", None)
        st.session_state.pop("active_model_alias", None)
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: a failed routing refresh must not arm an arbitrary raw model | date: 2026-09-09]
        return None, models
        # [AGENT EDIT END]
    if alias:
        resolved = core.resolve_alias(alias)
        if resolved and resolved.get("config"):
            st.session_state.active_model_alias = alias
            core.remember_pref("dlc_model_alias", alias)
            resolved["via"] = "alias"
            resolved["label"] = alias
            st.session_state.pop("unresolved_alias", None)
            return resolved, models
        st.session_state.unresolved_alias = alias

    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: keep explicit non-selection safe when metadata is unavailable | date: 2026-09-09]
    return (None, models) if st.session_state.get("routing_declined") else (raw_model(), models)
    # [AGENT EDIT END]


# ── session strip ──────────────────────────────────────────────────────────────
def on_video_path_change():
    candidate = (st.session_state.get("video_path_input_field") or "").strip()
    if not candidate:
        return
    if os.path.exists(candidate):
        st.session_state.video_path = candidate
        st.session_state.current_frame = None
        st.session_state.current_frame_key = None
        st.session_state.video_path_error = ""
    else:
        st.session_state.video_path_error = f"Not found: {candidate}"


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the full-width "In
# development" banner is dropped; which checkout queued jobs run stays one hover away, on
# the title | date: 2026-09-25]
_dev = core.code_banner()

# [AGENT EDIT START | agent: claude | user: wingertj | reason: renamed to Animal Tracking
# Tracker with a logo drawn from real DLC poses, and the video path moved up into the
# header row beside the session tags -- one row instead of two | date: 2026-09-25]
from logo import logo_svg

header_left, header_tail = st.columns([1.7, 3.9], vertical_alignment="center")
# the path is one more session badge, sitting immediately left of the others: one
# right-aligned flex row (core.inject_css styles st-key-session_badges), path first
header_right = header_tail.container(key="session_badges")
path_col = header_right
with header_left:
    st.markdown(f"<div style='display:flex;align-items:center;gap:14px'>"
                f"{logo_svg(72)}<span style='font-size:2rem;font-weight:700;"
                f"white-space:nowrap'>Animal Tracking Tracker</span></div>",
                unsafe_allow_html=True,
                help=(f"Queued jobs run `{_dev}`, not the deployed checkout."
                      if _dev else None))
# [AGENT EDIT END]

with path_col:
    # seed once, then let the widget own its key -- passing value= on every run while
    # the key is also in session_state is what makes Streamlit warn. This also applies
    # any pending retarget from core.open_session(), which cannot write these keys after
    # the widgets below exist; it must run before BOTH this box and the frame box.
    core.adopt_pending_session()
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: label collapsed now that the box sits in the header row | date: 2026-09-25]
    st.text_input("Video path", key="video_path_input_field",
                  on_change=on_video_path_change, label_visibility="collapsed",
                  placeholder="video path")
    # [AGENT EDIT END]
    if st.session_state.get("video_path_error"):
        st.caption(f":red[{st.session_state.video_path_error}]")

video_path = st.session_state.get("video_path", "")
video_ok = bool(video_path) and os.path.exists(video_path)

# Re-route BEFORE resolving, so a new video gets its own suggestion on this run rather
# than inheriting the previous video's alias from the selectbox's stored value.
if video_ok:
    core.sync_routing_for_video(video_path)

active, MODELS = resolve_active_model()
st.session_state.active_model = active
config_path = (active or {}).get("config") or ""
st.session_state.selected_dlc_model_config = config_path
st.session_state.available_dlc_models = MODELS
model_cfg = None
if config_path and os.path.exists(config_path):
    try:
        model_cfg = core.load_dlc_config_fields(config_path)
    except Exception as e:                                    # noqa: BLE001
        st.error(f"Failed to read {config_path}: {e}")

with header_right:
    session_name = (Path(video_path).parents[0].name if video_path else "—")
    alias_label = (active or {}).get("label") or "—"
    source = (active or {}).get("source", "—")
    chips = core.chip("session", session_name) + core.chip("model", alias_label, "accent")
    if source == "release":
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: "left_chimney_headphones-r0004" repeated the model tag beside it and pushed the tags onto two rows; show the release number, full id on hover | date: 2026-09-28]
        rid = (active or {}).get("release_id") or "?"
        short = rid[len(alias_label) + 1:] if rid.startswith(f"{alias_label}-") else rid
        chips += core.chip("release", short, "accent", title=rid)
        # [AGENT EDIT END]
    elif source in ("project", "raw", "guess"):
        # a fit from a mutable project carries no release id: nothing on disk records
        # which network produced it, which is what gDLCFit calls unattributable
        chips += core.chip("fits", "no provenance", "warn")
    if st.session_state.get("arena_config_dirty"):
        chips += core.chip("arena config", "unsaved", "warn")
    st.markdown(f"<div style='text-align:right;white-space:nowrap'>{chips}</div>",
                unsafe_allow_html=True)

if video_ok and st.session_state.get("auto_config_loaded_for") != str(Path(video_path)):
    core.load_sorted_config_for_session(video_path)

# [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame picker, the
# camera-override notice, the frame decode and the arena-config reload now live in the
# Arena markers tab (core.render_frame_picker / tabs.arena_markers), the only place that
# shows a frame from this box. Decoding here cost a seek+read on every tab | date: 2026-09-25]
# [AGENT EDIT END]
st.write("")

# [AGENT EDIT START | agent: Codex | user: wingertj | reason: Streamlit 1.40 tabs execute hidden panels; render only the selected section | date: 2026-09-09]
sections = {"Model": model, "Update model": update_model,
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: a cross-session
            # view of what still needs DLC, next to the other cross-session view
            # | date: 2026-09-15]
            "Find sessions": session_finder,
            # [AGENT EDIT END]
            "Database": database,
            "New DLC model": new_model, "Arena markers": arena_markers,
            "Arena geometry": arena_geometry}
section = core.tab_selector("Section", list(sections), key="app_section")
sections[section].render()
# [AGENT EDIT END]
