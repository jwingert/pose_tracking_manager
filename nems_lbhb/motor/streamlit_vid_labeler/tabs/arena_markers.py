"""Tab 5 — Arena markers: place annotation points on the frame, then save.

The frame sits on the left, shrinking to fit its column; add/remove, save/import and
the marker table sit beside it.  This is the only tab that takes clicks on a figure,
so the plotly_events key lives here alone.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_plotly_events import plotly_events

import labeler_core as core


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: marker clicks should rerun only the annotation editor | date: 2026-09-09]
@st.fragment
def render():
# [AGENT EDIT END]
    video_path = st.session_state.get("video_path", "")
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame box and the
    # arena-config status/reload moved here from the session strip -- this is the only
    # tab that shows a frame from them or edits the config | date: 2026-09-25]
    frame = _session_row(video_path)
    if frame is None:
        st.caption("no frame", help="Set a valid video path above; the frame loads "
                                    "from the session's ceiling view.")
        return
    # [AGENT EDIT END]

    _marker_toolbar(video_path)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the controls sat under
    # the frame while the page beside it was empty. The split is weighted by the
    # frame's native pixel width (640 or 1280 on these rigs) so a wide frame gets a wide
    # column; in a narrower window the frame shrinks to fit (fit_to_column)
    # | date: 2026-09-23]
    frame_col, control_col = st.columns([frame.width + 20, CONTROL_WIDTH])
    with frame_col:
        _frame_with_clicks(frame, video_path)
    with control_col:
        _add_remove(frame)
        st.divider()
        _save_import(video_path)
        st.divider()
        _positions()
    # [AGENT EDIT END]


# pixel weight of the control column against the frame's own width
CONTROL_WIDTH = 460
# room under the frame for the iframe body's margin; see fit_to_column
BOTTOM_MARGIN = 24


# [AGENT EDIT START | agent: claude | user: wingertj | reason: frame + arena-config row, moved from the session strip | date: 2026-09-25]
def _session_row(video_path):
    """Frame box, the config this session loaded, and reload -- one row."""
    frame_col, reload_col, config_col = st.columns([0.55, 0.12, 3.3])
    with frame_col:
        frame = core.render_frame_picker(video_path)
    with config_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        name = st.session_state.get("auto_config_name")
        if name:
            st.caption(f"config · {name}", help=st.session_state.get("auto_config_path"))
        elif video_path:
            st.caption("no config in sorted/",
                       help="Saving below writes a new one into this session's sorted/.")
        if core.arena_config_changed():
            st.caption(":orange[⚠ changed on disk]",
                       help="The loaded config was modified since it was read. Reload "
                            "to pick the changes up.")
        core.notes_caption(st.session_state.get("arena_config_notes") or [])
    with reload_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.button("↻", key="arena_reload",
                  disabled=not video_path or st.session_state.get("arena_config_dirty", False),
                  on_click=core.reload_arena_config,
                  help="Reload the arena config from disk. Save unsaved markers first.")
    return frame
# [AGENT EDIT END]


# ── toolbar: active point + overlay ────────────────────────────────────────────
def _marker_toolbar(video_path):
    points = st.session_state.points
    if points:
        if st.session_state.get("selected_point") not in points:
            st.session_state.selected_point = next(iter(points))

    point_col, overlay_col, file_col = st.columns([1.9, 0.7, 1.1])
    with point_col:
        if points:
            st.radio("Active point — then click the frame to place it",
                     options=list(points.keys()), horizontal=True, key="selected_point")
        else:
            st.caption("No points yet — add one below.")

    # deduped by real file and labelled by origin+release; see core.h5_options
    options = core.h5_options(core.session_h5_folders(video_path))
    with overlay_col:
        st.checkbox("DLC overlay", value=False, disabled=not options, key="am_overlay_on")
    with file_col:
        if options:
            core.h5_pick(options, "am_overlay_h5")            # drop a stale stored label
            st.selectbox("Prediction file", list(options),
                         key="am_overlay_h5", label_visibility="collapsed")
        else:
            st.caption("no .h5 for this session")


# ── the clickable frame ────────────────────────────────────────────────────────
def _frame_with_clicks(img, video_path):
    selected_name = st.session_state.get("selected_point")
    plotly_key_version = st.session_state.get("plotly_key_version", 0)
    click_key = f"plotly_click_{selected_name or 'none'}_{plotly_key_version}"

    click_data = st.session_state.get(click_key)
    if isinstance(click_data, str):
        try:
            import json
            click_data = json.loads(click_data)
        except Exception:                                     # noqa: BLE001
            click_data = None

    if click_data and selected_name in st.session_state.points:
        event = click_data[0]
        curve_num = event.get("curveNumber", None)
        # only clicks on the image trace (trace 0) move a marker
        if curve_num is None or curve_num == 0:
            x, y = int(event["x"]), int(event["y"])
        else:
            x = y = None
        old_x, old_y = st.session_state.points[selected_name]
        if x is not None and y is not None and (x, y) != (old_x, old_y):
            st.session_state.points[selected_name] = (x, y)
            st.session_state.arena_config_dirty = True

    fixation_point = core.compute_fixation_point_from_state()
    if fixation_point is not None:
        st.session_state.fixation_point = fixation_point

    fig = core.add_fixation_trace(core.create_plotly_figure(img))
    if st.session_state.get("am_overlay_on") and video_path:
        chosen = core.h5_pick(core.h5_options(core.session_h5_folders(video_path)),
                              "am_overlay_h5")
        if chosen:
            core.add_dlc_overlay(fig, chosen,
                                 st.session_state.get("frame_number_input", 0))

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame was drawn at
    # native pixel width, so a 1280 px frame was clipped in a column narrower than that.
    # Let it fill the column instead: clicks come back in DATA coordinates (image pixels),
    # so scaling the drawing does not move where a marker lands | date: 2026-09-23]
    fit_to_column(fig)
    plotly_events(
        core._PlotlyFigureWithConfig(fig, {"displayModeBar": False, "displaylogo": False,
                                           "responsive": True}),
        click_event=True, key=click_key, override_height=img.height + 60)


def fit_to_column(fig):
    """Size ``fig`` to its iframe rather than to the image's pixel size.

    The iframe is the column's width and ``img.height + 60`` tall, so the image shows at
    about native size when the column is wide enough and shrinks, aspect locked, when it
    is not. The bottom margin is not decoration: a figure filling the iframe loses its
    last ~16 px to the iframe body's margin, which cut markers on the bottom edge in
    half. (Pinning ``height`` instead was ignored on the first draw.)
    ``constrain='domain'`` shrinks the drawing area instead of padding the axis ranges,
    so the pixel ranges -- and therefore the clicked coordinates -- stay exactly the
    image's; anchoring left/top keeps the spare space at the right/bottom.
    """
    fig.update_layout(width=None, height=None, autosize=True,
                      margin=dict(l=0, r=0, t=0, b=BOTTOM_MARGIN),
                      # the click component does not follow the Streamlit theme; without
                      # this the spare space around the frame is a white slab on dark
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(constrain="domain", constraintoward="left")
    fig.update_yaxes(constrain="domain", constraintoward="top")
    return fig
# [AGENT EDIT END]


# ── add / remove ───────────────────────────────────────────────────────────────
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: callbacks edit marker selection before its widget is constructed | date: 2026-09-09]
def _add_point(frame):
    name = (st.session_state.get("am_new_point") or "").strip()
    if not name or name in st.session_state.points:
        return
    st.session_state.points[name] = (frame.width // 2, frame.height // 2)
    core.get_cmap_color()
    st.session_state.selected_point = name
    core.mark_arena_dirty()
    st.session_state.plotly_key_version = st.session_state.get("plotly_key_version", 0) + 1


def _add_remove(frame):
    st.markdown("##### Add / remove")
    with st.form("add_point", clear_on_submit=True):
        st.text_input("New point name", placeholder="e.g. corner_NW", key="am_new_point")
        st.form_submit_button("Add point", on_click=_add_point, args=(frame,))
# [AGENT EDIT END]

    if st.session_state.points:
        remove_col, btn_col = st.columns([1.4, 1])
        with remove_col:
            remove_point = st.selectbox("Point to remove",
                                        list(st.session_state.points.keys()),
                                        key="point_remove_select",
                                        label_visibility="collapsed")
        with btn_col:
            if st.button("Remove", key="am_remove_point"):
                del st.session_state.points[remove_point]
                core.get_cmap_color()
                st.session_state.arena_config_dirty = True
                st.session_state.plotly_key_version = (
                    st.session_state.get("plotly_key_version", 0) + 1)
                st.rerun()


def _positions():
    st.markdown("##### Marker positions")
    if not st.session_state.points:
        st.caption("No markers yet.")
        return
    rows = [{"marker": name, "x": round(float(x), 1), "y": round(float(y), 1)}
            for name, (x, y) in st.session_state.points.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if st.session_state.get("fixation_point"):
        fx, fy = st.session_state.fixation_point
        st.caption(f"fixation at {fx:.1f}, {fy:.1f} px "
                   f"(set in Arena geometry) · {st.session_state.ppm_avg:.0f} px/m")


# ── save / import ──────────────────────────────────────────────────────────────
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: imported widget values must be updated before constructing marker controls | date: 2026-09-09]
def _import_uploaded_config():
    uploaded = st.session_state.get("am_upload_config")
    if uploaded is not None:
        uploaded.seek(0)
        if core.import_config(uploaded):
            st.session_state.last_imported_config = f"{uploaded.name}:{uploaded.size}"
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: runs as an action_button job: a message on success, an exception on failure | date: 2026-09-28]
            return f"imported · {uploaded.name}"
        raise RuntimeError("; ".join(st.session_state.get("arena_config_notes")
                                     or ["import failed"]))
            # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: snap-confined browsers cannot
# read /auto, so the upload dialog can never reach a config on the share; browse and import
# from the SERVER, which is unconfined | date: 2026-09-16]
USERS_ROOT = "/auto/users"


def browse_default(video_path):
    """Where the browser opens: this session's ``sorted/``, else the user share.

    The session's own folder is the answer nine times out of ten -- the neighbouring
    configs you want to borrow are the ones sitting next to it.
    """
    if video_path:
        vp = Path(video_path)
        if len(vp.parents) >= 2:
            return str(vp.parents[1] / "sorted")
    return USERS_ROOT


def _jump(folder):
    st.session_state.am_browse_dir = str(folder)
    st.session_state.am_browse_pick = None


def browse_navigate():
    """Choosing a directory moves into it; choosing a file leaves it for Import.

    A callback rather than inline code because it rewrites ``am_browse_dir``, which is a
    text_input's key -- Streamlit refuses that once this run's widget exists.
    """
    label = st.session_state.get("am_browse_pick")
    if not label:
        return
    try:
        _folder, entries = core.browse_dir(st.session_state.get("am_browse_dir", ""))
    except OSError:
        return
    target = entries.get(label)
    if target is not None and target.is_dir():
        _jump(target)


def browse_selection():
    """The config currently picked, or None when it is a folder, absent, or gone.

    The folder is listed afresh every run, so a config can be renamed or moved between
    the run that offered it and the run that reads the pick.
    """
    label = st.session_state.get("am_browse_pick")
    if not label:
        return None
    try:
        _folder, entries = core.browse_dir(st.session_state.get("am_browse_dir", ""))
    except OSError:
        return None
    target = entries.get(label)
    return target if target is not None and target.is_file() else None


def _browse(video_path):
    """Pick a config off the server's own filesystem."""
    st.session_state.setdefault("am_browse_dir", browse_default(video_path))

    users_col, sorted_col = st.columns(2)
    with users_col:
        st.button(f"→ {USERS_ROOT}", key="am_jump_users", use_container_width=True,
                  on_click=_jump, args=(USERS_ROOT,))
    with sorted_col:
        st.button("→ session sorted/", key="am_jump_sorted", use_container_width=True,
                  disabled=not video_path, on_click=_jump,
                  args=(browse_default(video_path),))

    st.text_input("Folder", key="am_browse_dir",
                  help="Listed by the server, not the browser, so /auto is reachable. "
                       "Paste any path.")
    folder = st.session_state.am_browse_dir
    try:
        resolved, entries = core.browse_dir(folder)
    except OSError as e:                                      # noqa: BLE001
        st.warning(f"cannot list {folder} · {getattr(e, 'strerror', None) or e}")
        return

    labels = list(entries)
    if st.session_state.get("am_browse_pick") not in labels:
        st.session_state.am_browse_pick = None
    st.selectbox("Entries", labels, key="am_browse_pick", index=None,
                 placeholder=f"{len(labels)} item(s) in {resolved.name or '/'}",
                 on_change=browse_navigate)

    picked = browse_selection()
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
    core.action_button("Import from server", "am_browse_import", _import_from_server,
                       args=(picked,), type="primary", disabled=picked is None,
                       use_container_width=True)
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: action_button jobs for the
# arena config -- a message on success, an exception on failure | date: 2026-09-28]
def _import_from_server(path):
    if core.import_config_path(path):
        st.session_state.last_imported_config = str(path)
        return f"imported · {Path(path).name}"
    raise RuntimeError("; ".join(st.session_state.get("arena_config_notes")
                                 or [f"could not import {path}"]))


def _save_config(video_path):
    # read the name from the box NOW: a callback's args are frozen at the previous run,
    # so a name typed just before clicking would otherwise be ignored
    name = st.session_state.get("config_name_input") or core.default_config_name(video_path)
    os.makedirs(Path(name).parent, exist_ok=True)
    core.save_config(name, video_path)
# [AGENT EDIT END]
# [AGENT EDIT END]


def _save_import(video_path):
    st.markdown("##### Save configuration")
    if not video_path:
        st.info("Set a video path first.")
        return

    default_name = core.default_config_name(video_path)
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: defer directory creation to the explicit save action | date: 2026-09-09]
    # Browsing annotations does not create an output folder.
    # [AGENT EDIT END]
    config_name = st.text_input("Configuration name", value=default_name,
                                key="config_name_input")
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog. The output folder is still created only on save (_save_config) | date: 2026-09-28]
    core.action_button("Save configuration", "am_save", _save_config, args=(video_path,),
                       type="primary", use_container_width=True)
    # [AGENT EDIT END]
    st.caption("Writes the TOML plus the arena plot PNG, and a web JPG when the target "
               "is a sorted/ folder.")

    with st.expander("Import a config…", expanded=False):
        _browse(video_path)
        st.divider()
        uploaded_config = st.file_uploader("Upload config (.toml)", type=["toml"],
                                           help="Only for a file on the machine running "
                                                "your BROWSER. A snap-packaged browser "
                                                "cannot read /auto at all -- use the "
                                                "folder browser above for those.",
                                           key="am_upload_config")
        if uploaded_config is not None:
            file_id = f"{uploaded_config.name}:{uploaded_config.size}"
            if st.session_state.get("last_imported_config") != file_id:
                # [AGENT EDIT START | agent: Codex | user: wingertj | reason: run config import as a callback to avoid modifying already-rendered widgets | date: 2026-09-09]
                core.action_button("Import selected config", "am_import",
                                   _import_uploaded_config)
                # [AGENT EDIT END]
            else:
                st.caption("already imported", help="Upload a different file to re-import.")
        auto = st.session_state.get("auto_config_name")
        if auto:
            st.caption(f"auto-loaded from sorted/: {auto}")
