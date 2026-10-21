"""Inspect existing predictions without changing the model used for new jobs."""
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: independent fit inventory, linked visual selections and synchronized comparisons | date: 2026-09-10]
import hashlib
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import labeler_core as core
from nems_lbhb.motor import dlc_release


SLOTS = {"": "recording.avi", "_camB": "video_side.avi",
         "_camC": "video_side2.avi", "_pupil": "video_pupil.avi"}
VIEW_KEYS = {"Frame": "frame", "Tracking spread": "spread", "Labelled video": "video"}
VALUE_KEYS = ["inspect_a", "inspect_b", "inspect_unlinked", "inspect_cut",
              "inspect_frame", "inspect_clips", "inspect_seconds", "inspect_spread_plot", "model_view"]
VALUE_KEYS += [f"inspect_{view}_{side}" for view in VIEW_KEYS.values() for side in ("a", "b")]


def session_paths(video_path):
    vp = Path(video_path)
    if vp.name in SLOTS.values():
        return vp.parent.name, vp.parent, vp.parent.parent
    stem = re.sub(r"(_camB|_camC|_pupil)$", "", vp.stem)
    return stem, vp.parent / stem, vp.parent


@st.cache_data(ttl=15, max_entries=64, show_spinner=False)
def inventory(video_path, release_root):
    """Small file/manifest inventory only: never read HDF data or resolve routing."""
    session, session_dir, pen = session_paths(video_path)
    folders = {"temp": pen / "tmp" / "dlc" / session, "sorted": pen / "sorted"}
    found = {}
    for origin, folder in folders.items():
        for path in core.list_dlc_h5_files(folder):
            if not path.is_file():
                continue
            real = path.resolve()
            prefix = real.name.split("DLC", 1)[0]
            camera = next((slot for suffix, slot in SLOTS.items()
                           if prefix == session + suffix), None)
            # Old temp analyses used the raw camera-slot filename.
            if camera is None and origin == "temp":
                camera = next((slot for slot in SLOTS.values()
                               if prefix == Path(slot).stem), None)
            # A plain alias can still expose an unknown historical file. Do not
            # misattribute it to another session or silently guess its camera.
            named_alias = path.name in {session + suffix + ".dlc.h5" for suffix in SLOTS}
            if camera is None and not (origin == "temp" or named_alias):
                continue
            item = found.setdefault(str(real), {
                "id": str(real), "path": str(real), "origins": [], "roots": {},
                "camera": camera or "unknown", "video": str(session_dir / camera) if camera else "",
                "published": False, "release_id": None, "alias": None,
                "config": "", "provenance": "unknown", "warning": "",
            })
            if origin not in item["origins"]:
                item["origins"].append(origin)
            item["roots"][origin] = str(folder)
            item["published"] |= origin == "sorted" and named_alias
    for item in found.values():
        path = Path(item["path"])
        rid = path.parent.name
        if re.fullmatch(r".+-r\d+", rid):
            try:
                manifest = dlc_release.read_manifest(rid, root=release_root)
                suffix = manifest.get("model", {}).get("output_suffix", "")
                if not suffix or not path.stem.endswith(suffix) or manifest["release"]["id"] != rid:
                    raise ValueError("release manifest does not match prediction filename")
                item.update(release_id=rid, alias=manifest["release"]["alias"],
                            provenance="release",
                            warning=manifest["release"].get("provenance_warning", ""))
                config = Path(release_root) / rid / "config.yaml"
                if config.is_file():
                    item["config"] = str(config)
            except (OSError, ValueError, KeyError) as exc:
                item["warning"] = f"Unverified release {rid}: {exc}"
        item["label"] = " · ".join([
            item["release_id"] or path.stem,
            "/".join(item["origins"]), item["camera"],
            "published" if item["published"] else "not published"])
    return sorted(found.values(), key=lambda f: (not f["published"], f["label"], f["id"]))


def select_defaults(fits, state, session):
    """Preserve manual inspection across DB/route updates and file discovery."""
    if state.get("inspection_session") != session:
        state.pop("inspection_scores", None)
        state.pop("inspection_tables", None)
        for key in VALUE_KEYS:
            if key != "model_view":
                state.pop(key, None)
        state["inspection_session"] = session
    ids = [f["id"] for f in fits]
    for key in [k for k in VALUE_KEYS if k.endswith(("_a", "_b"))]:
        if key.endswith("_b") and key in state and state[key] is None:
            continue
        if state.get(key) not in ids:
            if key in ("inspect_a", "inspect_b"):
                state[key] = (ids[0] if ids else None) if key.endswith("_a") else None
            else:
                state.pop(key, None)


@st.cache_data(ttl=15, max_entries=64, show_spinner=False)
def reference_fingerprint(config_stamp):
    return core.hfqc.label_files_fingerprint(config_stamp[0])


def score_key(fit, cutoff):
    config = core.file_stamp(fit["config"]) if fit["config"] else None
    return (core.file_stamp(fit["path"]),
            config, reference_fingerprint(config) if config else None, float(cutoff))


@st.cache_data(max_entries=24, show_spinner=False)
def _score(key):
    prediction, config, labels_fp, cutoff = key
    reference, warning = None, None
    if config:
        cfg = core.read_yaml(config[0]) or {}
        if Path(cfg.get("project_path", Path(config[0]).parent)).resolve() != Path(config[0]).parent.resolve():
            warning = "Release config points outside its frozen directory; using within-video QC only."
        else:
            reference = core._hfqc_reference(config[0], config[1], labels_fp)
    result = core.hfqc.score_video(prediction[0], reference, p_cut=cutoff, track_p_cut=cutoff)
    if warning:
        result.warnings.append(warning)
    return result


def compute_scores(fits, cutoff):
    results = st.session_state.setdefault("inspection_scores", {})
    for fit in fits:
        try:
            key = score_key(fit, cutoff)
            results[fit["id"]] = (key, _score(key), None)
        except Exception as exc:
            results[fit["id"]] = (None, None, str(exc))


def existing_score(fit, cutoff):
    stored = st.session_state.get("inspection_scores", {}).get(fit["id"])
    if stored:
        if stored[2]:
            return None, stored[2]
        try:
            if stored[0] == score_key(fit, cutoff):
                return stored[1], None
        except OSError:
            pass
    return None, "not computed"


def fit_rows(fits, cutoff):
    rows = []
    for fit in fits:
        result, error = existing_score(fit, cutoff)
        summary = core.hfqc.summarize(result) if result is not None else {}
        rows.append({"Fit": fit["release_id"] or Path(fit["path"]).stem,
                     "Location": "/".join(fit["origins"]), "Camera": fit["camera"],
                     "Published": fit["published"], "Provenance": fit["provenance"],
                     "Tracked (mean)": summary.get("track_mean"),
                     "Tracked (worst)": summary.get("track_worst"),
                     "Spread (HL)": summary.get("rms_spread_proc_mean"),
                     "Spread / training": summary.get("ratio_mean"),
                     "QC": error or ("computed; no training reference" if not result.has_reference else "computed"),
                     "Path": fit["path"]})
    return pd.DataFrame(rows)


def pair_compatible(fits):
    return all(f["video"] and f["camera"] != "unknown" for f in fits) and len({f["video"] for f in fits}) == 1


def use_alias(alias):
    st.session_state.update(routing_alias=alias, routing_manual=True, use_raw_model=False)
    st.session_state["inspection_reroute_requested"] = True


def publish_fit(fit, video_path):
    """Explicit action only; revalidate selected bytes and current release at click time."""
    from nems_lbhb.motor import dlc_fit_script as fs
    resolved = core.resolve_alias(fit["alias"]) or {}
    if not fit["release_id"] or resolved.get("release_id") != fit["release_id"]:
        raise ValueError("Only the alias's currently deployed release can be published. No job queued.")
    root = fit["roots"].get("temp")
    session, _, _ = session_paths(video_path)
    expected_video, _ = core.ceiling_video(video_path)
    if os.path.realpath(expected_video) != os.path.realpath(fit["video"]):
        raise ValueError("This fit uses a different camera from the current session camera mapping.")
    candidate = fs.publishable_preview(root, session, fit["release_id"])
    if not candidate or os.path.realpath(candidate) != fit["path"]:
        raise ValueError("The temp release pointer no longer identifies this exact fit. Refresh existing fits.")
    return core.queue_dlc_job(video_path, None, None, "publish", alias=fit["alias"],
                              src=root, fallback=False, extra_note="publish_inspected_fit",
                              extra_args={"publish_token": fs.publication_token(candidate, fit["release_id"])})


def render_actions(fit, video_path):
    with st.expander("Actions for inspected fit A", expanded=False):
        st.caption(fit["label"])
        resolved = core.resolve_alias(fit["alias"]) if fit["alias"] else None
        if resolved:
            st.caption(f"next job · {resolved.get('release_id') or 'mutable project'}")
        st.button("Use this alias for next job", key="inspect_use_alias", disabled=not resolved,
                  on_click=use_alias, args=(fit["alias"],))
        if st.session_state.pop("inspection_reroute_requested", False):
            st.rerun()  # refresh the routing panel outside this fragment, too
        publishable = bool(resolved and fit["release_id"] and
                           resolved.get("release_id") == fit["release_id"] and
                           "temp" in fit["origins"] and not fit["published"])
        # Confirmation belongs to this exact fit, never a previously selected one.
        confirm_key = "inspect_publish_confirm_" + hashlib.sha256(fit["id"].encode()).hexdigest()[:12]
        confirmed = st.checkbox("Replace this session's published fit with fit A", key=confirm_key,
                                disabled=not publishable)
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
        core.action_button("Publish inspected fit → sorted", "inspect_publish", publish_fit,
                           args=(fit, video_path), disabled=not (publishable and confirmed))
        # [AGENT EDIT END]
        st.caption("publish · needs a temp fit from this alias's current release",
                   help="Browsing never publishes. The published fit is the one "
                        "`<session>.dlc.h5` points at.")


# [AGENT EDIT START | agent: claude | user: wingertj | reason: eleven widgets, five prose
# captions and a full table stood between the operator and the picture. Three controls
# steer; everything touched once a session moved behind one icon | date: 2026-09-16]
def _settings(fits, cutoff):
    """Setup, not steering: what you touch once a session, behind one icon."""
    with st.popover("⚙", use_container_width=True):
        if st.button("Refresh existing fits", key="inspect_refresh"):
            inventory.clear()
            reference_fingerprint.clear()
        st.checkbox("Override selection per view", key="inspect_unlinked")
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: "Per-marker table" and "Compare with recorded population" became plots in the Tracking spread view's strip | date: 2026-09-25]
        # [AGENT EDIT END]
        if st.button("Compute statistics for all fits", key="inspect_score_all"):
            with st.spinner("Scoring existing predictions (no inference)..."):
                compute_scores(fits, cutoff)
        st.dataframe(fit_rows(fits, cutoff), hide_index=True, use_container_width=True)
        st.caption("same cutoff for every row",
                   help="Spread is self-Procrustes RMS in head-lengths; training ratios "
                        "use each fit's own reference when one exists.")


def _per_view_selection(view, lookup, label, a, b):
    """Independent A/B for this view only. The first override starts where linked left."""
    suffix = VIEW_KEYS[view]
    ka, kb = f"inspect_{suffix}_a", f"inspect_{suffix}_b"
    st.session_state.setdefault(ka, a)
    st.session_state.setdefault(kb, b)
    col_a, col_b = st.columns(2)
    with col_a:
        a = st.selectbox(f"{view} A", list(lookup), format_func=label, key=ka)
    with col_b:
        b = st.selectbox(f"{view} B", [None] + list(lookup), format_func=label, key=kb)
    return a, b


def render(video_path):
    # Detach only value widgets, not buttons/confirmation events, before rendering.
    core.retain_widget_values(VALUE_KEYS)
    if not video_path:
        st.info("Select a session video to inspect existing fits.")
        return
    fits = inventory(str(video_path), str(dlc_release.RELEASE_ROOT))
    select_defaults(fits, st.session_state, str(video_path))
    if not fits:
        st.markdown("##### Inspect existing fits")
        st.info("No existing predictions found in temp or sorted for this session.")
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the session strip no longer decodes a frame on every tab, so decode it here when needed | date: 2026-09-25]
        frame = core.ensure_current_frame(video_path,
                                          st.session_state.get("frame_number_input", 0))
        if frame is not None:
            st.image(frame, use_container_width=True)
        # [AGENT EDIT END]
        return

    lookup = {f["id"]: f for f in fits}
    label = lambda value: lookup[value]["label"] if value is not None else "None"     # noqa: E731

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the title and the A/B
    # row were two lines; now one, like the model row above it. Widget labels would push
    # the boxes below the title, so A/B ride inside the dropdowns and the cutoff gets an
    # inline label | date: 2026-09-25]
    title_col, a_col, b_col, cut_label_col, cut_col, gear_col = st.columns(
        [0.95, 2.0, 2.0, 0.28, 0.5, 0.26], gap="small", vertical_alignment="center")
    with title_col:
        st.markdown("<p style='font-size:1.25rem;font-weight:600;margin:0;line-height:1.2;"
                    "white-space:nowrap;position:relative;top:-8px'>Inspect existing fits</p>",
                    unsafe_allow_html=True)
    with a_col:
        a = st.selectbox("A", list(lookup), key="inspect_a", label_visibility="collapsed",
                         format_func=lambda v: f"A · {label(v)}")
    with b_col:
        b = st.selectbox("B", [None] + list(lookup), key="inspect_b",
                         label_visibility="collapsed", placeholder="B · none",
                         format_func=lambda v: "B · none" if v is None else f"B · {label(v)}")
    with cut_label_col:
        st.markdown("<p style='font-size:14px;margin:0;text-align:right;position:relative;"
                    "top:-8px'>cutoff</p>", unsafe_allow_html=True)
    # [AGENT EDIT END]
    with cut_col:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: value= plus a stored
        # key raised Streamlit's "default value but also set via Session State" box on
        # every rerun; seed the key instead, as the other widgets do | date: 2026-09-25]
        st.session_state.setdefault("inspect_cut", float(core.SPREAD_P_CUT))
        # [AGENT EDIT END]
        cutoff = st.number_input("cutoff", min_value=0.0, max_value=1.0, step=0.05,
                                 key="inspect_cut", label_visibility="collapsed",
                                 help="Shared likelihood cutoff. Inspection is "
                                      "independent of routing, and low spread or high "
                                      "confidence alone does not prove correct tracking.")
    with gear_col:
        _settings(fits, cutoff)

    view = core.tab_selector("View", list(VIEW_KEYS), key="model_view")
    if st.session_state.get("inspect_unlinked"):
        a, b = _per_view_selection(view, lookup, label, a, b)

    selected = [lookup[a]] + ([lookup[b]] if b and b != a else [])
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: terse tags, full text in hover | date: 2026-09-25]
    for side, fit in zip(("A", "B"), selected):
        core.notes_caption([fit["warning"]], prefix=side if len(selected) > 1 else "",
                           fallback="release note")
    if len(selected) > 1 and not pair_compatible(selected):
        st.caption(":orange[⚠ A/B cameras differ]",
                   help="A/B needs the same known camera. Select each separately.")
        return
    # [AGENT EDIT END]

    if view == "Frame":
        render_frames(selected, cutoff)
    elif view == "Tracking spread":
        render_spread(selected, cutoff)
    else:
        render_montage(selected, cutoff)

    # below the view on purpose: you publish because of what you just saw
    render_actions(lookup[a], video_path)
# [AGENT EDIT END]


def prediction_table(fit):
    """Session LRU avoids copying entire HDF tables on every A/B frame change."""
    stamp = core.file_stamp(fit["path"])
    tables = st.session_state.setdefault("inspection_tables", {})
    if stamp not in tables:
        if len(tables) >= 2:
            tables.pop(next(iter(tables)))
        tables[stamp] = core._prediction_data(stamp)
    table = tables[stamp]
    if table.columns.nlevels != 3 or len(table.columns.get_level_values(0).unique()) != 1:
        raise ValueError("Viewer needs a single-animal, single-scorer prediction table.")
    return table


def marker_values(row, cutoff):
    sub = row.xs(row.index.get_level_values(0)[0], level=0)
    for part in sub.index.get_level_values(0).unique():
        values = sub[part]
        x, y, p = values.get("x", np.nan), values.get("y", np.nan), values.get("likelihood", 1.0)
        if np.isfinite([x, y, p]).all() and p >= cutoff:
            yield str(part), float(x), float(y)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame view overlaid
# markers unconditionally once _overlay_frame was orphaned, leaving no way to see the
# frame underneath -- which is the whole question when a fit looks wrong | date: 2026-09-16]
def markers_wanted():
    """Whether the frame view should draw markers. On unless turned off.

    The overlay is why this view exists, so ``off`` is the deliberate act.
    """
    return bool(st.session_state.get("inspect_markers", True))
# [AGENT EDIT END]


def render_frames(fits, cutoff):
    if not pair_compatible(fits) or not Path(fits[0]["video"]).is_file():
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: terse tags, full text in hover | date: 2026-09-25]
        st.caption(":orange[⚠ source video missing]",
                   help="The fit's source camera video is unknown or missing; refusing "
                        "to overlay on a different camera.")
        # [AGENT EDIT END]
        return
    try:
        tables = [prediction_table(f) for f in fits]
        count = core.video_frame_count(fits[0]["video"])
        if any(len(t) != count for t in tables):
            raise ValueError("Prediction/video frame counts differ; alignment must be resolved before overlaying.")
        st.session_state["inspect_frame"] = min(int(st.session_state.get("inspect_frame", 0)), count - 1)
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame box and the
        # Markers toggle took two full-width rows above the frame; they sit in a narrow
        # column beside it instead | date: 2026-09-25]
        control_col, view_col = st.columns([0.16, 1], gap="medium")
        with control_col:
            frame_number = st.number_input("Frame", min_value=0, max_value=count - 1,
                                           step=1, key="inspect_frame",
                                           help=f"Shared by A and B · {count} frames")
            st.checkbox("Markers", key="inspect_markers", value=True,
                        help="Off shows the raw frame, which is how you tell a marker in "
                             "the wrong place from an animal in an unusual pose.")
        # [AGENT EDIT END]
        frame = core.read_frame(fits[0]["video"], int(frame_number))
        import plotly.graph_objects as go
        show = markers_wanted()
        for col, side, table in zip(view_col.columns(len(fits)), ("A", "B"), tables):
            with col:
                fig = core.create_plotly_figure(frame)
                for name, x, y in (marker_values(table.iloc[int(frame_number)], cutoff)
                                   if show else ()):
                    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[name],
                                            marker=dict(color="cyan", size=8), showlegend=False))
                st.plotly_chart(fig, use_container_width=True, key=f"inspect_frame_{side}")
    except Exception as exc:
        st.error(f"Frame inspection unavailable: {exc}")


def render_spread(fits, cutoff):
    """Both fits overlaid in each plot, one plot at a time.

    Side-by-side columns made the comparison something you reconstructed by looking back
    and forth; overlaying them makes it something you see. Colour still means bodypart --
    which fit a point belongs to is its SYMBOL.

    [AGENT EDIT | agent: claude | user: wingertj | reason: the scatter, the bars, the
    database comparison and the per-marker table were stacked down the page, and two of
    them hid behind checkboxes in the settings popover. They are alternate answers to the
    same question, so they are now a strip of plots to switch between -- every one named,
    one click away, and only the chosen one built | date: 2026-09-25]
    """
    if st.button("Compute / refresh selected statistics", key="inspect_score_selected"):
        with st.spinner("Scoring selected predictions..."):
            compute_scores(fits, cutoff)

    scored = []
    for side, fit in zip(("A", "B"), fits):
        result, error = existing_score(fit, cutoff)
        if result is None:
            st.info(f"{side}: {error}")
        else:
            scored.append((side, fit, result))
    if not scored:
        return
    pairs = [(side, result) for side, _fit, result in scored]

    plot = core.tab_selector("Spread plot", list(SPREAD_PLOTS), key="inspect_spread_plot",
                             format_func=lambda k: SPREAD_PLOTS[k])

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: at full page width the
    # plots stretched to ~1900 px -- the 1:1 scatter drew a small square in a wide empty
    # frame and the bars ran the whole page. Each plot gets a column sized to its shape;
    # the summary and the caveats sit in the space beside it | date: 2026-09-28]
    share = PLOT_WIDTH[plot]
    if share < 1:
        plot_col, side_col = st.columns([share, 1 - share], gap="large")
    else:
        side_col, plot_col = st.container(), st.container()

    # the release note already sits above every view; do not repeat it per plot
    notes = {f["id"]: f.get("warning") for f in fits}
    with side_col:
        for side, fit, result in scored:
            st.caption(f"{side} · {core.hfqc_summary_chips(result)}",
                       help=core._hfqc_frame_caption(result))
            extra = [w for w in result.warnings if w != notes.get(fit["id"])]
            if not result.has_reference:
                extra.append("No frozen training reference: within-video QC only.")
            core.notes_caption(extra, prefix=side)

    with plot_col:
        if plot == "scatter":
            fig = core.hfqc_compare_figure(pairs)
            fig.update_layout(height=SCATTER_HEIGHT)      # ~square in its column
            st.plotly_chart(fig, use_container_width=True, key="inspect_spread_compare")
        elif plot == "bars":
            st.plotly_chart(core.hfqc_compare_bars(pairs), use_container_width=True,
                            key="inspect_bars_compare")
        elif plot == "database":
            # one column per fit: each is compared against its OWN model's population
            for col, (side, fit, result) in zip(st.columns(len(scored)), scored):
                with col:
                    if len(scored) > 1:
                        st.caption(f"**{side}** · {fit['alias'] or 'unknown model'}")
                    if fit["alias"]:
                        core._hfqc_population_section(result, cutoff, cutoff, stacked=True,
                                                      alias=fit["alias"],
                                                      key_prefix=f"inspect_pop_{side}",
                                                      fig_height=220)
                    else:
                        st.caption(f"{side} · unknown model · no population comparison")
        else:
            for side, result in pairs:
                st.caption(f"{side} · per-marker")
                st.dataframe(core._hfqc_style_table(result.table, result.has_reference),
                             use_container_width=True)
    # [AGENT EDIT END]


# share of the page each spread plot takes (the rest holds the summary); the table
# keeps the full width because its columns need it
PLOT_WIDTH = {"scatter": 0.36, "bars": 0.55, "database": 0.55, "table": 1.0}
SCATTER_HEIGHT = 620


# the Tracking spread view's plots, in the order you usually want them
SPREAD_PLOTS = {"scatter": "Spread", "bars": "Per marker", "database": "Vs database",
                "table": "Table"}


def montage_path(fits, cutoff, clips, seconds):
    identity = ("inspection-v1", tuple(core.file_stamp(f["path"]) for f in fits),
                core.file_stamp(fits[0]["video"]), float(cutoff), int(clips), float(seconds))
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()
    return core._montage_cache_dir() / f"{digest}.mp4"


def make_montage(fits, cutoff, clips, seconds):
    """One composed A/B video: identical source frames and a single playback clock."""
    if not pair_compatible(fits):
        raise ValueError("Synchronized montage requires the same known camera.")
    import imageio
    target = montage_path(fits, cutoff, clips, seconds)
    tables = [prediction_table(f) for f in fits]
    cap = cv2.VideoCapture(fits[0]["video"])
    scratch = None
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or any(len(t) != count for t in tables):
            raise ValueError("Prediction/video frame counts differ or source video is unreadable.")
        n = max(1, int(float(seconds) * fps))
        if count < int(clips) * n:
            raise ValueError("Video is too short for the requested montage.")
        starts = np.linspace(0, count - n, int(clips), dtype=int)
        fd, name = tempfile.mkstemp(dir=target.parent, suffix=".mp4")
        os.close(fd)
        scratch = Path(name)
        with imageio.get_writer(str(scratch), format="FFMPEG", fps=fps, codec="libx264",
                                pixelformat="yuv420p", output_params=["-movflags", "+faststart"]) as writer:
            for start in starts:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
                for index in range(int(start), int(start) + n):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Video decode failed at frame {index}; no partial montage saved.")
                    panels = []
                    for side, table in zip(("A", "B"), tables):
                        panel = frame.copy()
                        for part, x, y in marker_values(table.iloc[index], cutoff):
                            if 0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]:
                                cv2.circle(panel, (round(x), round(y)), 4, (255, 255, 0), -1)
                                cv2.putText(panel, part, (round(x) + 5, round(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                            .35, (255, 255, 0), 1)
                        cv2.putText(panel, f"{side} | frame {index} | {index/fps:.2f}s", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                        panels.append(panel)
                    writer.append_data(cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_BGR2RGB))
        check = cv2.VideoCapture(str(scratch))
        try:
            if not check.read()[0]:
                raise RuntimeError("Encoder produced no decodable video; check server FFmpeg output.")
        finally:
            check.release()
        os.replace(scratch, target)
        return target
    finally:
        cap.release()
        if scratch is not None and scratch.exists():
            scratch.unlink()


def render_montage(fits, cutoff):
    if not pair_compatible(fits) or not Path(fits[0]["video"]).is_file():
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: terse tags, full text in hover | date: 2026-09-25]
        st.caption(":orange[⚠ source video missing]",
                   help="Source camera video unknown or missing; montage unavailable.")
        # [AGENT EDIT END]
        return
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: default montage to 10 clips x 2 s | date: 2026-09-23]
    clips = st.number_input("Clips", min_value=1, max_value=10, value=10, key="inspect_clips")
    seconds = st.number_input("Seconds per clip", min_value=.5, max_value=2.0, value=2.0,
                              step=.5, key="inspect_seconds")
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: terse tags, full text in hover | date: 2026-09-25]
    st.caption("preview · no inference",
               help="Short preview from the selected predictions; no full labeled video "
                    "or inference job required. A/B shares one playback clock.")
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
    def _make():
        with st.spinner("Creating synchronized preview..."):
            return f"montage ready · {make_montage(fits, cutoff, clips, seconds).name}"
    core.action_button("Generate selected montage", "inspect_montage", _make)
    try:
        target = montage_path(fits, cutoff, clips, seconds)
        if target.is_file():
            st.video(str(target))
        else:
            st.caption("no montage yet")
    except Exception as exc:                                  # noqa: BLE001
        core.notes_caption([f"Montage unavailable ({type(exc).__name__}): {exc}"])
    # [AGENT EDIT END]
    with st.expander("Existing full labeled videos", expanded=False):
        for side, fit in zip(("A", "B"), fits):
            video = Path(fit["path"]).with_name(Path(fit["path"]).stem + "_labeled.mp4")
            if video.is_file():
                identity = hashlib.sha256(repr(core.file_stamp(video)).encode()).hexdigest()[:12]
                if st.checkbox(f"Load full video {side} ({video.stat().st_size / 1024**2:.0f} MB)", key=f"inspect_full_{side}_{identity}"):
                    st.video(str(video))
            else:
                st.caption(f"{side} · none", help="No full labeled video for this exact fit.")
# [AGENT EDIT END]
