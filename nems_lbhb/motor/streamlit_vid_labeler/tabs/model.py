"""Tab 1 — Model: what tracks this session, run it, check the result.

One question per section: which model (routing + override), what that resolves to,
run it, is the output any good, what is on record for this session.  Nothing here
changes a model — see the Update model tab.
"""
import os
from pathlib import Path

import streamlit as st

import labeler_core as core
from tabs import load_panel


def render():
    video_path = st.session_state.get("video_path", "")
    config_path = st.session_state.get("selected_dlc_model_config", "")

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the frame sat below the
    # routing, the run row and the QC section, so the one thing the user is here to LOOK at
    # was three scrolls down. Controls go in a narrow left column, the frame gets the right
    # half at the top of the page, and the QC/labelled-video views become tabs BESIDE it
    # rather than more page underneath it. | date: 2026-08-17]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the controls are all
    # text, the inspection is all graphics -- controls become one page-wide row on top
    # (which model | run it) and the inspection gets the full width under it | date: 2026-09-25]
    pick_col, run_col = st.columns([2.6, 1.2], gap="medium")
    with pick_col:
        _routing(video_path, config_path)
        _resolved_problems()
    with run_col:
        _run(video_path, config_path)
    st.divider()
    _viewer(video_path, config_path)
    # [AGENT EDIT END]

    st.divider()
    _outputs(video_path, config_path)
    # [AGENT EDIT END]


# ── which model ────────────────────────────────────────────────────────────────
def _routing(video_path, config_path):
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: no heading -- the picker
    # says what it is; the routing refresh and the Geometry/Legacy choice sit in the
    # picker row's ⚙ | date: 2026-09-25]
    if not video_path:
        st.caption("no video", help="Set a video path above to get a routing suggestion.")
        return
    # [AGENT EDIT END]

    panel = load_panel("routing")
    if panel is None:
        st.caption("routing panel unavailable — use the project override below.")
        _raw_override(config_path)
        return

    # The panel owns the method radio, the evidence line and the alias selectbox
    # (key 'routing_alias'), which the entry script has already seeded for this video.
    # expanded=None: this tab is already a section, and the panel nests an expander for
    # extra candidates -- Streamlit forbids expander-in-expander.
    result = core.routing_suggestion(video_path, core.routing_method())
    try:
        # the project override is the same decision one level down, so it rides in the
        # panel's picker row rather than stacking a fourth full-width control under it
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: compact routing row -- evidence, reasons and the resolved release behind one info popover | date: 2026-09-25]
        panel.render(video_path, expanded=None, key="routing", result=result,
                     extra=lambda: _raw_override(config_path),
                     info=lambda res: _routing_info(res, panel, config_path),
                     settings=core.routing_status)
        # [AGENT EDIT END]
    except TypeError:                                         # older panel signature
        panel.render(video_path, expanded=None, key="routing")
        _raw_override(config_path)


def _raw_override(config_path):
    """The same choice one level down: a project config instead of an alias."""
    models = st.session_state.get("available_dlc_models") or {}
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: shorter label | date: 2026-09-25]
    with st.expander("Advanced: raw project", expanded=False):
    # [AGENT EDIT END]
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: expose invalidation after external projects/jobs change | date: 2026-09-09]
        st.button("Refresh projects and files", key="model_refresh_files",
                  on_click=core.refresh_files)
        # [AGENT EDIT END]
        if not models:
            st.warning("No DLC models found under /auto/data/dlc/")
            return

        names = list(models)
        prefs = core.load_user_prefs()
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the removed "from
        # project config" checkbox ran the routed model's own training project; default
        # to that project so ticking the box below does the same by default | date: 2026-09-25]
        routed = core.routed_project_config()
        routed_name = Path(routed).parent.name if routed else None
        default = (st.session_state.get("dlc_model_select")
                   or (routed_name if routed_name in names else None)
                   or prefs.get("dlc_model_select") or names[0])
        # [AGENT EDIT END]

        def _remember():
            core.remember_pref("dlc_model_select", st.session_state.dlc_model_select)

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: index= plus a retained
        # key raised Streamlit's "default value but also set via Session State" box; seed
        # the key instead | date: 2026-09-25]
        if st.session_state.get("dlc_model_select") not in names:
            st.session_state.dlc_model_select = default if default in names else names[0]
        st.selectbox("DLC project", names, key="dlc_model_select", on_change=_remember)
        # [AGENT EDIT END]
        st.checkbox("use this project instead of the release", key="use_raw_model",
                    help="Bypasses the registry. The fit carries no release id, so "
                         "nothing on disk records which network produced it.")

        # notes belong to the mutable project; a release config is read-only
        project_config = core.project_config_path()
        if project_config and os.path.exists(project_config):
            st.text_area("Notes", key="dlc_notes", height=110,
                         on_change=core.save_notes_callback,
                         help=f"Saved into {Path(project_config).parent.name}/config.yaml")
            st.caption("bodyparts: "
                       + ", ".join(st.session_state.get("dlc_bodyparts", []) or ["—"]))


# ── what it resolved to ────────────────────────────────────────────────────────
# [AGENT EDIT START | agent: claude | user: wingertj | reason: the release was stated in
# the header chip, the suggestion banner and this line. The header keeps it; the evidence,
# the reasons and the resolved details sit behind one popover at the end of the picker
# row; only a problem is still said inline | date: 2026-09-25]
def _resolved_problems():
    """Inline, and only when something is wrong with what will run."""
    model = core.active_model()
    if st.session_state.get("unresolved_alias") and not st.session_state.get("use_raw_model"):
        st.caption(f":orange[⚠ `{st.session_state.unresolved_alias}` does not resolve]")
    if model.get("source") in ("project", "guess", "raw"):
        st.caption(":orange[⚠ training project · no provenance]",
                   help="Fits from a training project carry no release id, so nothing "
                        "on disk records which network produced them.")


def resolved_markdown(config_path):
    """What the selection resolved to, for the info popover."""
    model = core.active_model()
    source = model.get("source", "—")
    lines = [f"**release `{model.get('release_id')}`**" if source == "release" else
             "**training project** — fits carry no provenance"
             if source in ("project", "guess", "raw") else "**—**"]
    if not (config_path and os.path.exists(config_path)):
        return "\n".join(lines + ["- no config resolved yet"])
    if model.get("snapshot"):
        lines.append(f"- snapshot {model['snapshot']}")
    try:
        cfg, _project_path, _iterations = core.model_iterations(config_path)
        lines.append(f"- iteration-{cfg.get('iteration')} · snapshotindex "
                     f"{cfg.get('snapshotindex')}")
        videos, frames = core.labeled_frame_counts(config_path)
        if frames:
            lines.append(f"- {frames} labelled frames · {videos} label folder(s)")
    except Exception as e:                                    # noqa: BLE001
        lines.append(f"- config unreadable: {e}")
    if model.get("train_frame_size"):
        w, h = model["train_frame_size"]
        lines.append(f"- trained on {w}×{h}; wider video is analysed on a downscaled "
                     f"copy and converted back to native pixels")
    lines.append(f"- `{config_path}`")
    return "\n".join(lines)


def _routing_info(result, panel, config_path):
    """The one place that explains the routing row: evidence, reasons, what runs."""
    # rendered inside the routing row's ⚙, under the method and refresh controls
    if result.get("method") == "geometry":
        st.markdown("**Observed**")
        st.markdown(panel.observation_markdown(result.get("observed") or {}))
    st.markdown("**Suggested**")
    st.markdown(panel.suggestion_markdown(result))
    st.markdown("**Runs**")
    st.markdown(resolved_markdown(config_path))
# [AGENT EDIT END]


# ── run it ─────────────────────────────────────────────────────────────────────
def _run(video_path, config_path):
    """Two buttons side by side, level with the model picker.

    Running from a training project instead of the release is the Advanced expander's
    job (its project defaults to the routed model's own); a separate "from project
    config" checkbox here did the same thing a second way."""
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the run controls share the page-wide top row with the picker; no heading | date: 2026-09-25]
    alias = core.production_alias()
    ready = bool(video_path) and (bool(alias) or
                                  (bool(config_path) and os.path.exists(config_path)))

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: "from project config" folded into Advanced: raw project | date: 2026-09-25]
    from_project = False
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome shown by the button's colour and hover; failures open a dialog | date: 2026-09-28]
    temp_col, sorted_col = st.columns(2)
    with temp_col:
        core.action_button("Preview → temp", "run_temp", core.analyze_to_temp,
                           kwargs=dict(run_from_project=from_project), type="primary",
                           disabled=not ready, use_container_width=True)

    # the sorted button reuses a preview of the CURRENT release when there is one; say
    # so on the button rather than adding a second one that does the same job
    preview = None if from_project else core.publishable_preview(video_path)
    with sorted_col:
        core.action_button("Production → sorted", "run_sorted", core.analyze_to_sorted,
                           kwargs=dict(run_from_project=from_project), type="primary",
                           disabled=not ready, use_container_width=True)
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    if preview:
        st.caption("reuses the preview · no inference",
                   help="A preview of the release this session is routed to is "
                        "already in temp, so the job copies it instead of "
                        "recomputing identical output. Aliases, QC and the gDLCFit "
                        "row are written exactly as a fresh fit would.")

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: removed "Labelled video → temp"; the viewer's Labelled video view draws clips from the fit itself, so a GPU job to render one is redundant | date: 2026-09-23]
    # [AGENT EDIT END]

    if not ready:
        st.caption("no model", help="Needs a valid video path and a resolved model.")
    # (a project run's missing provenance is flagged once, under the picker)


# ── what exists for this session ───────────────────────────────────────────────
def _outputs(video_path, config_path):
    """Files on disk and rows in gDLCFit -- the same question, so one row of three.

    ``On record`` used to sit up beside the routing, where it duplicated what the sorted
    listing right here already shows; it reads as a caption on the file lists instead.
    """
    highlight_substr, highlight_tokens = core.get_dlc_highlight_tokens(config_path)
    temp_col, sorted_col, record_col = st.columns([1, 1, 1])

    with temp_col:
        if video_path:
            core.render_h5_list("Temp .h5 files", core.session_temp_dir(video_path),
                                highlight_substr=highlight_substr,
                                highlight_tokens=highlight_tokens)
        else:
            st.caption("Temp .h5 files: (set a video path)")
    with sorted_col:
        if video_path:
            vp = Path(video_path)
            core.render_h5_list("Sorted .h5 files", vp.parents[1] / "sorted",
                                match_substr=vp.parents[0].name,
                                highlight_substr=highlight_substr,
                                highlight_tokens=highlight_tokens)
        else:
            st.caption("Sorted .h5 files: (set a video path)")
    with record_col:
        st.caption("On record (gDLCFit)")
        panel = load_panel("provenance")
        if panel is None or not video_path:
            st.caption("—" if panel is not None else "provenance panel unavailable")
        else:
            try:
                panel.render_session(video_path, expanded=None)
            except Exception as e:                            # noqa: BLE001
                st.caption(f"gDLCFit unavailable: {e}")

    sidecars = core.scale_sidecars(video_path)
    if sidecars:
        with st.expander(f"Analysis resolution ({len(sidecars)} rescaled fit(s))",
                         expanded=False):
            for path, info in sidecars:
                st.write(f"**{Path(path).name}** — native {info.get('native')} → "
                         f"analysed at {info.get('analysis_size')} "
                         f"(scale {info.get('scale')})")


# ── look at it ─────────────────────────────────────────────────────────────────
# [AGENT EDIT START | agent: Codex | user: wingertj | reason: overlay and QC interactions should rerun only the visible viewer | date: 2026-09-09]
@st.fragment(run_every=30)
def _viewer(video_path, config_path):
    """The frame, plus the two other views of the same session, as sibling tabs.

    Spread QC and the labelled video answer "did this model work here?" about the very
    frame on screen, so they are alternate views of one thing rather than three sections
    that each deserve their own scroll.
    """
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: all diagnostics share an independent existing-fit selection | date: 2026-09-10]
    import fit_inspection
    fit_inspection.render(video_path)
    # [AGENT EDIT END]
# [AGENT EDIT END]


# The frame + overlay viewer that lived here was orphaned by commit 8f53b1a0, which
# replaced the three-tab viewer with fit_inspection.render(). Its marker toggle now
# lives there; the dead copy is removed rather than left looking live.
