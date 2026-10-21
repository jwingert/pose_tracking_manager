"""Tab 2 — Update model: make the routed model better, in order.

A vertical ladder, 1 → 5, each rung reporting its own state.  The states come from
``core.update_steps``: a refusal further down the loop (training against an uncurated
label set, a held training lock) is visible before you click, not after the job dies.

Every step operates on the routed model's **project** -- the mutable tree -- never on
the release inference runs from.  Nothing here creates a project; see the New DLC model
tab for that.
"""
import os
from pathlib import Path

import streamlit as st

import labeler_core as core

BADGE = {
    "ok":      ("#e6f4ea", "#1e6c33", "done"),
    "todo":    ("#e8eef7", "#294f7c", "next"),
    "warn":    ("#fdf0e3", "#8a4b00", "action needed"),
    "blocked": ("#f2f2f2", "#6b6b6b", "waiting"),
    "na":      ("#f2f2f2", "#6b6b6b", "n/a"),
}


def render():
    video_path = st.session_state.get("video_path", "")
    # Always a mutable project, never the frozen release the Model tab runs inference
    # from -- a release is read-only, holds one snapshot and has no curation log.
    config_path = _project_picker()

    if not config_path or not os.path.exists(config_path):
        st.warning("No project selected — pick one above or route a model in the "
                   "Model tab.")
        return

    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: refresh project diagnostics explicitly and reuse them between interactions | date: 2026-09-09]
    st.button("Refresh project status", key="update_refresh", on_click=core.refresh_files)
    steps = core.project_summary(video_path, config_path)
    # [AGENT EDIT END]

    # [AGENT EDIT | agent: claude | user: wingertj | reason: the ladder strip repeated
    # the tab labels one line above them. The at-a-glance state it carried moves INTO
    # the labels as a glyph, so all five states stay visible without a second row of
    # titles. Evaluate leads: the loop starts by measuring where the model stands, and
    # returns there after training to pick the snapshot. | date: 2026-08-18]
    bodies = [(_evaluate, (config_path,)),
              (_add_frames, (video_path, config_path)),
              (_curate, (config_path,)),
              (_retrain, (config_path,)),
              (_release, (config_path,))]
    glyph = {"ok": " ✓", "warn": " ⚠", "blocked": " ⏳", "todo": "", "na": ""}
    # [AGENT EDIT START | agent: Codex | user: wingertj | reason: execute only the selected update step | date: 2026-09-09]
    index = core.tab_selector(
        "Update step", range(len(steps)), key="update_section",
        format_func=lambda i: (f"{steps[i]['n']} · {steps[i]['title']}"
                               f"{glyph.get(steps[i]['state'], '')}"))
    _state_line(steps[index])
    body, args = bodies[index]
    body(*args)
    # [AGENT EDIT END]


# ── which project ──────────────────────────────────────────────────────────────
def _project_picker():
    """Pin the working project for every step below.

    [AGENT EDIT | agent: claude | user: wingertj | reason: the working project used to be
    whatever the Model tab's routing resolved, so updating a project the router does not
    point at (a successor like merged_left_chimney) meant a detour through the Advanced
    override on a different tab. Picking it HERE writes 'update_project_config', which
    core.project_config_path() serves to every project act -- add frames, curate, train,
    evaluate, release -- so all five rungs agree on the tree. | date: 2026-08-18]
    """
    models = core.mutable_projects()
    if not models:
        st.warning("No DLC projects found under /auto/data/dlc/")
        return ""
    names = list(models)

    # follow the routed model whenever routing changes, but let a manual pick survive
    # reruns -- the same seed-the-widget-key pattern as sync_routing_for_video
    routed = core.routed_project_config()
    routed_name = Path(routed).parent.name if routed else None
    if st.session_state.get("mu_project_stamp") != routed:
        st.session_state.mu_project_stamp = routed
        if routed_name in names:
            st.session_state.mu_project_select = routed_name
    if st.session_state.get("mu_project_select") not in names:
        st.session_state.pop("mu_project_select", None)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: name each project by
    # the release(s) cut from it and list those first, so "improve the model that runs
    # now" is picking a release, not recognising a directory name | date: 2026-09-25]
    releases = core.releases_by_project()
    served = {n: releases.get(os.path.abspath(str(Path(models[n]).parent)), [])
              for n in names}
    names = sorted(names, key=lambda n: (
        not any(st_ == "current" for _a, _r, st_ in served[n]),
        not any(st_ == "candidate" for _a, _r, st_ in served[n]), not served[n]))

    def label(name):
        rows = [(rid, status) for _a, rid, status in served.get(name) or []
                if status != "previous"]
        if not rows:
            return name
        tags = [rid if status == "current" else f"{rid} (candidate)"
                for rid, status in rows]
        return f"{', '.join(tags)}  ←  {name}"
    # [AGENT EDIT END]

    pick_col, info_col = st.columns([1.2, 2])
    with pick_col:
        # same seed-then-create pattern as routing_alias: pass the default only when
        # the key is unset, or Streamlit warns about the ignored index
        default = ({} if "mu_project_select" in st.session_state else
                   {"index": names.index(routed_name) if routed_name in names else 0})
        chosen = st.selectbox(
            "Project to update", names, key="mu_project_select", format_func=label,
            help="Every step below — frames, curation, training, evaluation, the "
                 "release — operates on this project's tree. A release is frozen, so "
                 "it is listed by the training project it was cut from. Defaults to "
                 "the project behind the Model tab's routed model.", **default)
    config_path = models.get(chosen, "")
    st.session_state.update_project_config = config_path
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: what a retrain here
    # starts from, in data terms | date: 2026-09-25]
    st.session_state.mu_project_releases = [rid for _a, rid, _s in served.get(chosen, [])]
    # [AGENT EDIT END]

    with info_col:
        bits = []
        if routed_name and chosen != routed_name:
            bits.append(f":orange[not the routed project (`{routed_name}`)]")
        model = core.active_model()
        if model.get("release_id") and chosen == routed_name:
            bits.append(f"inference keeps running `{model['release_id']}` until a new "
                        f"release is promoted (step 5)")
        if bits:
            st.markdown("<div style='padding-top:34px'></div>", unsafe_allow_html=True)
            st.caption(" · ".join(bits))
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: what a retrain here starts from, in data terms | date: 2026-09-25]
    _training_data_line(served.get(chosen, []), Path(config_path).parent)
    # [AGENT EDIT END]
    return config_path


# [AGENT EDIT START | agent: claude | user: wingertj | reason: say what data the release
# behind this project was trained on and how the project's labels differ now -- data is
# what a retrain changes | date: 2026-09-25]
def training_data_summary(data):
    """``(line, detail)`` for :func:`core.release_training_data`'s result."""
    frames = data.get("frames")
    head = (f"`{data['release_id']}` trained on "
            f"{frames if frames is not None else '?'} frames · "
            f"{len(data.get('folders') or [])} videos")
    changes = []
    if data.get("added"):
        changes.append(f"+{len(data['added'])} new")
    if data.get("changed"):
        changes.append(f"{len(data['changed'])} relabelled")
    if data.get("removed"):
        changes.append(f"{len(data['removed'])} removed")
    if changes:
        now = data.get("project_frames")
        tail = (f":orange[project labels changed since · {' · '.join(changes)} videos"
                + (f" · {now} frames now" if now is not None else "") + "]")
    else:
        tail = ":green[project labels = release]"
    detail = [f"iteration {data.get('iteration')} · snapshot {data.get('snapshot')}"]
    for name, key in (("Trained on", "folders"), ("New since release", "added"),
                      ("Relabelled since release", "changed"),
                      ("Removed since release", "removed")):
        if data.get(key):
            detail.append(f"{name}: " + ", ".join(data[key]))
    return f"{head} · {tail}", "\n\n".join(detail)


def _training_data_line(rows, project_dir):
    for _alias, rid, status in rows:
        if status != "current":
            continue
        data = core.release_training_data(rid, project_dir)
        if data is None:
            st.caption(f"`{rid}` · training data unreadable")
            continue
        line, detail = training_data_summary(data)
        st.caption(line, help=detail)
# [AGENT EDIT END]


def _state_line(step):
    """The step's state inside its tab — the tab label carries the title."""
    bg, fg, label = BADGE.get(step["state"], BADGE["na"])
    st.markdown(
        f"<span style='background:{bg};color:{fg};padding:2px 8px;font-size:11px;"
        f"border-radius:2px'>{label}</span> "
        f"<span style='font-size:12px;opacity:.75'>{step['detail']}</span>",
        unsafe_allow_html=True)


# ── 1 ──────────────────────────────────────────────────────────────────────────
def _add_frames(video_path, config_path):
    if not video_path:
        st.info("Set a video path in the strip above.")
        return

    trained = core.is_dlc_model_trained(config_path)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
    core.action_button("Add current video to dataset", "mu_add_video", _add_video,
                       type="primary", use_container_width=True)
    # [AGENT EDIT END]
    st.caption("queues " + ("fit + outlier extraction" if trained else
                            "frame extraction (no snapshot yet)"))

    # read the PINNED project's config directly -- session_state.dlc_videos is filled
    # from the Model tab's routed model at app entry, so with a different project pinned
    # here it listed the wrong project's videos (SQD037/SQD092 looked missing from the
    # merge when they were in videos/, labeled-data/ AND video_sets all along)
    try:
        import yaml as _yaml
        with open(config_path) as _f:
            _vs = list((_yaml.safe_load(_f).get("video_sets") or {}).keys())
    except Exception:                                         # noqa: BLE001
        _vs = []
    with st.expander(f"Videos in this project ({len(_vs)})", expanded=False):
        for v in _vs:
            st.write(f"- {Path(v).name}")


# ── 2 ──────────────────────────────────────────────────────────────────────────
def _curate(config_path):
    kind, command = core.labeling_command(config_path)
    st.caption(f"Run in a DLC shell — `{kind}`",
               help="refine_labels once a snapshot exists, label_frames before that.")
    st.code(command, language="python")

    st.markdown("**Accept the labels**")
    note = st.text_input("What changed", placeholder="e.g. added TAR017a04, refined LMD004a",
                         key="mu_curate_note")
    st.code(core.curate_command(note), language="bash")


# ── 1 ──────────────────────────────────────────────────────────────────────────
def _evaluate(config_path):
    """Train/test error and generalization, side by side.

    The two answer different halves of one question: the loss chart says how well the
    model fits frames LIKE its training labels; the generalization set says what it does
    on sessions it has never seen. Judging on either alone is how a model overfit to
    one animal ships.
    """
    loss_col, gen_col = st.columns([1.2, 1])
    with loss_col:
        st.markdown("**Train / test error (labelled frames)**")
        eval_files = core.all_eval_files(config_path)

        go_col, all_col = st.columns([1, 1.2])
        with go_col:
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
            core.action_button("Evaluate all snapshots", "mu_evaluate", _evaluate,
                               args=(config_path,), use_container_width=True)
            # [AGENT EDIT END]
        with all_col:
            overlay_all = st.checkbox("overlay every iteration", value=True,
                                      key="mu_overlay_iters")

        eval_files = core.all_eval_files(config_path)

        if eval_files:
            show = eval_files if overlay_all else {
                k: v for k, v in list(eval_files.items())[-1:]}
            st.plotly_chart(core.snapshot_error_figure(show), use_container_width=True,
                            config={"displayModeBar": False}, key="mu_eval_chart")

        _iteration_snapshot_picker(config_path)

    with gen_col:
        st.markdown("**Generalization set**",
                    help="Held-out sessions this geometry's models are judged on. "
                         "Never in the training data — the leakage check enforces it.")
        _generalization_set(config_path)


def _iteration_snapshot_picker(config_path):
    """Iteration + snapshot in use, written back into config.yaml on change."""
    try:
        cfg, project_path, iterations = core.model_iterations(config_path)
    except Exception as e:                                    # noqa: BLE001
        st.error(f"Failed to read DLC config: {e}")
        return
    if not iterations:
        st.caption("No iteration-* folders under dlc-models yet.")
        return

    current_iter = f"iteration-{cfg.get('iteration', 0)}"
    if current_iter not in iterations:
        current_iter = iterations[-1]

    it_col, snap_col = st.columns(2)
    with it_col:
        chosen_iter = st.selectbox("Iteration", iterations,
                                   index=iterations.index(current_iter),
                                   key="mu_iteration")
    snapshots = core.iteration_snapshots(project_path, chosen_iter)
    if not snapshots:
        with snap_col:
            st.caption(f"No snapshots found for {chosen_iter}")
        return

    stored_idx = cfg.get("snapshotindex", -1)
    if not isinstance(stored_idx, int) or stored_idx < 0 or stored_idx >= len(snapshots):
        stored_idx = len(snapshots) - 1
    with snap_col:
        chosen_snap = st.selectbox("Snapshot (training epoch)", snapshots,
                                   index=stored_idx, key="mu_snapshot")

    new_iter_num = int(chosen_iter.split("-")[-1])
    new_index = snapshots.index(chosen_snap)
    if new_iter_num != cfg.get("iteration") or new_index != cfg.get("snapshotindex"):
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
        core.action_button(f"Use iteration-{new_iter_num} / snapshot {chosen_snap}",
                           "mu_apply_snapshot", _apply_snapshot,
                           args=(config_path, new_iter_num, new_index))
        # [AGENT EDIT END]
        st.caption("Edits the project config, not what production runs.")


# ── 4 ──────────────────────────────────────────────────────────────────────────
def _retrain(config_path):
    # the pinned release alias (step 5), so transfer-from-release offers the releases of
    # the slot this model is heading for rather than whatever the router picked
    alias = _pinned_alias()
    current_release, history, _ = core.alias_release_info(alias)

    st.radio("Training mode",
             ["Train from scratch", "Transfer learning from snapshot",
              "Transfer learning from release"],
             key="training_mode")
    mode = st.session_state.get("training_mode")

    if mode == "Transfer learning from snapshot":
        try:
            _cfg, project_path, _ = core.model_iterations(config_path)
            snapshot_paths = sorted(str(s) for s in Path(project_path).rglob("snapshot-*.meta"))
        except Exception:                                     # noqa: BLE001
            snapshot_paths = []
        if snapshot_paths:
            st.selectbox("Initialise weights from snapshot", snapshot_paths,
                         index=len(snapshot_paths) - 1,
                         format_func=lambda s: "/".join(Path(s).parts[-3:]),
                         key="transfer_learning_snapshot")
        else:
            st.warning("No snapshots in this project to transfer from.")
    elif mode == "Transfer learning from release":
        options = list(dict.fromkeys([r for r in history + [current_release] if r]))
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: the releases cut
        # from the project being trained are the natural starting weights, whichever
        # alias step 5 is pinned to -- default to the project's own current release
        # | date: 2026-09-25]
        own = st.session_state.get("mu_project_releases") or []
        options = list(dict.fromkeys(options + own))
        if options and st.session_state.get("transfer_learning_release") not in options:
            st.session_state.pop("transfer_learning_release", None)
        default_index = options.index(own[0]) if own else len(options) - 1
        # [AGENT EDIT END]
        if options:
            st.selectbox("Initialise weights from release", options,
                         index=default_index, key="transfer_learning_release")
        else:
            st.warning(f"No releases recorded for alias {alias or '—'}.")

    st.checkbox("resume into an iteration that already holds snapshots",
                key="training_resume",
                help="Existing snapshots in that iteration WILL be overwritten. "
                     "Without this the run refuses.")

    st.session_state.setdefault("dlc_run_user", core.current_dlc_user())
    run_user = st.text_input("Your name (job attribution)", key="dlc_run_user")
    if run_user:
        core.remember_pref("dlc_run_user", run_user)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
    core.action_button("Queue training job", "mu_train", core.train_dlc_model,
                       type="primary", use_container_width=True)
    # [AGENT EDIT END]


# ── generalization set ─────────────────────────────────────────────────────────
def _generalization_set(config_path):
    """The win/loss matrix, always visible; curation folded into one expander.

    [AGENT EDIT | agent: claude | user: wingertj | reason: the panel was a video list, an
    evidence caption and a results table each in its own block, with the results hidden
    behind an expander. One matrix -- videos as rows, candidates as columns, spread as
    the cell -- shows which model wins where at a glance; a pinned mean row summarises;
    set problems (missing/leaked) surface only when they exist. | date: 2026-08-18]
    """
    gen = core.dlc_generalization
    if gen is None:
        st.caption("dlc_generalization unavailable.")
        return
    alias = _pinned_alias()
    if not alias:
        st.info("Pin an alias in step 5 to manage its generalization set.")
        return

    videos = gen.list_videos(alias)
    if not videos:
        st.caption(f"`{alias}` has no generalization set yet — add held-out sessions "
                   f"in the curation expander (other animals are the strongest "
                   f"evidence).")

    _gen_matrix(gen, alias, videos, config_path)
    _gen_run_row(gen, alias, config_path)
    _gen_curation(gen, alias, videos, config_path)


# What the matrix can show, and which direction is good. 'supervised' here means the
# stat leans on the candidate's OWN training labels (the ratio's denominator);
# 'unsupervised' needs nothing but the predictions. fmt is a d3 format string.
_GEN_METRICS = {
    "rms spread, axis frame (head-lengths) · unsupervised":
        dict(col="rms_spread_mean", better="lower", vmin=0.0, vmax=None, fmt=".3f",
             why="Residual scatter in the front-back head-axis frame, in head-lengths "
                 "-- the exact coordinates the downstream pipeline uses, so this is "
                 "the noise analyses inherit. Denominator-free: models are DIRECTLY "
                 "comparable on a row. Front/back's own error is absorbed into the "
                 "frame and shows up in the other markers."),
    "rms spread, self-Procrustes (head-lengths) · unsupervised":
        dict(col="rms_spread_proc_mean", better="lower", vmin=0.0, vmax=None, fmt=".3f",
             why="Residual scatter after per-frame Procrustes onto the video's OWN "
                 "mean shape -- all four rigid markers share the fit, so each shows "
                 "its own error (the per-marker diagnostic). Also denominator-free."),
    "spread ratio vs own labels · supervised":
        dict(col="ratio_mean", better="lower", vmin=0.0, vmax=None, fmt=".2f",
             why="Spread ÷ the candidate's own training-label scatter. 1.0 = tracks "
                 "held-out video as tightly as its labels. Self-normalised: good "
                 "within a model, misleading ACROSS models with different label sets."),
    "worst-marker ratio · supervised":
        dict(col="ratio_worst", better="lower", vmin=0.0, vmax=None, fmt=".2f",
             why="The single loosest rigid marker's ratio -- an order statistic; "
                 "jumpier than the mean but catches one-marker failures."),
    "frames tracked (p > cut) · unsupervised":
        dict(col="track_mean", better="higher", vmin=0.0, vmax=1.0, fmt=".0%",
             why="Fraction of sampled frames each rigid marker clears the likelihood "
                 "gate, averaged over markers."),
    "frames valid (all rigid > cut) · unsupervised":
        dict(col="frac_valid", better="higher", vmin=0.0, vmax=1.0, fmt=".0%",
             why="Fraction of frames where ALL four rigid markers clear the gate "
                 "together -- what the spread is actually measured on."),
}


def _gen_matrix(gen, alias, videos, config_path):
    """Videos × candidates heatmap: pick the stat, read the winner, see the scale."""
    rows = gen.sampled_results(alias)
    if not rows:
        st.caption("No sampled evaluations yet — queue one below.")
        return
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame(rows)
    for c in ("rms_spread_mean", "ratio_mean", "ratio_worst", "track_mean",
              "frac_valid", "p_cut", "n_sampled"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "snapshot" not in df.columns:
        df["snapshot"] = ""
    df["snapshot"] = df["snapshot"].fillna("").astype(str)
    # strip the alias prefix from column labels -- every candidate shares it, and the
    # heatmap's x labels are the scarcest space on the page
    short = df["candidate"].str.replace(f"{alias}[-_]?", "", n=1, regex=True)
    short = short.where(short != "", df["candidate"])
    df["model"] = short + " @" + df["snapshot"].replace("", "?")
    # a re-run of the same (model, video) supersedes the earlier rows
    df = df.drop_duplicates(["model", "session"], keep="last")

    metric_names = [m for m in _GEN_METRICS
                    if _GEN_METRICS[m]["col"] in df.columns
                    and df[_GEN_METRICS[m]["col"]].notna().any()]
    chosen = st.selectbox("stat", metric_names, key="mu_gen_metric",
                          label_visibility="collapsed",
                          help="\n\n".join(f"**{m}** — {_GEN_METRICS[m]['why']}"
                                            for m in metric_names))
    spec = _GEN_METRICS[chosen]

    animal = {e["session"]: e.get("animal") or "" for e in videos}
    mat = df.pivot(index="session", columns="model", values=spec["col"])
    mat.index = [f"{s} ({animal.get(s, '?')})" for s in mat.index]
    mat = pd.concat([pd.DataFrame([mat.mean()], index=["MEAN (all videos)"]), mat])

    # a heatmap rather than a styled table: cells and text scale with the window, the
    # colorbar keeps the ABSOLUTE scale visible even when cells get small, and long
    # matrices scroll inside a fixed-height container instead of eating the page
    vmax = spec["vmax"] if spec["vmax"] is not None else float(np.nanmax(mat.values))
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=list(mat.columns), y=list(mat.index),
        colorscale="RdYlGn" if spec["better"] == "higher" else "RdYlGn_r",
        zmin=spec["vmin"], zmax=vmax,
        texttemplate="%{z:" + spec["fmt"] + "}",
        hovertemplate="%{y}<br>%{x}<br>%{z:.4f}<extra></extra>",
        xgap=2, ygap=2,
        colorbar=dict(thickness=12, tickfont=dict(size=10),
                      title=dict(text=chosen.split("·")[0].strip(),
                                 font=dict(size=10), side="right"))))
    height = 90 + 44 * len(mat)
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                      xaxis=dict(side="top", tickangle=-20, tickfont=dict(size=11)))
    if height > 560:                                           # long matrix: scroll it
        with st.container(height=560):
            st.plotly_chart(fig, use_container_width=True, key="mu_gen_heatmap")
    else:
        st.plotly_chart(fig, use_container_width=True, key="mu_gen_heatmap")

    arrow = "lower/greener" if spec["better"] == "lower" else "higher/greener"
    cuts = sorted({f"{c:.2f}" for c in df["p_cut"].dropna().unique()})
    n = sorted({int(v) for v in df["n_sampled"].dropna().unique()})
    st.caption(f"**{arrow} is better** · likelihood gate p>"
               f"{', '.join(cuts) or '?'} · {'/'.join(str(x) for x in n)} frames "
               f"per video.", help=spec["why"])

    problems = []
    for e in videos:
        if not e["exists"]:
            problems.append(f":red[`{e['session']}` missing on disk]")
        elif e["link"] and gen.in_training(e["session"], config_path):
            problems.append(f":red[**`{e['session']}` IN TRAINING SET — leaked**]")
    if problems:
        st.markdown(" · ".join(problems))


def _gen_run_row(gen, alias, config_path):
    """Candidate, frames, queue and refresh on one line — no expander."""
    from nems_lbhb.motor import dlc_attribution
    options = {f"project: {Path(config_path).parent.name}": config_path}
    for rid in ([r for r in [core.alias_release_info(alias)[0]] if r]
                + core.unpromoted_releases(alias)):
        options[f"release: {rid}"] = os.path.join(
            dlc_attribution.RELEASE_ROOT, rid, "config.yaml")

    pick_col, n_col, go_col, r_col = st.columns([2, 0.9, 1.1, 0.4])
    with pick_col:
        pick = st.selectbox("Candidate", list(options), key="mu_gen_candidate",
                            label_visibility="collapsed")
    with n_col:
        n_frames = st.number_input("frames", min_value=100, max_value=5000, step=50,
                                   value=gen.DEFAULT_SAMPLE_FRAMES,
                                   key="mu_gen_nframes", label_visibility="collapsed")
    with go_col:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
        core.action_button("Queue eval", "mu_gen_queue", core.queue_generalization_eval,
                           args=(alias, options[pick]), kwargs=dict(n_frames=int(n_frames)),
                           type="primary", use_container_width=True,
                           help="~750 uniformly sampled frames per set video → candidate "
                                "inference → Procrustes spread, as a GPU queue job. "
                                "Appends to results_sampled.csv.")
        # [AGENT EDIT END]
    with r_col:
        if st.button("↻", key="mu_gen_refresh", use_container_width=True,
                     help="Re-read results_sampled.csv (a finished job does not "
                          "refresh the page on its own)."):
            st.rerun()


def _gen_curation(gen, alias, videos, config_path):
    """Add videos + ground-truth labelling — the rarely-used half, one expander."""
    with st.expander(f"Set curation ({len(videos)} video(s)) · labelling",
                     expanded=False):
        for e in videos:
            who = f" · {e['animal']}" if e.get("animal") else ""
            note = f" · {e['note']}" if e.get("note") else ""
            st.markdown(f"- `{e['session']}`{who}{note}",
                        help=f"added by {e.get('added_by') or '?'}"
                             + (f" {str(e['added'])[:10]}" if e.get("added") else ""))
        add_col, path_col = st.columns([1, 1.4])
        with add_col:
            video_path = st.session_state.get("video_path", "")
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
            core.action_button("Add current session", "mu_gen_add_current", _gen_add,
                               args=(gen, alias, video_path, config_path),
                               disabled=not video_path, use_container_width=True)
            # [AGENT EDIT END]
        with path_col:
            typed = st.text_input("or a video path", key="mu_gen_add_path",
                                  placeholder="/auto/data/daq/<animal>/<pen>/<sess>/recording.avi",
                                  label_visibility="collapsed")
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
            core.action_button("Add path", "mu_gen_add_typed", _gen_add,
                               args=(gen, alias, None, config_path), disabled=not typed)
            # [AGENT EDIT END]
        st.caption("Ground-truth labelling (companion project, never the training "
                   "project):")
        st.code(gen.labeling_recipe(alias, config_path), language="python")


def _gen_add(gen, alias, video_path, config_path):
    """One add, through the resolved ceiling view and the leakage check.

    An action_button job: ``video_path=None`` reads the typed box at click time (a
    callback's args are frozen at the previous run). Refusals raise.
    """
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: runs as an action_button job | date: 2026-09-28]
    video_path = video_path or st.session_state.get("mu_gen_add_path", "")
    resolved, _over = core.ceiling_video(video_path)
    link = gen.add_video(alias, resolved, by=core.current_dlc_user(),
                         project_config=config_path)
    return f"added · {Path(link).name}"
    # [AGENT EDIT END]


# ── 5 ──────────────────────────────────────────────────────────────────────────
def _pinned_alias():
    """The alias step 5 will cut the release under (widget state, may be None)."""
    return st.session_state.get("mu_release_alias") or core.active_model().get("alias")


def _release(config_path):
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the alias came only
    # from the Model tab's routing, so a project selected via the raw override had
    # alias=None and this step refused outright -- a successor project (the hemisphere
    # merges) could never be released without first being routed to. The alias is a
    # DECISION ("which deployment slot does this model serve?"), so it is a dropdown
    # here, defaulting to the routed alias when there is one. | date: 2026-08-18]
    project = str(Path(config_path).parent)
    names = core.alias_names()
    if not names:
        st.warning("No aliases in the registry — cannot cut a release.")
        return

    routed = core.active_model().get("alias")
    if st.session_state.get("mu_release_alias") not in names:
        st.session_state.pop("mu_release_alias", None)
    alias = st.selectbox(
        "Release under alias", names,
        index=names.index(routed) if routed in names else 0,
        key="mu_release_alias",
        help="The deployment slot this model serves. Cutting a release does NOT change "
             "what the alias runs — inference stays on the current release until you "
             "promote.")

    current_release, _history, next_id = core.alias_release_info(alias)
    c1, c2, c3 = st.columns(3)
    c1.metric("current release", current_release or "(none)")
    c2.metric("this would cut", next_id or "?")
    c3.metric("from project", Path(project).name)

    # cutting from a different project than the alias's registered one is exactly how a
    # successor model ships -- say it, don't block it
    try:
        registered = (core.resolve_alias(alias) or {}).get("project")
    except Exception:                                         # noqa: BLE001
        registered = None
    if registered and os.path.abspath(registered) != os.path.abspath(project):
        st.caption(f":orange[successor project] — `{alias}` currently trains in "
                   f"`{Path(registered).name}`; this release freezes "
                   f"`{Path(project).name}` instead. Legitimate for replacing the "
                   f"model behind an alias; the release records its true source.")
    # [AGENT EDIT END]

    try:
        cfg, project_path, _iterations = core.model_iterations(config_path)
        iteration = f"iteration-{cfg.get('iteration', 0)}"
        snapshots = core.iteration_snapshots(project_path, iteration)
    except Exception:                                         # noqa: BLE001
        snapshots = []

    snap_col, note_col = st.columns([1, 1.6])
    with snap_col:
        snapshot = st.selectbox("Snapshot to freeze", snapshots,
                                index=len(snapshots) - 1 if snapshots else 0,
                                key="mu_release_snapshot") if snapshots else None
    with note_col:
        note = st.text_input("Release note", key="mu_release_note",
                             placeholder="e.g. +31 TAR017 frames, refined left ear")

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: outcome on the button; failures in a dialog | date: 2026-09-28]
    core.action_button("Dry-run the build", "mu_release_dryrun", _dry_run,
                       args=(project, alias, snapshot), disabled=not snapshot)
    dry = st.session_state.get("mu_release_dryrun_result")
    if dry and dry.get("for") == (str(project), alias, snapshot):
        st.json(dry["report"], expanded=False)
    # [AGENT EDIT END]

    st.code(core.release_commands(project, alias, snapshot, note), language="python")
    st.caption("Run these in a shell — build and verify now, **promote only once the "
               "candidate clears the generalization bar in step 1**.",
               help="Promotion edits the git-tracked registry and changes what every "
                    "future fit runs, so it is deliberately not a button.")


def _generalization_evidence(alias, current_release, next_id):
    """One line: the deployed bar, plus any cut-but-untested candidates.

    [AGENT EDIT | agent: claude | user: wingertj | reason: this used to carry a header,
    three explanatory sentences and a five-step manual recipe. The queue button above
    IS that workflow now, and the sampled-results table is the evidence -- what remains
    worth a line is the bar and which candidates still lack numbers. | date: 2026-08-18]
    """
    bits = []
    bar = core.release_qc_summary(current_release) if current_release else None
    if bar and bar["ratio_median"] is not None:
        bits.append(f"deployed `{current_release}`: **{bar['ratio_median']:.2f}×** / "
                    f"**{bar['track_median']:.0%}** ({bar['n']} fits)")
    untested = [r for r in core.unpromoted_releases(alias)[:4]
                if not core.release_qc_summary(r)]
    if untested:
        bits.append(":orange[untested: " + ", ".join(f"`{r}`" for r in untested) + "]")
    if bits:
        st.caption(" · ".join(bits),
                   help="The deployed release's recorded QC (median rigid spread / "
                        "median tracking) is the bar a candidate must be no worse than "
                        "on the same sessions. Queue the evaluation above for any "
                        "candidate marked untested.")


# [AGENT EDIT START | agent: claude | user: wingertj | reason: action_button jobs for the
# update loop -- a message on success, an exception on failure | date: 2026-09-28]
def _add_video():
    with st.spinner("Copying video and queueing DLC job..."):
        return core.add_video_to_dataset()


def _evaluate(config_path):
    if core.all_eval_files(config_path):
        return "already evaluated · results shown"
    with st.spinner("Evaluating network (in-process, slow)..."):
        core.evaluate_all_snapshots(config_path)
    return "evaluated"


def _apply_snapshot(config_path, iteration, index):
    core.write_config_values(config_path, iteration=iteration, snapshotindex=index)
    return f"config.yaml · iteration {iteration} · snapshotindex {index}"


def _dry_run(project, alias, snapshot):
    note = st.session_state.get("mu_release_note", "")
    with st.spinner("Checking what would be frozen (writes nothing)..."):
        report = core.build_release_dry_run(project, alias, snapshot, note=note)
    st.session_state.mu_release_dryrun_result = {"for": (str(project), alias, snapshot),
                                                 "report": report}
    return "dry run ok · report below"
# [AGENT EDIT END]
