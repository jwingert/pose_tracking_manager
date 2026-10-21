# [AGENT EDIT START | agent: claude | user: wingertj | reason: the fit job now records its own
# gDLCFit row, and its QC must be the SAME document the sweeper produces -- two
# implementations of "score this output against the model's frozen labels" would drift
# | date: 2026-08-12]
"""Score a DLC output against the frozen labels of the release that produced it.

Extracted from ``scripts/dlc_sync_fits.py`` so the fit job and the sweeper share one
implementation. The sweeper's own wrapper still handles the harder cases it alone faces
(legacy files with no release, ambiguous scorers, live-project references); this module
covers the case the job has, which is the easy one: the release id is known for certain.

Deliberately NOT in ``dlc_fit_record``: that module stays free of numpy/pandas at import
time, and scoring pulls both in through ``dlc_headfix_qc``.
"""
import datetime
import glob
import os

from nems_lbhb.motor import dlc_attribution

# 1 -> 2 (2026-08-17): spread scored at the LOADER threshold (0.2) rather than 0.9, so it
# describes the frames baphy_io actually keeps; every rigid marker's residual recorded,
# not just the worst.
# 2 -> 3 (2026-08-17): frac_tracked also moves to the loader threshold -- v2 kept it at a
# strict 0.9 on a saturation argument generalised from two sessions, which is withdrawn.
# Both gates now match baphy_io exactly, including its strictly-greater comparison.
# 3 -> 4 (2026-08-18): the Procrustes alignment runs on head-fix coordinates, so the
# aligned frame is genuinely anatomical (+y toward the nose, +x toward the right ear) --
# it previously settled nose-DOWN because image coords have opposite chirality and
# rotation-only Kabsch cannot honour a mirrored seed. ratio_spread/frac_tracked are
# unchanged (rigid transforms preserve distances); bias_x/bias_y and mean_shape change
# sign/orientation, so v3 and v4 disagree on those fields only.
# 4 -> 5 (2026-08-18): spread becomes SELF-REFERENTIAL (user decision) -- each frame is
# aligned by its own front/headpost axis (the head-fix transform IS the fit) and
# rms_spread is the residual about the video's own per-marker mean in that frame, in
# head-lengths. Training labels no longer enter the measurement; ratio_*/bias_* become
# context against the labels' scatter measured in the same head-fix frame. The
# headpost-role marker has zero spread by construction (origin_part) and is excluded
# from headline means; rms_spread_mean/worst join the summary as the primary headline.
# 5 -> 6 (2026-08-19): adds the SELF-PROCRUSTES companion spread (user decision) --
# rms_spread_proc: per-frame Kabsch of the rigid quad onto the video's OWN Procrustes
# mean (still reference-free), so all four rigid markers share the fit and each shows
# its own residual (the axis-frame rms_spread hides front/back's error in the frame
# itself). rms_spread (axis frame) is unchanged and remains the pipeline-noise number;
# rms_spread_proc_mean/worst join the summary. v5 fields keep their meaning, so v5 and
# v6 agree wherever both define a value.
# Documents of different versions are NOT comparable; consumers must filter on this field
# (and on p_cut/track_p_cut, which record the gates each document actually used).
QC_MODULE_VERSION = 6


def reference_for_release(release_id, cache={}):
    """TrainingReference built from a release's FROZEN ``labels/`` directory.

    Not ``dlc_headfix_qc.build_training_reference``, which reads a LIVE project: the
    reference has to be the labels the model was actually trained on, or every QC number
    drifts as people keep labelling.
    """
    if release_id in cache:
        return cache[release_id]
    from nems_lbhb.motor import dlc_headfix_qc as qc
    root = os.path.join(dlc_attribution.RELEASE_ROOT, release_id, 'labels')
    folders = []
    for path in sorted(glob.glob(os.path.join(root, '*', 'CollectedData_*.h5'))):
        try:
            coords, _, parts = qc.load_dlc_h5(path)
        except Exception:                                     # noqa: BLE001
            continue
        folders.append((coords, parts, os.path.basename(os.path.dirname(path))))
    cache[release_id] = qc.build_reference_from_folders(folders) if folders else None
    return cache[release_id]


def release_labels_fingerprint(release_id, cache={}):
    """The release's own ``[source].labels_fingerprint``, from its RELEASE.toml."""
    if release_id in cache:
        return cache[release_id]
    from nems_lbhb.motor import dlc_release
    cache[release_id] = dlc_release.read_manifest(release_id)['source']['labels_fingerprint']
    return cache[release_id]


def score_release_output(h5_path, release_id, p_cut=None, track_p_cut=None):
    """The gDLCFit ``qc`` document for ``h5_path``, scored against ``release_id``'s labels.

    Returns ``{'error': '<type>: <message>'}`` rather than raising: QC is evidence about an
    analysis, and failing to produce it must never be able to fail the analysis itself or
    lose the row that records it.
    """
    try:
        from nems_lbhb.motor import dlc_headfix_qc as qc
        if p_cut is None:
            p_cut = qc.DEFAULT_P_CUT
        if track_p_cut is None:
            track_p_cut = qc.DEFAULT_TRACK_P_CUT
        reference = reference_for_release(release_id)
        if reference is None:
            raise RuntimeError('no usable reference in %s/labels (missing rigid markers, '
                               'or no readable label files)' % release_id)
        if not h5_path:
            raise RuntimeError('no output path to score')

        result = qc.score_video(h5_path, reference=reference, p_cut=p_cut,
                                track_p_cut=track_p_cut)
        # everything the fit produced, not just the worst rigid marker -- see
        # dlc_headfix_qc.summarize for why the whole table is kept
        doc = qc.summarize(result)
        doc.update({
            'p_median': float(result.table.loc[result.table['role'] == 'rigid',
                                               'p_median'].median()),
            'p_cut': p_cut, 'track_p_cut': track_p_cut,
            'reference_fingerprint': release_labels_fingerprint(release_id),
            'reference_source': 'release_labels',
            'computed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'qc_module_version': QC_MODULE_VERSION,
        })
        return doc
    except Exception as exc:                                  # noqa: BLE001
        return {'error': '%s: %s' % (type(exc).__name__, exc)}
# [AGENT EDIT END]
