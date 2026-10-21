"""Tests for head-aligned DLC marker spread QC.

Parity tests
------------
`nems_lbhb.motor.dlc_headfix_qc` deliberately carries its own copy of the head-fix
transform and of the rigid-marker resolution rules from `nems_lbhb.free_tools`,
because free_tools imports NEMS and the Streamlit app runs in a Python 3.8
environment without it. The `*_matches_free_tools` / `*_parity` tests are the only
thing keeping those copies honest, so they FAIL (not skip) when free_tools cannot
be imported -- a skip on an unrelated import breakage would leave the duplication
silently unguarded.

Set ``DLC_QC_ALLOW_NO_PARITY=1`` in the environment to downgrade that failure to a
skip. That is the intended way to run this suite from the app's Python 3.8
``streamlit_dlc`` environment, where NEMS genuinely is not installed.
"""
import os

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.motor import dlc_headfix_qc as qc

ALLOW_NO_PARITY_ENV = "DLC_QC_ALLOW_NO_PARITY"


def _require_free_tools():
    """Import nems_lbhb.free_tools, failing the test if it is unavailable.

    See the module docstring: importorskip() would turn any unrelated NEMS import
    breakage into a green skip, which is exactly the failure mode these tests
    exist to prevent. Opt out with DLC_QC_ALLOW_NO_PARITY=1.
    """
    try:
        import nems_lbhb.free_tools as free_tools
    except Exception as exc:                       # pragma: no cover - env dependent
        if os.environ.get(ALLOW_NO_PARITY_ENV) == "1":
            pytest.skip("nems_lbhb.free_tools not importable and {}=1: {}".format(
                ALLOW_NO_PARITY_ENV, exc))
        pytest.fail(
            "nems_lbhb.free_tools could not be imported, so the duplicated head-fix "
            "transform in dlc_headfix_qc is UNGUARDED: {!r}. Fix the import, or set "
            "{}=1 to knowingly run without the parity guard (e.g. in the Python 3.8 "
            "streamlit_dlc environment).".format(exc, ALLOW_NO_PARITY_ENV))
    return free_tools


def _write_h5(path, bodyparts, coords, likelihood=None, scorer="tester"):
    """Write a DLC-format h5. coords: [T, B, 2]. likelihood: [T, B] or None."""
    cols, data = [], []
    for b, part in enumerate(bodyparts):
        cols.append((scorer, part, "x"))
        data.append(coords[:, b, 0])
        cols.append((scorer, part, "y"))
        data.append(coords[:, b, 1])
        if likelihood is not None:
            cols.append((scorer, part, "likelihood"))
            data.append(likelihood[:, b])
    index = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(np.array(data).T, columns=index)
    df.to_hdf(str(path), key="df_with_missing", mode="w")
    return path


def test_resolve_rigid_indices_canonical_names():
    parts = ["front", "headpost", "leftear", "rightear", "shoulders"]
    idx = qc.resolve_rigid_indices(parts)
    assert idx == {"front": 0, "headpost": 1, "leftear": 2, "rightear": 3}


def test_resolve_rigid_indices_aliases():
    parts = ["nose", "back", "left_ear", "rightheadphone"]
    idx = qc.resolve_rigid_indices(parts)
    assert idx == {"front": 0, "headpost": 1, "leftear": 2, "rightear": 3}


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final review finding E(b) -
# alias resolution was duplicated between resolve_rigid_indices (alias priority) and
# free_tools.find_dlc_headfix_indices (last match in channel order) and the two diverged
# on projects carrying both a 'headpost' and a body 'back' marker | date: 2026-08-04]
def _chans_from_parts(bodyparts):
    """['front', 'back'] -> ['front_x', 'front_y', 'back_x', 'back_y'].

    free_tools.find_dlc_headfix_indices consumes flat DLC channel names and returns
    (x, y) channel-index pairs; resolve_rigid_indices consumes body-part names and
    returns body-part indices. Channel index i corresponds to body part i // 2.
    """
    chans = []
    for part in bodyparts:
        chans.append("{}_x".format(part))
        chans.append("{}_y".format(part))
    return chans


@pytest.mark.parametrize("bodyparts", [
    ["front", "headpost", "leftear", "rightear"],
    ["nose", "back", "left_ear", "rightheadphone"],
    # the ambiguous case: BOTH a headpost marker and a body 'back' marker.
    ["front", "headpost", "back", "leftear", "rightear"],
    ["front", "back", "headpost", "leftear", "rightear"],
    # duplicate aliases for front and rightear
    ["nose", "front", "headpost", "rightear", "right_ear", "leftear"],
    ["headpost", "front", "rightheadphone", "rightear", "shoulders"],
    # roles genuinely absent -> both must report "not found"
    ["front", "leftear", "shoulders"],
    ["shoulders", "midriff", "rear"],
])
def test_resolve_rigid_indices_parity_with_free_tools(bodyparts):
    """The QC must resolve front/headpost/rightear exactly as the fitting pipeline
    does, even where free_tools' behavior (scan in order, keep the LAST match) is
    incidental -- otherwise the QC measures a different skeleton than the models are
    fit on. leftear is not asserted here: free_tools does not resolve it at all.
    """
    free_tools = _require_free_tools()
    chans = _chans_from_parts(bodyparts)
    nose_idx, head_idx, ear_idx = free_tools.find_dlc_headfix_indices(chans)
    expected = {
        "front": None if nose_idx is None else nose_idx[0] // 2,
        "headpost": None if head_idx is None else head_idx[0] // 2,
        "rightear": None if ear_idx is None else ear_idx[0] // 2,
    }
    got = qc.resolve_rigid_indices(bodyparts)
    for role, want in expected.items():
        assert got[role] == want, (
            "role {!r} resolved to {} but free_tools resolved it to {} for {}".format(
                role, got[role], want, bodyparts))


def test_resolve_rigid_indices_headpost_and_back_takes_last():
    """Concrete statement of the case that used to diverge: alias priority picked
    'headpost' (index 1), free_tools picks the last match, i.e. 'back' (index 2).
    """
    parts = ["front", "headpost", "back", "leftear", "rightear"]
    assert qc.resolve_rigid_indices(parts)["headpost"] == 2
# [AGENT EDIT END]


def test_missing_rigid_roles_reports_gaps():
    parts = ["front", "headpost", "shoulders"]
    assert sorted(qc.missing_rigid_roles(parts)) == ["leftear", "rightear"]
    assert qc.resolve_rigid_indices(parts)["leftear"] is None


def test_load_dlc_h5_with_likelihood(tmp_path):
    parts = ["front", "headpost"]
    coords = np.arange(24, dtype=float).reshape(6, 2, 2)
    like = np.full((6, 2), 0.8)
    p = _write_h5(tmp_path / "analyzed.h5", parts, coords, likelihood=like)
    got_coords, got_like, got_parts = qc.load_dlc_h5(p)
    assert got_parts == parts
    assert got_coords.shape == (6, 2, 2)
    np.testing.assert_allclose(got_coords, coords)
    np.testing.assert_allclose(got_like, like)


def test_load_dlc_h5_without_likelihood_defaults_to_one(tmp_path):
    parts = ["front", "headpost"]
    coords = np.zeros((4, 2, 2))
    p = _write_h5(tmp_path / "labels.h5", parts, coords, likelihood=None)
    _, got_like, _ = qc.load_dlc_h5(p)
    assert got_like.shape == (4, 2)
    assert np.all(got_like == 1.0)


def test_load_dlc_h5_rejects_multi_animal(tmp_path):
    cols = pd.MultiIndex.from_tuples(
        [("s", "ind1", "front", "x"), ("s", "ind1", "front", "y")],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(np.zeros((3, 2)), columns=cols)
    p = tmp_path / "ma.h5"
    df.to_hdf(str(p), key="df_with_missing", mode="w")
    with pytest.raises(ValueError, match="[Mm]ulti-animal"):
        qc.load_dlc_h5(p)


def _rigid_template():
    """Head markers with |front - headpost| == 1.0, ordered front/headpost/L/R ear."""
    return np.array([[0.0, 1.0], [0.0, 0.0], [-0.4, 0.3], [0.4, 0.3]])


def _pose_frames(template, n, rng, noise=0.0):
    """Rotate + translate `template` into n frames, optionally adding noise."""
    ang = rng.uniform(-np.pi, np.pi, n)
    c, s = np.cos(ang), np.sin(ang)
    rot = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)
    shift = rng.uniform(-10, 10, (n, 1, 2))
    frames = np.einsum('tij,kj->tki', rot, template) + shift
    if noise:
        frames = frames + rng.normal(0, noise, frames.shape)
    return frames


def test_headfix_is_invariant_to_pose():
    rng = np.random.default_rng(0)
    template = _rigid_template()
    coords = _pose_frames(template, 200, rng)
    idx = {"front": 0, "headpost": 1, "leftear": 2, "rightear": 3}
    hf = qc.headfix_points(coords, idx)
    # every frame collapses onto the same head-fixed configuration
    assert np.allclose(hf, hf[0], atol=1e-9)
    # headpost at origin, nose on +y, right ear at +x
    np.testing.assert_allclose(hf[0, 1], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(hf[0, 0], [0.0, 1.0], atol=1e-9)
    assert hf[0, 3, 0] > 0
    assert hf[0, 2, 0] < 0


def test_headfix_matches_free_tools():
    nems_tools = _require_free_tools()
    rng = np.random.default_rng(1)
    template = _rigid_template()
    coords = _pose_frames(template, 150, rng, noise=0.05)
    chans = ["front_x", "front_y", "headpost_x", "headpost_y",
             "leftear_x", "leftear_y", "rightear_x", "rightear_y"]
    flat = coords.transpose(1, 2, 0).reshape(8, -1)  # interleaved x,y per marker
    ref, ref_chans = nems_tools.dlc_headfix(
        flat, nose_idx=(0, 1), head_idx=(2, 3), ear_idx=(6, 7), dlc_chans=chans)
    idx = {"front": 0, "headpost": 1, "leftear": 2, "rightear": 3}
    mine = qc.headfix_points(coords, idx)
    np.testing.assert_allclose(mine.transpose(1, 2, 0).reshape(8, -1), ref, atol=1e-9)
    assert ref_chans[:2] == ["front_x_hf", "front_y_hf"]


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: close ear-sign fill
# regression gap flagged in code review (parity test never exercised sign flips/zeros/NaNs) |
# date: 2026-08-04]
def test_headfix_matches_free_tools_ear_sign_stress():
    """Adversarial parity test that genuinely exercises `_fill_ear_sign`.

    The default parity test (`test_headfix_matches_free_tools`) keeps the right ear
    far from the forward axis with small noise, so `ear_dot` never crosses zero and
    the forward-fill / leading-gap / all-invalid branches of `_fill_ear_sign` are
    never entered. Here the right ear sits close to the forward axis with noise large
    enough to flip `ear_dot`'s sign often, plus injected exact-zero and NaN frames
    (including a NaN at frame 0, to force the leading-gap branch).
    """
    nems_tools = _require_free_tools()
    rng = np.random.default_rng(7)
    n = 200

    front = np.tile([0.0, 1.0], (n, 1))
    headpost = np.tile([0.0, 0.0], (n, 1))
    leftear = np.tile([-0.4, 0.3], (n, 1))
    # Right ear close to the forward axis (small lateral offset) with noise well
    # above the offset, so its sign flips frequently. With front/headpost fixed,
    # forward = (0, 1) and perp = (1, 0), so ear_dot == rightear_x exactly.
    rightear = np.empty((n, 2))
    rightear[:, 0] = 0.05 + rng.normal(0, 0.3, n)
    rightear[:, 1] = 0.3

    # Force some frames to exact-zero ear_dot.
    zero_idx = rng.choice(np.arange(1, n), size=10, replace=False)
    rightear[zero_idx, 0] = 0.0

    # Force some frames to NaN ear coordinates, including frame 0 (leading gap).
    remaining = np.setdiff1d(np.arange(1, n), zero_idx)
    nan_idx = np.concatenate([[0], rng.choice(remaining, size=8, replace=False)])
    rightear[nan_idx, 0] = np.nan
    rightear[nan_idx, 1] = np.nan

    # Sanity-check that the constructed data actually stresses the edge cases before
    # trusting the parity assertion below.
    ear_dot_raw = rightear[:, 0]
    assert np.isnan(ear_dot_raw[0]), "frame 0 must be NaN to exercise the leading-gap branch"
    assert np.isnan(ear_dot_raw).sum() >= 8, "too few NaN frames to stress forward-fill"
    assert (ear_dot_raw == 0).sum() >= 8, "too few exact-zero frames to stress forward-fill"
    finite = ear_dot_raw[~np.isnan(ear_dot_raw)]
    signs = np.sign(finite)
    signs[signs == 0] = 1.0  # zero-valued frames don't count as a "raw" flip on their own
    n_flips = int(np.sum(np.diff(signs) != 0))
    assert n_flips >= 30, "constructed ear_dot does not flip sign often enough to stress fill logic"

    coords = np.stack([front, headpost, leftear, rightear], axis=1)  # [T, 4, 2]
    chans = ["front_x", "front_y", "headpost_x", "headpost_y",
             "leftear_x", "leftear_y", "rightear_x", "rightear_y"]
    flat = coords.transpose(1, 2, 0).reshape(8, -1)  # interleaved x,y per marker
    ref, ref_chans = nems_tools.dlc_headfix(
        flat, nose_idx=(0, 1), head_idx=(2, 3), ear_idx=(6, 7), dlc_chans=chans)
    idx = {"front": 0, "headpost": 1, "leftear": 2, "rightear": 3}
    mine = qc.headfix_points(coords, idx)
    np.testing.assert_allclose(mine.transpose(1, 2, 0).reshape(8, -1), ref, atol=1e-9)
    assert ref_chans[:2] == ["front_x_hf", "front_y_hf"]


def test_fill_ear_sign_forward_fills_nan_and_zero():
    """NaN and exact-zero entries hold the most recent valid sign."""
    ear_dot = np.array([1.0, 0.0, np.nan, -3.0, np.nan, 0.0, 2.0])
    sign = qc._fill_ear_sign(ear_dot)
    expected = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
    np.testing.assert_array_equal(sign, expected)


def test_fill_ear_sign_leading_gap_takes_first_valid_sign():
    """Frames before the first valid entry all take that first valid entry's sign."""
    ear_dot = np.array([np.nan, 0.0, np.nan, -5.0, 2.0])
    sign = qc._fill_ear_sign(ear_dot)
    expected = np.array([-1.0, -1.0, -1.0, -1.0, 1.0])
    np.testing.assert_array_equal(sign, expected)


def test_fill_ear_sign_all_invalid_returns_ones():
    """An entirely NaN/zero trace falls back to all +1."""
    ear_dot = np.array([0.0, np.nan, 0.0, np.nan])
    sign = qc._fill_ear_sign(ear_dot)
    np.testing.assert_array_equal(sign, np.ones_like(ear_dot))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: Task 3 - Procrustes
# alignment and mean shape (kabsch_align, generalized_procrustes, apply_rigid) | date: 2026-08-04]
def test_kabsch_recovers_exact_pose():
    rng = np.random.default_rng(2)
    template = _rigid_template()
    coords = _pose_frames(template, 50, rng)
    aligned, rot, src_c, dst_c = qc.kabsch_align(coords, template)
    # np.testing.assert_allclose requires literal shape equality (no broadcasting,
    # per numpy's documented behavior since 1.24), so broadcast the expected value
    # explicitly rather than relying on template[None, :, :] to align against it.
    np.testing.assert_allclose(aligned, np.broadcast_to(template, aligned.shape), atol=1e-9)
    assert rot.shape == (50, 2, 2)
    assert src_c.shape == (50, 1, 2)
    assert dst_c.shape == (1, 2)


def test_generalized_procrustes_recovers_template_shape():
    rng = np.random.default_rng(3)
    template = _rigid_template()
    coords = _pose_frames(template, 500, rng, noise=0.01)
    mean_shape = qc.generalized_procrustes(coords)
    # mean shape is centred and matches the template up to a rotation
    np.testing.assert_allclose(mean_shape.mean(axis=0), [0.0, 0.0], atol=1e-9)
    centred = template - template.mean(axis=0)
    aligned, _, _, _ = qc.kabsch_align(mean_shape[None, :, :], centred)
    np.testing.assert_allclose(aligned[0], centred, atol=0.01)


def test_generalized_procrustes_orientation_follows_init():
    # init fixes the output frame; Task 4 relies on this to keep bias_x/bias_y anatomical
    rng = np.random.default_rng(20)
    template = _rigid_template()
    coords = _pose_frames(template, 300, rng, noise=0.005)
    init = template - template.mean(axis=0)
    mean_shape = qc.generalized_procrustes(coords, init=init)
    np.testing.assert_allclose(mean_shape, init, atol=0.01)


def test_procrustes_residual_recovers_injected_noise():
    # 4 points x 2 dims = 8 noise dims; a rigid fit absorbs 3 (2 shift + 1 rotation),
    # so per-marker RMS residual = sigma * sqrt(2 * 5/8) = 1.118 * sigma.
    rng = np.random.default_rng(4)
    template = _rigid_template()
    sigma = 0.02
    coords = _pose_frames(template, 4000, rng, noise=sigma)
    mean_shape = qc.generalized_procrustes(coords)
    aligned, _, _, _ = qc.kabsch_align(coords, mean_shape)
    resid = aligned - mean_shape[None, :, :]
    rms = np.sqrt((resid ** 2).sum(axis=2).mean())
    assert abs(rms / sigma - 1.118) < 0.06


def test_procrustes_residual_scales_with_noise():
    rng = np.random.default_rng(5)
    template = _rigid_template()

    def rms_for(sigma):
        coords = _pose_frames(template, 4000, rng, noise=sigma)
        mean_shape = qc.generalized_procrustes(coords)
        aligned, _, _, _ = qc.kabsch_align(coords, mean_shape)
        return np.sqrt(((aligned - mean_shape[None, :, :]) ** 2).sum(axis=2).mean())

    assert abs(rms_for(0.04) / rms_for(0.02) - 2.0) < 0.1


def test_apply_rigid_transports_extra_markers():
    rng = np.random.default_rng(6)
    template = _rigid_template()
    body = np.array([[0.0, -1.5]])            # one extra, non-rigid marker
    full_template = np.concatenate([template, body], axis=0)
    coords = _pose_frames(full_template, 30, rng)
    aligned, rot, src_c, dst_c = qc.kabsch_align(coords[:, :4, :], template)
    moved = qc.apply_rigid(coords, rot, src_c, dst_c)
    # See note above: assert_allclose needs literal shape equality, not broadcasting.
    np.testing.assert_allclose(moved, np.broadcast_to(full_template, moved.shape), atol=1e-9)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: Task 4 - training-label
# head-fix reference (head_scale, build_reference_from_folders, build_training_reference) |
# date: 2026-08-04]
_PARTS = ["front", "headpost", "leftear", "rightear", "shoulders"]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final review finding B -
# train_rms was a plain in-sample RMS over a handful of hand-placed labels, so one
# misplaced label inflated it ~2.8x and deflated every ratio ~0.35x, hiding exactly the
# degraded-tracking case this QC exists to catch | date: 2026-08-04]
def _plain_rms(resid):
    """The non-robust estimator robust_spread replaces, for comparison."""
    return float(np.sqrt((np.asarray(resid, float) ** 2).sum(axis=1).mean()))


def test_robust_spread_matches_plain_rms_on_clean_gaussian():
    """On clean isotropic Gaussian residuals the robust estimator must reproduce what
    a plain RMS would give -- otherwise switching estimators would silently rescale
    every ratio. Both should recover sigma * sqrt(2) (the RMS of a Rayleigh norm).
    """
    rng = np.random.default_rng(50)
    sigma = 0.037
    resid = rng.normal(0.0, sigma, (200000, 2))
    plain = _plain_rms(resid)
    robust = qc.robust_spread(resid)
    assert abs(robust / plain - 1.0) < 0.02
    assert abs(robust / (sigma * np.sqrt(2.0)) - 1.0) < 0.02


def test_robust_spread_resists_a_single_gross_outlier():
    """One misplaced label out of 36 must barely move the robust estimator, while it
    moves the plain RMS by a factor that would swamp a real 3.5x tracking failure.
    """
    rng = np.random.default_rng(51)
    resid = rng.normal(0.0, 0.01, (36, 2))
    plain_clean, robust_clean = _plain_rms(resid), qc.robust_spread(resid)

    dirty = resid.copy()
    dirty[0] = [0.15, 0.0]                    # one label misplaced by 0.15 head-lengths
    plain_dirty, robust_dirty = _plain_rms(dirty), qc.robust_spread(dirty)

    # a single 0.15-HL outlier among 36 clean sigma=0.01 residuals nearly doubles the
    # plain RMS (~1.9x here); that inflation propagates straight into the denominator
    # of every ratio, which is the miss direction.
    assert plain_dirty / plain_clean > 1.8, "the plain RMS should blow up here"
    assert robust_dirty / robust_clean < 1.10, "robust estimator moved too much"
    assert (plain_dirty / plain_clean) > 10 * (robust_dirty / robust_clean - 1.0) + 1.0


def test_robust_spread_all_nan_returns_nan():
    assert np.isnan(qc.robust_spread(np.full((5, 2), np.nan)))


def test_reference_train_rms_survives_one_misplaced_label():
    """End-to-end version of the finding: a single corrupted human label in a small
    label set must not inflate the reference's per-marker scale.
    """
    rng = np.random.default_rng(52)
    coords = _training_frames(36, rng, sigma=0.01, scale=1.0)
    clean = qc.build_reference_from_folders([(coords, _PARTS)])

    dirty_coords = coords.copy()
    dirty_coords[0, 2, :] += 0.15             # leftear misplaced in one labeled frame
    dirty = qc.build_reference_from_folders([(dirty_coords, _PARTS)])

    assert dirty.train_rms["leftear"] / clean.train_rms["leftear"] < 1.3


def _training_frames(n, rng, sigma=0.01, scale=1.0, seed_shift=0.0):
    """Synthetic labeled frames at an arbitrary camera scale."""
    template = np.concatenate([_rigid_template(), np.array([[0.0, -1.2]])], axis=0)
    frames = _pose_frames(template, n, rng, noise=sigma)
    return frames * scale + seed_shift


def test_head_scale_is_median_nose_head_distance():
    rng = np.random.default_rng(7)
    coords = _training_frames(200, rng, sigma=0.0, scale=37.0)
    idx = qc.resolve_rigid_indices(_PARTS)
    assert abs(qc.head_scale(coords, idx) - 37.0) < 1e-6


def test_reference_from_folders_is_scale_invariant():
    rng1 = np.random.default_rng(8)
    rng2 = np.random.default_rng(8)
    small = qc.build_reference_from_folders(
        [(_training_frames(300, rng1, sigma=0.01, scale=1.0), _PARTS)])
    big = qc.build_reference_from_folders(
        [(_training_frames(300, rng2, sigma=0.01, scale=50.0), _PARTS)])
    for part in ("front", "leftear", "rightear"):
        assert abs(small.train_rms[part] - big.train_rms[part]) < 1e-6


def test_reference_combines_multiple_folders():
    rng = np.random.default_rng(9)
    ref = qc.build_reference_from_folders([
        (_training_frames(120, rng, scale=1.0), _PARTS),
        (_training_frames(80, rng, scale=25.0), _PARTS),
    ])
    assert ref.n_frames == 200
    assert set(ref.train_rms) >= set(qc.RIGID_ROLES)
    assert ref.mean_shape.shape == (4, 2)
    assert ref.hf_cov["leftear"].shape == (2, 2)
    # shoulders rides along and gets a mean position behind the head
    assert ref.mean_full["shoulders"][1] < ref.mean_full["headpost"][1]


def test_build_training_reference_reads_project(tmp_path):
    import yaml
    rng = np.random.default_rng(10)
    project = tmp_path / "proj"
    for name, scale in (("vidA", 1.0), ("vidB", 20.0)):
        folder = project / "labeled-data" / name
        folder.mkdir(parents=True)
        _write_h5(folder / "CollectedData_tester.h5", _PARTS,
                  _training_frames(60, rng, scale=scale), likelihood=None)
    config_path = project / "config.yaml"
    config_path.write_text(yaml.dump({"project_path": str(project)}))

    ref = qc.build_training_reference(config_path)
    assert ref is not None
    assert ref.n_frames == 120
    assert len(ref.sources) == 2


def test_build_training_reference_returns_none_without_labels(tmp_path):
    import yaml
    project = tmp_path / "empty"
    (project / "labeled-data").mkdir(parents=True)
    config_path = project / "config.yaml"
    config_path.write_text(yaml.dump({"project_path": str(project)}))
    assert qc.build_training_reference(config_path) is None
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 6 review finding 1 -
# tests for label_files_fingerprint, which busts the Streamlit app's cached
# TrainingReference when labeled-data/*/CollectedData_*.h5 files change (config.yaml's
# mtime does not) | date: 2026-08-04]
def _make_labeled_project(tmp_path, rng, folders=(("vidA", 60, 1.0),)):
    """Build a tmp_path DLC project with the given (name, n_frames, scale) folders."""
    import yaml
    project = tmp_path / "proj"
    for name, n, scale in folders:
        folder = project / "labeled-data" / name
        folder.mkdir(parents=True)
        _write_h5(folder / "CollectedData_tester.h5", _PARTS,
                  _training_frames(n, rng, scale=scale), likelihood=None)
    config_path = project / "config.yaml"
    config_path.write_text(yaml.dump({"project_path": str(project)}))
    return config_path


def test_label_files_fingerprint_stable_for_same_files(tmp_path):
    rng = np.random.default_rng(40)
    config_path = _make_labeled_project(tmp_path, rng)
    fp1 = qc.label_files_fingerprint(config_path)
    fp2 = qc.label_files_fingerprint(config_path)
    assert fp1 == fp2
    assert isinstance(fp1, str) and fp1


def test_label_files_fingerprint_changes_when_file_added(tmp_path):
    rng = np.random.default_rng(41)
    config_path = _make_labeled_project(tmp_path, rng)
    before = qc.label_files_fingerprint(config_path)

    project = config_path.parent
    folder = project / "labeled-data" / "vidB"
    folder.mkdir(parents=True)
    _write_h5(folder / "CollectedData_tester.h5", _PARTS,
              _training_frames(60, rng, scale=5.0), likelihood=None)

    after = qc.label_files_fingerprint(config_path)
    assert after != before


def test_label_files_fingerprint_changes_when_file_modified(tmp_path):
    """Rewriting an existing label file with a different frame count changes the
    fingerprint. The frame count (and therefore file size) is deliberately changed,
    not just the contents, so the test isn't sensitive to mtime resolution -- two
    writes issued back-to-back can land in the same mtime tick on some filesystems.
    """
    rng = np.random.default_rng(42)
    config_path = _make_labeled_project(tmp_path, rng, folders=(("vidA", 60, 1.0),))
    before = qc.label_files_fingerprint(config_path)

    project = config_path.parent
    path = project / "labeled-data" / "vidA" / "CollectedData_tester.h5"
    _write_h5(path, _PARTS, _training_frames(120, rng, scale=1.0), likelihood=None)

    after = qc.label_files_fingerprint(config_path)
    assert after != before


def test_label_files_fingerprint_sentinel_without_labels(tmp_path):
    import yaml
    project = tmp_path / "empty"
    (project / "labeled-data").mkdir(parents=True)
    config_path = project / "config.yaml"
    config_path.write_text(yaml.dump({"project_path": str(project)}))
    fp = qc.label_files_fingerprint(config_path)
    assert isinstance(fp, str) and fp
    # same sentinel every time there are no label files
    assert fp == qc.label_files_fingerprint(config_path)


def test_label_files_fingerprint_missing_project_does_not_raise(tmp_path):
    config_path = tmp_path / "does_not_exist" / "config.yaml"
    fp = qc.label_files_fingerprint(config_path)
    assert isinstance(fp, str) and fp
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: close two review
# findings - the scale-invariance test only ever exercised a single folder (so it can't
# tell per-folder normalization apart from a global scale), and nothing locked in the
# anatomical orientation that generalized_procrustes' `init` wiring produces | date: 2026-08-04]
def test_reference_from_folders_normalizes_per_folder_not_globally():
    """Two folders at different scales must reduce to the same reference as the same
    two folders both at scale 1.0 -- because each folder is divided by its OWN
    head_scale before pooling. `test_reference_from_folders_is_scale_invariant` only
    ever passes a single folder, so a global-scale implementation (compute one scale
    over the pooled coords, e.g. accidentally hoisted out of the per-folder loop)
    would pass that test too. This test uses two folders per call specifically so a
    global scale -- which lets the large-scale folder dominate the pooled estimate --
    is distinguishable from the correct per-folder behavior.

    Same RNG seeds feed both constructions, so the only difference between `mixed`
    and `uniform` is the scale factor applied to the second folder; per-folder
    normalization is an exact algebraic cancellation (dividing by a linearly-scaled
    head_scale removes the scale factor exactly), so the two references should match
    to near floating-point precision, not just approximately.
    """
    n1, n2 = 150, 100
    seed1, seed2 = 30, 31

    def make(scale1, scale2):
        rng1 = np.random.default_rng(seed1)
        rng2 = np.random.default_rng(seed2)
        folder1 = _training_frames(n1, rng1, sigma=0.01, scale=scale1)
        folder2 = _training_frames(n2, rng2, sigma=0.01, scale=scale2)
        return qc.build_reference_from_folders([(folder1, _PARTS), (folder2, _PARTS)])

    mixed = make(1.0, 40.0)
    uniform = make(1.0, 1.0)

    for part in qc.RIGID_ROLES:
        assert abs(mixed.train_rms[part] - uniform.train_rms[part]) < 1e-6
    np.testing.assert_allclose(mixed.mean_shape, uniform.mean_shape, atol=1e-6)


def test_reference_from_folders_anatomical_orientation():
    """`mean_full` must sit in the anatomical frame (+y toward the nose, +x toward
    the right ear), which is only true because build_reference_from_folders seeds
    generalized_procrustes with `init` = the mean head-fixed configuration. This test
    exists specifically to catch that `init=init` wiring being dropped (or replaced
    by the default `shapes[0]` seed) -- without it the resulting mean_shape/mean_full
    would be rotated by whatever the first pooled frame's random pose happened to be,
    and the assertions below would fail. Do not "simplify" this test away: none of the
    other reference tests pin down orientation, only frame count/shape/coarse sign.

    mean_full is centred on the rigid-marker centroid (not the headpost), so we assert
    on relative orderings and the sign of the ear x-coordinate rather than on any
    marker's absolute position.
    """
    rng = np.random.default_rng(40)
    coords = _training_frames(400, rng, sigma=0.01, scale=1.0)
    ref = qc.build_reference_from_folders([(coords, _PARTS)])

    rigid_y = {role: ref.mean_full[role][1] for role in qc.RIGID_ROLES}
    assert max(rigid_y, key=rigid_y.get) == "front", \
        "nose/front should have the largest y among the rigid markers"
    assert min(rigid_y, key=rigid_y.get) == "headpost", \
        "headpost should have the smallest y among the rigid markers"

    assert ref.mean_full["rightear"][0] > 0, "right ear should sit at positive x"
    assert ref.mean_full["leftear"][0] < 0, "left ear should sit at negative x"

    assert ref.mean_full["shoulders"][1] < ref.mean_full["headpost"][1], \
        "shoulders (behind the head) should sit lower than the headpost"
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: jeremewingert | reason: Task 5 - per-video
# spread scoring (flag_for_ratio, score_video, worst_rigid) | date: 2026-08-04]
def _write_video_h5(path, n, rng, sigma=0.01, scale=1.0, p=0.99):
    coords = _training_frames(n, rng, sigma=sigma, scale=scale)
    like = np.full((n, len(_PARTS)), p)
    return _write_h5(path, _PARTS, coords, likelihood=like)


def _reference(seed=11, sigma=0.01):
    rng = np.random.default_rng(seed)
    return qc.build_reference_from_folders(
        [(_training_frames(400, rng, sigma=sigma), _PARTS)])


def test_flag_for_ratio_thresholds():
    assert qc.flag_for_ratio(1.0) == "ok"
    assert qc.flag_for_ratio(2.0) == "warn"
    assert qc.flag_for_ratio(5.0) == "fail"
    assert qc.flag_for_ratio(float("nan")) == ""


def test_flag_for_ratio_boundary_semantics():
    assert qc.flag_for_ratio(qc.RATIO_WARN) == "warn"
    assert qc.flag_for_ratio(qc.RATIO_WARN - 0.0001) == "ok"
    assert qc.flag_for_ratio(qc.RATIO_FAIL) == "fail"
    assert qc.flag_for_ratio(qc.RATIO_FAIL - 0.0001) == "warn"


def test_score_video_table_shape_and_columns(tmp_path):
    rng = np.random.default_rng(12)
    p = _write_video_h5(tmp_path / "v.h5", 600, rng)
    res = qc.score_video(p, _reference())
    assert list(res.table.index) == _PARTS
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 - column
    # list updated for TABLE_COLUMNS' new frac_tracked/p_median/flag_track entries
    # (this pre-existing test hardcoded a literal copy of the constant, which Step 4
    # of the task-1 brief explicitly and verbatim redefines) | date: 2026-08-04]
    assert list(res.table.columns) == [
        "role", "n_valid", "frac_valid", "frac_tracked", "p_median",
        "rms_spread", "rms_spread_px", "rms_spread_proc",
        "ratio_spread", "ratio_offset",
        "bias_x", "bias_y", "hf_cx", "hf_cy", "hf_sd", "radius",
        "flag", "flag_track"]
    # [AGENT EDIT END]
    assert res.table.loc["front", "role"] == "rigid"
    assert res.table.loc["shoulders", "role"] == "nonrigid"
    assert res.table.loc["shoulders", "flag"] == ""
    assert res.n_valid == 600
    assert res.has_reference


def test_score_video_matching_noise_gives_ratio_near_one(tmp_path):
    rng = np.random.default_rng(13)
    p = _write_video_h5(tmp_path / "v.h5", 2000, rng, sigma=0.01)
    res = qc.score_video(p, _reference(sigma=0.01))
    # QC v5: the headpost is the head-fix origin -- zero spread in both the video
    # and the labels, so its ratio is 0/0 and it carries no flag
    assert res.table.loc["headpost", "rms_spread"] == 0.0
    assert not np.isfinite(res.table.loc["headpost", "ratio_spread"])
    assert res.table.loc["headpost", "flag"] == ""
    for part in ("front", "leftear", "rightear"):
        assert abs(res.table.loc[part, "ratio_spread"] - 1.0) < 0.25
        assert res.table.loc[part, "flag"] == "ok"


def test_score_video_flags_degraded_tracking(tmp_path):
    rng = np.random.default_rng(14)
    p = _write_video_h5(tmp_path / "bad.h5", 2000, rng, sigma=0.05)
    res = qc.score_video(p, _reference(sigma=0.01))
    part, ratio = qc.worst_rigid(res)
    assert ratio > qc.RATIO_FAIL
    assert res.table.loc[part, "flag"] == "fail"


def test_score_video_is_scale_invariant(tmp_path):
    ref = _reference()
    a = qc.score_video(_write_video_h5(
        tmp_path / "a.h5", 800, np.random.default_rng(15), scale=1.0), ref)
    b = qc.score_video(_write_video_h5(
        tmp_path / "b.h5", 800, np.random.default_rng(15), scale=40.0), ref)
    for col in ("rms_spread", "ratio_spread", "ratio_offset", "hf_sd", "radius"):
        np.testing.assert_allclose(
            a.table[col].astype(float), b.table[col].astype(float), atol=1e-6)
    np.testing.assert_allclose(a.scale_px * 40.0, b.scale_px, rtol=1e-6)
    np.testing.assert_allclose(
        a.table["rms_spread_px"].astype(float) * 40.0,
        b.table["rms_spread_px"].astype(float), rtol=1e-6)


def test_score_video_applies_likelihood_cut(tmp_path):
    rng = np.random.default_rng(16)
    coords = _training_frames(500, rng)
    like = np.full((500, len(_PARTS)), 0.99)
    like[:200, 0] = 0.1           # front unreliable for the first 200 frames
    p = _write_h5(tmp_path / "v.h5", _PARTS, coords, likelihood=like)
    res = qc.score_video(p, _reference(), p_cut=0.9)
    assert res.n_valid == 300
    assert res.n_total == 500
    assert abs(res.frac_valid - 0.6) < 1e-9


def test_score_video_without_reference(tmp_path):
    rng = np.random.default_rng(17)
    p = _write_video_h5(tmp_path / "v.h5", 500, rng)
    res = qc.score_video(p, None)
    assert not res.has_reference
    assert res.table["ratio_spread"].isna().all()
    assert res.table["ratio_offset"].isna().all()
    assert (res.table["flag"] == "").all()
    # rms_spread means the SAME thing here as in the reference path -- scatter about
    # the video's own mean -- so this assertion now validates the quantity the
    # reference path actually computes (it previously did not).
    assert np.isfinite(res.table.loc["leftear", "rms_spread"])
    assert np.isfinite(res.table.loc["leftear", "rms_spread_px"])


def test_score_video_warns_on_few_valid_frames(tmp_path):
    rng = np.random.default_rng(18)
    p = _write_video_h5(tmp_path / "short.h5", 40, rng)
    res = qc.score_video(p, _reference())
    assert res.n_valid == 40
    assert any("valid frame" in w for w in res.warnings)


def test_score_video_subsamples_scatter(tmp_path):
    rng = np.random.default_rng(19)
    p = _write_video_h5(tmp_path / "v.h5", 1200, rng)
    res = qc.score_video(p, _reference(), max_scatter=100)
    assert res.scatter_hf["front"].shape == (100, 2)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final review findings A, C,
# D and F - spread vs offset separation, weak-reference warning, per-part likelihood
# masking, and the anatomical frame in the no-reference path | date: 2026-08-04]
def _wide_head_frames(n, rng, sigma=0.01, widen=1.06):
    """Training-like frames from an animal whose head is `widen` times wider in x.

    Only the ear x-coordinates move (front/headpost sit on x = 0), and the
    nose-to-headpost distance -- the normalizing head-length -- is unchanged, so
    this is a pure shape difference at identical tracking noise.
    """
    template = np.concatenate([_rigid_template(), np.array([[0.0, -1.2]])], axis=0)
    template = template.copy()
    template[:, 0] = template[:, 0] * widen
    return _pose_frames(template, n, rng, noise=sigma)


def test_spread_and_offset_separate_shape_difference_from_tracking_noise(tmp_path):
    """A 6%-wider head at IDENTICAL tracking noise must not read as tracking noise.

    Before the split, `ratio` mixed the systematic offset from the training mean into
    the same number as the residual scatter, so this case flagged both ears amber on
    perfect tracking. `ratio_spread` must stay near 1 (the noise really is the same)
    while `ratio_offset` carries the shape difference, and nothing may flag.
    """
    rng = np.random.default_rng(70)
    coords = _wide_head_frames(2000, rng, sigma=0.01, widen=1.06)
    like = np.full((2000, len(_PARTS)), 0.99)
    p = _write_h5(tmp_path / "wide.h5", _PARTS, coords, likelihood=like)
    res = qc.score_video(p, _reference(sigma=0.01))

    for ear in ("leftear", "rightear"):
        assert abs(res.table.loc[ear, "ratio_spread"] - 1.0) < 0.3, \
            "tracking noise is unchanged, so ratio_spread must stay near 1"
        assert res.table.loc[ear, "flag"] == "ok"
    # QC v5 measures the offset against the labels' head-fix scatter, which is
    # inflated by front/headpost axis-noise propagation relative to the old
    # Procrustes-frame denominator -- the same 6% widening now reads ~1.2x, still
    # clearly above the ~1.0 tracking-noise floor asserted above
    assert res.table.loc["rightear", "ratio_offset"] > 1.0, \
        "the systematic shape difference must show up in ratio_offset"
    assert res.table.loc["leftear", "ratio_offset"] > 1.0

    # and the offset really is lateral: ears pushed apart in x
    assert res.table.loc["rightear", "bias_x"] > 0
    assert res.table.loc["leftear", "bias_x"] < 0


def test_score_video_masks_each_body_part_by_its_own_likelihood(tmp_path):
    """DLC writes coordinates for every part in every frame, so a finite-only mask is
    all-True and the per-row n_valid/frac_valid used to be identical to the frame-level
    numbers for every row. Each row must instead reflect THAT part's likelihood.
    """
    rng = np.random.default_rng(71)
    n = 600
    coords = _training_frames(n, rng)
    like = np.full((n, len(_PARTS)), 0.99)
    like[:500, _PARTS.index("shoulders")] = 0.1     # shoulders mostly untracked
    p = _write_h5(tmp_path / "v.h5", _PARTS, coords, likelihood=like)
    res = qc.score_video(p, _reference(), p_cut=0.9)

    # the frame-level rigid gate is untouched: shoulders is not a rigid marker
    assert res.n_valid == n
    assert res.table.loc["front", "n_valid"] == n
    assert res.table.loc["shoulders", "n_valid"] == 100
    assert abs(res.table.loc["shoulders", "frac_valid"] - 100.0 / n) < 1e-9
    assert np.isfinite(res.table.loc["shoulders", "rms_spread"])


def test_score_video_untracked_part_is_nan_not_a_plausible_number(tmp_path):
    """A body part the model never learned must produce NaN, not a number that looks
    like a healthy measurement.
    """
    rng = np.random.default_rng(72)
    n = 400
    coords = _training_frames(n, rng)
    like = np.full((n, len(_PARTS)), 0.99)
    like[:, _PARTS.index("shoulders")] = 0.05       # never tracked
    p = _write_h5(tmp_path / "v.h5", _PARTS, coords, likelihood=like)
    res = qc.score_video(p, _reference(), p_cut=0.9)

    assert res.table.loc["shoulders", "n_valid"] == 0
    assert res.table.loc["shoulders", "frac_valid"] == 0.0
    for col in ("rms_spread", "rms_spread_px", "ratio_spread", "ratio_offset",
                "bias_x", "bias_y", "hf_cx", "hf_cy", "hf_sd", "radius"):
        assert np.isnan(float(res.table.loc["shoulders", col])), col
    # rigid rows are unaffected
    assert np.isfinite(res.table.loc["leftear", "rms_spread"])


def test_score_video_warns_when_reference_is_weak(tmp_path):
    """MIN_REFERENCE_FRAMES only ever emitted a log.warning, which nothing in the
    Streamlit app reads, so a reference built from a few dozen labels still rendered
    authoritative flags. Weak references must surface as a SpreadResult warning.
    """
    rng = np.random.default_rng(73)
    weak = qc.build_reference_from_folders(
        [(_training_frames(36, rng, sigma=0.01), _PARTS)])
    assert weak.n_frames == 36
    p = _write_video_h5(tmp_path / "v.h5", 400, np.random.default_rng(74))

    res = qc.score_video(p, weak)
    assert any("reference" in w for w in res.warnings), res.warnings
    assert any(str(qc.WEAK_REFERENCE_FRAMES) in w for w in res.warnings)

    strong = _reference()                            # 400 labeled frames
    assert strong.n_frames >= qc.WEAK_REFERENCE_FRAMES
    res_strong = qc.score_video(p, strong)
    assert not any("reference" in w for w in res_strong.warnings)


def test_score_video_without_reference_uses_anatomical_frame(tmp_path):
    """Without a reference the mean shape is fit from the video itself; it must be
    seeded from the mean head-fixed configuration (as build_reference_from_folders
    does), not from frame 0's arbitrary pose, or everything derived from the aligned
    coordinates lands in a randomly rotated frame.
    """
    rng = np.random.default_rng(75)
    p = _write_video_h5(tmp_path / "v.h5", 600, rng)
    res = qc.score_video(p, None)

    shape = np.asarray(res.mean_shape, dtype=float)
    assert shape.shape == (4, 2)
    order = list(qc.RIGID_ROLES)
    ys = {role: shape[order.index(role), 1] for role in order}
    assert max(ys, key=ys.get) == "front"
    assert min(ys, key=ys.get) == "headpost"
    assert shape[order.index("rightear"), 0] > 0
    assert shape[order.index("leftear"), 0] < 0
# [AGENT EDIT END]


def test_match_h5_files_requires_all_tokens(tmp_path):
    names = [
        "REI001aDLC_resnet50_freeAug1shuffle1_240000.h5",   # current model
        "REI001aDLC_resnet50_freeAug1shuffle1_100000.h5",   # older snapshot
        "REI001aDLC_resnet50_otherAug1shuffle1_240000.h5",  # other model
    ]
    paths = [tmp_path / n for n in names]
    got = qc.match_h5_files(paths, "free", ["shuffle1", "_240000"])
    assert [p.name for p in got] == [names[0]]


def test_match_h5_files_without_filters_passes_everything(tmp_path):
    paths = [tmp_path / "a.h5", tmp_path / "b.h5"]
    assert qc.match_h5_files(paths, None, None) == paths


def test_ellipse_path_traces_covariance():
    mean = np.array([1.0, -2.0])
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    pts = qc.ellipse_path(mean, cov, n_sd=2.0, n_points=128)
    assert pts.shape == (128, 2)
    # semi-axes are 2 sd: 4 in x, 2 in y, centred on the mean
    assert abs((pts[:, 0].max() - pts[:, 0].min()) / 2 - 4.0) < 1e-6
    assert abs((pts[:, 1].max() - pts[:, 1].min()) / 2 - 2.0) < 1e-6
    np.testing.assert_allclose(pts.mean(axis=0), mean, atol=1e-6)


def test_ellipse_path_handles_bad_covariance():
    pts = qc.ellipse_path(np.array([0.0, 0.0]), np.full((2, 2), np.nan))
    assert np.isnan(pts).all()


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 1 - per-marker
# tracking rate and confidence (flag_for_fraction, frac_tracked, p_median,
# flag_track, worst_rigid_tracking) | date: 2026-08-04]
def test_flag_for_fraction_is_inverted():
    assert qc.flag_for_fraction(1.0) == "ok"
    assert qc.flag_for_fraction(0.65) == "warn"
    assert qc.flag_for_fraction(0.2) == "fail"
    assert qc.flag_for_fraction(float("nan")) == ""
    assert qc.flag_for_fraction(None) == ""


def test_flag_for_fraction_boundaries():
    # lower is worse, so the threshold value itself is the BETTER band
    assert qc.flag_for_fraction(qc.TRACK_WARN) == "ok"
    assert qc.flag_for_fraction(qc.TRACK_WARN - 1e-6) == "warn"
    assert qc.flag_for_fraction(qc.TRACK_FAIL) == "warn"
    assert qc.flag_for_fraction(qc.TRACK_FAIL - 1e-6) == "fail"


def test_frac_tracked_is_measured_before_the_joint_gate(tmp_path):
    # front is unreliable in the first 400 of 1000 frames; leftear in a different
    # 200. The joint rigid gate drops the union (600 frames), so a post-gate
    # measurement would report both markers identically -- the bug this replaces.
    rng = np.random.default_rng(40)
    n = 1000
    coords = _training_frames(n, rng)
    like = np.full((n, len(_PARTS)), 0.99)
    like[:400, _PARTS.index("front")] = 0.1
    like[500:700, _PARTS.index("leftear")] = 0.1
    p = _write_h5(tmp_path / "v.h5", _PARTS, coords, likelihood=like)
    res = qc.score_video(p, _reference(), p_cut=0.9)

    assert abs(res.table.loc["front", "frac_tracked"] - 0.6) < 1e-9
    assert abs(res.table.loc["leftear", "frac_tracked"] - 0.8) < 1e-9
    assert abs(res.table.loc["rightear", "frac_tracked"] - 1.0) < 1e-9
    # the two bad markers must be distinguishable from each other
    assert res.table.loc["front", "frac_tracked"] < res.table.loc["leftear", "frac_tracked"]
    # frame-level gate drops the union of both bad spans
    assert res.n_valid == 400


def test_frac_tracked_does_not_collapse_like_frac_valid(tmp_path):
    rng = np.random.default_rng(41)
    n = 800
    coords = _training_frames(n, rng)
    like = np.full((n, len(_PARTS)), 0.99)
    like[:300, _PARTS.index("front")] = 0.1
    p = _write_h5(tmp_path / "v.h5", _PARTS, coords, likelihood=like)
    res = qc.score_video(p, _reference(), p_cut=0.9)
    rigid = [x for x in _PARTS if res.table.loc[x, "role"] == "rigid"]
    # frac_valid is identical across rigid markers (post-gate, by construction)
    assert res.table.loc[rigid, "frac_valid"].nunique() == 1
    # frac_tracked is not -- it isolates the offending marker
    assert res.table.loc[rigid, "frac_tracked"].nunique() > 1


def test_p_median_is_threshold_free(tmp_path):
    # _PARTS' nonrigid marker is "shoulders" (front/headpost/leftear/rightear are
    # rigid and already exercised elsewhere); the brief's own local part list here
    # renames it to "rear" to match the brief's variable names -- _PARTS itself
    # (shared by every other test) is left untouched.
    parts = _PARTS[:-1] + ["rear"]
    rng = np.random.default_rng(42)
    n = 600
    coords = _training_frames(n, rng)
    like = np.full((n, len(parts)), 0.99)
    like[:200, parts.index("rear")] = 0.30
    p = _write_h5(tmp_path / "v.h5", parts, coords, likelihood=like)
    a = qc.score_video(p, _reference(), p_cut=0.5)
    b = qc.score_video(p, _reference(), p_cut=0.95)
    # p_median depends only on the data, never on where the cut is placed
    for part in parts:
        np.testing.assert_allclose(a.table.loc[part, "p_median"],
                                   b.table.loc[part, "p_median"])
    assert abs(a.table.loc["rear", "p_median"] - 0.99) < 1e-9


def test_flag_track_applies_to_nonrigid_markers_too(tmp_path):
    # local part list, "shoulders" renamed to "rear" -- see note above.
    parts = _PARTS[:-1] + ["rear"]
    rng = np.random.default_rng(43)
    n = 1000
    coords = _training_frames(n, rng)
    like = np.full((n, len(parts)), 0.99)
    like[:700, parts.index("rear")] = 0.1        # rear tracked in only 30% of frames
    p = _write_h5(tmp_path / "v.h5", parts, coords, likelihood=like)
    res = qc.score_video(p, _reference(), p_cut=0.9)
    assert res.table.loc["rear", "role"] == "nonrigid"
    assert res.table.loc["rear", "flag_track"] == "fail"
    # the spread flag stays rigid-only
    assert res.table.loc["rear", "flag"] == ""
    assert res.table.loc["front", "flag_track"] == "ok"


def test_worst_rigid_tracking_picks_the_lowest(tmp_path):
    # local part list, "shoulders" renamed to "rear" -- see note above.
    parts = _PARTS[:-1] + ["rear"]
    rng = np.random.default_rng(44)
    n = 1000
    coords = _training_frames(n, rng)
    like = np.full((n, len(parts)), 0.99)
    like[:600, parts.index("leftear")] = 0.1     # rigid, 40% tracked
    like[:900, parts.index("rear")] = 0.1        # non-rigid, 10% tracked -- must be ignored
    p = _write_h5(tmp_path / "v.h5", parts, coords, likelihood=like)
    res = qc.score_video(p, _reference(), p_cut=0.9)
    part, frac = qc.worst_rigid_tracking(res)
    assert part == "leftear"
    assert abs(frac - 0.4) < 1e-9


def test_tracking_columns_survive_without_a_reference(tmp_path):
    rng = np.random.default_rng(45)
    n = 500
    coords = _training_frames(n, rng)
    like = np.full((n, len(_PARTS)), 0.99)
    like[:300, _PARTS.index("front")] = 0.1
    p = _write_h5(tmp_path / "v.h5", _PARTS, coords, likelihood=like)
    res = qc.score_video(p, None)
    assert not res.has_reference
    assert res.table["ratio_spread"].isna().all()
    # tracking needs no training labels, so it must still be populated
    assert abs(res.table.loc["front", "frac_tracked"] - 0.4) < 1e-9
    assert res.table.loc["front", "flag_track"] == "fail"
    assert np.isfinite(res.table.loc["front", "p_median"])


def test_table_column_order_includes_tracking():
    assert qc.TABLE_COLUMNS == [
        "role", "n_valid", "frac_valid", "frac_tracked", "p_median",
        "rms_spread", "rms_spread_px", "rms_spread_proc",
        "ratio_spread", "ratio_offset",
        "bias_x", "bias_y", "hf_cx", "hf_cy", "hf_sd", "radius",
        "flag", "flag_track"]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: qc v2 -- spread follows
# baphy_io's loader cut (0.2) so it describes the frames analyses actually keep, tracking
# keeps its own strict bar, and every marker's residual is recorded rather than only the
# worst. | date: 2026-08-17]
def test_spread_cut_matches_the_loader_threshold():
    # baphy_io.py: options['dlc_threshold'] = options.get('dlc_threshold', 0.2).
    # Scoring above that measured frames no analysis consumes.
    assert qc.LOADER_P_CUT == 0.2
    assert qc.DEFAULT_P_CUT == qc.LOADER_P_CUT


def test_both_gates_are_the_loader_threshold():
    # an earlier revision held frac_tracked at a strict 0.9 on a saturation argument
    # generalised from two sessions; withdrawn -- the ungated tracked error is known to
    # track HRTF improvement, so QC reports the frames analyses actually load.
    assert qc.DEFAULT_TRACK_P_CUT == qc.LOADER_P_CUT
    assert qc.DEFAULT_P_CUT == qc.LOADER_P_CUT


def test_gates_use_baphy_strictly_greater_comparison():
    # baphy_io.py:4769 -- `likelihood.values > dlc_threshold`. A `>=` gate would admit
    # frames baphy drops, so QC would not be scoring the set analyses receive.
    import inspect
    src = inspect.getsource(qc.score_video)
    assert '> p_cut' in src and '>= p_cut' not in src
    assert '> track_p_cut' in src and '>= track_p_cut' not in src


def test_track_p_cut_defaults_to_p_cut_when_unset():
    import inspect
    sig = inspect.signature(qc.score_video)
    assert sig.parameters['track_p_cut'].default is None


def _fake_result(rows, warnings=(), n_total=1000, n_valid=900):
    import pandas as pd
    import numpy as np

    class _Ref:
        n_frames = 75
        sources = ['A', 'B']

    class _R:
        table = pd.DataFrame(rows).set_index('part')
        scale_px = 20.0
        n_total, n_valid = 1000, 900
        frac_valid = 0.9
        has_reference = True
        reference = _Ref()
        mean_shape = np.zeros((4, 2))

    _R.warnings = list(warnings)
    return _R()


def test_summarize_keeps_every_marker_not_just_the_worst():
    result = _fake_result([
        {'part': 'front', 'role': 'rigid', 'ratio_spread': 0.5, 'ratio_offset': 0.1,
         'frac_tracked': 0.90, 'p_median': 0.99},
        {'part': 'leftear', 'role': 'rigid', 'ratio_spread': 1.5, 'ratio_offset': 0.3,
         'frac_tracked': 0.70, 'p_median': 0.95},
        {'part': 'rear', 'role': 'nonrigid', 'ratio_spread': 9.0, 'ratio_offset': 5.0,
         'frac_tracked': 0.10, 'p_median': 0.50},
    ])
    doc = qc.summarize(result)

    # non-rigid markers are recorded too -- they carry real information
    assert set(doc['parts']) == {'front', 'leftear', 'rear'}
    assert doc['parts']['front']['ratio_spread'] == 0.5

    # ...but the headline is over RIGID markers only, so an occluded body marker
    # cannot drive the verdict
    assert doc['ratio_mean'] == 1.0                       # mean(0.5, 1.5), rear excluded
    assert doc['ratio_worst'] == 1.5
    assert doc['ratio_worst_part'] == 'leftear'
    assert doc['track_mean'] == 0.80                      # mean(0.90, 0.70)


def test_summarize_is_json_safe_for_non_finite_values():
    import json
    result = _fake_result([
        {'part': 'front', 'role': 'rigid', 'ratio_spread': float('nan'),
         'ratio_offset': float('inf'), 'frac_tracked': 0.9, 'p_median': 0.99},
    ])
    doc = qc.summarize(result)
    assert doc['parts']['front']['ratio_spread'] is None
    assert doc['parts']['front']['ratio_offset'] is None
    json.dumps(doc, allow_nan=False)                      # would raise on NaN/Infinity


def test_summarize_carries_the_caveats_that_qualify_the_numbers():
    result = _fake_result(
        [{'part': 'front', 'role': 'rigid', 'ratio_spread': 0.5, 'ratio_offset': 0.1,
          'frac_tracked': 0.9, 'p_median': 0.99}],
        warnings=['only 12 valid frames (< 100); statistics are noisy'])
    doc = qc.summarize(result)
    assert doc['warnings'] == ['only 12 valid frames (< 100); statistics are noisy']
    assert doc['reference_frames'] == 75
    assert doc['scale_px'] == 20.0                        # the unit the ratios are in
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the aligned frame claimed to
# be anatomical but settled nose-DOWN -- the Procrustes ran on image coordinates, whose
# chirality is opposite to the head-fix basis, and rotation-only Kabsch cannot honour a
# mirrored seed. Pin the orientation so it cannot silently flip back. | date: 2026-08-18]
def _synthetic_rigid_video(n=300, seed=0):
    """Image-coordinate (y-down) frames of a rigid head at random poses + small noise."""
    rng = np.random.default_rng(seed)
    # anatomical template: nose forward, ears sideways, headpost behind
    template = np.array([[0.0, 1.0],      # front
                         [0.0, -0.4],     # headpost
                         [-0.9, 0.1],     # leftear
                         [0.9, 0.1]])     # rightear
    frames = []
    for _ in range(n):
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        world = template @ R.T * 40.0 + rng.uniform(100, 500, size=2)
        world = world * np.array([1.0, -1.0])          # image convention: y grows DOWN
        frames.append(world + rng.normal(0, 0.5, size=world.shape))
    return np.stack(frames)


def test_aligned_frame_is_anatomical_nose_up():
    coords = _synthetic_rigid_video()
    bodyparts = ['front', 'headpost', 'leftear', 'rightear']
    likelihood = np.ones(coords.shape[:2])
    import pandas as pd
    cols = pd.MultiIndex.from_product([['s'], bodyparts, ['x', 'y', 'likelihood']])
    rows = np.concatenate([np.concatenate([coords[:, b, :],
                                           likelihood[:, b:b + 1]], axis=1)
                           for b in range(4)], axis=1)
    df = pd.DataFrame(rows, columns=cols)
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'synthDLC_test.h5')
        df.to_hdf(p, key='df_with_missing')
        r = qc.score_video(p, reference=None, p_cut=0.2)

    ms = {role: xy for role, xy in zip(qc.RIGID_ROLES, np.asarray(r.mean_shape))}
    assert ms['front'][1] > 0, 'nose must point to +y'
    # QC v5: the head-fix origin IS the headpost, exactly, in every frame
    assert np.allclose(ms['headpost'], 0.0), 'headpost is the origin by construction'
    assert abs(ms['front'][0]) < 1e-9, 'front is pinned to the +y axis'
    assert ms['rightear'][0] > 0 and ms['leftear'][0] < 0, '+x must be the right ear'
    assert r.origin_part == 'headpost'
    # and the scatter lives in the same frame as the mean it is judged against
    for role in ('front', 'headpost'):
        cloud = np.nanmean(r.scatter_aligned[role], axis=0)
        assert abs(cloud[1] - ms[role][1]) < 0.1
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: QC v6 -- the
# self-Procrustes companion must distribute error to EVERY rigid marker, unlike the
# axis frame where front is pinned to x=0 and the headpost to (0,0) | date: 2026-08-19]
def test_v6_self_procrustes_distributes_error(tmp_path):
    rng = np.random.default_rng(21)
    p = _write_video_h5(tmp_path / "v.h5", 2000, rng, sigma=0.01)
    res = qc.score_video(p, reference=None)

    # axis frame: structural constraints hold
    assert res.table.loc["headpost", "rms_spread"] == 0.0
    front_ax = res.scatter_aligned["front"]
    assert np.nanstd(front_ax[:, 0]) < 1e-9, "front x is pinned in the axis frame"

    # self-Procrustes frame: every rigid marker carries its own 2-D residual
    for part in qc.RIGID_ROLES:
        assert res.table.loc[part, "rms_spread_proc"] > 0.0
    front_pr = res.scatter_proc["front"]
    assert np.nanstd(front_pr[:, 0]) > 1e-4, "front error is distributed, not absorbed"
    # the 4-point fit spreads the same total noise more evenly than the 2-point axis
    ears_ax = res.table.loc[["leftear", "rightear"], "rms_spread"].astype(float)
    ears_pr = res.table.loc[["leftear", "rightear"], "rms_spread_proc"].astype(float)
    assert (ears_pr < ears_ax).all(), \
        "ear residuals shrink once front/back stop dumping their noise into the frame"

    doc = qc.summarize(res)
    assert doc["rms_spread_proc_mean"] is not None
    assert doc["rms_spread_proc_worst_part"] in qc.RIGID_ROLES
# [AGENT EDIT END]
