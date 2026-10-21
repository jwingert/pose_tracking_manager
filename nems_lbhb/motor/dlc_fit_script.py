"""
process face video in dlc, ripped off of pupil_fit_script

very helpful tutorial/notes on refinement
https://guillermohidalgogadea.com/openlabnotebook/refining-your-dlc-model/

run in deeplabcut environment (tensorflow 2.4, cuda 11.0, etc) that
also has NEMS and nems_db installed

syntax:
   dlc_fit_script <avipath>/<avibase>.avi> [<action_list>] [<dlcmodelconfig filepath>] [<dest_path>] [key=value ...]

Arguments:
   video_file: Path to the video file to process
   action_list: Comma-separated list of actions (fit, video, summary, refine, extract,
                curate, training)
   dlcmodelconfig: Path to the DLC config.yaml.  Leave EMPTY ('') to resolve the
                 model through nems_lbhb/configs/dlc_registry.toml, which is the
                 production path -- see "Model resolution" below.
   dest_path: Destination path for results (or 'training' for training mode)

key=value arguments:
   alias=<name>        resolve the model from this registry alias instead of from
                       the video path's registry rules.
   allow_project=1     opt in to running from an alias's MUTABLE training project
                       when that alias has frozen releases.  Refused without it.
   init_weights=<path> transfer learning: snapshot path without .meta extension.
                       'training' action only; applied AFTER create_training_dataset().
   from_release=<id>   transfer learning from a frozen release's snapshot.  Same
                       effect as init_weights but resolved from the release and
                       checked against the configured net_type.
   resume=1            'training' action only: opt in to training into an iteration
                       that already holds snapshots (they WILL be overwritten).
   user=<name>         who launched this run; recorded in the training lock,
                       TRAINRUN.toml and the curation log.
   run_id=<id>         run identifier; recorded likewise.
   note=<text>         'curate' action only: free text recorded with the curation
                       entry ("added SQD032a, refined LMD004a").
   rescale_tmp=<dir>   'fit' action only: where the downscaled analysis copy of an
                       oversized video is written (default: the system temp dir).
                       An MJPEG copy of a 30-80k-frame arena video runs to ~1 GB,
                       so point this somewhere with room if /tmp is small.  The
                       directory is always deleted when the run ends.

Actions:
   curate    merge the labelled folders into the training set (deeplabcut.merge_datasets),
             refuse if that merge silently skipped, and append what changed to
             <project>/curation_history.log.  This used to happen implicitly inside
             'training'; it is now its own step, so accepting labels is a decision
             someone makes rather than a side effect of clicking Train.
   training  create the training dataset and train.  Refuses to start when the labels
             on disk are newer than the last 'curate' -- run curate first.

Model resolution (in order of precedence):
   1. an explicit config path in argv[3];
   2. alias=<name>            -> dlc_registry.resolve(alias)['config'];
   3. otherwise               -> dlc_registry.config_for_video(video_file).
   When the resolved model is a frozen release, it is verified against its own
   SHA256SUMS before anything is analyzed, and the run aborts on any problem.

results saved in <avipath>/sorted/<avibase>.dlc.h5, with a second, release-stamped
alias <avibase>.<release_id>.dlc.h5 recording which model produced them.

Analysis resolution:
   A ResNet pose model has no scale invariance to spare, and 13 legacy Prince
   sites (Nov 2022 - Feb 2023) were recorded at 1280x720 while every arena
   project's labelled frames are 640x480.  The 'fit' action therefore analyzes an
   aspect-preserving downscaled COPY of any video wider than the alias's
   train_frame_size (1280x720 -> 640x360), then converts the resulting x/y back to
   NATIVE pixels, so what lands in sorted/ is what a native run would have
   produced only better -- on PRN043a05 the worst rigid marker goes 0.334 -> 0.848.
   Because that output is deliberately indistinguishable, every fit also writes
   <avibase>.<release_id>.scale.json recording the native size, the analysis size,
   the scale factor and the reason.  See nems_lbhb/motor/dlc_video_scale.py.
"""

import os
import glob
import shutil as _shutil
import sys

import numpy as np

import nems_lbhb.db as nd
import nems0
from pathlib import Path
# [AGENT EDIT START | agent: claude | user: wingertj | reason: atomic alias; was os.system ln -s on newest mtime, silently a no-op when the alias existed | date: 2026-08-06]
from nems_lbhb.motor import dlc_alias
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- production inference
# runs from a verified release; training is locked and guarded | date: 2026-08-06]
# DeepLabCut is imported inside `if __name__ == '__main__'`, not here.  Everything
# above the __main__ block is plain model resolution and symlink bookkeeping, and
# keeping it importable without DeepLabCut is what lets tests/test_dlc_fit_script.py
# check the resolution precedence and the alias naming in psi_nems2 (which has no
# DLC) instead of only ever exercising them on a GPU.
from nems_lbhb.motor import dlc_registry
from nems_lbhb.motor import dlc_release
from nems_lbhb.motor import dlc_train_guard
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude | user: wingertj | reason: resolution-based inference rescaling -- 13 legacy Prince sites are 1280x720 while every arena model was trained on 640x480 frames | date: 2026-08-08]
from nems_lbhb.motor import dlc_video_scale
# [AGENT EDIT END]

import logging
import yaml

log = logging.getLogger(__name__)


def parse_key_value_args(argv):
    """Parse key=value arguments from command line."""
    kwargs = {}
    for arg in argv:
        if '=' in arg:
            key, value = arg.split('=', 1)
            kwargs[key] = value
    return kwargs


def update_pose_cfg_init_weights(config_path, init_weights_path):
    """
    Update init_weights in pose_cfg.yaml after create_training_dataset() has run.

    Args:
        config_path: Path to the DLC project config.yaml
        init_weights_path: Path to snapshot (without .meta extension)
    """
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(config_path)
    project_path = Path(config_path).parent

    # Find the pose_cfg.yaml in the training directory
    # It's typically at: dlc-models/iteration-X/PROJECTNAME-DATE-shuffle-1/train/pose_cfg.yaml
    model_folder = project_path / 'dlc-models'

    # Get the most recent iteration
    iterations = sorted(model_folder.glob('iteration-*'))
    if not iterations:
        log.error("No iteration folders found in dlc-models")
        return False

    latest_iteration = iterations[-1]

    # Find the training folder (shuffle directory)
    shuffle_folders = list(latest_iteration.glob('*shuffle*'))
    if not shuffle_folders:
        log.error(f"No shuffle folders found in {latest_iteration}")
        return False

    for shuffle_folder in shuffle_folders:
        pose_cfg_path = shuffle_folder / 'train' / 'pose_cfg.yaml'
        if pose_cfg_path.exists():
            log.info(f"Updating init_weights in: {pose_cfg_path}")

            # Read the pose_cfg.yaml
            with open(pose_cfg_path, 'r') as f:
                pose_cfg = yaml.safe_load(f)

            old_init_weights = pose_cfg.get('init_weights', 'not set')
            log.info(f"  Old init_weights: {old_init_weights}")
            log.info(f"  New init_weights: {init_weights_path}")

            # Update init_weights
            pose_cfg['init_weights'] = init_weights_path

            # Write back
            with open(pose_cfg_path, 'w') as f:
                yaml.dump(pose_cfg, f, default_flow_style=False)

            log.info(f"  Successfully updated pose_cfg.yaml")
        else:
            log.warning(f"pose_cfg.yaml not found at {pose_cfg_path}")

    return True


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- registry/release
# model resolution, release verification, two-alias provenance, training hyperparameters
# and TRAINRUN.toml, all as importable functions | date: 2026-08-06]

TRAINRUN_NAME = 'TRAINRUN.toml'

# hyperparameters that must be present in [training_defaults]; there are no
# code-side defaults, because a default here is a silent divergence between what
# the registry says a model was trained with and what it actually was.
_REQUIRED_TRAINING_KEYS = ('net_type', 'augmenter_type', 'max_snapshots_to_keep',
                           'saveiters', 'maxiters', 'displayiters', 'autotune',
                           'allow_growth')


def truthy(value):
    """Command-line flag parsing: '1'/'true'/'yes'/'on' are true, '' is false.

    Anything else raises, because ``allow_project=0`` silently meaning "yes" (as a
    bare ``bool(str)`` would make it) is precisely the class of accident this task
    is meant to remove.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('', '0', 'false', 'no', 'off'):
        return False
    raise ValueError('expected a boolean flag (1/0), got %r' % (value,))


def _alias_releases(block):
    """Every release id the registry records for one alias, oldest first."""
    ids = list(block.get('history', []))
    current = block.get('current', '')
    if current and current not in ids:
        ids.append(current)
    return ids


def release_id_of_config(config_path, root=None):
    """The release id when ``config_path`` is ``<root>/<release_id>/config.yaml``, else None.

    The Streamlit app hands ``dlc_fit_script`` an explicit config path for every
    button, including "Analyze DLC to sorted".  Recognising a release *by its path*
    is what lets that path still be verified and still write a stamped provenance
    alias, instead of being demoted to an unstamped ad-hoc run just because the
    caller spelled the model as a path rather than as an alias.
    """
    if root is None:
        root = dlc_release.RELEASE_ROOT
    parent = os.path.dirname(os.path.normpath(os.path.abspath(config_path)))
    if os.path.dirname(parent) != os.path.normpath(os.path.abspath(root)):
        return None
    if not os.path.isfile(os.path.join(parent, dlc_release.MANIFEST_NAME)):
        return None
    return os.path.basename(parent)


def _warn_if_config_shadows_a_release(config_path, registry):
    """Log loudly when an explicit config path is an alias's mutable project and
    that alias has frozen releases.

    Not a refusal: an explicit path is the caller saying which model they mean, and
    the brief makes case 1 imply ``allow_project``.  But it is exactly the situation
    releases exist to end, so it does not get to happen quietly.
    """
    project = os.path.dirname(os.path.normpath(os.path.abspath(config_path)))
    for name in registry['alias_names']:
        block = registry['aliases'][name]
        if os.path.normpath(os.path.abspath(block['project'])) != project:
            continue
        releases = _alias_releases(block)
        if releases:
            log.warning(
                'explicit config %s is alias %r\'s MUTABLE training project, but that '
                'alias serves release %s. These results will carry no provenance stamp '
                'and can change under the next train_network run. Drop the config path '
                '(pass an empty argv[3]) to run from the release instead.',
                config_path, name, block.get('current') or releases[-1])
        return name
    return None


def resolve_model(video_file, explicit_config=None, alias=None,
                  allow_project=False, registry=None):
    """Decide which DeepLabCut model analyzes ``video_file``.

    Precedence:

    1. ``explicit_config`` -- an explicit ``config.yaml`` path (``sys.argv[3]``).
       This is how the Streamlit app's "analyze to temp" runs an arbitrary model,
       so it is honoured as-is and *implies* ``allow_project``: someone who typed a
       path has already said which model they mean.
    2. ``alias`` -- a registry alias name.
    3. the registry rules for ``video_file``.

    Returns a dict with the keys :func:`dlc_registry.resolve` returns
    (``alias, release_id, config, project, source``) plus ``how``, a one-line
    description of which branch decided, for the log.  ``source`` is
    ``'release'``, ``'project'`` or ``'explicit'``.

    Raises ``RuntimeError`` when resolution lands on an alias's *mutable training
    project* while that alias has frozen releases.  Running production inference
    from a project directory is how the same alias came to mean two different
    models on two different days; ``allow_project=True`` is the explicit opt-in
    that says you mean it.
    """
    if registry is None:
        registry = dlc_registry.load_registry()

    if explicit_config:
        rid = release_id_of_config(explicit_config)
        if rid is not None:
            manifest_alias = dlc_release.read_manifest(rid).get('release', {}).get('alias')
            return {'alias': alias or manifest_alias, 'release_id': rid,
                    'config': explicit_config, 'project': None, 'source': 'release',
                    'how': 'explicit config path inside %s -> release %s'
                           % (dlc_release.RELEASE_ROOT, rid)}
        return {'alias': alias or _warn_if_config_shadows_a_release(explicit_config, registry),
                'release_id': None, 'config': explicit_config,
                'project': None, 'source': 'explicit',
                'how': 'explicit config path (implies allow_project)'}

    if alias:
        name = alias
        how = 'alias=%s' % alias
    else:
        name = dlc_registry.resolve_for_video(video_file, registry)
        how = 'registry rules matched alias %s' % name

    info = dlc_registry.resolve(name, registry)
    releases = _alias_releases(registry['aliases'][name])

    if allow_project:
        project_config = os.path.join(info['project'], 'config.yaml')
        return {'alias': name, 'release_id': None, 'config': project_config,
                'project': info['project'], 'source': 'project',
                'how': how + ' + allow_project=1 (MUTABLE training project)'}

    if info['source'] == 'project':
        if releases:
            raise RuntimeError(
                "alias %r resolves to its mutable training project %s, but frozen "
                "releases exist for it (%s; most recent %s). Analyzing from a "
                "training project means the next train_network run silently changes "
                "what this alias produced. Either promote a release "
                "(nems_lbhb.motor.dlc_release.promote(%r, '%s')) so `current` is set, "
                "or pass allow_project=1 to say you really want the project."
                % (name, info['project'], ', '.join(releases), releases[-1],
                   name, releases[-1]))
        info = dict(info)
        info['how'] = how + ' (no release cut for this alias yet)'
        return info

    info = dict(info)
    info['how'] = how
    return info


def verify_resolved_release(resolved, full=False):
    """Check a resolved release against its own SHA256SUMS; raise on any problem.

    A no-op for ``source != 'release'``.  ``full=False`` (presence + size) is what
    the fit path uses: re-hashing a 400 MB resnet on every video would cost more
    than the analysis.  A release whose files were deleted, truncated or replaced
    fails here, before anything is analyzed with it.
    """
    if resolved.get('source') != 'release':
        return []
    release_id = resolved['release_id']
    problems = dlc_release.verify(release_id, full=full)
    if problems:
        raise RuntimeError(
            'DLC release %s failed verification and will not be used:\n  %s'
            % (release_id, '\n  '.join(problems)))
    log.info('verified DLC release %s (%s checks)', release_id,
             'full sha256' if full else 'presence+size')
    return problems


def release_output_suffix(release_id):
    """The scorer suffix recorded in a release's ``RELEASE.toml``.

    Cross-checked against DeepLabCut's own ``GetScorerName`` at fit time; the two
    agreeing is what makes ``find_dlc_output`` deterministic rather than a guess.
    """
    manifest = dlc_release.read_manifest(release_id)
    suffix = manifest.get('model', {}).get('output_suffix', '')
    if not suffix:
        raise RuntimeError('release %s records no [model].output_suffix' % release_id)
    return suffix


# [AGENT EDIT START | agent: claude | user: wingertj | reason: a scorer string does not
# identify a release, so two releases can write the same filename -- give each release its
# own output directory | date: 2026-08-06]
def release_output_dir(dest_dir, release_id=None, create=True):
    """Directory DeepLabCut should analyze into: ``<dest_dir>/<release_id>``, or
    ``dest_dir`` itself when there is no release.

    DeepLabCut names its output ``<stem><scorer>.h5`` where
    ``scorer = DLC_<net>_<Task><date>shuffle<S>_<snapshot>``.  **Iteration is not in
    that string**, so two releases cut from the same project at the same snapshot
    number but different iterations produce byte-identical filenames while holding
    different weights -- e.g. ``two_chimney-r0001`` (iteration-0, md5 3386f312) and
    ``two_chimney-r0002`` (iteration-1, md5 55f0fc27) both emit
    ``DLC_resnet50_two_chimney3fbJan10shuffle1_60000``.

    Analyzing the second over the first would overwrite the file the first release's
    stamped provenance alias points at, so that alias would silently start describing
    a different model's output -- exactly what the stamp exists to prevent.

    A per-release directory makes that structurally impossible rather than merely
    refused, and it does so without touching the scorer string: the filename is
    unchanged, so it still matches what production has historically produced, and
    the plain ``<stem>.dlc.h5`` alias stays directly in ``sorted/`` where every
    downstream loader already looks.  Project-mode output also keeps writing straight
    into ``sorted/``, so the 552 existing files keep their layout.
    """
    if not release_id:
        return dest_dir
    out = os.path.join(dest_dir, release_id)
    if create and not os.path.isdir(out):
        os.makedirs(out, exist_ok=True)
    return out
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the gDLCFit recording block
# gated only on release_id, so the GUI's "analyze to temp" -- which passes a destfolder
# under tmp/dlc/ and, since Task 7, writes aliases there too -- would record a row whose
# alias_path points into scratch | date: 2026-08-14]
DURABLE_DEST_DIRNAME = 'sorted'

# Placeholder for "no explicit config" in the POSITIONAL config slot. See the use site in
# __main__: an empty argument cannot survive the queue's command-string round trip.
NO_CONFIG_SENTINEL = 'NOCONFIG'


def is_durable_dest(dest_dir):
    """True when analysis output at ``dest_dir`` is somewhere permanent.

    Only ``.../<animal>/<penname>/sorted`` counts.  A gDLCFit row asserts that a named
    file records which network produced a session's tracking, and the table is
    append-only by design -- so a row may only be written for output that is not
    expected to disappear.  Scratch previews (``tmp/dlc/<session>/``) are deletable at
    any time and nothing would ever retract the row, so they are recorded by nobody:
    not here, and not by the sweeper, whose glob is ``*/sorted/*.dlc.h5``.

    Vetting a preview and wanting it kept is a real workflow -- that is the ``promote``
    action, which copies it into ``sorted/`` and records it from there.
    """
    if not dest_dir:
        return False
    return (os.path.basename(os.path.normpath(str(dest_dir)))
            == DURABLE_DEST_DIRNAME)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: this rename only ran when no
# destfolder was given, so every "analyse to temp" preview was named after the SLOT
# (recordingDLC_...h5, identical for every session in the lab) instead of the session. Two
# consequences: the GUI's sorted listing filters on the session stem and would never show
# such a file, and publishable_preview looks up <session>.<release>.dlc.h5, so the
# publish-from-temp reuse could never fire on a preview it had just made.
# | date: 2026-08-17]
# psi slot -> suffix on the session-named symlink. '' means the ceiling view, whose link is
# just <session>.avi; baphy_io.load_dlc_trace reads the suffix back to pick the PSIVIDEO
# stream that aligns the trace, so these strings are an interface, not cosmetics.
_SLOT_LINK_SUFFIX = {'recording.avi': '',
                     'video_side.avi': '_camB',
                     'video_side2.avi': '_camC',
                     'video_pupil.avi': '_pupil'}


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: share read-only prediction resolution between GUI preflight and labeled-video jobs | date: 2026-09-10]
def analysis_video_stem(video_file):
    """The worker's session/camera name, without creating the video symlink."""
    vp = Path(video_file)
    if vp.name in _SLOT_LINK_SUFFIX:
        return vp.parent.name + _SLOT_LINK_SUFFIX[vp.name]
    return vp.stem


def labeled_video_prediction(dest_dir, video_file, output_suffix, release_id=None):
    """Find the exact unfiltered prediction and metadata DLC needs to label a video.

    Never glob across releases: different weights can emit the same scorer name.
    The caller applies camera overrides before calling this read-only helper.
    """
    folder = Path(release_output_dir(str(dest_dir), release_id, create=False))
    prediction = folder / (analysis_video_stem(video_file) + output_suffix + '.h5')
    if not prediction.is_file():
        raise FileNotFoundError(
            f"No prediction for the selected model ({release_id or 'project'}) at "
            f"{prediction}. Run 'Preview → temp' with this model first.")
    metadata = prediction.with_name(prediction.stem + '_meta.pickle')
    if not metadata.is_file():
        raise FileNotFoundError(f"Prediction exists, but DLC video metadata is missing: {metadata}")
    return prediction


def create_labeled_videos(dlc, config_path, video_paths, dest_dir, release_id=None):
    """Render beside each prediction, including the release-specific subdirectory."""
    suffix = (release_output_suffix(release_id) if release_id
              else dlc_alias.dlc_scorer_suffix(config_path))
    predictions = [labeled_video_prediction(dest_dir, video, suffix, release_id)
                   for video in video_paths]
    for video, prediction in zip(video_paths, predictions):
        log.info('Creating labeled video using %s', prediction)
        result = dlc.create_labeled_video(
            config_path, [video], videotype='avi', destfolder=str(prediction.parent))
        if result is not None and (not result or not all(result)):
            raise RuntimeError(f'DeepLabCut could not create the labeled video for {video}')
# [AGENT EDIT END]


def session_named_video(video_file):
    """``<session>/<slot>.avi`` -> a ``<session><suffix>.avi`` symlink beside the session.

    DLC names its output after the video file it was handed, so analysing ``recording.avi``
    in place yields ``recordingDLC_<scorer>.h5`` -- a name that identifies the slot but not
    the session, and that every session in the lab would share.  Everything downstream
    (alias stems, ``publishable_preview``, the GUI's per-session listings) keys on the
    session, so the video is linked under the session's name before analysis.

    Idempotent: an existing link is reused.  A path that is not a psi slot is returned
    unchanged, and must exist -- callers pass parent-level ``<session>.avi`` links too.
    """
    base = os.path.basename(video_file)
    if base not in _SLOT_LINK_SUFFIX:
        if not os.path.exists(video_file):
            raise ValueError(f"video_file {video_file} not found")
        return video_file
    linked = os.path.dirname(video_file) + _SLOT_LINK_SUFFIX[base] + '.avi'
    if not os.path.exists(linked):
        os.symlink(video_file, linked)
    return linked
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the raw-file lookup for a job's
# own gDLCFit row lives here, not in dlc_fit_record -- that module is the only code allowed to
# touch gDLCFit and is guarded against naming any other table | date: 2026-08-12]
def rawid_for_stem(session_stem):
    """``(rawid, siteid)`` for a session stem, or ``(None, None)``.

    ``gDataRaw.parmfile`` carries a trailing ``.m`` on older rows, so both spellings are
    tried. A video with no row (training sessions, scratch recordings) is legitimately
    un-joinable and yields ``(None, None)``; a rawid is never invented.
    """
    # %s with a tuple -- pd_query's documented style. It binds at the DRIVER level
    # (_exec_driver_sql_to_df), so pymysql's %s applies and named :params are passed
    # through to MySQL verbatim as a syntax error.
    got = nd.pd_query("SELECT id, cellid FROM gDataRaw "
                      "WHERE parmfile IN (%s, %s) LIMIT 1",
                      (session_stem, session_stem + '.m'))
    if not len(got):
        return None, None
    return int(got['id'].iloc[0]), got['cellid'].iloc[0]
# [AGENT EDIT END]


def alias_paths(dest_dir, video_stem, release_id=None):
    """``(stamped, plain)`` alias paths for one video.  ``stamped`` is None without
    a release."""
    stamped = (os.path.join(dest_dir, dlc_alias.stamped_alias_name(video_stem, release_id))
               if release_id else None)
    plain = os.path.join(dest_dir, video_stem + '.dlc.h5')
    return stamped, plain


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 1 -- the collision
# guard was only ever called from the release branch of the 'fit' block, so a project-mode
# or explicit-config run (which writes the SAME scorer-named .h5, since the project is what
# the release was cut from) could silently overwrite the file an existing stamped alias
# points at. Run the check on every path, with release_id=None standing in for "the run
# happening now is unstamped" | date: 2026-08-06]
def assert_no_stamped_collision(dest_dir, video_stem, release_id, output_suffix,
                                alias_stem=None):
    """Refuse when an *existing* stamped alias points at the file this run is
    about to write, unless that stamped alias belongs to the release now running.

    ``release_id`` is the release behind *this* run, or ``None`` for a project or
    explicit-config run.  Two distinct corruptions are caught by the same check:

    * **Two releases sharing an ``output_suffix``.**  The suffix encodes task,
      date, shuffle and snapshot *number*, not the release.
      ``left_chimney_headphones-r0001`` and ``-r0003`` both spell
      ``DLC_resnet50_headphones_left_chimney_LB1_pokesApr22shuffle1_60000``, as do
      ``two_chimney-r0001`` and ``-r0002``.  Analyzing with ``-r0003`` after
      ``-r0001`` was stamped would overwrite the .h5 ``-r0001``'s alias points at.
    * **A project-mode or explicit-config run reaches the same file a release
      already stamped.**  A release is cut *from* its project, so Task/date/
      shuffle/snapshot -- and therefore ``output_suffix`` -- usually match
      exactly.  A session stamped ``two_chimney-r0001``, re-analyzed with
      ``allow_project=1`` or an explicit project config, silently overwrites the
      file ``<stem>.two_chimney-r0001.dlc.h5`` points at, and that alias then
      describes a different (untracked) model's output.  This run has no
      ``release_id`` of its own to compare against, so *any* pre-existing stamped
      alias resolving to the target is refused.

    Neither promoted release collides with another promoted one on the current
    registry, so the first case never fires today; it exists so that promoting a
    colliding release later does not quietly corrupt the provenance of
    already-analyzed sessions. The second case fires on the current registry any
    time a stamped session is re-run in project/explicit mode.

    ``alias_stem`` is the stem the ALIASES are named after, which differs from
    ``video_stem`` only under a registry camera override (see ``write_aliases``).
    DeepLabCut names its output after the video it analysed, so the target filename
    uses ``video_stem``; the stamped aliases that might already point at it are named
    after the session, so the scan uses ``alias_stem``.
    """
    if not os.path.isdir(dest_dir):
        return
    target = os.path.realpath(os.path.join(dest_dir, video_stem + output_suffix + '.h5'))
    prefix = (alias_stem or video_stem) + '.'
    tail = '.dlc.h5'
    for name in sorted(os.listdir(dest_dir)):
        if not (name.startswith(prefix) and name.endswith(tail)):
            continue
        other = name[len(prefix):-len(tail)]
        if not other or other == release_id:
            continue
        link = os.path.join(dest_dir, name)
        if not os.path.islink(link):
            continue
        if os.path.realpath(link) == target:
            subject = ('release %s' % release_id) if release_id else 'this unstamped run'
            raise RuntimeError(
                'refusing to analyze %s here with %s: release %s already '
                'stamped %s, and this run would produce the same output filename '
                '(%s). Analyzing would overwrite that file and leave %s pointing at '
                'output it did not produce. Analyze into a different destfolder, or '
                'move the existing %s and its target aside first.'
                % (video_stem, subject, other, name, output_suffix, name, name))
# [AGENT EDIT END]


def write_aliases(target, dest_dir, video_stem, release_id=None, relative=True,
                  alias_stem=None):
    """Point both aliases at ``target``; return the paths written, stamped first.

    Two links, two jobs:

    * ``<stem>.<release_id>.dlc.h5`` -- durable provenance.  Its name carries the
      release, and its target lives in that release's own output directory (see
      :func:`release_output_dir`), so a later run of a *different* release writes
      into a different directory and can never overwrite this one.  Re-running the
      same release just repoints it at the same output.  The scorer string alone
      would NOT be enough: it omits the iteration, so two releases of one project
      can produce identical filenames from different weights.
    * ``<stem>.dlc.h5`` -- the "current" pointer every downstream loader opens,
      repointed atomically by ``dlc_alias.point_alias``.

    With no release (project or explicit-config mode) only the plain alias is
    written, and that is logged: those outputs carry no provenance stamp because
    there is no immutable thing to stamp them with.

    ``alias_stem`` names the aliases when that differs from the analysed video's own
    stem -- only under a registry camera override, where the ceiling view sits on a
    side-camera slot.  There the analysed stem is ``<session>_camC`` but the aliases
    must be ``<session>[.<release>].dlc.h5``, because that is the name every loader
    asks for.  The target keeps its camC name so ``baphy_io.load_dlc_trace`` can
    resolve the alias and align against that camera's own frame counts.
    """
    stamped, plain = alias_paths(dest_dir, alias_stem or video_stem, release_id)
    written = []
    if stamped is not None:
        dlc_alias.point_alias(target, stamped, relative=relative)
        written.append(stamped)
        log.info('provenance alias %s -> %s', os.path.basename(stamped),
                 os.path.relpath(target, dest_dir))
    else:
        log.warning('no release for %s: writing only the plain .dlc.h5 alias, so '
                    'these results carry NO model provenance stamp', video_stem)
    dlc_alias.point_alias(target, plain, relative=relative)
    written.append(plain)
    log.info('current alias %s -> %s', os.path.basename(plain),
             os.path.relpath(target, dest_dir))
    return written


# [AGENT EDIT START | agent: claude | user: wingertj | reason: vetting a preview and then
# re-running the same release on the GPU to get it into sorted/ recomputes bytes we already
# have; the preview only becomes publishable because it is stamped, which is what lets us
# prove it came from the release currently deployed | date: 2026-08-14]
def publishable_preview(src_dir, alias_stem, release_id):
    """Path of the stamped preview in ``src_dir`` for ``release_id``, or ``None``.

    A preview is publishable only when it carries the release stamp AND that stamp is
    the release being asked for.  Both halves matter:

    * **Unstamped** output (everything written before aliases were extended to explicit
      destfolders, and every project-mode run) has no record of which network made it.
      Copying it into ``sorted/`` would manufacture provenance that was never observed.
    * A stamp for a **different** release is a preview of a model that is no longer
      deployed.  Publishing it would put output in production under a name asserting the
      current release produced it -- the precise substitution this whole project exists
      to make impossible.

    Returns the resolved target (the real ``.h5``), not the alias.
    """
    if not src_dir or not release_id or not os.path.isdir(str(src_dir)):
        return None
    stamped, _plain = alias_paths(str(src_dir), alias_stem, release_id)
    if not stamped or not os.path.islink(stamped):
        return None
    target = os.path.realpath(stamped)
    if not os.path.isfile(target):
        return None                                           # dangling: scratch was cleaned
    return target


# [AGENT EDIT START | agent: Codex | user: wingertj | reason: pin explicitly inspected publications across queue delay and alias changes | date: 2026-09-10]
def publication_token(path, release_id):
    """Argv-safe fingerprint of the selected release, real path and file revision."""
    import hashlib
    import json
    real = os.path.realpath(path)
    stat = os.stat(real)
    identity = [release_id, real, stat.st_mtime_ns, stat.st_size]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def check_publication_token(path, release_id, expected):
    if expected and (not path or publication_token(path, release_id) != expected):
        raise ValueError('inspected fit or deployed release changed while queued; refresh and inspect again')
# [AGENT EDIT END]


def publish_preview(src_target, dest_dir, video_stem, release_id, alias_stem=None):
    """Copy a vetted preview into ``dest_dir`` and alias it there; return (target, written).

    The copy is ``copy2``, so the target keeps the mtime of the analysis that produced
    it rather than of the copy.  That matters beyond tidiness: ``gDLCFit``'s row key is
    ``sha256(alias_path, target_realpath, mtime)``, so preserving mtime keeps the row
    describing *when the pose was computed*.

    Same guards as a fresh fit, in the same order, because the destination cannot tell
    the difference between output that was analysed here and output that was copied here.
    """
    stem = alias_stem or video_stem
    suffix = os.path.basename(src_target)[len(video_stem):-len('.h5')]

    assert_no_stamped_collision(dest_dir, video_stem, release_id, suffix,
                                alias_stem=alias_stem)
    assert_aliases_writable(dest_dir, video_stem, release_id=release_id,
                            alias_stem=alias_stem)

    out_dir = release_output_dir(dest_dir, release_id, create=True)
    dest_target = os.path.join(out_dir, os.path.basename(src_target))
    if os.path.realpath(dest_target) == os.path.realpath(src_target):
        raise RuntimeError('refusing to publish %s onto itself' % src_target)
    _shutil.copy2(src_target, dest_target)
    log.info('published %s -> %s', src_target, dest_target)

    # the meta pickle and the scale sidecar are part of the record, not decoration: the
    # sidecar is the ONLY evidence that an oversized video was analysed on a rescaled copy
    for extra_src, extra_dst in _publish_companions(src_target, dest_target, stem,
                                                    release_id):
        if os.path.isfile(extra_src) and not os.path.exists(extra_dst):
            _shutil.copy2(extra_src, extra_dst)
            log.info('published companion %s', os.path.basename(extra_dst))

    written = write_aliases(dest_target, dest_dir, video_stem, release_id=release_id,
                            alias_stem=alias_stem)
    return dest_target, written


def _publish_companions(src_target, dest_target, alias_stem, release_id):
    """(src, dst) pairs for the files that travel with a published .h5."""
    pairs = [(src_target[:-len('.h5')] + '_meta.pickle',
              dest_target[:-len('.h5')] + '_meta.pickle')]
    sidecar = '%s.%s.scale.json' % (alias_stem, release_id)
    pairs.append((os.path.join(os.path.dirname(os.path.dirname(src_target)), sidecar),
                  os.path.join(os.path.dirname(os.path.dirname(dest_target)), sidecar)))
    return pairs
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 8 -- write_aliases
# can raise FileExistsError on the PLAIN alias after the stamped alias was already written
# and after analyze_videos (the expensive GPU work) completed, leaving partial state:
# a repointed stamped alias, no plain alias, and GPU time spent. Check both alias paths
# up front, before analyze_videos runs, so a real file sitting at either path is a cheap
# refusal instead | date: 2026-08-06]
def assert_aliases_writable(dest_dir, video_stem, release_id=None, alias_stem=None):
    """Refuse up front if either alias path is a real file ``write_aliases``
    would later refuse to clobber.

    ``write_aliases`` (via ``dlc_alias.point_alias``) already never overwrites a
    non-symlink regular file, but it checks one alias at a time, and only after
    ``analyze_videos`` has produced the .h5 it would point at. Calling this
    *before* ``analyze_videos`` turns a post-hoc, partial-state failure into a
    cheap one: nothing has been analyzed yet, so there is nothing to clean up.

    ``alias_stem`` as in ``write_aliases``: the paths checked here must be exactly
    the paths that will later be written, or the check guards the wrong names.
    """
    stamped, plain = alias_paths(dest_dir, alias_stem or video_stem, release_id)
    for path in (stamped, plain):
        if path is None:
            continue
        if os.path.lexists(path) and not os.path.islink(path):
            raise FileExistsError(
                'refusing to analyze %s: %s already exists and is not a symlink '
                '(it looks like real data, not an alias). Move it aside before '
                'analyzing so this run does not need to clobber it after '
                'spending GPU time producing results.' % (video_stem, path))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: resolution-based inference
# rescaling -- pick the training frame size from the alias that actually ran | date: 2026-08-08]
def resolved_train_frame_size(resolved, registry=None):
    """``(width, height)`` of the frames the resolved model was labelled on.

    Read from the registry entry for ``resolved['alias']``.  Falls back to
    ``dlc_registry.DEFAULT_TRAIN_FRAME_SIZE`` when the run named a config path
    that belongs to no alias -- every DLC project in this lab that analyzes arena
    video is 640x480, so the default is the measured value rather than a guess,
    and it is also the conservative direction: it only ever triggers a downscale
    of a video that is *larger* than 640 wide, never an upscale.
    """
    alias = resolved.get('alias')
    if alias:
        try:
            return dlc_registry.resolve(alias, registry)['train_frame_size']
        except (KeyError, ValueError, OSError):
            log.warning('alias %r is not in the registry; using the default '
                        'training frame size %s to decide the analysis scale',
                        alias, dlc_registry.DEFAULT_TRAIN_FRAME_SIZE)
    return dlc_registry.DEFAULT_TRAIN_FRAME_SIZE
# [AGENT EDIT END]


def training_hyperparams(registry=None):
    """``[training_defaults]`` from the registry, with every key required.

    Replaces the literals that used to sit in the ``training`` branch
    (``net_type='resnet_50'``, ``maxiters=60000`` ...).  Missing keys raise rather
    than falling back, so the registry and the code cannot disagree about what a
    model was trained with.
    """
    if registry is None:
        registry = dlc_registry.load_registry()
    defaults = registry.get('training_defaults', {})
    missing = [k for k in _REQUIRED_TRAINING_KEYS if k not in defaults]
    if missing:
        raise KeyError(
            '[training_defaults] in %s is missing %s'
            % (registry.get('_path', dlc_registry.DEFAULT_REGISTRY_PATH),
               ', '.join(missing)))
    return dict(defaults)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: run length is a per-run
# decision (an A/B arm may need more iterations than the default), but there was no way to
# set it short of editing the git-tracked registry -- which would silently restate what
# every OTHER model was trained with | date: 2026-08-15]
OVERRIDABLE_TRAINING_KEYS = ('maxiters', 'saveiters', 'displayiters')


def apply_training_overrides(hp, extra_kwargs):
    """Apply ``maxiters=``/``saveiters=``/``displayiters=`` from the command line.

    Deliberately narrow: only run-length knobs, never ``net_type`` or
    ``augmenter_type``, which define what the model *is* and must keep coming from the
    registry alone.  Overrides are returned in ``hp``, so they land in TRAINRUN.toml
    with everything else -- the registry stays the default, and what a given run
    actually used is still recorded next to its weights.
    """
    hp = dict(hp)
    applied = {}
    for key in OVERRIDABLE_TRAINING_KEYS:
        raw = extra_kwargs.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError('%s= must be an integer, got %r' % (key, raw))
        if value <= 0:
            raise ValueError('%s= must be positive, got %d' % (key, value))
        applied[key] = value
        hp[key] = value
    if applied:
        log.warning('training hyperparameters overridden for this run: %s '
                    '(registry defaults unchanged; TRAINRUN.toml records what ran)',
                    ', '.join('%s=%d' % kv for kv in sorted(applied.items())))
        hp['_overridden'] = sorted(applied)
    return hp
# [AGENT EDIT END]


def release_init_weights(release_id, expect_net_type=None):
    """Path (no extension) of a release's snapshot, for transfer learning.

    Refuses when the release's ``net_type`` differs from ``expect_net_type``:
    initialising a resnet_50 from a mobilenet checkpoint fails deep inside
    TensorFlow with a shape error that says nothing about why.
    """
    manifest = dlc_release.read_manifest(release_id)
    net_type = manifest.get('model', {}).get('net_type')
    if expect_net_type is not None and net_type != expect_net_type:
        raise RuntimeError(
            'from_release=%s was trained as net_type=%r but this run is configured '
            'for net_type=%r; the checkpoint cannot initialise a different network.'
            % (release_id, net_type, expect_net_type))
    snapshot = manifest['source']['snapshot']
    shuffle = manifest['source'].get('shuffle', 1)
    rdir = os.path.join(dlc_release.RELEASE_ROOT, release_id)
    train_dir, _model_dir, _iteration = dlc_release.find_snapshot(
        rdir, snapshot, iteration=0, shuffle=shuffle)
    return os.path.join(train_dir, 'snapshot-%d' % snapshot)


def _toml_value(value):
    """Minimal TOML scalar/array rendering for TRAINRUN.toml.

    Hand-rolled rather than via ``tomli_w`` because TRAINRUN.toml is written from
    the production DLC environment, which is not guaranteed to have it, and a
    missing writer must not be the reason a training run's provenance goes
    unrecorded.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[%s]' % ', '.join(_toml_value(v) for v in value)
    text = str(value)
    return '"%s"' % text.replace('\\', '\\\\').replace('"', '\\"')


def render_trainrun(sections):
    """``sections`` (an ordered list of ``(name, dict)``) as TOML text."""
    lines = []
    for name, table in sections:
        lines.append('[%s]' % name)
        for key in sorted(table):
            lines.append('%s = %s' % (key, _toml_value(table[key])))
        lines.append('')
    return '\n'.join(lines)


def write_trainrun(run_dir, sections):
    """Write ``TRAINRUN.toml`` into ``run_dir``; return its path."""
    path = os.path.join(run_dir, TRAINRUN_NAME)
    tmp = path + '.tmp-%d' % os.getpid()
    with open(tmp, 'w') as f:
        f.write(render_trainrun(sections))
    os.replace(tmp, path)
    log.info('recorded training run provenance in %s', path)
    return path


def project_iteration(config_path):
    """The ``iteration`` a DLC project config currently points at.

    Read with plain yaml, never ``auxiliaryfunctions.read_config``, which rewrites
    ``project_path`` and writes the file back.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return int(cfg['iteration'])
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- both the
# 'curate' and the 'training' action have to refuse a frozen release, and both want to
# name the writable project the caller probably meant | date: 2026-08-06]
def alias_project_hint(resolved):
    """The mutable training project behind a resolved model, or ``None``.

    ``resolved['project']`` is ``None`` when a release was reached via an explicit
    ``RELEASE_ROOT`` config path, so fall back to the registry entry for the alias
    the release manifest names.  Used only to make a refusal actionable.
    """
    hint = resolved.get('project')
    if hint:
        return hint
    if resolved.get('alias'):
        try:
            return dlc_registry.resolve(resolved['alias'])['project']
        except (KeyError, RuntimeError, FileNotFoundError):
            return None
    return None


# [AGENT EDIT START | agent: claude | user: wingertj | reason: extract/refine run as the
# QUEUE account (lbhb) and create labeled-data/<session>/ with the default 0755, so the
# LABELLER (a real user) cannot write the CollectedData file into it and napari dies with
# PermissionError. It has bitten three times; only root can fix it afterwards, because
# renaming or chmod-ing a directory needs write access ON that directory. Every folder a
# job creates under the project is opened up as it is created instead.
# | date: 2026-08-23]
def open_up_project_paths(project, subdirs=('labeled-data',)):
    """chmod a+rwX everything the job just created under `project`, best effort."""
    for sub in subdirs:
        root = os.path.join(project, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for p, mode in ([(dirpath, 0o777)]
                            + [(os.path.join(dirpath, f), 0o666) for f in filenames]):
                try:
                    os.chmod(p, mode)
                except OSError:
                    pass          # someone else's file: leave it, never fail the job
    log.info('opened permissions under %s/%s', project, ','.join(subdirs))
# [AGENT EDIT END]


def assert_not_a_release(resolved, action):
    """Refuse to mutate a frozen release with ``action``.

    A release is 0444/0555 precisely so that neither training nor curation can
    reach the weights and labels production is using.  Both writing actions land
    here rather than failing later with a permission error that says nothing
    about why.
    """
    if resolved.get('source') != 'release':
        return
    hint = alias_project_hint(resolved)
    raise RuntimeError(
        "refusing to run '%s' against release %s: a release is immutable (0444 files, "
        "0555 directories) and exists precisely so that neither training nor curation "
        "can reach the model production uses. Work in the alias's project %s instead "
        "-- pass allow_project=1, or give the project's config.yaml explicitly as "
        "argv[3]."
        % (action, resolved.get('release_id'),
           hint or '(unknown -- pass alias=<name> to identify it)'))
# [AGENT EDIT END]


try:
    import nems_lbhb.db as n
    db_exists = True
except Exception as e:
    # If there's an error import nems0.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems0.db, can't update tQueue")
    print(e)
    db_exists = False

if __name__ == '__main__':
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- DeepLabCut
    # imported here, not at module scope, so the resolution/alias helpers above stay
    # importable (and unit-testable) without DLC | date: 2026-08-06]
    import deeplabcut as dlc
    # [AGENT EDIT END]

    if 'QUEUEID' in os.environ:
        import nems_lbhb.utils
        queueid = int(os.environ['QUEUEID'])
        nems_lbhb.utils.progress_fun = nd.update_job_tick
    else:
        queueid = 0

    if db_exists & queueid>0:
        log.info("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)
    log.info(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

    # figure out filenames and paths to process
    video_file = sys.argv[1]
    create_labeled_video = False

    # figure out action(s) to perform
    if len(sys.argv) > 2:
        action_list = sys.argv[2].split(",")
    else:
        #action_list = ['fit','summary']
        action_list = ['fit']

    # Parse any key=value arguments (e.g., init_weights=/path/to/snapshot)
    extra_kwargs = parse_key_value_args(sys.argv)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- resolve the
    # model through the registry and verify the release before analyzing anything
    # | date: 2026-08-06]
    # An argv[3] that is really a key=value argument is not a config path.
    explicit_config = sys.argv[3] if len(sys.argv) > 3 else ''
    if '=' in explicit_config:
        explicit_config = ''
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the config slot is
    # POSITIONAL, but a caller with no config (the production alias route) cannot pass an
    # empty one: db.add_job_to_queue joins args into a single shell string, and the queue
    # daemon then strips both bare and quoted empties, so the destfolder slid into argv[3]
    # and the job tried to read its own output directory as a DLC config. NOCONFIG is a
    # token that survives whitespace and quote normalisation. | date: 2026-08-17]
    if explicit_config == NO_CONFIG_SENTINEL:
        explicit_config = ''
    # [AGENT EDIT END]

    resolved = resolve_model(
        video_file,
        explicit_config=explicit_config,
        alias=extra_kwargs.get('alias'),
        allow_project=truthy(extra_kwargs.get('allow_project')),
    )
    path_config = resolved['config']
    release_id = resolved['release_id']

    log.info(f"DLC model: {path_config}")
    log.info(f"  resolved by: {resolved['how']}")
    log.info(f"  alias={resolved['alias']} source={resolved['source']} "
             f"release_id={release_id}")

    verify_resolved_release(resolved)
    if release_id is not None:
        log.info(f"  release output_suffix: {release_output_suffix(release_id)}")

    if not os.path.exists(path_config):
        raise FileNotFoundError(f"DLC config not found: {path_config}")
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: the sampled
    # generalization evaluation ran only as a paste-into-a-GPU-shell recipe; as a queue
    # action it uses the same wrapper, GPU pool and logging as every other DLC job.
    # Placed HERE, after resolve_model/verify: the candidate arrives as the explicit
    # config (or alias=), so a release candidate is checksum-verified before a single
    # frame is scored. The unit of work is the ALIAS's whole set -- argv[1] is unused --
    # so this exits before any of the per-video destination machinery below.
    # | date: 2026-08-18]
    if 'geneval' in action_list:
        from nems_lbhb.motor import dlc_generalization
        gen_alias = extra_kwargs.get('gen_alias') or resolved.get('alias')
        if not gen_alias:
            raise ValueError("geneval needs gen_alias=<alias> (or alias=) to know "
                             "which generalization set to run")
        n_frames = int(extra_kwargs.get('n_frames')
                       or dlc_generalization.DEFAULT_SAMPLE_FRAMES)
        log.info("geneval: alias=%s candidate=%s n_frames=%d",
                 gen_alias, path_config, n_frames)
        rows = dlc_generalization.evaluate_candidate(gen_alias, path_config,
                                                     n_frames=n_frames)
        for r in rows:
            log.info("  %-22s ratio_mean=%s rms_spread=%s track_mean=%s",
                     r['session'], r['ratio_mean'], r['rms_spread_mean'],
                     r['track_mean'])
        log.info("geneval complete: %d video(s) -> %s/results_sampled.csv",
                 len(rows), dlc_generalization.gen_dir(gen_alias))
        if db_exists & (queueid > 0):
            nd.update_job_complete(queueid)
        sys.exit(0)
    # [AGENT EDIT END]

    init_weights = extra_kwargs.get('init_weights', None)
    if init_weights:
        log.info(f"Transfer learning: will use init_weights={init_weights}")

    if len(sys.argv) > 4:
        dest_path = sys.argv[4]
        if dest_path == 'training':
            pass
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- argv[4]
        # may now be any key=value arg (alias=, allow_project=, resume=, ...), not just
        # init_weights=; the old startswith('init_weights=') test would have treated
        # those as a destination directory and tried to mkdir them | date: 2026-08-06]
        elif '=' in dest_path:
            # This is a key=value arg, not a dest_path
            dest_path = None
        # [AGENT EDIT END]
        else:
            try:
                # Set umask to 0 so makedirs creates with full permissions
                old_umask = os.umask(0o000)
                os.makedirs(dest_path, mode=0o777, exist_ok=True)
                os.makedirs(Path(dest_path).parent, mode=0o777, exist_ok=True)
                os.umask(old_umask)  # Restore original umask
                # Try to chmod in case directory already existed
                os.chmod(dest_path, 0o777)
                os.chmod(Path(dest_path).parents[0], 0o777)
            except:
                log.info(f"chmod error, but maybe ok?")
            log.info(f'Results destination provided: saving results to {dest_path}')
    else:
        dest_path = None

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: honour per-session camera
    # overrides so a re-run cannot silently go back to the wrong camera | date: 2026-08-12]
    # psi names video files after the camera SLOT, not the content. When a USB port
    # re-enumerates the slots swap, and the ceiling view ends up in video_side/video_side2
    # while recording.avi holds something else. That is not derivable from the files, so
    # the registry asserts it per session; see [camera_override] in dlc_registry.toml.
    #
    # Redirecting here (before the symlink block below) means the normal machinery does the
    # rest: video_side2.avi is linked as <stem>_camC.avi, DLC writes a camC-tagged output,
    # and baphy_io.load_dlc_trace reads that tag to align against PSIVIDEO_camC -- this
    # session's OWN frame counts -- instead of the base camera's.
    alias_stem_override = None
    try:
        _cam_over = dlc_registry.camera_override(video_file)
    except Exception as e:
        log.warning(f"camera_override lookup failed for {video_file}: {e}")
        _cam_over = None
    if _cam_over:
        _sess = os.path.dirname(video_file)
        if not os.path.basename(video_file).startswith(('recording', 'video_')):
            _sess = os.path.splitext(video_file)[0]      # given as <stem>.avi
        _want = os.path.join(_sess, _cam_over['video'])
        if not os.path.exists(_want):
            raise ValueError(
                f"registry camera_override says this session's ceiling view is "
                f"{_cam_over['video']}, but {_want} does not exist")
        log.warning("CAMERA OVERRIDE for this session: analysing %s instead of %s "
                    "(cam_tag=%s, verified by %s on %s). Reason: %s",
                    _cam_over['video'], os.path.basename(video_file),
                    _cam_over.get('cam_tag'), _cam_over.get('verified_by'),
                    _cam_over.get('verified_date'), (_cam_over.get('note') or '').strip()[:160])
        video_file = _want
        # Name the ALIASES after the session, not after the camera we ended up
        # analysing.  The redirect above sends the rest of this script down the
        # video_side2 path, so the analysed stem becomes '<session>_camC' and the
        # aliases would land at '<session>_camC.dlc.h5' -- but every loader asks for
        # '<session>.dlc.h5' (baphy_experiment.dlcfile), so the session would simply
        # have no tracking as far as downstream is concerned.  The TARGET keeps its
        # camC name, which is what baphy_io.load_dlc_trace resolves the alias to in
        # order to align against PSIVIDEO_camC -- this session's own frame counts.
        #
        # This is deliberately NOT derived from the filename: a genuine camC analysis
        # (tracking the side camera on purpose, e.g. for 3D) must keep its camC alias.
        # The difference is the override assertion itself, so it is carried from here.
        alias_stem_override = os.path.basename(_sess)
    # [AGENT EDIT END]

    if dest_path is None:
        if 'training' in video_file:
            new_parent = os.path.join(Path(video_file).parents[1], '_'.join(Path(video_file).parents[0].stem.split('_')[:-2]))
            os.makedirs(new_parent, exist_ok=True)
            video_base = os.path.basename(video_file)
            if video_base.startswith('recording'):
                # create softlink in parent dir
                video_path = os.path.dirname(video_file)
                old_video_file = video_file
                video_file = os.path.join(new_parent, Path(video_file).parent.stem+'.avi')
                if os.path.exists(video_file):
                    pass
                else:
                    os.symlink(old_video_file, video_file)
            elif video_base.startswith('video_side'):
                # create softlink in parent dir
                video_path = os.path.dirname(video_file)
                old_video_file = video_file
                video_file = os.path.join(new_parent, Path(video_file).parent.stem+'_camB.avi')
                if os.path.exists(video_file):
                    pass
                else:
                    os.symlink(old_video_file, video_file)
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: `~` is bitwise
            # NOT, not logical negation: ~True == -2 and ~False == -1, both truthy, so this
            # branch fired unconditionally and could raise "not found" on a video that is
            # present | date: 2026-08-12]
            elif not os.path.exists(video_file):
                raise ValueError(f"video_file {video_file} not found")
            # [AGENT EDIT END]

        else:
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: a relative
            # video path (e.g. "videos/<session>.avi", the form queue_dlc_job emits) is
            # relative to the PROJECT, not to the job's working directory -- without a
            # destfolder, 7 extract jobs died in session_named_video's existence check
            # because the daemon's cwd is elsewhere | date: 2026-08-19]
            if not os.path.isabs(video_file) and not os.path.exists(video_file) \
                    and path_config:
                candidate = os.path.join(os.path.dirname(path_config), video_file)
                if os.path.exists(candidate):
                    video_file = candidate
            # [AGENT EDIT END]
            # the four psi slots collapse to one rule; see session_named_video
            video_file = session_named_video(video_file)

        vid_dir, base_file = os.path.split(video_file)
        animal_path, penname = os.path.split(vid_dir)
        daq_path, animal = os.path.split(animal_path)
        path_sorted = os.path.join(vid_dir, 'sorted')

        vids = [base_file]
        if not os.path.exists(path_sorted):
            os.makedirs(path_sorted)
            log.info(f'Created sorted directory in {vid_dir}')

        log.info(f'Number of videos to be analyzed: {len(vids)}')
        log.info(f'Results will be saved in: {path_sorted}')

        vid_paths = [os.path.join(vid_dir, v) for v in vids]

    elif (dest_path is not None) and (dest_path != 'training'):
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: same
        # project-relative fallback as the dest-less branch above -- a relative
        # "videos/<session>.avi" is relative to the PROJECT, not the daemon's cwd
        # | date: 2026-08-19]
        if not os.path.isabs(video_file) and not os.path.exists(video_file) \
                and path_config:
            candidate = os.path.join(os.path.dirname(path_config), video_file)
            if os.path.exists(candidate):
                video_file = candidate
        # [AGENT EDIT END]
        # name the analysed video after the SESSION here too -- see session_named_video.
        # Without this a preview into tmp/ was called recordingDLC_<scorer>.h5 while the
        # identical run into sorted/ was called <session>DLC_<scorer>.h5, so the two paths
        # disagreed about the one name everything downstream looks the output up by.
        video_file = session_named_video(video_file)
        vid_paths = [video_file]
        vid_dir, base_file = os.path.split(video_file)
        vids = [base_file]
        path_sorted = dest_path
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- aliases are
        # now written into an explicit destfolder too (they used to be skipped here, so an
        # "analyze to temp" run produced a bare scorer-named .h5 that nothing downstream
        # could open by a stable name).  The alias paths are derived in write_aliases from
        # path_sorted + the video stem, so the old `output_aliased` list is gone.
        # | date: 2026-08-06]
        # [AGENT EDIT END]
        log.info(f'Number of videos to be analyzed: {len(vids)}')
        log.info(f'Results will be saved in: {path_sorted}')

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: "analyze to sorted" after
    # vetting a preview re-ran the GPU to recompute bytes already on disk. Publishing copies
    # them instead -- legitimate only because the preview is release-stamped, which is what
    # proves it came from the release still deployed | date: 2026-08-14]
    if 'publish' in action_list:
        _stem = vids[0].replace('.avi', '')
        _alias_stem = alias_stem_override or _stem
        _src_dir = extra_kwargs.get('src') or os.path.join(
            os.path.dirname(path_sorted), 'tmp', 'dlc', _alias_stem)
        # fallback=1 makes this "publish if you can, otherwise analyze": the GUI's
        # sorted button decides which is possible when it queues, but scratch can be
        # cleaned between then and now, and a vanished preview should cost a GPU run
        # rather than a failed job someone has to notice and re-click.
        _fallback = truthy(extra_kwargs.get('fallback'))

        _src_target, _blocked = None, None
        if release_id is None:
            _blocked = ('there is no release: an unstamped preview records nothing about '
                        'which network produced it, so copying it into %s would '
                        'manufacture provenance that was never observed' % path_sorted)
        elif not is_durable_dest(path_sorted):
            _blocked = 'the destination %s is not a sorted/ directory' % path_sorted
        else:
            _src_target = publishable_preview(_src_dir, _alias_stem, release_id)
            if _src_target is None:
                _blocked = (
                    'no preview stamped %s for %s in %s -- either nothing was analyzed '
                    'there, or what is there came from a different release (or from a '
                    'project run, which carries no stamp)'
                    % (release_id, _alias_stem, _src_dir))

        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: an explicit inspected-fit publication must never substitute new bytes or fall back to inference | date: 2026-09-10]
        if extra_kwargs.get('publish_token'):
            check_publication_token(_src_target, release_id, extra_kwargs['publish_token'])
            if _blocked:
                raise ValueError('cannot publish inspected fit: %s' % _blocked)
        # [AGENT EDIT END]
        if _blocked and not _fallback:
            raise ValueError('cannot publish: %s.' % _blocked)

    if 'publish' in action_list and _blocked:
        # 'fit' is appended, not substituted, so the ordinary fit block below does the
        # work -- there is exactly one code path that analyzes and records a session
        log.warning('not publishing (%s); running a full fit instead', _blocked)
        if 'fit' not in action_list:
            action_list.append('fit')

    elif 'publish' in action_list:
        log.info('publishing vetted preview %s -> %s', _src_target, path_sorted)
        _target, _written = publish_preview(_src_target, path_sorted, _stem,
                                            release_id, alias_stem=alias_stem_override)
        try:
            from nems_lbhb.motor import dlc_fit_record, dlc_fit_qc
            _qc = dlc_fit_qc.score_release_output(_target, release_id)
            _session = dlc_alias.strip_stream_suffix(_alias_stem)
            _rawid, _siteid = rawid_for_stem(_session)
            _rec = dlc_fit_record.record_job_fit(
                alias_path=_written[0], target_path=_target, release_id=release_id,
                rawid=_rawid, siteid=_siteid, routing_alias=resolved.get('alias'),
                source_project=resolved.get('project'),
                scorer=dlc_alias.parse_scorer(_target).get('scorer'),
                snapshot=dlc_alias.parse_scorer(_target)['snapshot'],
                queueid=(queueid or None), requested_by=extra_kwargs.get('user'),
                code_git_hash=dlc_release._git_hash(
                    os.path.dirname(os.path.abspath(__file__)))[0] or None,
                qc=_qc)
            log.info('recorded gDLCFit row for published %s (release %s, rawid %s)',
                     os.path.basename(_rec['alias_path']), release_id, _rec.get('rawid'))
        except Exception as e:
            log.error('FAILED to record a gDLCFit row for published %s: %s: %s. The '
                      'published file is fine -- run scripts/dlc_sync_fits.py --go to '
                      'backfill it.', _stem, type(e).__name__, e)
    # [AGENT EDIT END]

    if 'fit' in action_list:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- suffix
        # cross-check before analyzing, then BOTH aliases (stamped provenance + current).
        # review fix 1 -- the stamped-collision guard now runs on EVERY path (release,
        # project, and explicit-config), not only when release_id is set: a project/
        # explicit-config run writes the same scorer-named .h5 a stamped alias may already
        # point at, since the release was cut from that very project. review fix 8 -- the
        # alias-writability check now runs before analyze_videos, not after, so a real file
        # sitting at either alias path is a cheap refusal instead of a post-GPU one.
        # | date: 2026-08-06]
        suffix = dlc_alias.dlc_scorer_suffix(path_config)
        if release_id is not None:
            recorded = release_output_suffix(release_id)
            if recorded != suffix:
                raise RuntimeError(
                    f"release {release_id} records output_suffix {recorded!r} but "
                    f"DeepLabCut's GetScorerName says {suffix!r} for the same config. "
                    f"One of them is wrong, so the alias would point at a file that "
                    f"either does not exist or was produced by a different model.")

        out_dir = release_output_dir(path_sorted, release_id)
        for v in vids:
            stem = v.replace('.avi', '')
            assert_no_stamped_collision(out_dir, stem, release_id, suffix,
                                        alias_stem=alias_stem_override)
            assert_aliases_writable(path_sorted, stem, release_id=release_id,
                                    alias_stem=alias_stem_override)

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: analyze at the
        # resolution the model was trained at. 13 legacy Prince sites are 1280x720 while
        # every arena project's labelled frames are 640x480; at native scale the model
        # simply fails to FIND the animal (PRN043a05: worst rigid marker 0.334 tracked at
        # 1280x720 vs 0.848 on a 640x360 copy, with p_median already 0.954 at native --
        # a scale-detection failure, not a localisation one) | date: 2026-08-08]
        import shutil as _shutil
        import tempfile as _tempfile

        train_size = resolved_train_frame_size(resolved)
        log.info(f"training frame size for this model: {train_size[0]}x{train_size[1]}")

        scale_info = {}
        analyze_paths = list(vid_paths)
        rescale_tmp = None
        try:
            for i, vp in enumerate(vid_paths):
                info = dlc_video_scale.analysis_scale(vp, train_frame_size=train_size)
                scale_info[vp] = info
                if not info['rescale']:
                    log.info('%s: %s', os.path.basename(vp), info['reason'])
                    continue
                # Loud, because the results this produces are DELIBERATELY
                # indistinguishable from a native run's once the coordinates are
                # converted back (see the sidecar below).
                log.warning('RESCALING FOR ANALYSIS -- %s: %s',
                            os.path.basename(vp), info['reason'])
                if rescale_tmp is None:
                    # A downscaled MJPEG copy of a 30-80k-frame arena video runs to
                    # ~1 GB, so honour an explicit rescale_tmp=/dir when /tmp is small.
                    rescale_tmp = _tempfile.mkdtemp(
                        prefix='dlc_rescale_',
                        dir=extra_kwargs.get('rescale_tmp') or None)
                    log.info('rescaled copies go in %s (deleted when this run ends)',
                             rescale_tmp)
                analyze_paths[i] = dlc_video_scale.rescaled_copy(
                    vp, info['analysis_size'], rescale_tmp)

            # [AGENT EDIT START | agent: claude | user: wingertj | reason: DLC 2.3.6's
            # AnalyzeVideo ends `finally: return DLCscorer`, and a return inside finally
            # SWALLOWS any in-flight exception -- so a failed save_data still prints the
            # success banner and analyze_videos returns cleanly with no .h5 written.
            # Jobs 2650040/41 each burned ~55 GPU-minutes and died downstream at
            # find_dlc_output with the real error already discarded. Wrap save_data so
            # the failure is at least LOGGED before DLC eats it. | date: 2026-08-18]
            try:
                from deeplabcut.utils import auxiliaryfunctions as _aux
                if not getattr(_aux.save_data, '_loud', False):
                    _orig_save_data = _aux.save_data

                    def _loud_save_data(*a, **k):
                        try:
                            return _orig_save_data(*a, **k)
                        except BaseException as e:
                            log.error('DLC save_data FAILED (DLC will swallow this '
                                      'exception -- see the finally/return bug in '
                                      'predict_videos.AnalyzeVideo): %s: %s',
                                      type(e).__name__, e, exc_info=True)
                            raise
                    _loud_save_data._loud = True
                    _aux.save_data = _loud_save_data
            except Exception:                                 # noqa: BLE001
                log.warning('could not instrument DLC save_data', exc_info=True)
            # [AGENT EDIT END]
            dlc.analyze_videos(path_config, analyze_paths, videotype='avi',
                               destfolder=out_dir)

            for v, vp in zip(vids, vid_paths):
                stem = v.replace('.avi', '')
                target = dlc_alias.find_dlc_output(out_dir, stem, suffix)
                info = scale_info[vp]
                if info['rescale']:
                    # Put the coordinates back in NATIVE pixels before anything
                    # downstream (arena ppm, head angle, HRTF) sees them.  This has to
                    # happen before the aliases are written, so no reader can ever
                    # observe the analysis-resolution numbers through a stable name.
                    n_rows = dlc_video_scale.rescale_coordinates(target, info['scale'])
                    log.warning('%s: converted %d rows of pose back to native %dx%d '
                                'pixels (x %.6f)', stem, n_rows, info['native'][0],
                                info['native'][1], 1.0 / info['scale'])
                sidecar = dlc_video_scale.write_scale_sidecar(
                    path_sorted, stem, info, release_id=release_id,
                    extra={'video': vp, 'config': path_config,
                           'alias': resolved['alias'] or '',
                           'train_frame_size': list(train_size)})
                if info['rescale']:
                    log.warning('provenance: %s records that this analysis ran on a '
                                'rescaled copy', os.path.basename(sidecar))
                written = write_aliases(target, path_sorted, stem, release_id=release_id,
                                        alias_stem=alias_stem_override)

                # [AGENT EDIT START | agent: claude | user: wingertj | reason: nothing wired
                # recording into the fit job, so gDLCFit only ever gained a row when someone
                # remembered to run the sweeper -- it was a periodic snapshot of the
                # filesystem, not a log of analyses | date: 2026-08-12]
                # Recorded AFTER the aliases exist, so the row always describes an output
                # reachable by a stable name, and only for a release run: a project or
                # explicit-config fit has no stamped alias to key the row on, and its
                # provenance is exactly what nothing can vouch for. Those stay the sweeper's
                # to infer.
                #
                # A DB failure must never fail an analysis that already succeeded on disk --
                # the .h5 and its aliases are the real product, and the sweeper will pick up
                # anything missed here. So this is caught, loudly, and never re-raised.
                if release_id is not None and is_durable_dest(path_sorted):
                    try:
                        from nems_lbhb.motor import dlc_fit_record, dlc_fit_qc
                        stamped_alias = written[0]
                        qc_doc = dlc_fit_qc.score_release_output(target, release_id)
                        if qc_doc and qc_doc.get('error'):
                            log.warning('QC scoring failed for %s: %s -- recording the row '
                                        'with the error, not dropping it',
                                        stem, qc_doc['error'])
                        _session = dlc_alias.strip_stream_suffix(
                            alias_stem_override or stem)
                        _rawid, _siteid = rawid_for_stem(_session)
                        rec = dlc_fit_record.record_job_fit(
                            alias_path=stamped_alias, target_path=target,
                            release_id=release_id, rawid=_rawid, siteid=_siteid,
                            routing_alias=resolved.get('alias'),
                            source_project=resolved.get('project'),
                            scorer=suffix, snapshot=dlc_alias.parse_scorer(target)['snapshot'],
                            queueid=(queueid or None),
                            requested_by=extra_kwargs.get('user'),
                            # _git_hash returns (hash, warning), not a bare string --
                            # binding the tuple makes MySQL reject the whole INSERT
                            code_git_hash=dlc_release._git_hash(
                                os.path.dirname(os.path.abspath(__file__)))[0] or None,
                            qc=qc_doc)
                        log.info('recorded gDLCFit row for %s (release %s, rawid %s)',
                                 os.path.basename(rec['alias_path']), release_id,
                                 rec.get('rawid'))
                    except Exception as e:
                        log.error('FAILED to record a gDLCFit row for %s: %s: %s. The '
                                  'analysis itself is fine -- run scripts/dlc_sync_fits.py '
                                  '--go to backfill it.', stem, type(e).__name__, e)
                elif release_id is None:
                    log.info('no release for %s: not recording a gDLCFit row (project and '
                             'explicit-config runs carry no provenance stamp; the sweeper '
                             'infers those)', stem)
                else:
                    log.info('%s was analyzed into %s, not a sorted/ directory: not '
                             'recording a gDLCFit row. Scratch output is deletable at any '
                             'time and the table is append-only, so a row for it would '
                             'become a dangling pointer nothing cleans up.',
                             stem, path_sorted)
                # [AGENT EDIT END]
        finally:
            if rescale_tmp is not None:
                _shutil.rmtree(rescale_tmp, ignore_errors=True)
                log.info('removed rescale scratch directory %s', rescale_tmp)
        # [AGENT EDIT END]
        # [AGENT EDIT END]

    if 'video' in action_list:
        # [AGENT EDIT START | agent: Codex | user: wingertj | reason: label from the same release directory that inference writes into | date: 2026-09-10]
        create_labeled_videos(dlc, path_config, vid_paths, path_sorted, release_id)
        # [AGENT EDIT END]

    if 'summary' in action_list:
        from nems_lbhb.motor import face_tools

        fig=face_tools.summary_plot(vid_paths)

    if 'refine' in action_list:
        # identify "bad" frames and save in training set
        dlc.extract_outlier_frames(path_config, vid_paths, destfolder=path_sorted, automatic=True)
        # same reason as 'extract': a human refines these next (see open_up_project_paths)
        open_up_project_paths(os.path.dirname(os.path.abspath(path_config)))

        # gui to relabel the bad frames
        # dlc.refine_labels(path_config)
    if 'extract' in action_list:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: DLC 2.3.6
        # extract_frames signature is (config, mode=..., videos_list=...); the old
        # videos=/automatic= kwargs raise TypeError | date: 2026-08-18]
        dlc.extract_frames(path_config, mode='automatic', userfeedback=False,
                           videos_list=vid_paths)
        # [AGENT EDIT END]
        # the frames are for a HUMAN to label next, so the folder this job just created
        # has to be writable by that human -- see open_up_project_paths
        open_up_project_paths(os.path.dirname(os.path.abspath(path_config)))

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: evaluation existed only
    # as a BLOCKING in-GUI call (labeler_core.evaluate_all_snapshots), so scoring a fresh
    # training run meant holding the Streamlit session open on a GPU box. As a queue
    # action it runs where the other GPU work runs. snapshotindex is set to 'all' so every
    # snapshot is scored, and restored in a finally: without it a crash leaves the project
    # config on 'all', which then breaks every later inference (todo.txt item 8).
    # | date: 2026-08-23]
    if 'evaluate' in action_list:
        assert_not_a_release(resolved, 'evaluate')
        _eval_cfg = yaml.safe_load(open(path_config))
        _orig_index = _eval_cfg.get('snapshotindex')
        _eval_cfg['snapshotindex'] = 'all'
        with open(path_config, 'w') as _fh:
            yaml.safe_dump(_eval_cfg, _fh, sort_keys=False)
        try:
            log.info('evaluating every snapshot of %s (iteration-%s)',
                     path_config, _eval_cfg.get('iteration'))
            dlc.evaluate_network(path_config, plotting=False)
        finally:
            _eval_cfg = yaml.safe_load(open(path_config))
            _eval_cfg['snapshotindex'] = _orig_index
            with open(path_config, 'w') as _fh:
                yaml.safe_dump(_eval_cfg, _fh, sort_keys=False)
            log.info('restored snapshotindex=%r', _orig_index)
        _res = os.path.join(os.path.dirname(os.path.abspath(path_config)),
                            'evaluation-results',
                            'iteration-%s' % _eval_cfg.get('iteration'),
                            'CombinedEvaluation-results.csv')
        log.info('evaluation results: %s (exists=%s)', _res, os.path.exists(_res))
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- curation is
    # its own action. It used to run inside 'training' (dlc.merge_datasets right before
    # create_training_dataset), so labels were accepted as a side effect of clicking
    # Train, DLC's silent skip went unnoticed, and nothing recorded what changed between
    # one release's label set and the next | date: 2026-08-06]
    if 'curate' in action_list:
        assert_not_a_release(resolved, 'curate')

        curate_project = os.path.dirname(os.path.abspath(path_config))
        curate_user = extra_kwargs.get('user') or os.environ.get('USER', 'unknown')
        curate_note = extra_kwargs.get('note', '')

        # The same lock training takes: merge_datasets rewrites config.yaml's
        # `iteration`, and doing that under a running training run is how the two
        # disagree about which iteration is being trained.
        lock = dlc_train_guard.acquire_lock(curate_project, {
            'user': curate_user, 'queueid': queueid,
            'run_id': extra_kwargs.get('run_id') or str(queueid or os.getpid()),
            'action': 'curate', 'config': path_config,
        })
        try:
            before = dlc_train_guard.label_inventory(curate_project)
            # Capture the newest label mtime BEFORE the merge, and stamp the log with
            # that. The DLC labeling GUI does not take this lock, so a label written
            # while merge_datasets runs was never merged -- stamping a post-merge stat
            # would record it as curated and let the staleness gate wave it through.
            labels_seen_mtime = dlc_train_guard.newest_label_mtime(curate_project) or 0.0
            iteration_before = project_iteration(path_config)
            log.info(f"curating {curate_project}: {len(before)} labelled folder(s) at "
                     f"iteration-{iteration_before}")

            dlc.merge_datasets(path_config)

            # merge_datasets prints and returns when it skips; this is what turns that
            # into a refusal instead of a training run against the old iteration.
            dlc_train_guard.assert_merge_succeeded(curate_project, iteration_before)

            after = dlc_train_guard.label_inventory(curate_project)
            delta = dlc_train_guard.curation_delta(before, after)
            delta['iteration_before'] = iteration_before
            delta['iteration_after'] = project_iteration(path_config)
            delta['alias'] = resolved['alias'] or ''
            delta['config'] = path_config

            curation_log = dlc_train_guard.append_curation_log(
                curate_project, delta, curate_user, curate_note,
                labels_newest_mtime=labels_seen_mtime)
            log.info(f"curation recorded in {curation_log}: "
                     f"added {delta['folders_added']}, changed {delta['folders_changed']}, "
                     f"frames {delta['frames_before']} -> {delta['frames_after']} "
                     f"(iteration-{delta['iteration_before']} -> "
                     f"iteration-{delta['iteration_after']})")
        finally:
            # same rationale as the training block: a release_lock failure must not
            # replace the exception the caller actually needs to see.
            try:
                dlc_train_guard.release_lock(lock)
            except Exception:
                log.exception(
                    'failed to release DLC training lock %s; it may need manual cleanup '
                    'via dlc_train_guard.release_lock(%r, force=True)', lock, lock)
    # [AGENT EDIT END]

    if 'training' in action_list:
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- lock the
        # project, refuse an in-place retrain, take hyperparameters from the registry,
        # support from_release=, and record TRAINRUN.toml | date: 2026-08-06]
        shuffle = 1

        # review fix 4 -- init_weights=/from_release= mutual exclusion is parsed here,
        # before merge_datasets/create_training_dataset mutate the project, so a
        # malformed command line fails before anything is written. It used to be
        # checked only after both of those had already run.
        from_release = extra_kwargs.get('from_release')
        if from_release and init_weights:
            raise ValueError('pass either init_weights= or from_release=, not both')

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- this
        # refusal (and the Task-7 "review fix 3" project hint behind it: resolved
        # ['project'] is None when the release was reached via an explicit RELEASE_ROOT
        # config path, which used to print "Train in the alias's project None instead")
        # now lives in assert_not_a_release/alias_project_hint, because `curate` has to
        # refuse a frozen release for exactly the same reason | date: 2026-08-06]
        assert_not_a_release(resolved, 'training')
        # [AGENT EDIT END]

        train_project = os.path.dirname(os.path.abspath(path_config))
        train_user = extra_kwargs.get('user') or os.environ.get('USER', 'unknown')
        train_run_id = extra_kwargs.get('run_id') or str(queueid or os.getpid())
        modelprefix = dlc_train_guard.run_modelprefix(train_user, train_run_id)
        resume = truthy(extra_kwargs.get('resume'))

        hp = apply_training_overrides(training_hyperparams(), extra_kwargs)
        log.info(f"training hyperparameters: {hp}")

        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- training
        # no longer merges. Since merge_datasets is now a separate 'curate' action, the
        # new failure mode is labelling a session and going straight to Train, which
        # would silently train the previous label set; refuse that here, before the lock
        # and before anything is written | date: 2026-08-06]
        dlc_train_guard.assert_curation_current(train_project)
        # [AGENT EDIT END]

        lock = dlc_train_guard.acquire_lock(train_project, {
            'user': train_user, 'queueid': queueid, 'run_id': train_run_id,
            'modelprefix': modelprefix, 'config': path_config,
        })
        try:
            # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 --
            # dlc.merge_datasets(path_config) used to run right here. It is now the
            # 'curate' action: merging is a decision (it changes which labels every
            # future model is trained on, and DLC's version of it fails silently),
            # not a side effect of asking to train. assert_curation_current above is
            # what makes sure it has happened | date: 2026-08-06]
            iteration = project_iteration(path_config)
            # [AGENT EDIT END]
            log.info(f"training into iteration-{iteration} of {train_project}")

            dlc_train_guard.assert_fresh_iteration(train_project, iteration,
                                                   shuffle=shuffle, resume=resume)

            # Create training dataset
            # NOTE: This regenerates pose_cfg.yaml with default init_weights!
            dlc.create_training_dataset(
                path_config,
                net_type=hp['net_type'],
                augmenter_type=hp['augmenter_type'],
            )

            # Apply transfer learning init_weights if specified.
            # This MUST happen after create_training_dataset() since it overwrites pose_cfg.yaml
            if from_release:
                init_weights = release_init_weights(from_release,
                                                    expect_net_type=hp['net_type'])
                log.info(f"from_release={from_release}: init_weights={init_weights}")
            if init_weights:
                log.info(f"Applying transfer learning init_weights: {init_weights}")
                if not update_pose_cfg_init_weights(path_config, init_weights):
                    raise RuntimeError(
                        f"could not write init_weights={init_weights} into pose_cfg.yaml; "
                        f"refusing to start a run that would silently train from the "
                        f"default ImageNet weights instead.")
                log.info("Successfully applied transfer learning weights")

            git_hash, git_warning = dlc_release._git_hash(
                os.path.dirname(os.path.abspath(__file__)))
            train_dirs = sorted(glob.glob(os.path.join(
                train_project, 'dlc-models', f'iteration-{iteration}',
                f'*shuffle{shuffle}', 'train')))
            if not train_dirs:
                raise RuntimeError(
                    f"create_training_dataset produced no "
                    f"dlc-models/iteration-{iteration}/*shuffle{shuffle}/train under "
                    f"{train_project}")
            for train_dir in train_dirs:
                write_trainrun(train_dir, [
                    ('run', {'user': train_user, 'run_id': train_run_id,
                             'queueid': queueid, 'modelprefix': modelprefix,
                             'started_utc': dlc_train_guard._utcnow(),
                             'host': os.uname()[1], 'git_hash': git_hash,
                             'git_warning': git_warning,
                             'config': path_config, 'project': train_project,
                             'alias': resolved['alias'] or '',
                             'iteration': iteration, 'shuffle': shuffle,
                             'resume': resume}),
                    ('init', {'from_release': from_release or '',
                              'init_weights': init_weights or ''}),
                    ('hyperparameters', hp),
                    ('lock', dlc_train_guard.lock_holder(train_project) or {}),
                ])

            # Train the model
            dlc.train_network(path_config, shuffle=shuffle,
                              max_snapshots_to_keep=hp['max_snapshots_to_keep'],
                              autotune=hp['autotune'],
                              displayiters=hp['displayiters'],
                              saveiters=hp['saveiters'],
                              maxiters=hp['maxiters'],
                              allow_growth=hp['allow_growth'])
        finally:
            # review fix 5 -- release_lock can itself raise (e.g. a corrupt lockfile,
            # or a lock this process no longer owns for some reason); doing that inside
            # a bare `finally` would replace/mask whatever exception the training block
            # itself raised. Log and swallow instead: the training exception (if any) is
            # what the caller needs to see, and a lock left behind on release failure is
            # recoverable by hand (dlc_train_guard.release_lock(path, force=True)).
            try:
                dlc_train_guard.release_lock(lock)
            except Exception:
                log.exception(
                    'failed to release DLC training lock %s; it may need manual cleanup '
                    'via dlc_train_guard.release_lock(%r, force=True)', lock, lock)
        # [AGENT EDIT END]

    if 0:
        dlc.merge_datasets(path_config)
        dlc.create_training_dataset(path_config, net_type='resnet_50', augmenter_type='imgaug')

    if 0:
        # before running, update pose_cfg.yaml to use last snapshot from previous iteration as initial condition
        # (rather than starting over from visnet)
        dlc.train_network(path_config, shuffle=1, displayiters=100)

    log.info(f"face_fit_script complete db_exists={db_exists} qid={queueid}")

    if db_exists & (queueid > 0):
        log.info('Marking job complete in celldb')
        nd.update_job_complete(queueid)
