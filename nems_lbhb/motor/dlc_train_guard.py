"""Refusals that make the two DLC-model-destroying accidents impossible.

Two mechanisms destroyed deployed models, and both were previously only
*documented*.  This module turns each into a refusal:

* **An in-place retrain into a populated ``train/``.**  ``create_training_dataset``
  regenerates ``dlc-models/iteration-<i>/<model>/train/`` and ``train_network``
  then writes ``snapshot-*`` into it.  If that directory already holds the
  snapshots a deployed model was cut from, the run overwrites them and the
  deployed model is gone.  :func:`assert_fresh_iteration` refuses to start unless
  the iteration is empty, with ``resume=1`` as the explicit, typed-by-a-human
  opt-in.
* **Two concurrent runs sharing one project directory.**  DeepLabCut keeps no
  lock of any kind; two ``train_network`` calls on one project interleave their
  checkpoint writes.  :func:`acquire_lock` puts an ``O_EXCL`` lockfile at
  ``<project>/.dlc_train.lock`` -- atomic even on NFS, which is where every
  project here lives -- and the second run is refused by name.

**There are no silent fallbacks anywhere in this module.**  A held lock raises and
names its holder; a corrupt lockfile raises rather than being treated as absent;
releasing a lock this process does not own raises unless ``force=True``.  A
guard that quietly steps aside under an unusual condition is not a guard, and
both defects above happened under exactly such conditions.

Task 8 adds the third refusal, on the *label* side:

* **A ``merge_datasets`` that silently did nothing.**  DeepLabCut sets
  ``flagged=True`` and leaves ``iteration`` **unchanged** when a ``labeled-data``
  folder is missing its expected files, so a training run started right after it
  proceeds against the OLD iteration with no error.
  :func:`assert_merge_succeeded` refuses on an iteration that did not advance and
  names the folders that most likely caused it, and
  :func:`assert_curation_current` refuses to *train* at all when the labels on
  disk are newer than the last recorded curation.

Deliberately **standard library only** -- no NEMS, no DeepLabCut, no numpy, no
pandas -- and Python 3.8 compatible, because it is imported by the production
``dlc5`` environment (py3.8.17), by ``streamlit_dlc`` (py3.8) and by ``psi_nems2``
(py3.9) alike, and a guard that cannot be imported is a guard that gets commented
out.  Two deviations, both narrow: ``yaml`` is imported at module scope inside a
``try`` (it is a hard dependency of DeepLabCut itself, and its absence disables
only :func:`assert_merge_succeeded`, loudly), and ``pandas`` is imported *inside*
:func:`label_inventory`, whose frame counts degrade to ``None`` with an explicit
``frames_source='unavailable'`` rather than to a guessed number.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 7 -- new module:
# exclusive training lock + fresh-iteration guard against in-place retrains | date: 2026-08-06]
import calendar
import datetime
import errno
import getpass
import glob
import hashlib
import json
import logging
import os
import re
import socket
import time

# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- assert_merge_succeeded
# has to re-read the project's config.yaml; yaml ships with DeepLabCut so it is present
# everywhere this module actually runs, but the import stays optional so that the LOCK
# guards remain importable in an environment that lacks it | date: 2026-08-06]
try:
    import yaml
except ImportError:                     # pragma: no cover - yaml is present everywhere here
    yaml = None
# [AGENT EDIT END]

log = logging.getLogger(__name__)

LOCK_NAME = '.dlc_train.lock'

# how often acquire_lock() re-checks while waiting out a `timeout`
_POLL_SECONDS = 2.0

# [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 7 -- bound the
# "lock vanished between the O_EXCL failure and the read" retry, which previously could
# spin without ever checking the deadline or sleeping | date: 2026-08-06]
# how many times acquire_lock() will retry immediately (no deadline check, no sleep)
# when the lockfile disappears between its O_EXCL failure and the follow-up read
_MAX_VANISHED_RETRIES = 20
# [AGENT EDIT END]

_SLUG_RE = re.compile(r'[^A-Za-z0-9._]+')


def _utcnow():
    """``YYYY-MM-DDTHH:MM:SSZ`` -- same spelling the release manifests use."""
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def _slug(value, fallback='unknown'):
    """``value`` reduced to ``[A-Za-z0-9._]`` runs joined by ``-``.

    Used on the user name and run id that go into :func:`run_modelprefix`, whose
    result is a *path* component: an unslugged ``/`` there would silently nest the
    run one directory deeper than every tool expects.
    """
    out = _SLUG_RE.sub('-', str(value)).strip('-')
    return out or fallback


def _pid_alive(pid):
    """True/False if we can tell, None if we cannot (foreign uid, bad pid).

    Only ever used to *annotate* a refusal, never to decide one: a lock is never
    broken automatically, because a dead pid on another host is indistinguishable
    from a live pid on this one.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True      # exists, owned by someone else
        return None


# ---------------------------------------------------------------------------
# the training lock
# ---------------------------------------------------------------------------

def lock_path(project):
    """``<project>/.dlc_train.lock``."""
    return os.path.join(project, LOCK_NAME)


def _read_lock(path):
    """The lock record at ``path``, or ``None`` if there is no lock there.

    A file that exists but does not parse raises: an unreadable lock is a lock in
    an unknown state, and treating it as absent is exactly the silent fallback
    this module exists to remove.
    """
    try:
        with open(path) as f:
            text = f.read()
    except IOError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise
    try:
        rec = json.loads(text)
    except ValueError as exc:
        raise RuntimeError(
            'DLC training lockfile %s is unreadable (%s). It was written by a run '
            'that died mid-write, or by hand. Inspect it, then remove it with '
            'dlc_train_guard.release_lock(%r, force=True) once you have confirmed '
            'no training run is active.' % (path, exc, path))
    if not isinstance(rec, dict):
        raise RuntimeError(
            'DLC training lockfile %s is unreadable: expected a JSON object, got %s'
            % (path, type(rec).__name__))
    return rec


def _describe(rec, path):
    """One-line human description of a lock holder, for a refusal message."""
    alive = _pid_alive(rec.get('pid'))
    if alive is False:
        liveness = ' (pid is NOT running on this host -- likely a crashed run)'
    elif alive is None:
        liveness = ' (liveness unknown)'
    else:
        liveness = ''
    return ('user=%s host=%s pid=%s queueid=%s run_id=%s started_utc=%s%s [%s]'
            % (rec.get('user'), rec.get('host'), rec.get('pid'),
               rec.get('queueid'), rec.get('run_id'), rec.get('started_utc'),
               liveness, path))


def lock_holder(project):
    """The record of whoever holds ``project``'s training lock, or ``None``.

    Keys: ``user, host, pid, queueid, run_id, started_utc`` plus anything extra
    the acquirer passed in ``info``.  Raises ``RuntimeError`` if a lockfile is
    present but unparseable.
    """
    return _read_lock(lock_path(project))


def acquire_lock(project, info=None, timeout=0):
    """Take ``project``'s exclusive training lock; return the lockfile path.

    ``os.open(..., O_CREAT | O_EXCL)`` is the whole mechanism.  It is atomic on
    NFS (the server resolves the race, not the client), which matters because
    every DLC project here lives on ``/auto/data``.  A check-then-create pair
    would not be.

    ``info`` contributes ``user``, ``queueid``, ``run_id`` and any other
    provenance the caller wants recorded.  ``pid``, ``host`` and ``started_utc``
    are stamped *after* ``info`` is merged and therefore cannot be overridden --
    diagnosing a stale lock depends on those three being the truth about the
    process that actually holds it.

    ``timeout`` (seconds) polls instead of failing immediately; ``0`` (the
    default) means fail now.  Either way the failure is a ``RuntimeError`` naming
    the holder's user, host, pid and start time -- never a wait forever, and never
    a silent steal.
    """
    if not os.path.isdir(project):
        raise ValueError(
            'cannot take a DLC training lock: %s is not a directory' % project)

    path = lock_path(project)

    rec = {'user': getpass.getuser(), 'queueid': 0, 'run_id': ''}
    rec.update(info or {})
    # non-negotiable, stamped last:
    rec['pid'] = os.getpid()
    rec['host'] = socket.gethostname()
    rec['started_utc'] = _utcnow()

    payload = json.dumps(rec, indent=2, sort_keys=True, default=str) + '\n'

    deadline = time.time() + float(timeout or 0)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 7 -- bound
    # the "lock vanished" retry below | date: 2026-08-06]
    vanished_retries = 0
    # [AGENT EDIT END]
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        else:
            try:
                os.write(fd, payload.encode('utf-8'))
            finally:
                os.close(fd)
            log.info('took DLC training lock %s (%s)', path, _describe(rec, path))
            return path

        if time.time() >= deadline:
            held = _read_lock(path)
            if held is None:
                # [AGENT EDIT START | agent: claude | user: wingertj | reason: review fix 7
                # -- this branch (the lock disappeared between the O_EXCL failure just
                # above and this read) used to `continue` unconditionally, skipping both
                # the deadline check and the sleep on every iteration. If something kept
                # recreating and removing the lockfile at exactly the wrong moment, this
                # would spin without bound. Cap it and fail loudly instead. | date: 2026-08-06]
                vanished_retries += 1
                if vanished_retries > _MAX_VANISHED_RETRIES:
                    raise RuntimeError(
                        'DLC training lock %s kept appearing and disappearing while this '
                        'run tried to acquire it (retried %d times without success). '
                        'Refusing to spin forever; something is repeatedly creating and '
                        'removing this lockfile.' % (path, vanished_retries))
                # released between the O_EXCL failure and the read -- retry, bounded above.
                continue
                # [AGENT EDIT END]
            raise RuntimeError(
                'DLC training lock is already held on %s by %s. Concurrent training '
                'runs on one project interleave their checkpoint writes and destroy '
                'both models, so this run is refused. If that run is definitely over, '
                'clear the lock with '
                'dlc_train_guard.release_lock(%r, force=True).'
                % (project, _describe(held, path), path))

        time.sleep(min(_POLL_SECONDS, max(0.05, deadline - time.time())))


def release_lock(path, force=False):
    """Drop the lockfile at ``path``.  Returns True if one was removed.

    Absence is tolerated (``False``) so this is safe in a ``finally``.  A lock
    held by a *different* process is not: releasing someone else's lock is how
    the concurrency it prevents comes back, so it raises unless ``force=True``.
    """
    rec = None
    try:
        rec = _read_lock(path)
    except RuntimeError:
        if not force:
            raise
    if rec is None and not os.path.lexists(path):
        return False

    if not force and rec is not None:
        mine = (rec.get('pid') == os.getpid()
                and rec.get('host') == socket.gethostname())
        if not mine:
            raise RuntimeError(
                'refusing to release a DLC training lock this process does not '
                'hold: %s. Pass force=True only after confirming that run is over.'
                % _describe(rec, path))

    try:
        os.remove(path)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return False
        raise
    log.info('released DLC training lock %s', path)
    return True


# ---------------------------------------------------------------------------
# the fresh-iteration guard
# ---------------------------------------------------------------------------

def iteration_snapshots(project, iteration, shuffle=1):
    """``(at_risk, other_shuffles)`` -- snapshot ``.index`` paths under one iteration.

    ``at_risk`` are the ones inside a model directory ending ``shuffle<shuffle>``,
    i.e. exactly the ones a run with that shuffle would overwrite.
    """
    pattern = os.path.join(project, 'dlc-models', 'iteration-%s' % iteration,
                           '*shuffle*', 'train', 'snapshot-*.index')
    hits = sorted(glob.glob(pattern))
    tail = 'shuffle%s' % shuffle
    at_risk, other = [], []
    for h in hits:
        model_dir = os.path.basename(os.path.dirname(os.path.dirname(h)))
        (at_risk if model_dir.endswith(tail) else other).append(h)
    return at_risk, other


def assert_fresh_iteration(project, iteration, shuffle=1, resume=False):
    """Refuse to train into an iteration that already holds snapshots.

    This is the direct guard against the defect that destroyed a deployed model:
    ``train_network`` writes ``snapshot-<N>`` into
    ``dlc-models/iteration-<i>/<model>-shuffle<s>/train/``, and if that directory
    is already populated the previous run's checkpoints are overwritten in place
    with no warning and no backup.

    Scoped to ``shuffle`` because a run with shuffle 1 does not touch shuffle 2's
    directory; snapshots found under other shuffles are logged, not refused.

    ``resume=True`` is the explicit opt-in for the legitimate case (continuing an
    interrupted run of *this* model).  It is deliberately not inferable from
    anything on disk -- a human has to say it.
    """
    at_risk, other = iteration_snapshots(project, iteration, shuffle)

    if other:
        log.info('iteration-%s of %s also holds %d snapshot(s) under other shuffles '
                 '(not at risk from shuffle=%s): %s',
                 iteration, project, len(other), shuffle,
                 ', '.join(os.path.basename(p) for p in other))

    if not at_risk:
        return

    if resume:
        log.warning('resume=1: training into a POPULATED iteration-%s (shuffle=%s) of '
                    '%s. These snapshots will be overwritten as training proceeds: %s',
                    iteration, shuffle, project, ', '.join(at_risk))
        return

    listed = '\n  '.join(at_risk)
    raise RuntimeError(
        'refusing to train into iteration-%s (shuffle=%s) of %s: it already holds '
        'trained snapshots, which train_network would overwrite in place. This is '
        'how a deployed model was destroyed before.\n  %s\n'
        'If you really mean to continue this same run, pass resume=1. If you mean to '
        'train a NEW model, bump the project\'s `iteration` (deeplabcut.merge_datasets '
        'does this when every labeled folder has been refined) so the new run gets its '
        'own directory, or cut a release of the current snapshots first '
        '(nems_lbhb.motor.dlc_release.build_release) so they survive.'
        % (iteration, shuffle, project, listed))


# ---------------------------------------------------------------------------
# run identity
# ---------------------------------------------------------------------------

def run_modelprefix(user, run_id):
    """``runs/<UTC date>-<user>-<run_id>`` -- a unique name for one training run.

    Spelled to be usable directly as DeepLabCut's ``modelprefix`` argument, which
    relocates ``dlc-models`` to ``<project>/<modelprefix>/dlc-models`` and so gives
    concurrent runs disjoint output trees.  ``dlc_fit_script`` does **not** pass it
    to DeepLabCut today -- doing so would move where every existing tool looks for
    trained models -- it records it in ``TRAINRUN.toml`` and in the lock so a run
    can be named after the fact.  Wiring it into ``modelprefix`` is a deliberate,
    separate decision.
    """
    date = datetime.datetime.utcnow().strftime('%Y-%m-%d')
    return 'runs/%s-%s-%s' % (date, _slug(user, 'unknown-user'),
                              _slug(run_id, 'unknown-run'))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- curation:
# unbundle merge_datasets from training, guard its silent skip, and record what the
# labels did between one release and the next | date: 2026-08-06]
# ---------------------------------------------------------------------------
# curation: the label side of the same problem
# ---------------------------------------------------------------------------

CURATION_LOG_NAME = 'curation_history.log'

# How far a log entry's wall-clock timestamp may lag the newest label file before
# the labels count as newer than the curation.  Only used for log entries written
# before `labels_newest_mtime` existed (or by a hand edit): `_utcnow()` truncates
# to whole seconds, so a curation recorded 0.4 s after the file it merged would
# otherwise read as stale by rounding alone.
_CURATION_MTIME_TOLERANCE = 2.0


def _sha256_file(path, _chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(_chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _parse_utc(text):
    """``YYYY-MM-DDTHH:MM:SSZ`` -> POSIX seconds.  Raises on anything else."""
    return calendar.timegm(time.strptime(str(text), '%Y-%m-%dT%H:%M:%SZ'))


LABEL_FILE_PATTERNS = ('CollectedData_*.h5', 'MachineLabelsRefine.h5')


def label_files(project):
    """Every label file under ``project``'s ``labeled-data/``, sorted.

    A folder counts when it holds ``CollectedData_*.h5`` **or**
    ``MachineLabelsRefine.h5``.  Both matter, and for different reasons:

    * ``CollectedData_<scorer>.h5`` -- frames labelled by hand.
    * ``MachineLabelsRefine.h5`` -- outlier frames extracted by
      ``extract_outlier_frames`` and corrected by ``refine_labels``.

    DeepLabCut's own ``merge_datasets`` accepts a folder holding *either*
    (``refine_training_dataset/outlier_frames.py``, ~line 1113), so refined data is
    training data as far as DLC is concerned.  Watching only ``CollectedData_*``
    would leave the staleness gate blind to a freshly refined folder -- fail-open,
    and precisely in the workflow the ``curate`` action exists to serve.

    A bare folder name is NOT enough: ``right_chimney_PRNfb``'s
    ``labeled-data/SQD032a07_a_NTD_recording/`` holds neither file and contributes
    no training frames.

    Known, deliberate difference from DLC: DLC matches ``CollectedData_`` +
    ``cfg['scorer']`` + ``.h5`` exactly, whereas this globs, so it also picks up the
    duplicate downloads that sit beside the real file in four projects
    (``CollectedData_svd [1].h5``).  That over-inclusion is kept on purpose --
    ``dlc_release`` freezes all of them into a release's ``labels/``, so an inventory
    that silently picked one would disagree with the release it describes -- and it
    errs toward demanding a curate rather than skipping one.
    """
    out = []
    for pattern in LABEL_FILE_PATTERNS:
        out.extend(glob.glob(os.path.join(project, 'labeled-data', '*', pattern)))
    return sorted(out)


def _label_files_by_folder(project):
    out = {}
    for path in label_files(project):
        out.setdefault(os.path.basename(os.path.dirname(path)), []).append(path)
    return out


def label_inventory(project):
    """``{folder: {'file', 'files', 'file_count', 'frames', 'frames_source',
    'sha256', 'mtime'}}`` for every labelled folder in ``project``.

    ``sha256`` is a content fingerprint over the folder's label files (each
    file's basename and its own sha256), so it changes when a label changes and
    does **not** change when a file is merely touched -- that is what makes the
    before/after comparison in :func:`curation_delta` mean "someone relabelled
    this", not "something rewrote the mtime".

    Five real folders hold a duplicate download beside the real file
    (``CollectedData_svd [1].h5``).  All of them are folded into one record, and
    the duplicates are logged, because ``dlc_release`` freezes all of them too and
    an inventory that silently picked one would disagree with the release it is
    meant to describe.

    ``frames`` is the row count from ``pandas.read_hdf``.  pandas is imported
    **here**, not at module scope, and when it is missing (or a file will not
    read) ``frames`` is ``None`` and ``frames_source`` says why.  A guessed count
    would propagate into the curation log and from there into a release manifest.
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None

    inventory = {}
    for folder, files in sorted(_label_files_by_folder(project).items()):
        files = sorted(files)
        if len(files) > 1:
            log.warning('labeled-data/%s holds %d CollectedData_*.h5 files (%s); all of '
                        'them are counted, as dlc_release freezes all of them',
                        folder, len(files), ', '.join(os.path.basename(f) for f in files))

        digest = hashlib.sha256()
        frames = 0 if pd is not None else None
        source = 'pandas' if pd is not None else 'unavailable'

        for path in files:
            digest.update(os.path.basename(path).encode('utf-8'))
            digest.update(b'\x00')
            digest.update(_sha256_file(path).encode('ascii'))
            digest.update(b'\n')
            if pd is None or frames is None:
                continue
            try:
                frames += len(pd.read_hdf(path))
            except Exception as exc:
                # only the first line: an h5py failure drags a 20-line HDF5 trace
                # with it, and this string ends up in a one-line JSON log record.
                text = str(exc).strip().splitlines()
                frames = None
                source = 'unreadable: %s: %s' % (type(exc).__name__,
                                                 text[0] if text else '')

        inventory[folder] = {
            'file': files[0],
            'files': files,
            'file_count': len(files),
            'frames': frames,
            'frames_source': source,
            'sha256': digest.hexdigest(),
            'mtime': max(os.path.getmtime(p) for p in files),
        }
    return inventory


def _sum_frames(inventory):
    """Total frames, or ``None`` if any folder's count is unknown."""
    total = 0
    for rec in inventory.values():
        frames = rec.get('frames')
        if frames is None:
            return None
        total += int(frames)
    return total


def curation_delta(before, after):
    """What one ``merge_datasets`` did to the labels, as a JSON-able dict.

    Keys: ``folders_added``, ``folders_changed``, ``folders_removed``,
    ``frames_before``, ``frames_after``, ``frames_added``, ``frames_counted``,
    ``folders_before``, ``folders_after``.

    When any folder's frame count is unknown (no pandas, or an unreadable .h5)
    the three frame totals are ``None`` and ``frames_counted`` is ``False``.
    Treating an unknown count as zero would silently report "0 frames added" for
    a curation that added hundreds.
    """
    before = before or {}
    after = after or {}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(f for f in (set(before) & set(after))
                     if before[f].get('sha256') != after[f].get('sha256'))

    frames_before = _sum_frames(before)
    frames_after = _sum_frames(after)
    counted = frames_before is not None and frames_after is not None
    return {
        'folders_added': added,
        'folders_changed': changed,
        'folders_removed': removed,
        'folders_before': len(before),
        'folders_after': len(after),
        'frames_before': frames_before,
        'frames_after': frames_after,
        'frames_added': (frames_after - frames_before) if counted else None,
        'frames_counted': counted,
    }


def unlabelled_folders(project, scorer=None):
    """``labeled-data/*`` directories DeepLabCut's ``merge_datasets`` would flag.

    Mirrors ``merge_datasets`` (DLC 2.3.6,
    ``refine_training_dataset/outlier_frames.py``) exactly: it walks
    ``labeled-data/*``, skipping names containing ``_labeled`` and names starting
    with ``.``, and flags any folder holding **neither** ``MachineLabelsRefine.h5``
    **nor** ``CollectedData_<scorer>.h5``.  One flagged folder and the whole merge
    returns without advancing ``iteration``.

    ``scorer`` comes from the project config; without it the check falls back to
    ``CollectedData_*.h5``, which is the same rule :func:`label_inventory` and
    ``dlc_release`` use for "this folder contributes frames".
    """
    out = []
    for path in sorted(glob.glob(os.path.join(project, 'labeled-data', '*'))):
        name = os.path.basename(path)
        if not os.path.isdir(path) or '_labeled' in name or name.startswith('.'):
            continue
        if os.path.isfile(os.path.join(path, 'MachineLabelsRefine.h5')):
            continue
        if scorer:
            if os.path.isfile(os.path.join(path, 'CollectedData_%s.h5' % scorer)):
                continue
        elif glob.glob(os.path.join(path, 'CollectedData_*.h5')):
            continue
        out.append(name)
    return out


def read_project_config(project_or_config):
    """``(cfg, path)`` for a DLC project config, read with plain ``yaml.safe_load``.

    Never DeepLabCut's ``auxiliaryfunctions.read_config``, which rewrites
    ``project_path`` and writes the file back -- on a frozen release that is a
    crash, and on a project it is an unannounced mutation.
    """
    if yaml is None:
        raise RuntimeError(
            'reading a DLC project config needs PyYAML, which failed to import in '
            'this environment. Install it, or run the curate action from the dlc5 '
            'environment, where DeepLabCut already depends on it.')
    path = project_or_config
    if os.path.isdir(path):
        path = os.path.join(path, 'config.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, path


def project_iteration(project_or_config):
    """The ``iteration`` in a project's ``config.yaml``.

    Accepts the project directory or the config path itself.
    """
    cfg, path = read_project_config(project_or_config)
    if 'iteration' not in cfg:
        raise RuntimeError('%s has no `iteration` key; it is not a DLC project config'
                           % path)
    return int(cfg['iteration'])


def assert_merge_succeeded(project, iteration_before):
    """Refuse when ``merge_datasets`` did not advance the project's ``iteration``.

    DeepLabCut's ``merge_datasets`` sets ``flagged=True`` and **returns without
    touching ``iteration``** whenever a ``labeled-data`` folder is missing its
    expected files.  It logs, it does not raise.  Training started right after it
    then creates its dataset from the OLD iteration -- the new labels are simply
    absent from the model, and nothing anywhere says so.

    The refusal names every ``labeled-data/*`` folder with no
    ``CollectedData_*.h5``, because that is the condition that causes it in
    practice.  When there is no such folder the message says so instead of
    inventing a culprit: the merge still did not happen, and the cause is
    somewhere else (an already-current iteration, a hand-edited config).
    """
    cfg, _cfg_path = read_project_config(project)
    if 'iteration' not in cfg:
        raise RuntimeError('%s has no `iteration` key; it is not a DLC project config'
                           % _cfg_path)
    iteration_after = int(cfg['iteration'])
    if iteration_after > iteration_before:
        log.info('merge_datasets advanced %s from iteration-%s to iteration-%s',
                 project, iteration_before, iteration_after)
        return

    missing = unlabelled_folders(project, scorer=cfg.get('scorer'))
    if missing:
        cause = ('These labeled-data folders hold neither MachineLabelsRefine.h5 nor '
                 'CollectedData_%s.h5, which is exactly what makes merge_datasets flag '
                 'the project and skip. Label or remove them, then curate again:\n  %s'
                 % (cfg.get('scorer') or '<scorer>', '\n  '.join(missing)))
    else:
        cause = ('every labeled-data folder does hold its labels, so the usual cause '
                 'does not apply here -- check merge_datasets\' own output for the '
                 'folder it flagged, and check whether config.yaml was edited by hand '
                 '(merge_datasets is also a no-op when it has already run since the '
                 'last labelling).')

    raise RuntimeError(
        'deeplabcut.merge_datasets did not advance the iteration of %s: it was %s '
        'before and is still %s. merge_datasets does not raise when it skips -- it '
        'flags the project and returns -- so training started now would silently '
        'train the OLD iteration and none of the new labels.\n%s'
        % (project, iteration_before, iteration_after, cause))


def curation_log_path(project):
    """``<project>/curation_history.log``."""
    return os.path.join(project, CURATION_LOG_NAME)


def read_curation_log(project):
    """Every entry in the curation log, oldest first; ``[]`` when there is none.

    Blank lines are skipped; a line that does not parse raises, for the same
    reason a corrupt lockfile does: a curation history in an unknown state must
    not be read as an empty one.
    """
    path = curation_log_path(project)
    entries = []
    try:
        with open(path) as f:
            lines = f.readlines()
    except IOError as exc:
        if exc.errno == errno.ENOENT:
            return []
        raise
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise RuntimeError(
                '%s line %d is not valid JSON (%s). The curation history is '
                'append-only and one object per line; repair or remove that line '
                'before curating or training again.' % (path, n, exc))
        if not isinstance(rec, dict):
            raise RuntimeError('%s line %d is not a JSON object' % (path, n))
        entries.append(rec)
    return entries


def newest_label_mtime(project):
    """mtime of the most recently written ``CollectedData_*.h5``, or ``None``."""
    files = label_files(project)
    if not files:
        return None
    return max(os.path.getmtime(p) for p in files)


def append_curation_log(project, delta, user=None, note='', labels_newest_mtime=None):
    """Append one JSON object to ``<project>/curation_history.log``; return its path.

    Opened ``'a'`` and never rewritten: the history of what was accepted into a
    model's training set is exactly the kind of record that must not be
    editable-in-passing by the code that writes it.

    ``labels_newest_mtime`` is stamped alongside the delta so
    :func:`assert_curation_current` can compare like with like -- the wall-clock
    ``utc`` is truncated to whole seconds and would make a curation recorded a
    fraction of a second after the file it merged look stale.
    """
    record = dict(delta or {})
    record.update({
        'utc': _utcnow(),
        'user': _slug(user or getpass.getuser(), 'unknown'),
        'note': str(note or ''),
        'host': socket.gethostname(),
        'pid': os.getpid(),
        # Stamp the newest mtime the merge actually SAW, passed in from the caller's
        # pre-merge inventory -- not a fresh stat now. The DLC labeling GUI does not
        # take this lock, so a label written DURING the curate run would otherwise be
        # recorded as curated without having been merged, and the staleness gate would
        # then wave it through. Falling back to a fresh stat only when the caller gives
        # us nothing keeps older callers working, but the curate action always passes it.
        'labels_newest_mtime': float(labels_newest_mtime
                                     if labels_newest_mtime is not None
                                     else (newest_label_mtime(project) or 0.0)),
    })
    path = curation_log_path(project)
    with open(path, 'a') as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + '\n')
    log.info('recorded curation in %s: +%s folder(s), %s changed, frames %s -> %s',
             path, len(record.get('folders_added') or []),
             len(record.get('folders_changed') or []),
             record.get('frames_before'), record.get('frames_after'))
    return path


def assert_curation_current(project):
    """Refuse to train when the labels on disk are newer than the last curation.

    ``merge_datasets`` used to run inside the ``training`` action, so labels were
    accepted as a side effect of clicking Train and nothing recorded what
    changed.  Now that it is its own ``curate`` action, the failure mode moves:
    someone labels a new session and trains without curating, and the run
    silently trains the previous label set.

    Refuses when there is no ``curation_history.log`` at all, and when the newest
    ``CollectedData_*.h5`` is newer than the last entry -- naming the files that
    are newer.  A project with no labels at all passes: there is nothing to merge
    and so nothing to be stale about.
    """
    files = label_files(project)
    if not files:
        return

    entries = read_curation_log(project)
    path = curation_log_path(project)
    if not entries:
        raise RuntimeError(
            'refusing to train %s: there is no curation record (%s is %s), so nothing '
            'here has confirmed that deeplabcut.merge_datasets actually accepted the '
            '%d labelled folder(s) on disk. merge_datasets no longer runs as a side '
            'effect of training -- run the `curate` action on this project first '
            '(dlc_fit_script.py <video> curate ... ), which merges, checks that the '
            'iteration advanced, and records what changed.'
            % (project, path, 'absent' if not os.path.exists(path) else 'empty',
               len(_label_files_by_folder(project))))

    last = entries[-1]
    reference = last.get('labels_newest_mtime')
    if reference is None:
        # No exact reference to compare against. The previous behaviour fell back to
        # the entry's wall clock plus a tolerance, which can wave a label file through
        # -- a fail-OPEN softening of the one gate that stops training on stale data,
        # on the single input class this code does not control (hand-edited or foreign
        # entries). This module's doctrine is no silent fallbacks, and a false refusal
        # is the safe direction, so refuse and say exactly how to clear it.
        raise RuntimeError(
            'refusing to train %s: the last entry in %s records no '
            '"labels_newest_mtime", so there is no exact reference to check the '
            'labels on disk against. It was probably hand-edited or written by an '
            'older version. Run the `curate` action to record a fresh entry.'
            % (project, path))
    reference = float(reference)

    newer = [p for p in files if os.path.getmtime(p) > reference]
    if not newer:
        return

    raise RuntimeError(
        'refusing to train %s: %d label file(s) are newer than the last recorded '
        'curation (%s by %s, note %r). Training now would build its dataset from the '
        'labels merge_datasets last accepted, not from these:\n  %s\n'
        'Run the `curate` action on this project first.'
        % (project, len(newer), last.get('utc'), last.get('user'), last.get('note'),
           '\n  '.join(newer)))
# [AGENT EDIT END]
