"""Immutable, checksummed, single-snapshot DeepLabCut model releases.

Two failure modes motivated this module, and a *release* makes both structurally
impossible rather than merely discouraged:

* **Defect 2 -- a deployed model destroyed by an in-place retrain.**  Production
  pointed at a live training project, so the next ``train_network`` run
  overwrote the weights that had already analyzed months of data.  A release is
  a separate directory whose files are ``0o444`` and whose directories are
  ``0o555``; retraining the project cannot reach into it.
* **Defect 3 -- an alias that resolves by snapshot *index*.**  ``snapshotindex``
  selects from the sorted list of snapshots in a training directory, and one
  such directory held two disjoint training runs, so the same index meant two
  different models on two different days.  A release directory contains
  **exactly one** snapshot and pins ``snapshotindex: 0``, so there is nothing to
  choose between.

Layout produced by :func:`build_release`::

    <root>/<alias>-r<NNNN>/
        RELEASE.toml
        SHA256SUMS
        config.yaml
        dlc-models/iteration-0/<Task><date>-trainset<F>shuffle<S>/
            test/pose_cfg.yaml
            train/pose_cfg.yaml
            train/snapshot-<N>.{index,meta,data-00000-of-00001}
        labels/<folder>/CollectedData_<scorer>.h5     (frozen copy)
        evaluation.csv, train_log.txt, learning_stats.csv   (when present)

The copied model directory always lands under ``iteration-0`` no matter which
iteration it came from, and the release ``config.yaml`` says ``iteration: 0``,
``snapshotindex: 0``, ``project_path: <release dir>``.  The *true* source
iteration is recorded in ``RELEASE.toml``, which is provenance, not routing.

``project_path`` matters more than it looks: DeepLabCut's
``auxiliaryfunctions.read_config`` rewrites ``project_path`` to the config's own
directory **and writes the file back** -- but only when the two differ.  Making
the release config already say the right thing is what keeps ``0o444`` from
turning every read into a crash.

Dependencies are deliberately thin: standard library plus PyYAML.  No NEMS, no
DeepLabCut, no pandas, no numpy at module scope.  ``tomli_w`` is imported lazily
inside the writer, so every read path (:func:`verify`, :func:`read_manifest`)
works in an environment that does not have it.  In particular a release can be
built *and verified* without DeepLabCut installed, which is why
``output_suffix`` is reconstructed from the copied artifacts rather than by
calling ``dlc_alias.dlc_scorer_suffix`` (that one imports DeepLabCut).

Registry edits (:func:`promote`, :func:`rollback`) are **surgical line edits**,
not a parse-and-re-serialize round-trip.  ``nems_lbhb/configs/dlc_registry.toml``
carries 61 comment lines transcribing the legacy ``if``/``elif`` condition that
each rule replaced; those comments are the migration's audit trail, and
``tomli_w`` -- like every non-style-preserving TOML writer -- would delete all of
them on the first promote.  ``tomlkit`` is not installed in any environment here,
so the editor below finds the ``[alias.<name>]`` block by its header line and
rewrites only the ``current`` and ``history`` lines inside it, then re-parses the
result and asserts the intended values are present so a botched edit fails loudly
instead of corrupting the file.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 5 -- new module: build,
# verify, promote and roll back immutable single-snapshot DLC releases | date: 2026-08-06]
import csv
import datetime
import difflib
import glob
import hashlib
import logging
import os
import re
import shutil
import socket
import subprocess
import stat
import tempfile

import yaml

from nems_lbhb.motor import dlc_registry

try:                       # python >= 3.11
    import tomllib
except ImportError:        # python <= 3.10
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

log = logging.getLogger(__name__)

# Where frozen releases live.  Single source of truth: dlc_registry.resolve()
# builds an alias's config path from the same constant.
RELEASE_ROOT = dlc_registry.RELEASE_ROOT

DEFAULT_REGISTRY_PATH = dlc_registry.DEFAULT_REGISTRY_PATH

MANIFEST_NAME = 'RELEASE.toml'
SUMS_NAME = 'SHA256SUMS'

# The three files TensorFlow writes per checkpoint.  All three or nothing.
SNAPSHOT_EXTS = ('.index', '.meta', '.data-00000-of-00001')

# <Task><date>-trainset<F>shuffle<S>
_MODEL_DIR_RE = re.compile(r'^(?P<taskdate>.+)-trainset(?P<frac>\d+)shuffle(?P<shuffle>\d+)$')

# pose_cfg.yaml keys worth freezing into [training].  A whitelist, because
# pose_cfg carries numpy-ish and path-ish junk that TOML cannot represent.
_TRAINING_KEYS = (
    'net_type', 'init_weights', 'dataset', 'dataset_type', 'batch_size',
    'global_scale', 'pos_dist_thresh', 'max_input_size', 'min_input_size',
    'scale_jitter_lo', 'scale_jitter_up', 'rotation', 'rotratio', 'cropratio',
    'mirror', 'display_iters', 'save_iters', 'decay_steps', 'lr_init',
    'multi_step', 'optimizer', 'intermediate_supervision', 'location_refinement',
    'locref_loss_weight', 'locref_stdev', 'num_joints', 'alpha_r', 'apply_prob',
    'augmenter_type', 'crop_size', 'crop_sampling', 'max_shift',
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _toml_parser():
    if tomllib is None:
        raise ImportError(
            "reading a release manifest needs a TOML parser: python>=3.11 has "
            "'tomllib' in the standard library, on older pythons run `pip install tomli`")
    return tomllib


def _sha256(path, _chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(_chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _iter_release_files(rdir):
    """Every regular file in the release, as sorted paths relative to ``rdir``."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(rdir):
        for name in filenames:
            full = os.path.join(dirpath, name)
            out.append(os.path.relpath(full, rdir))
    return sorted(out)


def _release_dir(release_id_or_path, root=RELEASE_ROOT):
    """Accept either a release id or a path to a release directory."""
    arg = str(release_id_or_path)
    if os.path.isdir(arg) or os.sep in arg:
        return os.path.abspath(arg)
    return os.path.join(root, arg)


def _git_hash(cwd):
    """``git rev-parse HEAD`` for the repo holding this file, or '' plus a warning."""
    try:
        proc = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=cwd,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return '', 'git_hash unavailable: %s' % exc
    if proc.returncode != 0:
        return '', ('git rev-parse HEAD failed (rc=%d): %s'
                    % (proc.returncode, proc.stderr.decode('utf-8', 'replace').strip()))
    return proc.stdout.decode('utf-8', 'replace').strip(), ''


def _pkg_version(name):
    """Installed version of ``name``, or ''.

    Uses ``importlib.metadata`` rather than importing the package: importing
    tensorflow costs seconds and importing deeplabcut is exactly the dependency
    this module refuses to take on.
    """
    try:
        import importlib.metadata as md
    except ImportError:                                     # pragma: no cover (py<3.8)
        return ''
    try:
        return md.version(name)
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# locating the snapshot
# ---------------------------------------------------------------------------

def _iteration_of(train_dir):
    """The ``iteration-<i>`` component above ``train_dir``, as an int."""
    parts = os.path.normpath(train_dir).split(os.sep)
    for part in reversed(parts):
        m = re.match(r'^iteration-(\d+)$', part)
        if m:
            return int(m.group(1))
    raise ValueError('no iteration-<N> component in %s' % train_dir)


def find_snapshot(project, snapshot, iteration=None, shuffle=1):
    """Locate exactly one ``snapshot-<N>`` inside ``project``.

    Returns ``(train_dir, model_dir_name, source_iteration)``.

    Raises ``FileNotFoundError`` when nothing matches (naming the snapshot
    numbers that *do* exist) or when the ``.index``/``.meta``/``.data-*``
    triplet is incomplete (naming the missing extension), and ``ValueError``
    when the same snapshot number exists in more than one iteration and the
    caller did not say which -- that ambiguity is defect 3 itself, so it is
    refused rather than resolved by a rule the caller cannot see.
    """
    pattern = os.path.join(project, 'dlc-models', 'iteration-*', '*shuffle*',
                           'train', 'snapshot-%d.index' % snapshot)
    hits = sorted(glob.glob(pattern))
    hits = [h for h in hits
            if os.path.basename(os.path.dirname(os.path.dirname(h)))
            .endswith('shuffle%d' % shuffle)]
    if iteration is not None:
        hits = [h for h in hits if _iteration_of(os.path.dirname(h)) == iteration]

    if not hits:
        available = set()
        for path in glob.glob(os.path.join(project, 'dlc-models', 'iteration-*',
                                           '*shuffle*', 'train', 'snapshot-*.index')):
            m = re.match(r'^snapshot-(\d+)\.index$', os.path.basename(path))
            if m:
                available.add(int(m.group(1)))
        raise FileNotFoundError(
            'no snapshot-%d found in %s (shuffle=%s, iteration=%s); snapshots present: %s'
            % (snapshot, project, shuffle, iteration,
               ', '.join(str(s) for s in sorted(available)) or '<none>'))

    if len(hits) > 1:
        cands = ['iteration-%d (%s)' % (_iteration_of(os.path.dirname(h)),
                                        os.path.dirname(h)) for h in hits]
        raise ValueError(
            'snapshot-%d is ambiguous in %s: it exists in %d iterations -- %s. '
            'These are disjoint training runs; pass iteration=<N> to say which one '
            'this release is cut from.'
            % (snapshot, project, len(hits), '; '.join(cands)))

    train_dir = os.path.dirname(hits[0])
    for ext in SNAPSHOT_EXTS:
        sib = os.path.join(train_dir, 'snapshot-%d%s' % (snapshot, ext))
        if not os.path.isfile(sib):
            raise FileNotFoundError(
                'snapshot-%d in %s is incomplete: %s is missing. A TensorFlow '
                'checkpoint is only loadable with all of %s present.'
                % (snapshot, train_dir, ext, ', '.join(SNAPSHOT_EXTS)))

    model_dir = os.path.dirname(train_dir)
    return train_dir, os.path.basename(model_dir), _iteration_of(train_dir)


# ---------------------------------------------------------------------------
# release ids
# ---------------------------------------------------------------------------

def _release_numbers(alias, root=RELEASE_ROOT):
    """``[(number, release_id), ...]`` for one alias under ``root``, ascending."""
    out = []
    pat = re.compile(r'^%s-r(\d{4})$' % re.escape(alias))
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            m = pat.match(name)
            if m and os.path.isdir(os.path.join(root, name)):
                out.append((int(m.group(1)), name))
    return sorted(out)


def next_release_id(alias, root=RELEASE_ROOT):
    """``<alias>-r<NNNN>``, one past the highest already present under ``root``."""
    numbers = _release_numbers(alias, root)
    highest = numbers[-1][0] if numbers else 0
    return '%s-r%04d' % (alias, highest + 1)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- a release records
# the curation entries logged since the alias's previous release, so "what changed about
# the labels between these two models" is answerable from the manifests | date: 2026-08-06]
def previous_release_id(alias, root=RELEASE_ROOT, exclude=None):
    """The alias's highest-numbered existing release, or ``None``.

    By directory number rather than by the registry's ``current``: this is asked
    while *building* release N, before anything is promoted, and the question is
    "which release last froze this alias's labels", not "which one is routed".
    ``exclude`` drops the release currently being built -- its directory already
    exists by the time the manifest is assembled.
    """
    for _n, rid in reversed(_release_numbers(alias, root)):
        if rid != exclude:
            return rid
    return None


def curation_section(project, alias, root=RELEASE_ROOT, exclude=None):
    """``[curation]`` for a release: what the labels did since the previous one.

    Reads ``<project>/curation_history.log`` (written by the ``curate`` action via
    ``dlc_train_guard.append_curation_log``) and keeps the entries at or after the
    previous release's ``created_utc``.  ``state`` says which situation produced
    the result, so an empty ``entries`` list is never ambiguous:

    * ``no_log`` -- the project has never been curated through the ``curate``
      action (every project predates it, so this is the normal state for a while);
    * ``no_previous_release`` -- first release for this alias, so the whole log
      describes it;
    * ``since_previous_release`` -- the usual case;
    * ``log_unreadable`` -- a corrupt line.  The release is still cut (the weights
      are the irreplaceable part) and the problem is reported in
      ``provenance_warning`` rather than silently read as "no curation".

    The boundary is inclusive: an entry stamped in the same second as the previous
    release appears in both sections.  An overlap is recoverable by eye; a gap
    would be a curation no release records at all.

    Returns ``(section, warning)``.
    """
    from nems_lbhb.motor import dlc_train_guard

    log_path = os.path.join(project, dlc_train_guard.CURATION_LOG_NAME)
    prev = previous_release_id(alias, root, exclude=exclude)
    prev_created = ''
    if prev:
        try:
            prev_created = str(read_manifest(prev, root=root)
                               .get('release', {}).get('created_utc', ''))
        except (FileNotFoundError, ImportError, ValueError) as exc:
            log.warning('could not read %s of the previous release %s: %s',
                        MANIFEST_NAME, prev, exc)
            prev_created = ''

    section = {
        'log_present': os.path.isfile(log_path),
        'log_path': log_path,
        'previous_release': prev or '',
        'previous_created_utc': prev_created,
        'entries': [],
        'entry_count': 0,
        'folders_added': [],
        'folders_changed': [],
        'frames_counted': True,
    }

    if not section['log_present']:
        section['state'] = 'no_log'
        section['note'] = (
            'no %s in %s: this project has not been curated through the `curate` '
            'action, so what changed about its labels before this release is not '
            'recorded anywhere.' % (dlc_train_guard.CURATION_LOG_NAME, project))
        return section, ''

    try:
        entries = dlc_train_guard.read_curation_log(project)
    except RuntimeError as exc:
        section['state'] = 'log_unreadable'
        section['note'] = str(exc)
        warning = ('curation not recorded in this release: %s'
                   % str(exc).strip().splitlines()[0])
        return section, warning

    if prev and prev_created:
        kept = [e for e in entries if str(e.get('utc', '')) >= prev_created]
        section['state'] = 'since_previous_release'
        section['note'] = ('curation entries at or after %s, when %s was cut'
                           % (prev_created, prev))
    else:
        kept = list(entries)
        section['state'] = 'no_previous_release'
        section['note'] = ('no earlier release exists for alias %r, so the whole '
                           'curation log describes this one' % alias)
        if prev and not prev_created:
            section['state'] = 'previous_release_undated'
            section['note'] = ('release %s records no created_utc, so the whole '
                               'curation log is kept rather than a window of it' % prev)

    frames = 0
    folders_added, folders_changed = set(), set()
    for entry in kept:
        added = entry.get('frames_added')
        if added is None or entry.get('frames_counted') is False:
            section['frames_counted'] = False
        elif section['frames_counted']:
            frames += int(added)
        folders_added.update(entry.get('folders_added') or [])
        folders_changed.update(entry.get('folders_changed') or [])
        section['entries'].append(_toml_safe(entry))

    section['entry_count'] = len(kept)
    section['folders_added'] = sorted(folders_added)
    section['folders_changed'] = sorted(folders_changed)
    if section['frames_counted']:
        section['frames_added'] = frames
    return section, ''


def _toml_safe(entry):
    """One log entry with the values TOML cannot hold removed or stringified.

    ``None`` has no TOML spelling, so a key whose value is unknown is *dropped*
    rather than written as 0 or "" -- the same rule
    ``dlc_train_guard.curation_delta`` follows, and for the same reason: a
    manufactured zero here would read as "this curation added no frames".
    """
    out = {}
    for key, value in sorted(entry.items()):
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = [v if isinstance(v, (bool, int, float, str)) else str(v)
                        for v in value if v is not None]
        else:
            out[key] = str(value)
    return out
# [AGENT EDIT END]


# ---------------------------------------------------------------------------
# permissions
# ---------------------------------------------------------------------------

def _chmod(path, mode):
    """chmod, returning a problem string instead of raising.

    On NFS (which is where the real release root lives) permission bits are
    advisory and chmod can fail outright.  Recording that honestly in the
    manifest is better than pretending the tree is frozen when it is not.
    """
    try:
        os.chmod(path, mode)
    except OSError as exc:
        return 'chmod %o failed on %s: %s' % (mode, path, exc)
    return None


def _freeze_files(rdir):
    problems = []
    for rp in _iter_release_files(rdir):
        problem = _chmod(os.path.join(rdir, rp), 0o444)
        if problem:
            problems.append(problem)
    return problems


def _freeze_dirs(rdir):
    """Directories last and deepest-first, or the writer locks itself out."""
    problems = []
    dirs = []
    for dirpath, _dirnames, _filenames in os.walk(rdir):
        dirs.append(dirpath)
    for d in sorted(dirs, key=len, reverse=True):
        problem = _chmod(d, 0o555)
        if problem:
            problems.append(problem)
    return problems


def _unfreeze(path):
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# manifest + checksums
# ---------------------------------------------------------------------------

def _write_manifest(rdir, manifest):
    import tomli_w                       # lazy: read paths must work without it
    tmp = os.path.join(rdir, MANIFEST_NAME + '.tmp')
    with open(tmp, 'wb') as f:
        tomli_w.dump(manifest, f)
    os.replace(tmp, os.path.join(rdir, MANIFEST_NAME))


def _write_sums(rdir):
    lines = []
    for rp in _iter_release_files(rdir):
        if rp == SUMS_NAME:
            continue
        lines.append('%s  %s\n' % (_sha256(os.path.join(rdir, rp)), rp))
    tmp = os.path.join(rdir, SUMS_NAME + '.tmp')
    with open(tmp, 'w') as f:
        f.writelines(lines)
    os.replace(tmp, os.path.join(rdir, SUMS_NAME))


def _read_sums(rdir):
    """``[(sha256, relpath), ...]`` from ``SHA256SUMS``."""
    path = os.path.join(rdir, SUMS_NAME)
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            out.append((line[:64], line[66:]))
    return out


def read_manifest(release_id_or_path, root=RELEASE_ROOT):
    """Parse a release's ``RELEASE.toml`` and return it as a dict."""
    rdir = _release_dir(release_id_or_path, root)
    path = os.path.join(rdir, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError('no %s in %s' % (MANIFEST_NAME, rdir))
    with open(path, 'rb') as f:
        return _toml_parser().load(f)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _copy(src, dst):
    dstdir = os.path.dirname(dst)
    if not os.path.isdir(dstdir):
        os.makedirs(dstdir)
    shutil.copyfile(src, dst)


def _labels_fingerprint(rdir, label_rel_paths):
    """sha256 over sorted ``(relpath, size, sha256)`` of every frozen label file.

    Content-hashed, not mtime-hashed like
    ``dlc_headfix_qc.label_files_fingerprint``, because it has to survive the
    copy into the release and still equal the source's fingerprint.
    """
    parts = []
    for rp in sorted(label_rel_paths):
        full = os.path.join(rdir, rp)
        parts.append('%s:%d:%s' % (rp, os.path.getsize(full), _sha256(full)))
    return hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()


def _count_label_frames(paths):
    """Rows across the frozen ``CollectedData_*.h5`` files, or ``(0, warning)``.

    pandas is imported lazily and its absence (or an unreadable file) is
    reported in ``provenance_warning`` rather than silently recorded as zero.
    """
    try:
        import pandas as pd
    except ImportError:
        return 0, 'label_frames not counted: pandas is not installed in the build environment'
    total = 0
    for p in paths:
        try:
            total += len(pd.read_hdf(p))
        except Exception as exc:
            # only the first line: an h5py failure drags a 20-line HDF5 back trace
            # along with it, and the manifest is meant to stay readable.
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else ''
            return 0, 'label_frames not counted: could not read %s (%s: %s)' % (
                os.path.basename(p), type(exc).__name__, first)
    return total, ''


# [AGENT EDIT START | agent: claude | user: wingertj | reason: review finding 2 --
# net_type.replace('_','') is only correct for DLC's plain-resnet branch of
# auxiliaryfunctions.get_scorer_name; refuse rather than silently mis-name
# output_suffix for multi_stage/mobilenet/efficientnet nets | date: 2026-08-06]
_RESNET_RE = re.compile(r'^resnet_\d+$')


def _net_suffix(net_type, multi_stage=False):
    """The network-name segment of DLC's scorer string, for a plain resnet only.

    DeepLabCut's ``auxiliaryfunctions.get_scorer_name`` branches on the model:
    ``'dlcrnetms5'`` when ``multi_stage`` is true, ``'mobnet_<int(x*100)>'`` for
    mobilenet, ``'effnet_<b>'`` for efficientnet, and only
    ``net_type.replace('_', '')`` for a plain resnet. Every project on disk today
    is ``net_type: resnet_50, multi_stage: false``, so the plain-resnet form is
    correct for everything this module has ever routed -- but guessing it for
    anything else would silently produce a wrong ``output_suffix`` and point an
    alias at a file that will never exist. Refuse instead.
    """
    if multi_stage:
        raise ValueError(
            "cannot compute output_suffix for a multi_stage net: DeepLabCut's "
            "get_scorer_name uses the literal 'dlcrnetms5' in that case, not "
            "net_type.replace('_', ''), and this module does not reproduce that "
            "branch. Extend _net_suffix if a multi_stage release is genuinely "
            "needed.")
    net_type = str(net_type)
    if not _RESNET_RE.match(net_type):
        raise ValueError(
            "cannot compute output_suffix for net_type %r: only plain resnet_<N> "
            "nets are supported. DeepLabCut's get_scorer_name uses a different "
            "naming scheme for mobilenet ('mobnet_<int>') and efficientnet "
            "('effnet_<b>') that this module does not reproduce -- guessing here "
            "would silently point an alias at a file that will never exist. "
            "Extend _net_suffix if a mobilenet/efficientnet release is genuinely "
            "needed." % net_type)
    return net_type.replace('_', '')
# [AGENT EDIT END]


def _read_evaluation(project, iteration, snapshot, shuffle, train_pct):
    """``(dict, src_csv_path_or_None)`` from ``evaluation-results``.

    Always returns a dict carrying an explicit ``available`` flag; an
    un-evaluated model is a fact worth recording, not a hole to leave out.
    """
    src = os.path.join(project, 'evaluation-results', 'iteration-%d' % iteration,
                       'CombinedEvaluation-results.csv')
    if not os.path.isfile(src):
        return {'available': False,
                'note': 'no CombinedEvaluation-results.csv for iteration-%d' % iteration}, None

    with open(src) as f:
        rows = [{(k or '').strip(): (v or '').strip() for k, v in row.items()}
                for row in csv.DictReader(f)]

    def _f(row, key):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return float('nan')

    for row in rows:
        try:
            same = (int(float(row['Training iterations:'])) == snapshot
                    and int(float(row['Shuffle number'])) == shuffle
                    and int(float(row['%Training dataset'])) == train_pct)
        except (KeyError, TypeError, ValueError):
            continue
        if same:
            out = {'available': True,
                   'train_error_px': _f(row, 'Train error(px)'),
                   'test_error_px': _f(row, 'Test error(px)'),
                   'p_cutoff': _f(row, 'p-cutoff used'),
                   'train_error_px_pcutoff': _f(row, 'Train error with p-cutoff'),
                   'test_error_px_pcutoff': _f(row, 'Test error with p-cutoff')}
            out = {k: v for k, v in out.items()
                   if not (isinstance(v, float) and v != v)}   # drop NaN: TOML has none
            out['available'] = True
            return out, src

    return {'available': False,
            'note': ('CombinedEvaluation-results.csv for iteration-%d has no row for '
                     'snapshot %d / shuffle %d / trainset %d'
                     % (iteration, snapshot, shuffle, train_pct))}, src


def build_release(project, alias, snapshot, root=RELEASE_ROOT, iteration=None,
                  shuffle=1, note='', provenance_warning='', dry_run=False):
    """Freeze one snapshot of a DLC training project into an immutable release.

    Returns the release id (``<alias>-r<NNNN>``).  With ``dry_run=True`` the
    snapshot is still located and validated -- so the call still fails on a
    missing or ambiguous snapshot -- but nothing is written.
    """
    project = os.path.abspath(project)
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: review finding 1 --
    # a relative root left project_path (and therefore RELEASE.toml paths) relative,
    # which makes DLC's read_config rewrite the 0444 config on first read and break
    # the release's own checksums | date: 2026-08-06]
    root = os.path.abspath(root)
    # [AGENT EDIT END]
    src_config = os.path.join(project, 'config.yaml')
    if not os.path.isfile(src_config):
        raise FileNotFoundError('%s is not a DLC project: no config.yaml' % project)

    train_dir, model_dir_name, src_iteration = find_snapshot(
        project, snapshot, iteration=iteration, shuffle=shuffle)

    m = _MODEL_DIR_RE.match(model_dir_name)
    if m is None:
        raise ValueError(
            "model directory %r does not look like '<Task><date>-trainset<F>shuffle<S>'; "
            "the release layout and output_suffix are both derived from that name"
            % model_dir_name)
    train_pct = int(m.group('frac'))
    train_fraction = train_pct / 100.0

    rid = next_release_id(alias, root)
    if dry_run:
        return rid

    rdir = os.path.join(root, rid)
    if not os.path.isdir(root):
        os.makedirs(root)
    os.makedirs(rdir)                     # refuse to overwrite an existing release

    try:
        return _build_into(rdir, rid, alias, project, src_config, train_dir,
                           model_dir_name, src_iteration, snapshot, shuffle,
                           train_pct, train_fraction, note, provenance_warning)
    except BaseException:
        # a half-written release is worse than none: it would verify as "clean"
        # against its own truncated SHA256SUMS.
        for dirpath, _dirnames, filenames in os.walk(rdir):
            _chmod(dirpath, 0o755)
            for name in filenames:
                _unfreeze(os.path.join(dirpath, name))
        shutil.rmtree(rdir, ignore_errors=True)
        raise


def _build_into(rdir, rid, alias, project, src_config, train_dir, model_dir_name,
                src_iteration, snapshot, shuffle, train_pct, train_fraction,
                note, provenance_warning):
    warnings = [provenance_warning] if provenance_warning else []

    # --- the one snapshot, always re-homed under iteration-0 -----------------
    rel_model = os.path.join('dlc-models', 'iteration-0', model_dir_name)
    rel_train = os.path.join(rel_model, 'train')
    for ext in SNAPSHOT_EXTS:
        name = 'snapshot-%d%s' % (snapshot, ext)
        _copy(os.path.join(train_dir, name), os.path.join(rdir, rel_train, name))
    src_train_cfg = os.path.join(train_dir, 'pose_cfg.yaml')
    if os.path.isfile(src_train_cfg):
        _copy(src_train_cfg, os.path.join(rdir, rel_train, 'pose_cfg.yaml'))
    else:
        # DeepLabCut cannot load the release without this file; the release is
        # still cut (the weights are the irreplaceable part) but it is labelled
        # broken rather than quietly shipped as if it were runnable.
        warnings.append('INCOMPLETE RELEASE: no train/pose_cfg.yaml in %s -- DeepLabCut '
                        'cannot load this release as-is' % train_dir)

    src_test_cfg = os.path.join(os.path.dirname(train_dir), 'test', 'pose_cfg.yaml')
    if os.path.isfile(src_test_cfg):
        _copy(src_test_cfg, os.path.join(rdir, rel_model, 'test', 'pose_cfg.yaml'))
    else:
        warnings.append('no test/pose_cfg.yaml in %s' % os.path.dirname(train_dir))

    pose_cfg = {}
    copied_train_cfg = os.path.join(rdir, rel_train, 'pose_cfg.yaml')
    if os.path.isfile(copied_train_cfg):
        with open(copied_train_cfg) as f:
            pose_cfg = yaml.safe_load(f) or {}

    with open(src_config) as f:
        cfg = yaml.safe_load(f) or {}

    # --- output_suffix, reconstructed from the COPIED artifacts --------------
    # DLC 2.3.6: scorer = "DLC_" + net_type.replace('_','') + Task + date
    #                     + "shuffle" + shuffle + "_" + iters
    # Deriving it here (rather than from dlc_alias.dlc_scorer_suffix, which
    # imports DeepLabCut) keeps build and verify usable in a plain python env,
    # and means the string describes what is actually in the release.
    net_type = pose_cfg.get('net_type') or cfg.get('net_type')
    if not net_type:
        raise ValueError(
            'cannot determine net_type for %s: neither the training pose_cfg.yaml nor '
            "%s carries one, so the model's DLC scorer string is unknowable"
            % (model_dir_name, src_config))
    multi_stage = bool(pose_cfg.get('multi_stage') or cfg.get('multi_stage') or False)
    net = _net_suffix(net_type, multi_stage=multi_stage)
    taskdate = _MODEL_DIR_RE.match(model_dir_name).group('taskdate')
    output_suffix = 'DLC_%s_%sshuffle%d_%d' % (net, taskdate, shuffle, snapshot)
    assert output_suffix and output_suffix.endswith('_%d' % snapshot), (
        'computed output_suffix %r is not a usable DLC scorer for snapshot %d'
        % (output_suffix, snapshot))

    # --- frozen labels -------------------------------------------------------
    label_rel = []
    label_folders = []
    for src in sorted(glob.glob(os.path.join(project, 'labeled-data', '*',
                                             'CollectedData_*.h5'))):
        folder = os.path.basename(os.path.dirname(src))
        rp = os.path.join('labels', folder, os.path.basename(src))
        _copy(src, os.path.join(rdir, rp))
        label_rel.append(rp)
        label_folders.append(folder)
    if not label_rel:
        warnings.append('no labeled-data/*/CollectedData_*.h5 found in %s' % project)
        os.makedirs(os.path.join(rdir, 'labels'))
    labels_fingerprint = _labels_fingerprint(rdir, label_rel)
    label_frames, frames_warning = _count_label_frames(
        [os.path.join(rdir, rp) for rp in label_rel])
    if frames_warning:
        warnings.append(frames_warning)

    # --- side artifacts, copied when present ---------------------------------
    evaluation, eval_csv = _read_evaluation(project, src_iteration, snapshot,
                                            shuffle, train_pct)
    if eval_csv:
        _copy(eval_csv, os.path.join(rdir, 'evaluation.csv'))
    for src_name, dst_name in (('log.txt', 'train_log.txt'),
                               ('learning_stats.csv', 'learning_stats.csv')):
        src = os.path.join(train_dir, src_name)
        if os.path.isfile(src):
            _copy(src, os.path.join(rdir, dst_name))

    # --- the release config --------------------------------------------------
    cfg['project_path'] = rdir          # already correct => read_config never rewrites it
    cfg['iteration'] = 0                  # the copied model always lives in iteration-0
    cfg['snapshotindex'] = 0              # exactly one snapshot, so index 0 is unambiguous
    cfg['TrainingFraction'] = [train_fraction]   # only the shipped fraction; index 0 is it
    with open(os.path.join(rdir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    git_hash, git_warning = _git_hash(os.path.dirname(os.path.abspath(__file__)))
    if git_warning:
        warnings.append(git_warning)

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- record what
    # the labels did between the alias's previous release and this one. `rdir`'s parent
    # is the release root, and `rid`'s own directory already exists by now, so it is
    # excluded when looking for the predecessor | date: 2026-08-06]
    curation, curation_warning = curation_section(
        project, alias, root=os.path.dirname(rdir), exclude=rid)
    if curation_warning:
        warnings.append(curation_warning)
    # [AGENT EDIT END]

    training = {}
    for key in _TRAINING_KEYS:
        if key in pose_cfg and pose_cfg[key] is not None:
            training[key] = pose_cfg[key]

    manifest = {
        'release': {
            'id': rid,
            'alias': alias,
            # datetime.utcnow() is deprecated (removed behaviour) from py3.12;
            # this stays timezone-aware and still runs on py3.8.
            'created_utc': datetime.datetime.now(datetime.timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%SZ'),
            'created_by': os.environ.get('USER') or _getuser(),
            'host': socket.gethostname(),
            'note': note,
            'provenance_warning': '',     # filled in below, after the chmod pass
            'git_hash': git_hash,
        },
        'source': {
            'project': project,
            'iteration': src_iteration,
            'shuffle': shuffle,
            'train_fraction': train_fraction,
            'snapshot': snapshot,
            'labels_fingerprint': labels_fingerprint,
            'label_folders': sorted(set(label_folders)),
            'label_frames': label_frames,
        },
        'model': {
            'net_type': str(net_type),
            'bodyparts': list(cfg.get('bodyparts', [])),
            'output_suffix': output_suffix,
            'dlc_version': _pkg_version('deeplabcut'),
            'tf_version': _pkg_version('tensorflow') or _pkg_version('tensorflow-gpu'),
        },
        'training': training,
        'evaluation': evaluation,
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: Task 8 -- was an
        # empty table; now the curation entries logged since the alias's previous release
        # | date: 2026-08-06]
        'curation': curation,
        # [AGENT EDIT END]
        'files': {},                      # sizes, filled in below
    }

    # Sizes are what fast verify compares against.  RELEASE.toml cannot record
    # its own size (writing the number changes it) and SHA256SUMS is written
    # afterwards, so both are presence-only in fast mode and fully hashed in
    # full mode like everything else.
    #
    # [AGENT EDIT | reason: review minor finding -- a 'count' computed here would
    # be wrong (RELEASE.toml and SHA256SUMS do not exist on disk yet, so it would
    # undercount the finished tree by 2); dropped rather than recomputed after
    # those two files land, since nothing reads it]
    manifest['files'] = {
        'size': {rp: os.path.getsize(os.path.join(rdir, rp))
                 for rp in _iter_release_files(rdir)},
    }

    manifest['release']['provenance_warning'] = '; '.join(w for w in warnings if w)
    _write_manifest(rdir, manifest)
    _write_sums(rdir)

    # --- freeze, last of all -------------------------------------------------
    problems = _freeze_files(rdir)
    problems += _freeze_dirs(rdir)
    if problems:
        # chmod is advisory on NFS.  Say so in the manifest rather than let the
        # release claim an immutability it does not have.  Nothing was frozen,
        # so the rewrite below is possible by construction.
        log.warning('release %s could not be made read-only: %s', rid, '; '.join(problems))
        warnings.extend(problems)
        manifest['release']['provenance_warning'] = '; '.join(w for w in warnings if w)
        _unfreeze(os.path.join(rdir, MANIFEST_NAME))
        _unfreeze(os.path.join(rdir, SUMS_NAME))
        _write_manifest(rdir, manifest)
        _write_sums(rdir)
        _chmod(os.path.join(rdir, MANIFEST_NAME), 0o444)
        _chmod(os.path.join(rdir, SUMS_NAME), 0o444)

    return rid


def _getuser():
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def verify(release_id_or_path, full=False, root=RELEASE_ROOT):
    """Check a release against its own ``SHA256SUMS``.

    ``full=False`` checks presence and recorded size of every listed file;
    ``full=True`` re-hashes.  Returns a list of human-readable problem strings,
    empty when the release is clean.

    ``RELEASE.toml`` and ``SHA256SUMS`` themselves are presence-only checks in
    fast mode: ``RELEASE.toml`` cannot record its own size (writing the number
    would change it) and ``SHA256SUMS`` is written after the size table is
    computed, so neither appears in ``[files].size``.  Both are still fully
    hashed like every other file when ``full=True``.
    """
    rdir = _release_dir(release_id_or_path, root)
    problems = []
    if not os.path.isdir(rdir):
        return ['release directory does not exist: %s' % rdir]
    if not os.path.isfile(os.path.join(rdir, SUMS_NAME)):
        return ['%s is missing from %s' % (SUMS_NAME, rdir)]

    sizes = {}
    if os.path.isfile(os.path.join(rdir, MANIFEST_NAME)):
        try:
            sizes = read_manifest(rdir, root).get('files', {}).get('size', {})
        except Exception as exc:
            problems.append('%s is unreadable: %s' % (MANIFEST_NAME, exc))
    else:
        problems.append('%s is missing from %s' % (MANIFEST_NAME, rdir))

    listed = _read_sums(rdir)
    for digest, rp in listed:
        full_path = os.path.join(rdir, rp)
        if not os.path.isfile(full_path):
            problems.append('%s is missing' % rp)
            continue
        if full:
            actual = _sha256(full_path)
            if actual != digest:
                problems.append('%s has changed: sha256 %s, expected %s'
                                % (rp, actual, digest))
        elif rp in sizes:
            actual = os.path.getsize(full_path)
            if actual != sizes[rp]:
                problems.append('%s has changed size: %d bytes, expected %d'
                                % (rp, actual, sizes[rp]))

    known = {rp for _d, rp in listed} | {SUMS_NAME}
    for rp in _iter_release_files(rdir):
        if rp not in known:
            problems.append('%s is present but not listed in %s' % (rp, SUMS_NAME))

    return problems


# ---------------------------------------------------------------------------
# rename (re-seal under a new alias)
# ---------------------------------------------------------------------------
# [AGENT EDIT START | agent: claude | user: wingertj | reason: consolidation task step 1 --
# aliases are being renamed to name what they track (eyecloseup->pupil, multivid->face),
# which moves the release directories too. A release is immutable, so this is an explicit
# RE-SEAL -- unfreeze, move, rewrite the two fields that name the release, regenerate
# SHA256SUMS (it covers RELEASE.toml, so editing the manifest invalidates it), re-freeze,
# re-verify -- not an edit in place. Reusable because more aliases will be renamed when the
# consolidated per-geometry models land | date: 2026-08-07]

_RELEASE_ID_RE = re.compile(r'^(?P<alias>.+)-r(?P<num>\d{4})$')

# What may name an alias.  Deliberately narrow: the alias becomes a directory
# name, a release-id prefix and a TOML table key, and it is baked into
# `<stem>.<release_id>.dlc.h5` output filenames.
_ALIAS_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')


def _unfreeze_tree(rdir):
    """Make every directory and file under ``rdir`` writable again.

    Top-down (``os.walk`` default) so a directory is writable before its own
    files are chmod'ed -- the mirror image of :func:`_freeze_dirs`, which must
    go deepest-first or it locks itself out.
    """
    problems = []
    for dirpath, _dirnames, filenames in os.walk(rdir):
        problem = _chmod(dirpath, 0o755)
        if problem:
            problems.append(problem)
        for name in filenames:
            problem = _chmod(os.path.join(dirpath, name), 0o644)
            if problem:
                problems.append(problem)
    return problems


def _size_table(rdir):
    """``{relpath: size}`` for every file except the two the manifest cannot size.

    ``RELEASE.toml`` cannot record its own size (writing the number changes it)
    and ``SHA256SUMS`` is written after the table is computed, so both are
    presence-only in fast verify.  Reproduced here so a re-sealed release's
    ``[files].size`` has exactly the same key set a freshly built one would.
    """
    return {rp: os.path.getsize(os.path.join(rdir, rp))
            for rp in _iter_release_files(rdir)
            if rp not in (MANIFEST_NAME, SUMS_NAME)}


def _reseal(rdir, manifest):
    """Write ``manifest`` + a fresh ``SHA256SUMS`` into ``rdir`` and re-freeze it.

    Mirrors the tail of :func:`_build_into`, including its honesty about NFS:
    when chmod fails the failure is appended to ``provenance_warning`` rather
    than letting the release claim an immutability it does not have.
    """
    manifest['files'] = {'size': _size_table(rdir)}
    _write_manifest(rdir, manifest)
    _write_sums(rdir)

    problems = _freeze_files(rdir)
    problems += _freeze_dirs(rdir)
    if problems:
        log.warning('release %s could not be made read-only: %s',
                    os.path.basename(rdir), '; '.join(problems))
        existing = manifest['release'].get('provenance_warning', '') or ''
        manifest['release']['provenance_warning'] = '; '.join(
            w for w in ([existing] + problems) if w)
        _unfreeze(os.path.join(rdir, MANIFEST_NAME))
        _unfreeze(os.path.join(rdir, SUMS_NAME))
        manifest['files'] = {'size': _size_table(rdir)}
        _write_manifest(rdir, manifest)
        _write_sums(rdir)
        _chmod(os.path.join(rdir, MANIFEST_NAME), 0o444)
        _chmod(os.path.join(rdir, SUMS_NAME), 0o444)


def rename_release(old_id, new_alias, root=RELEASE_ROOT, reason='', dry_run=False):
    """Re-seal release ``old_id`` under ``new_alias``, keeping its release number.

    ``eyecloseup-r0001`` + ``'pupil'`` -> ``pupil-r0001``.  The directory moves,
    ``[release].id`` and ``[release].alias`` are rewritten, a ``renamed_from``
    entry records where it came from and why, and ``SHA256SUMS`` is regenerated
    -- it covers ``RELEASE.toml``, so editing the manifest invalidates it.  The
    release is then frozen again and re-verified with ``full=True``; anything
    less would leave a half-sealed release behind.

    ``config.yaml``'s ``project_path`` is repointed at the new directory as part
    of the same re-seal.  It is not cosmetic: DeepLabCut's ``read_config``
    rewrites ``project_path`` to the config's own directory **and writes the file
    back** when the two differ, which on a 0444 config is either a crash or a
    silent checksum break.

    Returns the new release id.  ``dry_run=True`` runs every check and returns
    the id without touching anything.

    Raises
    ------
    ValueError
        Malformed ``old_id`` or ``new_alias``, or no ``reason``.
    FileExistsError
        The destination release id already exists -- release ids are never reused.
    FileNotFoundError
        No such release directory.
    RuntimeError
        The release did not verify before the rename, could not be unfrozen, or
        did not verify after it.  In the last two cases the release is restored
        to its previous name and contents.
    """
    root = os.path.abspath(root)

    m = _RELEASE_ID_RE.match(str(old_id))
    if m is None:
        raise ValueError(
            "release id %r is not '<alias>-r<NNNN>'; rename_release moves a release "
            "between aliases and has to know which part is which" % (old_id,))
    old_alias, number = m.group('alias'), m.group('num')

    if not _ALIAS_RE.match(str(new_alias)):
        raise ValueError(
            "alias %r is not usable: an alias becomes a directory name, a release-id "
            "prefix, a TOML table key and part of every output filename, so it must "
            "match %s" % (new_alias, _ALIAS_RE.pattern))

    reason = str(reason).strip()
    if not reason:
        raise ValueError(
            'rename_release needs a reason: the whole point of re-sealing rather than '
            'editing in place is that the manifest records why the id changed. Pass '
            'reason="...".')

    new_id = '%s-r%s' % (new_alias, number)
    if new_id == old_id:
        raise ValueError('%s already has alias %r; nothing to rename' % (old_id, new_alias))

    old_rdir = os.path.join(root, old_id)
    new_rdir = os.path.join(root, new_id)
    if not os.path.isdir(old_rdir):
        raise FileNotFoundError('no release directory at %s' % old_rdir)
    if os.path.exists(new_rdir):
        raise FileExistsError(
            'cannot rename %s to %s: %s already exists. Release ids are never reused -- '
            'a stale one still names whatever analyzed a session with it.'
            % (old_id, new_id, new_rdir))

    # Refuse to re-seal something that is already broken: the new SHA256SUMS
    # would bless the damage and the release would verify clean forever after.
    problems = verify(old_rdir, full=True, root=root)
    if problems:
        raise RuntimeError(
            'refusing to rename %s: it does not verify as it stands, and re-sealing it '
            'would bless the damage into a fresh SHA256SUMS:\n  %s'
            % (old_id, '\n  '.join(problems)))

    manifest = read_manifest(old_rdir, root=root)
    if dry_run:
        return new_id

    # Kept so a failure anywhere below can put the release back exactly as found.
    with open(os.path.join(old_rdir, MANIFEST_NAME), 'rb') as f:
        manifest_bytes = f.read()
    with open(os.path.join(old_rdir, SUMS_NAME), 'rb') as f:
        sums_bytes = f.read()
    config_path = os.path.join(old_rdir, 'config.yaml')
    config_bytes = None
    if os.path.isfile(config_path):
        with open(config_path, 'rb') as f:
            config_bytes = f.read()

    unfreeze_problems = _unfreeze_tree(old_rdir)
    if unfreeze_problems:
        _freeze_files(old_rdir)
        _freeze_dirs(old_rdir)
        raise RuntimeError(
            'refusing to rename %s: it could not be made writable, so the re-seal '
            'would be half-applied:\n  %s' % (old_id, '\n  '.join(unfreeze_problems)))

    os.rename(old_rdir, new_rdir)
    try:
        # config.yaml must keep pointing at its own directory -- see the docstring.
        if config_bytes is not None:
            new_config = os.path.join(new_rdir, 'config.yaml')
            with open(new_config) as f:
                cfg = yaml.safe_load(f) or {}
            cfg['project_path'] = new_rdir
            with open(new_config, 'w') as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

        release = manifest.setdefault('release', {})
        release['id'] = new_id
        release['alias'] = new_alias
        history = list(release.get('renamed_from', []))
        history.append({
            'old_id': old_id,
            'old_alias': old_alias,
            'new_id': new_id,
            'utc': datetime.datetime.now(datetime.timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%SZ'),
            'by': os.environ.get('USER') or _getuser(),
            'host': socket.gethostname(),
            'reason': reason,
        })
        release['renamed_from'] = history

        _reseal(new_rdir, manifest)

        problems = verify(new_rdir, full=True, root=root)
        if problems:
            raise RuntimeError(
                'release %s did not verify after the re-seal:\n  %s'
                % (new_id, '\n  '.join(problems)))
    except BaseException:
        # Put it back under its old name with its old bytes.  A half-renamed
        # release is worse than none: it would verify clean against a manifest
        # that names a directory it no longer lives in.
        try:
            _unfreeze_tree(new_rdir)
            if os.path.exists(new_rdir):
                os.rename(new_rdir, old_rdir)
            with open(os.path.join(old_rdir, MANIFEST_NAME), 'wb') as f:
                f.write(manifest_bytes)
            with open(os.path.join(old_rdir, SUMS_NAME), 'wb') as f:
                f.write(sums_bytes)
            if config_bytes is not None:
                with open(os.path.join(old_rdir, 'config.yaml'), 'wb') as f:
                    f.write(config_bytes)
            _freeze_files(old_rdir)
            _freeze_dirs(old_rdir)
        except Exception as exc:                       # pragma: no cover
            log.error('rename of %s failed AND could not be undone: %s', old_id, exc)
        raise

    log.info('renamed release %s -> %s (%s)', old_id, new_id, reason)
    return new_id
# [AGENT EDIT END]


# ---------------------------------------------------------------------------
# the surgical registry editor
# ---------------------------------------------------------------------------
# A parse-and-re-serialize round-trip through tomli_w would silently delete the
# 61 comment lines in dlc_registry.toml that transcribe each legacy if/elif
# condition.  tomlkit (which does preserve style) is not installed in any
# environment here, so the registry is edited line by line and the result is
# re-parsed as a guard.

def _strip_strings(line):
    """``line`` with the contents of quoted spans blanked out.

    Only used to find table headers and bracket depth, so blanking is enough --
    the string's *contents* never matter, but its brackets and '#' must not be
    mistaken for structure.
    """
    out = []
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote is None:
            if line.startswith('"""', i) or line.startswith("'''", i):
                quote = line[i:i + 3]
                out.append('   ')
                i += 3
                continue
            if ch in '"\'':
                quote = ch
                out.append(' ')
                i += 1
                continue
            if ch == '#':
                break                      # comment: nothing structural after it
            out.append(ch)
            i += 1
        else:
            if len(quote) == 3 and line.startswith(quote, i):
                quote = None
                out.append('   ')
                i += 3
                continue
            if len(quote) == 1 and ch == quote:
                quote = None
            out.append(' ')
            i += 1
    return ''.join(out)


def _bracket_delta(line):
    bare = _strip_strings(line)
    return bare.count('[') - bare.count(']') + bare.count('{') - bare.count('}')


def _alias_block_span(lines, alias):
    """``(start, end)`` line indices of the ``[alias.<name>]`` block.

    ``start`` is the header line itself; ``end`` is the first line of the next
    table (or ``len(lines)``).  Raises when the header is missing or duplicated
    -- an ambiguous registry must not be edited by guesswork.
    """
    header = '[alias.%s]' % alias
    hits = [i for i, line in enumerate(lines) if line.strip() == header]
    if not hits:
        raise ValueError(
            'alias %r has no %s block in the registry; cannot promote a release for an '
            'alias that does not exist' % (alias, header))
    if len(hits) > 1:
        raise ValueError('registry defines %s %d times (lines %s); refusing to edit an '
                         'ambiguous file'
                         % (header, len(hits), ', '.join(str(i + 1) for i in hits)))

    start = hits[0]
    depth = 0
    for j in range(start + 1, len(lines)):
        if depth <= 0 and re.match(r'^\[[^\]]*\]\s*(#.*)?$', lines[j].strip()) \
                and lines[j].startswith('['):
            return start, j
        depth += _bracket_delta(lines[j])
    return start, len(lines)


def _key_span(lines, start, end, key):
    """``(i, j)`` half-open span of ``key = ...`` inside ``[start, end)``, or None.

    Handles a value spread over several lines (``history`` is rendered one entry
    per line) by counting brackets.
    """
    pat = re.compile(r'^\s*%s\s*=' % re.escape(key))
    for i in range(start, end):
        if pat.match(lines[i]):
            depth = _bracket_delta(lines[i])
            j = i + 1
            while depth > 0 and j < end:
                depth += _bracket_delta(lines[j])
                j += 1
            return i, j
    return None


def _render_history(entries):
    if not entries:
        return ['history = []']
    out = ['history = [']
    out += ['  "%s",' % e for e in entries]
    out.append(']')
    return out


def _edit_alias_block(text, alias, current, history):
    """Return ``text`` with only this alias's ``current``/``history`` lines rewritten.

    Every other byte of the file -- comments included -- is passed through.
    """
    lines = text.split('\n')
    start, end = _alias_block_span(lines, alias)

    hist_span = _key_span(lines, start, end, 'history')
    cur_span = _key_span(lines, start, end, 'current')
    if cur_span is None:
        raise ValueError('[alias.%s] has no `current` key to set' % alias)

    new_lines = list(lines)
    # Replace/insert history first when it sits after current, so the indices of
    # the earlier edit stay valid; do them in descending order of position.
    edits = [(cur_span, ['current = "%s"' % current])]
    if hist_span is not None:
        edits.append((hist_span, _render_history(history)))
    else:
        edits.append(((cur_span[1], cur_span[1]), _render_history(history)))
    for (i, j), repl in sorted(edits, key=lambda e: e[0][0], reverse=True):
        new_lines[i:j] = repl

    return '\n'.join(new_lines)


def _write_atomic(path, text):
    d = os.path.dirname(os.path.abspath(path)) or '.'
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: mkstemp creates 0600 and
    # os.replace carries that mode onto the target, so every promote()/rollback() silently
    # stripped the registry's group/other READ bit. Queue jobs run as `lbhb` and resolve
    # their model through this file, so the next fit after a promotion died with
    # PermissionError -- the promotion succeeded and broke inference in the same call.
    # Carry the original mode across the replace. | date: 2026-08-25]
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        mode = 0o644                      # new file: readable, like every other config
    fd, tmp = tempfile.mkstemp(prefix='.dlc_registry.', suffix='.toml', dir=d)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    # [AGENT EDIT END]


def _apply_registry_edit(registry_path, alias, current, history):
    """Edit, write atomically, then re-parse and assert.  Returns a unified diff."""
    with open(registry_path) as f:
        before = f.read()

    after = _edit_alias_block(before, alias, current, history)
    _write_atomic(registry_path, after)

    # The guard.  A line editor that got the span wrong would otherwise leave a
    # corrupted registry on disk and be discovered by whichever job read it next.
    try:
        with open(registry_path, 'rb') as f:
            parsed = _toml_parser().load(f)
        block = parsed['alias'][alias]
        ok = (block.get('current') == current
              and list(block.get('history', [])) == list(history))
        problem = None if ok else ('after the edit, [alias.%s] reads current=%r history=%r'
                                   % (alias, block.get('current'), block.get('history')))
    except Exception as exc:
        problem = '%s: %s' % (type(exc).__name__, exc)

    if problem:
        _write_atomic(registry_path, before)      # restore, byte for byte
        raise RuntimeError(
            'refusing a botched registry edit for alias %r: %s. %s was restored '
            'to its previous contents.' % (alias, problem, registry_path))

    return ''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=registry_path + ' (before)', tofile=registry_path + ' (after)'))


def _registry_block(registry_path, alias):
    with open(registry_path, 'rb') as f:
        parsed = _toml_parser().load(f)
    aliases = parsed.get('alias', {})
    if alias not in aliases:
        raise ValueError('unknown DLC alias %r; %s defines: %s'
                         % (alias, registry_path, ', '.join(sorted(aliases))))
    return aliases[alias]


def promote(alias, release_id, registry_path=None, root=RELEASE_ROOT, skip_verify=False):
    """Point ``alias`` at ``release_id`` in the registry.  Returns the diff.

    Verifies the release exists and hashes clean first (``skip_verify=True``
    opts out, e.g. when editing a registry copy in a test).  Appends the id to
    ``history`` if it is not already there; never removes anything.  Does not
    commit -- it prints the git command for a human to run.
    """
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH

    block = _registry_block(registry_path, alias)

    if not skip_verify:
        rdir = _release_dir(release_id, root)
        if not os.path.isdir(rdir):
            raise FileNotFoundError(
                'cannot promote %s: no release directory at %s' % (release_id, rdir))
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: review finding 4 --
        # a missing train/pose_cfg.yaml is recorded in the manifest as
        # 'INCOMPLETE RELEASE' but is structurally clean (verify has nothing to
        # compare it against, since the file was never in SHA256SUMS), so verify()
        # alone would wave an unloadable release through | date: 2026-08-06]
        try:
            manifest = read_manifest(rdir, root)
        except FileNotFoundError:
            manifest = None
        if manifest is not None:
            warning = manifest.get('release', {}).get('provenance_warning', '') or ''
            if 'INCOMPLETE RELEASE' in warning:
                raise ValueError(
                    'cannot promote %s: its manifest records provenance_warning=%r; '
                    'an incomplete release cannot be safely deployed' % (release_id, warning))
        # [AGENT EDIT END]
        problems = verify(rdir, full=True, root=root)
        if problems:
            raise ValueError('release %s does not verify; refusing to promote:\n  %s'
                             % (release_id, '\n  '.join(problems)))

    history = list(block.get('history', []))
    if release_id not in history:
        history.append(release_id)

    diff = _apply_registry_edit(registry_path, alias, release_id, history)

    log.info('promoted %s -> %s in %s', alias, release_id, registry_path)
    print('registry updated. Review and commit it yourself:\n'
          '  git add %s && git commit -m "chore(dlc): %s -> %s"'
          % (registry_path, alias, release_id))
    return diff


def rollback(alias, registry_path=None, root=RELEASE_ROOT, skip_verify=False):
    """Point ``alias`` back at the release immediately before its current one.

    ``history`` is never truncated: rolling back is itself a fact about what was
    deployed, and the forward entries are how you roll forward again.

    Refuses (``FileNotFoundError``) when the target release's directory no
    longer exists on disk -- rolling back to a release that isn't there would
    leave the alias pointing at nothing, which is worse than refusing.
    ``skip_verify=True`` opts out (e.g. when editing a registry copy in a test
    that never cut a real release).
    """
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH

    block = _registry_block(registry_path, alias)
    current = block.get('current', '')
    history = list(block.get('history', []))

    if not current:
        raise ValueError('alias %r has no current release to roll back from' % alias)
    if current not in history:
        raise ValueError('alias %r has current=%r which is absent from history=%r; '
                         'the registry is inconsistent, fix it by hand'
                         % (alias, current, history))
    idx = history.index(current)
    if idx == 0:
        raise ValueError('alias %r is already at its oldest release (%s); nothing to '
                         'roll back to' % (alias, current))

    previous = history[idx - 1]
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: review finding 5 (minor) --
    # rollback never checked the target release directory still exists; a release can be
    # deleted from disk (retention, a bad build cleaned up) without the registry knowing,
    # and rolling back to it would silently point the alias at nothing | date: 2026-08-06]
    if not skip_verify:
        rdir = _release_dir(previous, root)
        if not os.path.isdir(rdir):
            raise FileNotFoundError(
                'cannot roll back %s to %s: no release directory at %s'
                % (alias, previous, rdir))
    # [AGENT EDIT END]
    diff = _apply_registry_edit(registry_path, alias, previous, history)
    log.info('rolled %s back: %s -> %s', alias, current, previous)
    print('registry rolled back. Review and commit it yourself:\n'
          '  git add %s && git commit -m "chore(dlc): rollback %s -> %s"'
          % (registry_path, alias, previous))
    return diff
# [AGENT EDIT END]
