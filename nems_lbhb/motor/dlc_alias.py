"""Atomic `.dlc.h5` alias management and deterministic DLC output selection.

Defect this module fixes: `dlc_fit_script.py` used to point a video's `.dlc.h5`
alias at its newest DeepLabCut output with `os.system(f"ln -s {target} {alias}")`.
Two problems followed from that one line. First, `ln -s` refuses to create a
symlink that already exists, `os.system`'s return code was discarded, and the
old target silently survived -- every re-analysis of an already-aliased video
was a downstream no-op. Second, picking the "newest" output by file mtime means
a stray `.h5` left in `sorted/` (e.g. from an abandoned shuffle or an older
DeepLabCut project) can outrank the intended one just because it happened to be
touched more recently.

`os.system("ln -s ...")` is banned in this codebase from here on. `point_alias`
replaces it with an atomic create-temp-then-`os.replace` sequence, and
`find_dlc_output` replaces mtime-based discovery with a deterministic filename
construction from the DeepLabCut scorer suffix that was actually requested.

This module is deliberately stdlib-only (no NEMS, no DeepLabCut, no numpy, no
pandas, no yaml) so it imports cleanly from the Python 3.8 `streamlit_dlc`
environment, the Python 3.9 `psi_nems2` environment, and the production `dlc5`
environment alike.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: atomic alias creation replacing os.system ln -s | date: 2026-08-06]
import os
import re

_SCORER_RE = re.compile(
    r'DLC_(?P<net>[a-zA-Z0-9]+)_(?P<task>.+?)(?P<date>[A-Z][a-z]{2}\d{1,2})'
    r'shuffle(?P<shuffle>\d+)_(?P<snapshot>\d+)'
)


def point_alias(target, alias_path, relative=True):
    """Atomically (re)point the symlink at `alias_path` to `target`.

    Replaces `os.system(f"ln -s {target} {alias}")`, which fails silently (the
    return code is discarded) whenever `alias_path` already exists, leaving the
    old target in place. Here the new link is created under a unique temporary
    name in the alias's own directory and installed with `os.replace`, which is
    atomic on POSIX: any concurrent reader sees either the old link or the new
    one, never a missing file and never a half-written one.

    Order of checks:
      1. `target` must exist, else `FileNotFoundError`.
      2. If something already sits at `alias_path` and is NOT a symlink (i.e. a
         regular file -- real data, not a pointer), raise `FileExistsError`
         rather than silently destroying it. `os.path.lexists` is used for this
         check so a dangling symlink still counts as "a symlink" and is
         replaced, not treated as a blocking regular file.
      3. Create the temp link and `os.replace` it over `alias_path`.

    `relative=True` (the default) stores the link as a relative path computed
    with `os.path.relpath(target, os.path.dirname(alias_path))`, so the alias
    keeps working if the containing directory tree is moved or renamed.

    Returns the resolved (absolute, symlink-free) path of `target`.
    """
    target_abs = os.path.abspath(target)
    if not os.path.exists(target_abs):
        raise FileNotFoundError("point_alias target does not exist: {}".format(target))

    alias_dir = os.path.dirname(os.path.abspath(alias_path)) or '.'

    if os.path.lexists(alias_path) and not os.path.islink(alias_path):
        raise FileExistsError(
            "refusing to overwrite a non-symlink at alias path: {}".format(alias_path))

    link_value = (os.path.relpath(target_abs, alias_dir) if relative
                  else target_abs)

    counter = 0
    pid = os.getpid()
    while True:
        tmp = "{}.tmp-{}-{}".format(alias_path, pid, counter)
        try:
            os.symlink(link_value, tmp)
            break
        except FileExistsError:
            counter += 1
            continue

    try:
        os.replace(tmp, alias_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

    return os.path.realpath(target_abs)


def find_dlc_output(dest_dir, video_stem, scorer_suffix):
    """Deterministically locate a DeepLabCut output file, never by mtime.

    Constructs `os.path.join(dest_dir, video_stem + scorer_suffix + '.h5')` and
    asserts it is a regular file. This is a construction, not a glob or a
    "pick the newest" scan -- determinism is the entire point, since picking by
    mtime lets a stray `.h5` from an old or abandoned shuffle silently become
    the alias target.

    `scorer_suffix` is passed without a leading separator and without the
    `.h5` extension, e.g. `'DLC_resnet50_two_chimney3fbJan10shuffle1_60000'`.

    Raises `FileNotFoundError` listing the `*.h5` files actually present in
    `dest_dir` when the constructed path does not exist.
    """
    candidate = os.path.join(dest_dir, video_stem + scorer_suffix + '.h5')
    if os.path.isfile(candidate):
        return candidate

    try:
        present = sorted(
            f for f in os.listdir(dest_dir) if f.endswith('.h5')
        )
    except OSError:
        present = []
    raise FileNotFoundError(
        "expected DLC output not found: {} (*.h5 present in {}: {})".format(
            candidate, dest_dir, present))


def dlc_scorer_suffix(path_config, shuffle=1, train_fraction=None):
    """Return the scorer suffix DeepLabCut will append to its outputs.

    e.g. `'DLC_resnet50_two_chimney3fbJan10shuffle1_60000'`, which is exactly
    what `find_dlc_output` expects -- these two are a pair, which is why they
    live together.

    Asks DeepLabCut rather than reimplementing its naming, so this cannot
    drift. `auxiliaryfunctions.get_scorer_name` with
    `trainingsiterations='unknown'` resolves the real snapshot number by the
    identical procedure `analyze_videos` uses (read `snapshotindex`, list
    `iteration-<n>/<model folder>/train/*.index`, sort numerically, index), so
    the two cannot disagree.

    NOTE: this is the one function in this module that needs DeepLabCut. The
    import is deliberately inside the body so the module itself stays
    importable in environments without DLC.

    NOTE: `auxiliaryfunctions.read_config` rewrites `project_path` to the
    config file's own directory and writes the file back -- but only when the
    two differ, so a release whose config already points at itself is not
    written to, and can safely be read-only.
    """
    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(path_config)
    if train_fraction is None:
        train_fraction = cfg['TrainingFraction'][0]
    dlc_scorer, _ = auxiliaryfunctions.GetScorerName(
        cfg, shuffle, train_fraction, trainingsiterations='unknown')
    return dlc_scorer


def parse_scorer(h5_path):
    """Parse a DeepLabCut output filename into its scorer components.

    Returns a dict with keys `scorer`, `net`, `task`, `date`, `shuffle`,
    `snapshot`. `scorer` is the whole matched substring (e.g.
    `'DLC_resnet50_two_chimney3fbJan10shuffle1_60000'`); `shuffle` and
    `snapshot` are ints. When `h5_path`'s basename does not match the
    DeepLabCut naming convention (e.g. an already-aliased `*.dlc.h5`), every
    key is `None`.
    """
    name = os.path.basename(h5_path)
    m = _SCORER_RE.search(name)
    if not m:
        return {
            'scorer': None, 'net': None, 'task': None,
            'date': None, 'shuffle': None, 'snapshot': None,
        }
    return {
        'scorer': m.group(0),
        'net': m.group('net'),
        'task': m.group('task'),
        'date': m.group('date'),
        'shuffle': int(m.group('shuffle')),
        'snapshot': int(m.group('snapshot')),
    }


def stamped_alias_name(video_stem, release_id):
    """Alias filename for `video_stem` under a specific model `release_id`.

    e.g. `stamped_alias_name('SLJ017a07_a_NTD', 'two_chimney-r0001')` ->
    `'SLJ017a07_a_NTD.two_chimney-r0001.dlc.h5'`.
    """
    return "{}.{}.dlc.h5".format(video_stem, release_id)


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the release id has to reach
# the recording cache hash, so a loader needs to answer "which model produced the file this
# alias currently points at?" from the filesystem alone | date: 2026-08-12]
# A release-stamped alias name, e.g. 'SLJ017a07_a_NTD.two_chimney-r0001', matched against
# the stem (the '.dlc.h5' suffix already stripped).  Canonical here because
# `stamped_alias_name` above is what writes it; scripts/dlc_sync_fits.py imports it.
STAMPED_RE = re.compile(r'^(?P<base>.+)\.(?P<alias>[A-Za-z0-9_]+)-r(?P<num>\d{4})$')


def parse_stamped_alias(name):
    """`('SLJ017a07_a_NTD', 'two_chimney-r0001')` for a stamped alias, else `(stem, None)`.

    Accepts a bare stem, a basename or a full path; a trailing `.dlc.h5` is stripped.
    """
    stem = os.path.basename(str(name))
    if stem.endswith('.dlc.h5'):
        stem = stem[: -len('.dlc.h5')]
    m = STAMPED_RE.match(stem)
    if not m:
        return stem, None
    return m.group('base'), '%s-r%s' % (m.group('alias'), m.group('num'))


# (suffix, stream name). Mutually exclusive in practice -- no stem carries two.
STREAM_SUFFIXES = (('_camB', 'camB'), ('_camC', 'camC'), ('.lick', 'lick'), ('_pupil', 'pupil'))


def stream_for(stem):
    """'camB'/'camC'/'lick'/'pupil' when `stem` ends with the matching suffix, else ''."""
    for suffix, name in STREAM_SUFFIXES:
        if str(stem).endswith(suffix):
            return name
    return ''


def strip_stream_suffix(stem):
    """`stem` with a trailing stream suffix removed -- the key a session is known by."""
    stem = str(stem)
    for suffix, _name in STREAM_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def analysed_stem(target_path):
    """The video stem DeepLabCut named its output after: `<stem><scorer>.h5` -> `<stem>`.

    `None` when the name is not a scorer-named DLC output (a dangling link, or an
    alias that resolves to another alias).
    """
    name = os.path.basename(str(target_path))
    scorer = parse_scorer(name)['scorer']
    if not scorer or scorer not in name:
        return None
    return name[: name.index(scorer)]


def stream_of(alias_stem, target_path):
    """Which camera produced a file, read from the TARGET rather than the alias name.

    The alias name agrees with the target for every file written before per-session
    camera overrides existed. It disagrees for exactly the case an override creates: the
    ceiling view sat on a side-camera slot, so a SESSION-named alias -- the name every
    loader asks for -- points at a `_camC`-named target. Reading the alias there would
    record the base camera and align the trace to the wrong frame counts.

    Falls back to `alias_stem` when the target cannot be parsed, which is the honest
    answer for a dangling link.
    """
    stem = analysed_stem(target_path)
    return stream_for(stem if stem is not None else alias_stem)


def release_for_alias(alias_path):
    """Which release produced the file `alias_path` currently points at, or `None`.

    `dlc_fit_script` writes two links per analysed video: `<stem>.<release>.dlc.h5`
    (durable, one per release ever run) and `<stem>.dlc.h5` (the "current" pointer,
    atomically repointed).  The plain alias is the one every loader opens and the one
    that is authoritative about *what is current*, but it carries no provenance in its
    own name.  This recovers the provenance by asking which stamped sibling resolves to
    the same file.

    Deliberately not "parse the newest stamped alias": if the current pointer was moved
    back to an older release, the stamp that matches it is the older one, and that is the
    honest answer.  Also deliberately not the scorer string, which omits the iteration --
    two releases of one project can produce byte-identical scorer names from different
    weights.

    Returns `None` for a legacy (unstamped) file, a dangling link, or a real file that is
    not a link at all.  When the alias is itself a stamped name, that stamp is returned
    directly.  When more than one stamp resolves to the same target the ids are returned
    joined by '+' and sorted, so the answer stays deterministic and the ambiguity is
    visible rather than silently resolved.
    """
    _base, own = parse_stamped_alias(alias_path)
    if own is not None:
        return own
    try:
        target = os.path.realpath(alias_path)
        if not os.path.exists(target):
            return None
        directory = os.path.dirname(os.path.abspath(alias_path))
        stem = os.path.basename(str(alias_path))
        if stem.endswith('.dlc.h5'):
            stem = stem[: -len('.dlc.h5')]
        prefix, tail = stem + '.', '.dlc.h5'
        found = []
        for name in os.listdir(directory):
            if not (name.startswith(prefix) and name.endswith(tail)):
                continue
            _b, rid = parse_stamped_alias(name)
            if rid is None:
                continue
            if os.path.realpath(os.path.join(directory, name)) == target:
                found.append(rid)
    except OSError:
        return None
    if not found:
        return None
    return '+'.join(sorted(set(found)))
# [AGENT EDIT END]


def audit(paths, expected=None):
    """Report the alias state of every path in `paths`, without modifying any of them.

    Returns one dict per path with keys `alias, target, exists, broken,
    is_link, scorer, task, date, shuffle, snapshot, expected, stale`.

    - `is_link` is `os.path.islink(alias)`.
    - `exists` is `os.path.exists(alias)` (False for a dangling link, since
      `os.path.exists` follows the link).
    - `target` is `os.readlink(alias)` when `alias` is a link, else `None`.
    - `broken` is `is_link and not exists` -- a symlink whose target is gone.
    - The scorer fields (`scorer`/`net`/`task`/`date`/`shuffle`/`snapshot`) come
      from `parse_scorer` applied to the resolved target's basename, not to the
      alias's own name (the alias is named `<stem>.dlc.h5` and carries no
      scorer info itself). For a broken link the readlink string is still
      resolved and parsed even though the file is absent, so `scorer`/
      `snapshot`/etc. are still reported when parseable; they are `None` when
      the target name does not match the DeepLabCut naming convention.
    - `expected` is the `expected` argument (unchanged, for the caller's
      convenience); `stale` is `None` when `expected is None`, else
      `scorer != expected`.

    Repair is intentionally out of scope here -- `audit` only reports.
    """
    rows = []
    for alias in paths:
        is_link = os.path.islink(alias)
        exists = os.path.exists(alias)
        broken = is_link and not exists

        target = os.readlink(alias) if is_link else None

        if target is not None:
            scorer_fields = parse_scorer(os.path.basename(target))
        else:
            scorer_fields = parse_scorer('')

        stale = None if expected is None else (scorer_fields['scorer'] != expected)

        row = {
            'alias': alias,
            'target': target,
            'exists': exists,
            'broken': broken,
            'is_link': is_link,
            'scorer': scorer_fields['scorer'],
            'task': scorer_fields['task'],
            'date': scorer_fields['date'],
            'shuffle': scorer_fields['shuffle'],
            'snapshot': scorer_fields['snapshot'],
            'expected': expected,
            'stale': stale,
        }
        rows.append(row)
    return rows
# [AGENT EDIT END]
