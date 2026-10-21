"""Registry-driven routing from a video path to a DeepLabCut model.

The routing rules used to live in a 45-line if/elif chain in
``nems_lbhb.motor.nems_dlc_settings.get_dlc_model``.  They now live in
``nems_lbhb/configs/dlc_registry.toml``, which is git-tracked and reviewable, and
this module evaluates them.  ``get_dlc_model`` is a one-line shim over
:func:`config_for_video`.

``tests/test_dlc_registry.py`` holds a frozen copy of the pre-migration function and
asserts that this module reproduces it -- same answer, and same *exception type* --
on every video in ``/auto/data/dlc/inventory/dlc_inventory.csv``.  Several of the
legacy function's quirks are therefore reproduced deliberately; see
:func:`_site_number` and the ``[known_issues]`` table in the TOML.

Deliberately dependency-free: standard library only, no NEMS, no DeepLabCut, no
pandas, numpy or yaml.  It is imported by py3.8 (``streamlit_dlc``), py3.9
(``psi_nems2``) and the production DLC environment.

Shape of the dict returned by :func:`load_registry`::

    {'schema_version':    int,
     'training_defaults': {...},
     'known_issues':      {...},
     'aliases':           {name: block} in file order,
     'alias_names':       [name, ...] in file order}

An alias block may carry an optional ``train_frame_size = [w, h]``; see
:data:`DEFAULT_TRAIN_FRAME_SIZE` and :func:`resolve`.

``'aliases'`` is the mapping of alias name -> block; ``'alias_names'`` is the
plain list of names, the convenience key callers use to enumerate.  The raw
TOML table is spelled ``[alias.<name>]`` (singular, so each block reads
naturally), but the key ``'alias'`` is deliberately *removed* from the returned
dict: leaving both ``'alias'`` and ``'aliases'`` in place put two keys one
character apart holding the same mapping, which is a footgun for every caller.
"""
# [AGENT EDIT START | agent: claude | user: wingertj | reason: new module -- DLC model routing moved out of the nems_dlc_settings if/elif chain into a git-tracked registry | date: 2026-08-06]
import os

try:                       # python >= 3.11
    import tomllib
except ImportError:        # python <= 3.10
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


DEFAULT_REGISTRY_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'dlc_registry.toml'))

# Where a frozen release lives once Task 6 cuts one.  Until an alias's `current`
# field is non-empty, resolve() points at the live training project instead.
RELEASE_ROOT = '/auto/data/dlc/releases'

# [AGENT EDIT START | agent: claude | user: wingertj | reason: resolution-based inference rescaling -- an alias may declare the frame size its labelled data was at | date: 2026-08-08]
# Every labelled frame in every DLC project in this lab is 640x480 (all 17
# projects, checked), so this is the value used when an alias says nothing.  It
# is a property of a model's TRAINING DATA, which is why it lives on the alias
# rather than being hardcoded: the day someone trains on frames of a different
# size, exactly one line of TOML has to change and every video routed to that
# alias is rescaled accordingly.
DEFAULT_TRAIN_FRAME_SIZE = (640, 480)
# [AGENT EDIT END]

# Predicates a rule may carry.  Anything else in a rule is a registry authoring
# error and raises rather than being silently ignored.
_RULE_KEYS = ('path_contains', 'site_max', 'site_min', 'site_in', 'site_not_in',
              'animal', 'runclass')

# path -> (mtime, registry).  Keyed on (path, mtime): editing the TOML busts the
# entry by itself.  clear_cache() forces a reload for tests.
_CACHE = {}


def _toml_parser():
    """Return the TOML module, or raise an ImportError that says what to install."""
    if tomllib is None:
        raise ImportError(
            "reading %s needs a TOML parser: python>=3.11 has 'tomllib' in the "
            "standard library, on older pythons run `pip install tomli`"
            % DEFAULT_REGISTRY_PATH)
    return tomllib


def clear_cache():
    """Drop the parsed-registry cache.  Only needed when a test rewrites the TOML
    within the same mtime granularity."""
    _CACHE.clear()


def load_registry(path=None):
    """Parse the registry TOML and return it as a dict (see the module docstring).

    Cached on ``(path, mtime)``, so an edit to the file is picked up automatically.
    """
    if path is None:
        path = DEFAULT_REGISTRY_PATH
    path = os.path.normpath(os.path.abspath(path))

    mtime = os.path.getmtime(path)
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with open(path, 'rb') as f:
        raw = _toml_parser().load(f)

    # The TOML spells the per-model blocks `[alias.<name>]`, so the parser hands
    # them back under 'alias'.  Move them to 'aliases', expose the names in file
    # order (dicts preserve insertion order on every python this runs on), and
    # DROP 'alias' -- two keys one character apart holding the same mapping is a
    # footgun for every caller.
    aliases = raw.get('alias', {})
    registry = dict(raw)
    registry.pop('alias', None)
    registry['aliases'] = aliases
    registry['alias_names'] = list(aliases.keys())
    registry['_path'] = path

    # Validate every rule's predicate keys once, here, rather than lazily when a
    # path happens to reach that rule.  Without this, a typo in an unreachable
    # alias's rules (behavior_only, face) would never be caught by any test.
    for name, block in aliases.items():
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: validate the optional train_frame_size at load time, like the rule predicates | date: 2026-08-08]
        try:
            train_frame_size(block)
        except ValueError as e:
            raise ValueError("registry {}: alias '{}': {}".format(path, name, e))
        # [AGENT EDIT END]
        for i, rule in enumerate(block.get('rules', [])):
            unknown = sorted(set(rule) - set(_RULE_KEYS))
            if unknown:
                raise ValueError(
                    "registry {}: alias '{}' rule {} has unknown predicate(s) {} "
                    "(valid: {})".format(path, name, i, unknown, list(_RULE_KEYS)))

    # [AGENT EDIT START | agent: claude | user: wingertj | reason: validate camera overrides at load time, like the rule predicates | date: 2026-08-12]
    _validate_camera_override(path, registry.get('camera_override'))
    # [AGENT EDIT END]

    _CACHE[path] = (mtime, registry)
    return registry


_CAMERA_KEYS = ('video', 'cam_tag', 'verified_by', 'verified_date', 'note')
_CAM_TAG_FOR_VIDEO = {'recording.avi': None,
                      'video_side.avi': 'camB',
                      'video_side2.avi': 'camC'}


def camera_override(video_path, registry=None):
    """Per-session camera override, or None.

    Returns ``{'video': 'video_side2.avi', 'cam_tag': 'camC', ...}`` when this session's
    ceiling view is NOT in the file the pipeline would pick by default.  See the
    ``[camera_override]`` comment block in the registry TOML for why this cannot be
    derived from the files themselves.

    Keyed on the session stem, so both ``.../PRN013a02_a_NTD.avi`` and
    ``.../PRN013a02_a_NTD/recording.avi`` resolve to the same entry.
    """
    if registry is None:
        registry = load_registry()
    over = registry.get('camera_override') or {}
    if not over:
        return None
    p = str(video_path)
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    if base in ('recording.avi', 'video_side.avi', 'video_side2.avi'):
        stem = os.path.basename(os.path.dirname(p))     # .../<stem>/recording.avi
    return over.get(stem)


def _validate_camera_override(path, over):
    for stem, block in (over or {}).items():
        if not isinstance(block, dict):
            raise ValueError("registry {}: [camera_override.{}] must be a table"
                             .format(path, stem))
        unknown = sorted(set(block) - set(_CAMERA_KEYS))
        if unknown:
            raise ValueError("registry {}: [camera_override.{}] unknown key(s) {} "
                             "(valid: {})".format(path, stem, unknown, list(_CAMERA_KEYS)))
        for req in ('video', 'cam_tag'):
            if req not in block:
                raise ValueError("registry {}: [camera_override.{}] missing '{}'"
                                 .format(path, stem, req))
        vid = block['video']
        if vid not in _CAM_TAG_FOR_VIDEO:
            raise ValueError("registry {}: [camera_override.{}] video={!r} is not a psi "
                             "camera slot (expected one of {})"
                             .format(path, stem, vid, sorted(_CAM_TAG_FOR_VIDEO)))
        # the cam tag decides which PSIVIDEO stream aligns the trace, so a tag that
        # disagrees with the slot would mis-align every trial while looking deliberate
        expect = _CAM_TAG_FOR_VIDEO[vid]
        got = block['cam_tag'] or None
        if got != expect:
            raise ValueError("registry {}: [camera_override.{}] video={!r} is psi slot "
                             "{!r} but cam_tag={!r}; alignment would use the wrong "
                             "PSIVIDEO stream".format(path, stem, vid, expect, got))


def resolve(alias, registry=None):
    """Resolve an alias name to the model it currently serves.

    Returns a dict with keys ``alias, release_id, config, project, snapshot,
    bodyparts, qc, source, train_frame_size``.  ``source`` is ``'release'`` when
    the alias names a frozen release in its ``current`` field, otherwise
    ``'project'`` and ``config`` points at the live training project's
    ``config.yaml``.

    ``train_frame_size`` is the ``(width, height)`` of the frames this model was
    labelled on, defaulting to :data:`DEFAULT_TRAIN_FRAME_SIZE` when the alias
    does not declare one.  ``nems_lbhb.motor.dlc_video_scale`` uses it to decide
    whether a video has to be downscaled before inference.

    Raises ``KeyError`` for an unknown alias -- there is no fallback.
    """
    if registry is None:
        registry = load_registry()
    aliases = registry['aliases']
    if alias not in aliases:
        raise KeyError('unknown DLC alias %r; registry %s defines: %s'
                       % (alias, registry.get('_path', DEFAULT_REGISTRY_PATH),
                          ', '.join(registry['alias_names'])))
    block = aliases[alias]

    project = block['project']
    release_id = block.get('current', '')
    if release_id:
        config = os.path.join(RELEASE_ROOT, release_id, 'config.yaml')
        source = 'release'
    else:
        release_id = None
        config = os.path.join(project, 'config.yaml')
        source = 'project'

    return {
        'alias': alias,
        'release_id': release_id,
        'config': config,
        'project': project,
        'snapshot': block.get('snapshot'),
        'bodyparts': list(block.get('bodyparts', [])),
        'qc': block.get('qc'),
        'source': source,
        # [AGENT EDIT START | agent: claude | user: wingertj | reason: resolution-based inference rescaling | date: 2026-08-08]
        'train_frame_size': train_frame_size(block),
        # [AGENT EDIT END]
    }


# [AGENT EDIT START | agent: claude | user: wingertj | reason: resolution-based inference rescaling | date: 2026-08-08]
def train_frame_size(block):
    """``(width, height)`` an alias block was labelled at.

    Absent means :data:`DEFAULT_TRAIN_FRAME_SIZE`, because that is what every
    project in this lab is.  A malformed value raises rather than falling back:
    a typo here would silently pick the analysis resolution for every video the
    alias routes, and the resulting pose files look perfectly well-formed.
    """
    raw = block.get('train_frame_size')
    if raw is None:
        return DEFAULT_TRAIN_FRAME_SIZE
    try:
        width, height = (int(raw[0]), int(raw[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        raise ValueError('train_frame_size must be [width, height], got %r' % (raw,))
    if len(raw) != 2 or width <= 0 or height <= 0:
        raise ValueError('train_frame_size must be two positive ints, got %r' % (raw,))
    return (width, height)
# [AGENT EDIT END]


def _site_number(avipath):
    """``int(avipath.split('/')[5][3:])`` -- the legacy site-number extraction, verbatim.

    Index 5 hard-codes the ``/auto/data/daq/<animal>/<penname>/<video>`` layout
    (``['', 'auto', 'data', 'daq', animal, penname, ...]``).  That assumption is
    inherited from the pre-migration router and is why any path not of that shape
    raises here:

    * fewer than six ``/``-separated components -> ``IndexError``
    * a non-numeric penname (``.../LemonDisco/training/x.avi`` -> ``int('ning')``)
      -> ``ValueError``

    Both are load-bearing: the golden test compares exception *types*.
    """
    pthsplit = avipath.split('/')
    return int(pthsplit[5][3:])


def _animal(avipath):
    """``<animal>`` from ``/auto/data/daq/<animal>/<penname>/...``.  None if absent."""
    parts = avipath.split('/')
    return parts[4] if len(parts) > 4 else None


def _runclass(avipath):
    """Trailing runclass token of the video stem, e.g. ``PRN015a01_a_NTD.avi`` -> ``NTD``.

    Everything from the first ``.`` of the basename is dropped first, so the
    ``.lick`` / ``.pupil`` sub-camera suffixes do not end up in the token.
    """
    base = os.path.basename(avipath).split('.')[0]
    if '_' not in base:
        return None
    return base.rsplit('_', 1)[1]


def _contains_any(avipath, needle):
    if isinstance(needle, str):
        return needle in avipath
    return any(n in avipath for n in needle)


def _equals_any(value, expected):
    if isinstance(expected, str):
        return value == expected
    return value in expected


def _rule_matches(rule, avipath, sitenum):
    """True when every predicate in ``rule`` holds.  Predicates AND; an empty rule
    would match everything, so it is rejected as an authoring error."""
    unknown = [k for k in rule if k not in _RULE_KEYS]
    if unknown:
        raise ValueError('unknown predicate(s) %s in DLC registry rule %r; supported: %s'
                         % (unknown, rule, list(_RULE_KEYS)))
    if not rule:
        raise ValueError('empty DLC registry rule would match every path')

    if 'path_contains' in rule and not _contains_any(avipath, rule['path_contains']):
        return False
    if 'site_max' in rule and not sitenum <= rule['site_max']:
        return False
    if 'site_min' in rule and not sitenum >= rule['site_min']:
        return False
    if 'site_in' in rule and sitenum not in rule['site_in']:
        return False
    if 'site_not_in' in rule and sitenum in rule['site_not_in']:
        return False
    if 'animal' in rule and not _equals_any(_animal(avipath), rule['animal']):
        return False
    if 'runclass' in rule and not _equals_any(_runclass(avipath), rule['runclass']):
        return False
    return True


def resolve_for_video(video_path, registry=None):
    """Return the alias whose rules first match ``video_path``.

    Aliases are tried in registry file order and each alias's rules in list order;
    the first match wins, exactly as the legacy if/elif chain did.  Raises
    ``ValueError`` naming the path when nothing matches.
    """
    if registry is None:
        registry = load_registry()

    # DELIBERATELY EAGER.  The legacy router computed the site number on its first
    # two lines, before testing any branch, so a path whose penname is not
    # <3 letters><digits> raised there rather than falling through to a rule that
    # would have matched it (notably '.../training/' -> ValueError, which is why the
    # behavior_only alias is unreachable by path).  Evaluating this lazily -- only
    # for rules that actually mention a site predicate -- would be a behaviour
    # change, not a migration.  See [known_issues].eager_site_parse in the TOML.
    sitenum = _site_number(video_path)

    aliases = registry['aliases']
    for name in registry['alias_names']:
        for rule in aliases[name].get('rules', []):
            if _rule_matches(rule, video_path, sitenum):
                return name

    raise ValueError('No DLC model specified for this path: %s' % video_path)


def config_for_video(video_path, registry=None):
    """Path to the ``config.yaml`` of the DLC model that analyzes ``video_path``."""
    if registry is None:
        registry = load_registry()
    return resolve(resolve_for_video(video_path, registry), registry)['config']
# [AGENT EDIT END]
