"""Golden test: the registry-driven DLC router must reproduce the legacy if/elif chain exactly.

The whole point of this file is that `_legacy_get_dlc_model` below is a frozen, verbatim
snapshot of `nems_lbhb.motor.nems_dlc_settings.get_dlc_model` as it stood immediately before
the migration to `nems_lbhb/configs/dlc_registry.toml`.  If a path ever diverges, the registry
is wrong -- never this file.
"""
import os

import pytest

pd = pytest.importorskip('pandas')

from nems_lbhb.motor import dlc_registry as reg

INVENTORY = '/auto/data/dlc/inventory/dlc_inventory.csv'


# --- frozen snapshot of nems_dlc_settings.get_dlc_model as of 2026-08-06 -------------
# DO NOT EDIT.  The golden test's whole value is that this is the pre-migration behaviour,
# faults and all:
#   * the `int(avipath.split('/')[5][3:])` runs EAGERLY, before any branch is tested, so
#     '.../LemonDisco/training/x.avi' raises ValueError and never reaches its own branch;
#   * short paths raise IndexError, not ValueError;
#   * the `free_top_2022_11` assignment is dead (overwritten on the next line).
# All of that is behaviour the migration must preserve.
def _legacy_get_dlc_model(avipath):
    # get siteid number dumb but quick
    pthsplit = avipath.split('/')
    sitenum = int(pthsplit[5][3:])
    rei_headphone_nums = [68, 70, 74]
    # single chimney lh?
    print(avipath)
    if ('video_pupil' in avipath):
        dlc_model = '/auto/data/dlc/eyecloseup/config.yaml'

    elif ('PRN00' in avipath) | ('PRN01' in avipath) | ('PRN02' in avipath) | ('PRN03' in avipath):
        # single chimney, LH
        dlc_model = '/auto/data/dlc/free_top_2022_11/config.yaml'
        # single chimney, LH, back of implant instead of rear headpost
        dlc_model = '/auto/data/dlc/left_chimney_PRNfb-svd-2024-01-11/config.yaml'

    elif ('PRN04' in avipath) | ('PRN05' in avipath) | ('PRN06' in avipath) | ('PRN07' in avipath):
        # single chimney, RH
        #dlc_model = '/auto/data/dlc/free_top_RH-jereme-2023-02-17/config.yaml'
        #dlc_model = '/auto/data/dlc/right_chimney_PRN-svd-2023-11-21/config.yaml'
        # back of implant instead of rear headpost
        dlc_model = '/auto/data/dlc/right_chimney_PRNfb-svd-2023-11-30/config.yaml'
    elif ('LemonDisco/training' in avipath) or \
            ('SlipperyJack/training' in avipath):
        # NO CHIMNEYS
        dlc_model = '/auto/data/dlc/free_train-svd-2023-09-26/config.yaml'
    elif ('SlipperyJack/SLJ' in avipath) or \
            ('LemonDisco/LMD' in avipath):
        # TWO CHIMNEYS
        # coordinates on corners of chimney
        # dlc_model = '/auto/data/dlc/two_chimney-svd-2023-09-28/config.yaml'
        # switched to tether entry points.
        # dlc_model = '/auto/data/dlc/two_chimney2-svd-2023-11-03/config.yaml'
        # back of implant instead of rear headpost
        dlc_model = '/auto/data/dlc/two_chimney3fb-svd-2024-01-10/config.yaml'
    # put correct REI LH last pen
    elif (('StinkySquid/SQD' in avipath) & (sitenum <= 20)) or (('Reishi/REI' in avipath) & (sitenum <= 42) & (sitenum not in rei_headphone_nums) ) or ('StinkySquid/training2024' in avipath):
        # RIGHT CHIMNEY in LB-1 - Includes multiple extra points for anipose
        dlc_model = '/auto/data/dlc/right_chimney_LB1_pokes-jcw-2024-10-22/config.yaml'
        # old, simple model
        #dlc_model = '/auto/data/dlc/right_chimney_LB1-svd-2024-09-10/config.yaml'
    # model with old cameras REI left hemisphere & headphones
    elif (('Reishi/REI' in avipath) & (42 < sitenum <= 86) & (sitenum in rei_headphone_nums)):
        dlc_model = '/auto/data/dlc/headphones_left_chimney_LB1_pokes-jcw-2025-04-22/config.yaml'
    else:
        raise ValueError('No DLC model specified for this path')

    return dlc_model
# --- end frozen snapshot -------------------------------------------------------------


# [AGENT EDIT START | agent: claude | user: wingertj | reason: the left headphone
# TRAINING PROJECT was renamed on disk (2026-08-23, user decision: the name said
# "pokes", which belongs to the port-tracking models, and the right-hemisphere folders
# were split out). Routing is unchanged -- the same videos reach the same alias -- so
# the frozen legacy snapshot above stays frozen and the rename is declared here instead.
# A directory rename is the ONLY thing this map may ever encode; a genuine routing
# change must fail the golden test. | date: 2026-08-23]
PROJECT_RENAMES = {
    '/auto/data/dlc/headphones_left_chimney_LB1_pokes-jcw-2025-04-22':
        '/auto/data/dlc/left_chimney_headphones-jcw-2025-04-22',
}


def _rename(path):
    """Apply the on-disk project renames to a legacy config path."""
    for old, new in PROJECT_RENAMES.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


#: Aliases whose TRAINING HOME deliberately moved to a different project (not a
#: rename: a new project, with a different label set, that a future release will be
#: cut from). Unlike PROJECT_RENAMES these are not path-equivalent, so the golden
#: comparison cannot silently absorb them -- each is declared with the project the
#: legacy chain named and the project the registry now names, and the test asserts
#: BOTH, so accidental drift still fails.
TRAINING_HOME_MOVES = {
    # merged left-chimney project (2026-08-10): PRNfb + Prince coverage merged in
    'left_chimney': ('/auto/data/dlc/left_chimney_PRNfb-svd-2024-01-11',
                     '/auto/data/dlc/merged_left_chimney-jcw-2026-08-10'),
}
# [AGENT EDIT END]


def _moved(path):
    """Apply declared training-home moves to a legacy config path."""
    for legacy_proj, current_proj in TRAINING_HOME_MOVES.values():
        if path.startswith(legacy_proj):
            return current_proj + path[len(legacy_proj):]
    return path


def _legacy_or_error(p):
    try:
        return _moved(_rename(_legacy_get_dlc_model(p)))
    except Exception as e:
        return 'ERROR:%s' % type(e).__name__


def _new_or_error(p):
    """Route `p` and report the SOURCE PROJECT config the legacy chain would have named.

    Once an alias is promoted, `config_for_video` returns the frozen release's config
    instead of the mutable project's -- that is the whole point of the release store.
    The golden test exists to prove the *routing* migration changed nothing, so it must
    compare the alias's `project`, not wherever that alias currently resolves to.
    `test_alias_projects_match_the_legacy_config_paths` covers the other half: that each
    alias's `project` really is the directory the legacy chain named.
    """
    try:
        alias = reg.resolve_for_video(p)
    except Exception as e:
        return 'ERROR:%s' % type(e).__name__
    block = reg.load_registry()['aliases'][alias]
    return os.path.join(block['project'], 'config.yaml')


SYNTHETIC = [
    '/auto/data/daq/Prince/PRN015/PRN015a01_a_NTD.avi',
    '/auto/data/daq/Prince/PRN048/PRN048a01_a_NTD.avi',
    '/auto/data/daq/LemonDisco/training/x.avi',
    '/auto/data/daq/SlipperyJack/training/x.avi',
    '/auto/data/daq/SlipperyJack/SLJ017/SLJ017a07_a_NTD.avi',
    '/auto/data/daq/LemonDisco/LMD004/LMD004a01_a_NTD.avi',
    '/auto/data/daq/StinkySquid/SQD016/SQD016a01_a_NTD.avi',
    '/auto/data/daq/StinkySquid/SQD022/SQD022b03_a_NTD.avi',
    '/auto/data/daq/StinkySquid/training2024/x.avi',
    '/auto/data/daq/Reishi/REI030/REI030a01_a_NTD.avi',
    '/auto/data/daq/Reishi/REI068/REI068a01_a_NTD.avi',
    '/auto/data/daq/Reishi/REI074/REI074a07_p_NFC.avi',
    '/auto/data/daq/Reishi/REI097/REI097a01_a_NTD.avi',      # legacy: ValueError
    '/auto/data/daq/Reishi/REI102/REI102a01_a_NFC.avi',      # legacy: ValueError
    '/auto/data/daq/Prince/PRN044/PRN044a_video_pupil.avi',

    # --- added beyond the brief's list: boundaries and the eager-int() failure modes.
    # These only strengthen the comparison -- each one is still checked against the
    # frozen legacy function, never against a hand-written expectation.
    '/auto/x.avi',                                            # legacy: IndexError, not ValueError
    '/auto/data/daq/LemonDisco/training/x_video_pupil.avi',   # eager int() beats 'video_pupil'
    '/auto/data/daq/StinkySquid/SQD020/SQD020a01_a_NTD.avi',  # site_max = 20, on the boundary
    '/auto/data/daq/StinkySquid/SQD021/SQD021a01_a_NTD.avi',  # one past it
    '/auto/data/daq/Reishi/REI042/REI042a01_a_NTD.avi',       # site_max = 42, on the boundary
    '/auto/data/daq/Reishi/REI043/REI043a01_a_NTD.avi',       # one past it, not a headphone site
    '/auto/data/daq/Reishi/REI070/REI070a01_a_NTD.avi',       # headphone site
    '/auto/data/daq/Reishi/REI086/REI086a01_a_NTD.avi',       # site_max = 86, but not in [68,70,74]
]


@pytest.mark.parametrize('p', SYNTHETIC)
def test_router_matches_legacy_on_synthetic_paths(p):
    assert _new_or_error(p) == _legacy_or_error(p)


@pytest.mark.skipif(not os.path.exists(INVENTORY), reason='run scripts/dlc_inventory.py first')
def test_router_matches_legacy_on_every_inventoried_video():
    """The migration is behaviour-preserving over every video the lab has actually analyzed."""
    inv = pd.read_csv(INVENTORY)
    avis = sorted({
        os.path.join(os.path.dirname(os.path.dirname(a)), s + '.avi')
        for a, s in zip(inv['alias_path'], inv['video_stem'])
        if isinstance(a, str) and isinstance(s, str)
    })
    assert len(avis) > 400
    mismatches = [(p, _legacy_or_error(p), _new_or_error(p))
                  for p in avis if _legacy_or_error(p) != _new_or_error(p)]
    assert not mismatches, 'router diverges from legacy on %d paths, e.g. %s' % (
        len(mismatches), mismatches[:5])


def test_resolve_unknown_alias_raises():
    with pytest.raises(KeyError):
        reg.resolve('no_such_alias')


def test_every_alias_config_exists_on_disk():
    r = reg.load_registry()
    missing = [a for a in r['alias_names'] if not os.path.exists(reg.resolve(a)['config'])]
    assert not missing, 'aliases point at configs that do not exist: %s' % missing


def test_bodyparts_recorded_match_the_project_config():
    """Guards the registry against drifting from the model it names."""
    yaml = pytest.importorskip('yaml')
    r = reg.load_registry()
    for a in r['alias_names']:
        info = reg.resolve(a)
        cfg = yaml.safe_load(open(info['config']))
        assert list(cfg['bodyparts']) == list(info['bodyparts']), a


def test_load_registry_drops_the_ambiguous_alias_key():
    """'alias' (the raw TOML table) must not survive beside 'aliases'."""
    r = reg.load_registry()
    assert 'alias' not in r
    assert set(r['aliases']) == set(r['alias_names'])
    assert list(r['aliases']) == r['alias_names'], 'file order must be preserved'


def test_load_registry_rejects_an_unknown_rule_predicate(tmp_path):
    """A typo in an UNREACHABLE alias's rules must still be caught at load time."""
    bad = tmp_path / 'bad.toml'
    bad.write_text(
        'schema_version = 1\n'
        '[alias.never_reached]\n'
        'project = "/tmp"\n'
        'current = ""\n'
        'rules = [{ path_contian = "typo" }]\n')
    with pytest.raises(ValueError) as e:
        reg.load_registry(str(bad))
    assert 'path_contian' in str(e.value) and 'never_reached' in str(e.value)


def test_load_registry_accepts_every_valid_predicate(tmp_path):
    ok = tmp_path / 'ok.toml'
    ok.write_text(
        'schema_version = 1\n'
        '[alias.a]\n'
        'project = "/tmp"\n'
        'current = ""\n'
        'rules = [{ path_contains = "x", site_min = 1, site_max = 2, site_in = [1],'
        ' site_not_in = [3], animal = "A", runclass = "NTD" }]\n')
    assert reg.load_registry(str(ok))['alias_names'] == ['a']


def test_alias_projects_match_the_legacy_config_paths():
    """Each alias's `project` is exactly the directory the legacy chain pointed at.

    Together with the golden test (which compares `project`, not the resolved release),
    this pins both halves: the right alias is chosen, and that alias names the right model.
    """
    r = reg.load_registry()
    seen = {}
    for p in SYNTHETIC:
        try:
            alias = reg.resolve_for_video(p)
        except Exception:
            continue
        seen[alias] = _legacy_get_dlc_model(p)
    assert seen, 'no synthetic path routed'
    for alias, legacy_cfg in seen.items():
        expected = os.path.dirname(_moved(_rename(legacy_cfg)))
        assert r['aliases'][alias]['project'] == expected, alias
        if alias in TRAINING_HOME_MOVES:
            legacy_proj, current_proj = TRAINING_HOME_MOVES[alias]
            assert os.path.dirname(legacy_cfg) == legacy_proj, alias
            assert r['aliases'][alias]['project'] == current_proj, alias


def test_promoted_aliases_resolve_to_a_verified_release():
    """Every promoted alias points at a release that exists and passes a fast verify."""
    rel = pytest.importorskip('nems_lbhb.motor.dlc_release')
    r = reg.load_registry()
    for alias in r['alias_names']:
        info = reg.resolve(alias)
        if info['source'] != 'release':
            continue
        assert os.path.exists(info['config']), alias
        assert rel.verify(info['release_id']) == [], alias
