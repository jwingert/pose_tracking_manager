"""The GUI's routing-suggestion cache must invalidate on a promotion.

`labeler_core` imports streamlit and plotly, which are not in the test env, so this
reads the source rather than importing it. That is enough to pin the contract that
broke: the cache key must carry the registry's identity.

Motivation: after promoting left_chimney_headphones-r0004 the GUI banner still read
"release left_chimney_headphones-r0003" while the resolved model correctly said r0004.
The banner's cache was keyed on (video_path, method) alone, so a promotion -- which
changes only the registry -- could never evict it.
"""
import os
import re

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'nems_lbhb', 'motor', 'streamlit_vid_labeler', 'labeler_core.py')


def _src():
    with open(CORE) as fh:
        return fh.read()


def test_routing_suggestion_cache_key_includes_the_registry():
    src = _src()
    m = re.search(r'def _routing_suggestion_cached\(([^)]*)\)', src)
    assert m, '_routing_suggestion_cached not found'
    params = [p.strip() for p in m.group(1).split(',')]
    assert 'registry_mtime' in params, (
        'cache key is %s -- a promotion changes only the registry, so without it the '
        'suggestion outlives the release it names' % params)


def test_caller_passes_a_live_registry_stamp():
    src = _src()
    assert '_routing_suggestion_cached(str(video_path), method, _registry_mtime())' in src
    assert 'def _registry_mtime()' in src
    # the stamp must come from the registry file itself, not a constant
    body = src.split('def _registry_mtime()', 1)[1].split('\ndef ', 1)[0]
    assert 'getmtime' in body and 'DEFAULT_REGISTRY_PATH' in body


# [AGENT EDIT START | agent: claude | user: wingertj | reason: saving an arena config as
# .toml raised "Object of type <class 'NoneType'> is not TOML serializable" for any
# session WITHOUT a camera override -- i.e. the ordinary case. TOML cannot represent
# null, and the config carries several optional fields that are legitimately None.
# | date: 2026-08-26]
def _drop_none():
    """Load _toml_drop_none from source (labeler_core imports streamlit)."""
    import ast
    tree = ast.parse(_src())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == '_toml_drop_none')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<t>', 'exec'), ns)
    return ns['_toml_drop_none']


def test_toml_drop_none_makes_an_unset_config_serializable():
    tomli_w = __import__('pytest').importorskip('tomli_w')
    drop = _drop_none()
    cfg = {'metadata': {'video_path': '/x/recording.avi', 'camera_override': None,
                        'frame_height': None, 'frame_number': 10},
           'fixation_base_marker': None,
           'xaxis_pair': {'xaxis_pair': None},
           'points': {'a': [1.0, 2.0]}}
    try:
        tomli_w.dumps(cfg)
        raise AssertionError('expected raw dump to fail on None')
    except TypeError:
        pass
    text = tomli_w.dumps(drop(cfg))          # must not raise
    assert 'camera_override' not in text
    assert 'video_path' in text and 'frame_number' in text


def test_toml_drop_none_is_recursive_and_keeps_falsey_values():
    drop = _drop_none()
    out = drop({'a': {'b': None, 'c': 0, 'd': False, 'e': ''},
                'l': [1, None, 2], 'keep': 0.0})
    assert out == {'a': {'c': 0, 'd': False, 'e': ''}, 'l': [1, 2], 'keep': 0.0}, out


def test_save_config_uses_the_sanitizer_for_toml_only():
    src = _src()
    assert 'tomli_w.dump(_toml_drop_none(config), f)' in src
    # yaml/json can represent null, so they must NOT be silently reshaped
    assert 'yaml.dump(config, f)' in src
    assert 'json.dump(config, f, indent=2)' in src
# [AGENT EDIT END]
