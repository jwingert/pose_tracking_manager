import datetime
import os

import pytest

from nems_lbhb.motor import dlc_attribution as at

INVENTORY = '/auto/data/dlc/inventory/dlc_inventory.csv'

TWO_CHIMNEY = 'DLC_resnet50_two_chimney3fbJan10shuffle1_60000'
HEADPHONES60 = 'DLC_resnet50_headphones_left_chimney_LB1_pokesApr22shuffle1_60000'
PUPIL = 'DLC_resnet50_eyecloseupJun22shuffle1_120000'


def _utc(text):
    return datetime.datetime.fromisoformat(text).replace(tzinfo=datetime.timezone.utc)


def test_release_index_maps_suffix_to_release_ids():
    idx = at.release_index()
    assert idx[TWO_CHIMNEY] == ['two_chimney-r0001', 'two_chimney-r0002']


def test_unique_scorer_resolves_by_scorer_alone():
    got = at.attribute('DLC_resnet50_right_chimney_LB1_pokesOct22shuffle1_50000',
                       _utc('2024-11-01T00:00:00'))
    assert got['release_id'] == 'right_chimney_pokes-r0001'
    assert got['attribution'] == 'inferred_scorer'


def test_two_chimney_before_cutoff_is_iteration_zero():
    got = at.attribute(TWO_CHIMNEY, _utc('2024-01-11T02:47:49'))
    assert got['release_id'] == 'two_chimney-r0001'
    assert got['attribution'] == 'inferred_mtime'
    assert 'iteration-0' in got['evidence']


def test_two_chimney_after_cutoff_is_iteration_one():
    got = at.attribute(TWO_CHIMNEY, _utc('2024-01-13T09:00:00'))
    assert got['release_id'] == 'two_chimney-r0002'
    assert got['attribution'] == 'inferred_mtime'


def test_headphone_sixty_thousand_resolves_to_r0001():
    """All 28 live outputs predate the 2025-12-22 dataset regeneration."""
    got = at.attribute(HEADPHONES60, _utc('2025-12-16T20:28:56'))
    assert got['release_id'] == 'left_chimney_headphones-r0001'
    assert got['attribution'] == 'inferred_mtime'


def test_pupil_is_forced_ambiguous_until_archive_releases_exist():
    """One scorer, three sets of weights (defect 9). A scorer match is not a conclusion."""
    got = at.attribute(PUPIL, _utc('2023-05-01T00:00:00'))
    assert got['release_id'] is None
    assert got['attribution'] == 'ambiguous'
    assert 'iteration' in got['evidence'].lower()


def test_unknown_scorer_is_unknown_not_a_guess():
    got = at.attribute('DLC_resnet50_no_such_modelJan01shuffle1_10000', _utc('2024-01-01T00:00:00'))
    assert got['release_id'] is None
    assert got['attribution'] == 'unknown'


def test_missing_scorer_is_unknown():
    got = at.attribute(None, _utc('2024-01-01T00:00:00'))
    assert got['attribution'] == 'unknown'


def test_left_chimney_is_attributed_but_not_reproducible():
    """The weights that made these 65 outputs were destroyed by an in-place retrain. Attribution
    (high confidence, inferred_scorer) and reproducibility (0, weights gone) are independent
    facts about this row; the story behind the 0 lives in RELEASE.toml, not in evidence -- see
    test_reproducible_zero_story_lives_in_the_release_manifest_not_in_evidence below."""
    got = at.attribute('DLC_resnet50_left_chimney_PRNfbJan11shuffle1_60000',
                       _utc('2024-02-01T00:00:00'))
    assert got['release_id'] == 'left_chimney-r0001'
    assert got['attribution'] == 'inferred_scorer'
    assert got['reproducible'] is False


def test_a_reproducible_release_says_so():
    got = at.attribute(TWO_CHIMNEY, _utc('2024-01-13T09:00:00'))
    assert got['reproducible'] is True


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-fix -- attribution_evidence
# and reproducible are orthogonal questions (which release produced this row vs. do the weights
# still exist to regenerate it); reproducibility prose was being appended into
# attribution_evidence, a category error. `reproducible` is now a bare flag, set regardless of
# how it was verified, and NEITHER of its two possible justifications (the reproducible=1 policy
# default, or a reproducible=0 release's provenance_warning) is ever written into `evidence`
# again -- these tests replace the old ones that asserted the opposite. | date: 2026-08-12]
def test_reproducible_default_is_not_written_into_evidence():
    """The reproducible=1 default's rationale is policy (REPRODUCIBLE_BY_DEFAULT_EVIDENCE's own
    docstring / the spec), not row text -- it must never appear in a row's `evidence`, on a
    default-reproducible row or any other."""
    got = at.attribute(TWO_CHIMNEY, _utc('2024-01-13T09:00:00'))
    assert got['reproducible'] is True
    assert at.REPRODUCIBLE_BY_DEFAULT_EVIDENCE not in (got['evidence'] or '')
    # the substantive attribution evidence (the mtime-rule text) is still there, untouched
    assert 'iteration-1' in got['evidence']


def test_reproducible_zero_story_lives_in_the_release_manifest_not_in_evidence():
    """The DAMAGED PROVENANCE story for a reproducible=0 row is no longer duplicated into
    `evidence` -- it lives exactly once, in the producing release's RELEASE.toml
    `provenance_warning`, one hop away via release_id -> release_path."""
    got = at.attribute('DLC_resnet50_left_chimney_PRNfbJan11shuffle1_60000',
                       _utc('2024-02-01T00:00:00'))
    assert got['reproducible'] is False
    assert got['evidence'] is None
    manifest_path = os.path.join(at.RELEASE_ROOT, got['release_id'], 'RELEASE.toml')
    with open(manifest_path, 'rb') as f:
        manifest = at._toml_parser().load(f)
    assert '2025-06-12' in manifest['release']['provenance_warning']
    assert 'DAMAGED PROVENANCE' in manifest['release']['provenance_warning']


def test_an_unattributed_row_makes_no_reproducibility_claim_either_way():
    got = at.attribute(PUPIL, _utc('2023-05-01T00:00:00'))
    assert got['reproducible'] is None
    assert at.REPRODUCIBLE_BY_DEFAULT_EVIDENCE not in (got['evidence'] or '')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude | user: wingertj | reason: schema change 2 -- write
# attribution_evidence only when it is non-generic (updated by the final-fix round: with
# reproducibility prose no longer appended in _finalize, EVERY plain inferred_scorer row's
# evidence is now exactly GENERIC_SCORER_MATCH_EVIDENCE and nothing else -- including the
# 64 left_chimney rows, which used to keep DAMAGED PROVENANCE as a "substantive override" but
# no longer do, since that text is reproducibility prose and does not belong in this column
# either). Measured 2026-08-12 over the 552-row inventory: 440 inferred_scorer rows (all of
# them, including left_chimney) collapse to NULL; the 112 non-null rows are exactly the 83
# inferred_mtime + 28 ambiguous + 1 unknown rows, none of which ever contained the generic
# sentence in the first place. | date: 2026-08-12]
def test_generic_scorer_match_evidence_is_a_named_constant():
    """Exposed as a module constant, not a repeated literal, so attribute()'s ladder step 4
    and _strip_generic_evidence's stripping rule cannot drift apart."""
    assert at.GENERIC_SCORER_MATCH_EVIDENCE == 'scorer matches exactly one release manifest'


def test_bare_scorer_match_with_default_reproducibility_is_null_evidence():
    """A unique scorer match, default reproducibility -- nothing in the stored evidence says
    anything about THIS row beyond what attribution/reproducible already carry, so it is NULL
    rather than a restated boilerplate sentence."""
    got = at.attribute('DLC_resnet50_right_chimney_LB1_pokesOct22shuffle1_50000',
                       _utc('2024-11-01T00:00:00'))
    assert got['attribution'] == 'inferred_scorer'
    assert got['reproducible'] is True
    assert got['evidence'] is None


def test_generic_scorer_match_with_a_reproducibility_override_is_also_null_evidence():
    """The 64-row left_chimney case: a unique scorer match whose release happens to have a
    reproducible=0 entry. That entry's DAMAGED PROVENANCE text is reproducibility evidence, not
    attribution evidence, so it is never appended here either -- evidence collapses to NULL
    exactly like the reproducible=1 case, and `reproducible` alone carries the 0."""
    got = at.attribute('DLC_resnet50_left_chimney_PRNfbJan11shuffle1_60000',
                       _utc('2024-02-01T00:00:00'))
    assert got['evidence'] is None
    assert got['reproducible'] is False


def test_mtime_rule_evidence_is_untouched_by_the_generic_strip():
    """An mtime-rule row never contains the generic sentence in the first place (it is a
    different ladder step entirely) -- its own substantive text survives unstripped, and no
    reproducibility clause is ever appended to it."""
    got = at.attribute(TWO_CHIMNEY, _utc('2024-01-13T09:00:00'))
    assert got['evidence'] is not None
    assert at.GENERIC_SCORER_MATCH_EVIDENCE not in got['evidence']
    assert 'iteration-1' in got['evidence']
    assert at.REPRODUCIBLE_BY_DEFAULT_EVIDENCE not in got['evidence']


def test_forced_ambiguous_evidence_is_untouched_by_the_generic_strip():
    got = at.attribute(PUPIL, _utc('2023-05-01T00:00:00'))
    assert got['evidence'] is not None
    assert at.GENERIC_SCORER_MATCH_EVIDENCE not in got['evidence']


def test_strip_generic_evidence_unit_cases():
    """Direct unit coverage of the one place the rule lives, independent of attribute()'s
    own ladder wiring. Simplified by the final-fix round: since _finalize no longer appends
    anything to evidence, the only case left to strip is an EXACT match on the generic
    sentence -- there is no longer a "generic sentence followed by something else" shape for
    this function to see."""
    strip = at._strip_generic_evidence

    assert strip(None) is None
    assert strip(at.GENERIC_SCORER_MATCH_EVIDENCE) is None
    # a hand-built string that merely starts with the generic sentence is NOT the shape
    # _finalize ever produces anymore, and is returned unchanged rather than truncated
    assert strip(at.GENERIC_SCORER_MATCH_EVIDENCE + '\n\nsomething substantive') \
        == at.GENERIC_SCORER_MATCH_EVIDENCE + '\n\nsomething substantive'
    # text that never contained the generic sentence passes through byte-for-byte
    assert strip('a wholly unrelated evidence string') == 'a wholly unrelated evidence string'


# [AGENT EDIT START | agent: claude | user: wingertj | reason: final-fix -- the golden count
# moves from 177/375 to 112/440: the final-fix round stops appending EITHER reproducibility
# clause into evidence (not just the default-reproducible one this test used to describe), so
# all 440 inferred_scorer rows -- including the 64 left_chimney rows that used to keep DAMAGED
# PROVENANCE as a kept "substantive override" -- now collapse to NULL, not just the 375 that
# were already bare-default. The remaining 112 non-null rows are unchanged: 83 inferred_mtime +
# 28 ambiguous + 1 unknown, none of which were ever touched by the generic-match stripping or
# the reproducibility clauses in the first place. | date: 2026-08-12]
@pytest.mark.skipif(not os.path.exists(INVENTORY), reason='run scripts/dlc_inventory.py first')
def test_golden_evidence_breakdown_over_the_real_inventory():
    """The concrete count the user asked to see, measured directly against the frozen
    552-row inventory (2026-08-12): 112 non-null, 440 null. Every `inferred_scorer` row (a bare
    scorer match, whether or not its release has a reproducibility override) is NULL now that
    `attribution_evidence` never carries reproducibility prose; the 112 non-null rows are the
    83 `inferred_mtime` + 28 `ambiguous` + 1 `unknown` rows, each of which carries its own
    substantive (non-reproducibility) evidence text.

    This is `attribute()` alone, over the ladder-only 552-row inventory snapshot -- distinct
    from (and one row larger than) the live sweeper's 551-row `--out` CSV, whose REI070 file
    is additionally hash-promoted from `unknown` to `inferred_hash` by
    scripts/dlc_sync_fits.py (out of scope for this module) and whose corpus has since lost
    one file on disk; both land on 112 non-null by coincidence of bucket sizes, not because the
    two corpora agree row-for-row. Same golden-data pattern as
    test_golden_breakdown_over_the_real_inventory above: a rule-file change that alters how
    much evidence survives fails here.
    """
    pd = pytest.importorskip('pandas')
    inv = pd.read_csv(INVENTORY)
    idx = at.release_index()

    non_null = 0
    total = 0
    for scorer, mtime in zip(inv['scorer'], inv['mtime_utc']):
        got = at.attribute(scorer if isinstance(scorer, str) else None,
                           pd.to_datetime(mtime).to_pydatetime(),
                           release_index=idx)
        total += 1
        if got['evidence'] is not None:
            non_null += 1

    assert total == 552
    assert non_null == 112
    assert total - non_null == 440
# [AGENT EDIT END]
# [AGENT EDIT END]


def test_every_release_id_the_rules_emit_exists_on_disk():
    """The rule file must not drift from the release store."""
    rules = at.load_rules()
    ids = set()
    for rule in rules['mtime_rules']:
        ids.add(rule['before'])
        ids.add(rule['after'])
    ids.update(rules['reproducibility'].keys())
    # [AGENT EDIT START | agent: claude | user: wingertj | reason: final-review Minor --
    # forced_ambiguous[].candidates was the one release-naming key this drift check did not
    # cover, and it stopped being inert when the sweeper started DERIVING the stored `alias`
    # from those candidates (dlc_sync_fits._alias_for_ambiguous_scorer splits each candidate
    # on '-r'). A typo there would not fail anything -- it would quietly relabel the 28
    # ambiguous rows with a misspelled alias, or drop the alias entirely. | date: 2026-08-12]
    for entry in rules['forced_ambiguous']:
        ids.update(entry.get('candidates', []))
    # [AGENT EDIT END]
    missing = [r for r in sorted(ids)
               if not os.path.isdir(os.path.join(at.RELEASE_ROOT, r))]
    assert not missing, 'rules name releases that do not exist: %s' % missing


@pytest.mark.skipif(not os.path.exists(INVENTORY), reason='run scripts/dlc_inventory.py first')
def test_golden_breakdown_over_the_real_inventory():
    """Two breakdowns, measured 2026-08-12. A rule edit that reclassifies files fails here.

    The raw table is evidence; the final table is conclusions. A rule that promotes
    one to the other moves the second and not the first, which is exactly what this
    test exists to make visible.
    """
    pd = pytest.importorskip('pandas')
    inv = pd.read_csv(INVENTORY)
    idx = at.release_index()

    raw = {'one': 0, 'many': 0, 'none': 0}
    for scorer in inv['scorer']:
        n = len(idx.get(scorer, [])) if isinstance(scorer, str) else 0
        raw['one' if n == 1 else 'many' if n > 1 else 'none'] += 1
    assert raw == {'one': 468, 'many': 83, 'none': 1}

    final = {}
    for scorer, mtime in zip(inv['scorer'], inv['mtime_utc']):
        got = at.attribute(scorer if isinstance(scorer, str) else None,
                           pd.to_datetime(mtime).to_pydatetime(),
                           release_index=idx)
        final[got['attribution']] = final.get(got['attribution'], 0) + 1

    assert final.get('inferred_scorer') == 440
    assert final.get('inferred_mtime') == 83
    assert final.get('ambiguous') == 28
    assert final.get('unknown') == 1        # the REI070 copy; step 4 resolves it in Task 4
    assert sum(final.values()) == 552


@pytest.mark.skipif(not os.path.exists(INVENTORY), reason='run scripts/dlc_inventory.py first')
def test_two_chimney_splits_forty_two_thirteen():
    pd = pytest.importorskip('pandas')
    inv = pd.read_csv(INVENTORY)
    idx = at.release_index()
    rows = inv[inv['scorer'] == TWO_CHIMNEY]
    got = [at.attribute(TWO_CHIMNEY, pd.to_datetime(m).to_pydatetime(),
                        release_index=idx)['release_id'] for m in rows['mtime_utc']]
    assert got.count('two_chimney-r0001') == 42
    assert got.count('two_chimney-r0002') == 13
