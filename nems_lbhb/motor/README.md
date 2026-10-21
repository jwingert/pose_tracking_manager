# `nems_lbhb/motor` — DeepLabCut pose tracking

Everything that turns arena video into pose: which model runs, where its weights come from,
what produced a given `.dlc.h5`, and whether the result is any good.

The organising idea is **provenance**. Historically nothing recorded which network produced
which output, so routing had to be restrictive to be safe, and a wrong model was silent and
permanent. Now every production fit runs from an immutable release, is stamped in its own
filename, and is recorded in `gDLCFit`. Because the record is trustworthy, the *router* can
relax into a suggestion — you can run any model you like, as long as what ran is auditable.

---

## The modules

**Model identity and routing**

| module | what it owns |
|---|---|
| `dlc_registry.py` | the registry TOML: aliases, their current release, geometry declarations, legacy path/site rules, per-session camera overrides |
| `dlc_routing.py` | model *suggestions* — `geometry` (what the recording is) and `legacy` (the migrated if/elif chain). Never raises, never decides |
| `dlc_release.py` | immutable releases: `build_release`, `verify`, `promote`, `rollback`, `read_manifest` |
| `dlc_alias.py` | filename semantics: atomic symlink repointing, scorer parsing, release stamps, stream/stem helpers |
| `dlc_camera_override.py` | asserting which psi camera slot holds the ceiling view, shared by the CLI and the GUI |

**Running and recording a fit**

| module | what it owns |
|---|---|
| `dlc_fit_script.py` | the entry point: resolve → verify → analyse → write aliases → record the row |
| `dlc_fit_record.py` | the `gDLCFit` schema, row identity, and the *only* code that touches that table |
| `dlc_attribution.py` | the evidence ladder that attributes a legacy output to a release |
| `dlc_fit_qc.py` | scoring an output against its release's frozen labels (shared by the fit job and the sweeper) |
| `dlc_train_guard.py` | locking and fresh-iteration guards, so a retrain cannot destroy deployed weights |
| `dlc_video_scale.py` | analysing at the resolution the model was trained on, then converting back |

**Quality**

| module | what it owns |
|---|---|
| `dlc_headfix_qc.py` | head-aligned rigid-marker spread and tracking rate; the flag thresholds |
| `dlc_lumgate.py` | masking frames the camera could not see (lights-out timeout punishment) |
| `dlc_video_qc.py` | catching silently rotted session videos |
| `dlc_geometry.py` | arena geometry helpers |

`streamlit_vid_labeler/panels/` holds thin GUI panels that render these modules. Panels
render; the modules hold the rules, so the CLI, the queue jobs and the GUI cannot drift apart,
and the rules stay testable without Streamlit.

Scripts live in `scripts/dlc_*.py`: `dlc_sync_fits.py` (the filesystem sweeper that backfills
legacy outputs), `dlc_set_camera_override.py`, `dlc_build_lumgate.py`, `dlc_inventory.py`.

---

## How a production fit works

```
video ──► resolve_model()          registry alias, or alias=<name>, or an explicit config
      ──► verify_resolved_release() SHA256 check before anything is analysed
      ──► [camera_override]         redirect to the slot that really holds the ceiling view
      ──► analyze_videos()          into sorted/<release_id>/, at the trained resolution
      ──► write_aliases()           <stem>.<release>.dlc.h5  +  <stem>.dlc.h5
      ──► record_job_fit()          one gDLCFit row, with QC
```

Two aliases, two jobs. `<stem>.<release>.dlc.h5` is durable provenance — one per release
ever run, never repointed. `<stem>.dlc.h5` is the *current pointer*, repointed atomically,
and is the name every loader opens (`baphy_experiment.dlcfile`).

---

## Invariants worth not undoing

Each of these was a bug once.

**The scorer string is not a model identity.** It omits the iteration, so two releases cut
from one project can spell it identically from different weights. That is why releases carry
an id, analyse into their own output directory, and stamp it into the alias name.

**`ln -s` is banned.** It refuses to overwrite an existing link and `os.system` discards the
return code, so every re-analysis of an already-aliased video was a silent no-op. Use
`dlc_alias.point_alias`.

**`gDLCFit` is append-only** (`INSERT IGNORE` on `row_key`). A re-analysis mints a row; nothing
rewrites history. Anything that would need an `UPDATE` — staleness, QC flags, `is_current` —
is *derived at read time* instead, which is also why retuning a threshold reclassifies all of
history for free.

**One row per analysis, not per symlink.** The sweeper skips the current pointer when a stamped
sibling resolves to the same file — but only then, since the ~550 legacy outputs have no stamp
and an unconditional skip would erase them from the table.

**The cam tag comes from the resolved target, never the alias name.** A camera override points
a session-named alias at a `_camC` target on purpose; reading the alias would align the trace
to the wrong camera's frame counts. That is a wrong frame→time map, not a cosmetic mislabel.

**`dlc_fit_record` touches exactly one table.** A guard test asserts it never so much as names
another. The `gDataRaw` lookup therefore lives in the caller.

**`db.sql_command()` does not persist** under the SQLAlchemy in this environment — a bare
`connect()+execute()` silently drops the write. Everything here writes through
`db.Engine().begin()`. (Fixed on the `cluster_sqlalchemy_upgrade` branch by one word:
`connect()` → `begin()`.)

---

## Routing: geometry vs legacy

Routing keyed on site numbers was never a design — it was an `if/elif` chain migrated
faithfully into data, and site numbers were a proxy for facts nobody had recorded.

The `geometry` method reads exactly four signals and nothing else:

| signal | source | note |
|---|---|---|
| chimney side | `gPenetration.ear` | **contralateral** — `ear='R'` means the chimney is on the LEFT |
| audio | `gData` `name='audio'` | `In-ear` means headphones are on the head, which changes what the camera sees |
| probes | open-ephys `settings.xml` | two probes **establish** two chimneys; one probe establishes nothing |
| stream | the DLC cam tag | arena / pupil / face; side cameras are still the arena model |

Each alias declares in the registry what it is validated *for*; an omitted key means "any".
Those declarations were read off the corpus, not assumed — which is how we know
`left_chimney` already spans three animals on one geometry, and that
`left_chimney_headphones` spans both chimney sides (its name is historical, not a claim).

Two rules carry the weight, both about not asserting what nothing recorded. `ear` gives side,
never count, so a two-chimney model comes back *unknown* against a side observation rather
than contradicted. And an unobserved dimension neither supports nor refutes an alias; it is
reported. Ranking is matched dimensions → fewest unresolved requirements → registry file order.

Measured against all 552 recorded fits: the model that actually ran is the top suggestion for
**91%** and among the candidates for **99%**. The residue is entirely two-chimney animals that
ran a single probe, which nothing on disk distinguishes from a single-chimney recording. The
router says so instead of guessing, and the GUI plots a frame beside the evidence.

`legacy` remains selectable so a historical file can be re-analysed with the model that
actually produced it.

---

## Where things stand

**Done and verified**

- Releases (25 on disk), the registry and its 9 aliases, training guards, resolution handling.
- `gDLCFit`, 552 rows, with the attribution ladder and the sweeper.
- Camera overrides end to end: `REI084a10_a_NFB` re-analysed from `left_chimney-r0001`
  aligns against `PSIVIDEO_camC` at 30.23 fps (the base stream would have been 100.77 —
  3.3× wrong), 14 markers, 100% finite, `p_median` 0.990.
- The DLC release id is part of the recording cache hash, gated so legacy caches are not
  invalidated to record "we still don't know".
- GUI panels: routing, camera-slot override, provenance, staleness.

**Remaining**

- **Config-mutation-on-view** (`behavior_config_labeler_plotly.py`): three write sites mutate
  `config.yaml` on a selectbox change, with no lock, atomicity or mtime check. Needs a
  `dlc_config.update()` with `O_EXCL` + mtime check + `ruamel.yaml` round-trip, and a
  `try/finally` on `evaluate_all_snapshots`.
- **Curation panel**: labelled frames per folder, which are new since the release's frozen
  `labels/`, whether a project has uncommitted curation.
- **Promotion gate**: `promote()` verifies checksums but has no quality gate. The piece that
  matters is the regression comparison — re-score the same videos under the current release and
  require the candidate to be no worse. Blocked on the 120-frame ground-truth set at
  `generalization_eval_labels-jcw-2026-08-10` (0 labelled), which is what would establish
  whether `rms_spread` tracks accuracy or is only a proxy.
- **`right_chimney_merged` is unrouted** — three releases on disk, no alias points at them, so
  nothing in production uses the merged right-chimney model.
- **`left_chimney` retrain**: 150 frames still to label in `merged_left_chimney-jcw-2026-08-10`.
- 12 lumgate sidecars blocked by `PermissionError`; need a run as `lbhb`.
- Whether `dlc_valid` should exclude paw channels — a 7.7–31× data-retention swing at 4 SQD
  sites, so it needs a decision rather than a quiet default flip.

**Known unverified:** the ~30 lines in `dlc_fit_script` that record a job's own `gDLCFit` row
have been exercised against real data with the same arguments, but have not yet run inside a
live fit. The next release-based fit is the first true run.

**Known inconsistencies:** 7 of the 552 rows have metadata that contradicts what ran — three
In-ear sessions analysed with a free-field model, three the reverse, and one pupil file whose
filename lost its stream tag. Could be mislabels either way; unexamined.
