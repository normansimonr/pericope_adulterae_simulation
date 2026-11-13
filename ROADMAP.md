# ROADMAP: From Simulator to Scientific Workbench

> **Status:** authoritative roadmap
> **Version:** 1.0
> **Date:** 2026-11-13
> **Scope:** the `pasim` repository and the scientific programme around the
> *Pericope Adulterae* simulation.
>
> This document sets the direction for the next phase of the project. Where it
> conflicts with older planning documents, this roadmap wins on **direction**;
> the code and `project_overview.md` remain authoritative on **current
> behaviour** until they are updated.

---

## 0. How to use this document

This is a constitution, not a wish list. It defines:

- the scientific principles that must not be violated;
- the definitions of the quantities we report;
- the evidence standards for inputs and claims;
- the five workstreams that take the repository from "simulator" to
  "scientific workbench";
- the acceptance criteria for each workstream;
- the order in which the work should happen.

Anyone — human or AI agent — working in this repository should:

1. Read this document and the relevant module before changing code.
2. Preserve the principles in §2 unless this document is explicitly amended.
3. Add tests and provenance for every new scientific quantity.
4. Prefer explicit configuration over hidden constants.
5. Keep commits small and reviewable.
6. Update this roadmap when a decision changes, rather than silently diverging.

---

## 1. Purpose and starting point

`pasim` simulates the transmission of the *Pericope Adulterae* (PA, John
7:53–8:11) through a genealogy of manuscripts, with two textual regimes:

- **insertion:** the autograph lacks the PA; an innovator later introduces it;
- **omission:** the autograph contains the PA; an innovator later removes it.

The simulator already has a strong architecture: a demographic scaffold that is
independent of the text, deterministic dual-regime replay, parallel Monte Carlo
execution, persistence, and a substantial test suite.

The next phase is not "add more features to the simulator." It is to make the
repository capable of **scientific inference under sparse historical evidence**:
traceable inputs, explicit model variants, quantified uncertainty, model
discrimination, reproducible reports, and performance sufficient for
simulation-based exploration.

### 1.1 Blank-slate rule

The existing `experiments/exp001_baseline/results.csv`,
`experiment_metadata.json`, and the earlier paper draft are **historical
artefacts**, not validation targets. They may be cited for provenance, but:

- do not tune the model to reproduce them;
- do not treat their numbers as ground truth;
- do not spend a workstream reconstructing a stale draft;
- label them as superseded when they are no longer authoritative.

The current code is the engineering starting point. The scientific starting
point is the evidence ledger described in §4.

---

## 2. Non-negotiable principles

### P1. Evidence before inference

Every scientific claim must be traceable to an input, a model variant, a seed,
and a command. No number is reported without a denominator and a provenance
record.

### P2. No fabricated data

There is no manuscript-level dataset in this project. Do not invent one, scrape
an unlicensed catalogue, or present model output as observation. Use:

- historical input parameters with cited sources and uncertainty;
- coarse published aggregates with explicit denominators;
- qualitative historical facts as constraints.

### P3. Construct validity

Prevalence, majority status, raw production, and survival-weighted corpus
composition are different quantities. Never compare one to another without
saying so. Every metric must have a written definition and denominator.

### P4. Determinism is a scientific feature

Same code + same config + same seed must produce the same result. Any change to
the random-number stream or event order creates a **new engine version** and
must be documented, versioned, and validated.

### P5. No silent model changes

Performance work must not change the model. New mechanisms (institutional
policies, new parent selection, new survival models) must be explicit model
variants behind configuration and plugins, with the default path preserved.

### P6. Two engines

Keep a **reference engine** that represents the current semantics exactly.
Add a **fast engine** only with equivalence evidence. Never delete the reference
engine until the fast engine has passed the equivalence programme.

### P7. Evidence grades

Every input carries an evidence grade:

| Grade | Meaning | Examples |
|---|---|---|
| A | Direct primary data | A collated manuscript catalogue (not currently available) |
| B | Published aggregate or qualitative historical fact | ~85/15 continuous-text prevalence; early Egyptian omission |
| C | Expert judgment / scholarly consensus | Plausible migration rates, scriptoria influence |
| D | Modelling assumption | Grid geometry, mutation kernel, policy strength |

The workbench must make the grade visible in every report.

### P8. Uncertainty is the output

With sparse evidence, the honest output is a **plausible region**, a
**robustness statement**, or an **equifinality report** — not a single "best"
number. Avoid false precision.

### P9. Reports are generated, not hand-edited

Every figure and table in a scientific report must be reproducible from a
command and stored with the run that produced it.

### P10. No performance optimisation that breaks the agent logic

Batch only where the agent dependency is absent. In this model that means:

- within a tick, using the pre-tick alive snapshot;
- by topological generation in the text replay;
- never across ticks.

---

## 3. Definitions and metrics

### 3.1 Core entities

- **Manuscript:** a physical object with birth/death tick, material, region,
  location, and script.
- **Witness:** the textual content associated with one manuscript.
- **Witness instance / genealogy node:** the graph representation of one
  witness.
- **Tagged string:** a vector of `text_length` integer segments. `0` means the
  PA is absent; non-zero means the PA is present. Values `1–5` are variant
  readings within the PA-present state.
- **Tick:** one simulation time step. The current baseline uses annual ticks;
  any change to the time unit is a model-version change.

### 3.2 The metric contract

All reported quantities must use these definitions.

| Metric | Definition | Denominator | Notes |
|---|---|---|---|
| `prevalence_all` | Fraction of **all** witness instances with PA | All instances, excluding the autograph | Raw production |
| `prevalence_sampled` | Fraction of **sampled survivors** with PA | Sampled survivor set | Compare to real-world extant corpus |
| `majority_status` | `prevalence_sampled > 0.5` | One value per run | Not a prevalence |
| `omission_share` | `1 - prevalence` | Same as prevalence | Report alongside prevalence |
| `regional_prevalence(r, c)` | Prevalence in region `r` and century `c` | Witnesses in that cell | Primary observable for institutional hypotheses |
| `time_to_majority(r)` | First tick at which regional prevalence exceeds 0.5 | Region | Report with uncertainty |
| `minority_share` | Fraction of the corpus omitting the PA | Defined corpus | Real-world target is ~15% continuous-text |
| `monophyly` | Fraction of the promoted reading descending from the policy origin | Promoted-reading witnesses | Distinguishes institutional promotion from independent events |
| `policy_effect` | Difference between policy variant and matching baseline | Paired runs | Use common random numbers where possible |

### 3.3 Real-world constraints

The only external empirical aggregate currently available is the composition of
the extant Greek manuscript corpus:

- Among **continuous-text** manuscripts of John, roughly **84.7% include** the
  PA and **15.3% omit** it (Robinson 2016, as cited in the paper draft).
- Among **all** manuscripts, including lectionaries, roughly **67.8% include**
  and **32.3% omit**. Lectionaries omit passages for liturgical reasons, so the
  continuous-text figure is the primary comparison.
- Early Egyptian witnesses (e.g., P66, P75, Sinaiticus, Vaticanus) omit the PA;
  later Byzantine/Asia Minor witnesses generally include it.

These are **coarse constraints**, not a likelihood. They should be used as a
plausibility band, and only after accounting for the survivor sampler. The
simulation does not currently model lectionaries; that denominator must be
stated explicitly in any comparison.

### 3.4 What must never be conflated

- Real-world **corpus prevalence** (~85/15) is not the same as the fraction of
  simulation runs that reach **majority status**.
- **Raw production** prevalence is not **survival-weighted** prevalence.
- A **majority-status** statistic is not a **prevalence** statistic.
- The earlier paper draft's 20/90 is an **old model output**, not data.

Every report must print the metric name, definition, denominator, and whether it
is raw or survival-weighted.

---

## 4. The evidence model

### 4.1 No manuscript-level data

The project does not have access to a manuscript-level dataset. The scientific
inputs are:

1. **Historical parameter estimates** from scholarship (migration, lifespan,
   demand, script transitions, persecution dates, etc.).
2. **Coarse corpus aggregates** (the ~85/15 and ~68/32 figures above).
3. **Qualitative historical facts** (regional and chronological patterns).
4. **Explicit modelling assumptions** (grid geometry, mutation kernel, policy
   strength).

All four must be represented in a machine-readable evidence ledger with
provenance.

### 4.2 Evidence ledger

Each input must record:

```yaml
p_region_migration:
  value: 0.04
  range: [0.02, 0.08]
  distribution: uniform
  source: "Historical estimate; see citation field"
  citation: "..."
  evidence_grade: C
  notes: "Interpreted as per-tick probability of inter-region movement."
```

The ledger is the single source of truth for parameter uncertainty. A scenario
is a point or distribution in that ledger, not a hand-edited `params.yaml`.

### 4.3 Prior predictive checks

Before inference, simulate from the evidence ranges and ask:

- Does the model produce corpora inside the ~85/15 plausibility band?
- Does it produce the qualitative regional pattern?
- Which inputs are so influential that the answer is dominated by grade-D
  assumptions?

The output is a **prior predictive report**, not a posterior.

### 4.4 What replaces ABC

ABC is optional. It becomes meaningful only if the coarse aggregates are treated
as observations with an explicit observation model and summary statistics. The
default workflow is:

- **prior predictive checks** for plausibility;
- **history matching** for the not-ruled-out region;
- **sensitivity analysis** for input importance;
- **model discrimination** for choosing between structural hypotheses.

This is more honest than fitting a posterior to a handful of aggregates and
calling it calibration.

---

## 5. Model family

The scientific object is no longer a single insertion/omission model. It is a
family of models that differ in **origin** and **promotion mechanism**.

| Model | Origin of PA | Promotion mechanism | Purpose |
|---|---|---|---|
| **H0** | Present | None (neutral omission) | Null model for an original PA |
| **H1** | Absent (single insertion) | None (neutral insertion) | Null model for a later insertion |
| **H2** | Present | Alexandrian institutional omission | Test the Alexandrian recension hypothesis |
| **H3** | Absent | Lucianic/Byzantine institutional insertion | Test the Antiochene/Byzantine counterpart |
| **H4** | Present | Alexandrian omission + Lucianic restoration/promotion | Two-centre dynamic |
| **H5** | Absent | Lucianic promotion + Alexandrian omission | Exploratory interaction |

H4 and H5 are exploratory. They should not be presented as equally supported
without explicit evidence.

### 5.1 Institutional policies

An institutional policy is a **time-bounded, region- or centre-specific bias**,
not a single mutation event. The origin event (the innovator) and the promotion
mechanism are distinct.

Schema (illustrative):

```yaml
institutional_policies:
  - name: alexandrian_omission
    centre: alexandria
    start_tick: 180
    end_tick: 450
    target_reading: absent
    strength: 0.6
    mechanism: copying_bias

  - name: lucianic_insertion
    centre: antioch
    start_tick: 300
    end_tick: 900
    target_reading: present
    strength: 0.5
    mechanism: copying_bias
```

Supported mechanisms should include, at minimum:

- `copying_bias`: the institution's scriptoria are more likely to produce the
  target reading;
- `demand_multiplier`: the institution commissions more copies;
- `exemplar_authority`: the institution's manuscripts are preferred as
  exemplars.

The first is the least invasive and should be implemented first because it
preserves the fixed demographic scaffold.

### 5.2 Policy centres, not new regions

The current model has three coarse regions (Asia Minor, Egypt, Levant). Do not
multiply administrative regions just to represent Alexandria, Antioch, and
Constantinople. Add a **policy-centre** layer independent of `Region`:

```yaml
policy_centres:
  - name: alexandria
    location: [250, 250]   # inside Egypt
    radius: 40
  - name: antioch
    location: [450, 450]   # inside the Levant
    radius: 40
  - name: constantinople
    location: [80, 20]     # inside Asia Minor
    radius: 30
```

This gives geographical granularity for the recension hypotheses without
changing demand, migration, or survival. If finer geography is later needed,
use a hierarchy: coarse regions for demand/migration/survival, policy centres or
settlements for institutional effects and reporting, and continuous coordinates
for parent selection.

Any expansion of the region set must be accompanied by the region-scaling
benchmark in §6.1 and a fallback-rate report.

---

## 6. The five workstreams

Each workstream is a substantial project. It has an objective, deliverables,
acceptance criteria, dependencies, and an effort estimate. The estimates are for
an experienced developer or a frontier coding agent; "MVP" is the smallest
scientifically useful slice.

### 6.1 Workstream 1 — Fast, faithful core

**Objective:** make the simulator fast enough for ensemble exploration without
silently changing the model.

**Why it matters:** the current baseline is roughly 209 CPU-seconds and 742 MB
for one `exp001` run (276k nodes, two regimes); the checked-in batch metadata
records ~7 hours for 1,000 runs at 16-way parallelism. History matching and
sensitivity analysis need thousands to tens of thousands of runs.

**Reference facts**

- One `exp001` run: ~3 min 27 s wall, ~209 s CPU, ~742 MB peak RSS.
- Demographic phase dominates (~73% in the `exp004` profile; more in `exp001`).
- Hot spots: migration/relocation loops, spawn/parent selection, per-tick KDTree
  rebuilds, per-node RNG creation in replay, NetworkX attribute overhead.

**Phase A — bit-exact optimisations**

These must not change any output for a fixed seed/config.

- Replace NetworkX in the hot generation loop with compact arrays/caches; build
  `nx.DiGraph` only for snapshot/persistence/tests.
- Replay by topological generation, mapping each node's existing RNG to its node
  ID rather than its position.
- Lazy spatial index: build a region KDTree only when it is needed (spawns or
  policy-centre queries).
- `__slots__` and reduced registry lookups.
- I/O: Parquet/Arrow + zstd for large artefacts; survivor-only text persistence;
  atomic, append-friendly aggregation.
- Parallel scheduling: chunking, memory-aware worker count, no oversubscription.

**Phase B — engine v2 (new reproducibility contract)**

These preserve the agent rules but change the random stream, so they are a new
engine version.

- Per-event RNG streams derived from `(tick, manuscript_id, purpose)` or an
  equivalent documented scheme.
- Tick-snapshot spawning: generate all newborns from the pre-tick alive pool;
  batch KDTree queries and ranking.
- Sequential slow paths for the two dependencies that are genuinely
  order-sensitive: fallback parent selection and cultural replacement.
- Optional compiled kernel (Numba/Cython/Rust) for mutation/spatial loops.
- No batching across ticks. That would break the genealogy.

**Equivalence programme**

- Engine A (reference) and Engine B (fast) coexist behind `--engine`.
- Phase A: golden hashes of snapshots and replay texts.
- Phase B: distributional equivalence across N seeds — parent-selection
  distributions, per-tick population, regional prevalence, mutation counts,
  majority outcomes (KS/Wasserstein).
- Invariants: single autograph, DAG, PA-presence lock, event counts, no
  same-tick parent except documented fallback.
- Decision-trace comparison on a small config: log every parent choice and
  mutation decision; compare sequences.

**Region-scaling benchmark**

- 3, 6, and 12 regions at constant total population and demand.
- Measure wall time, RSS, KDTree build time, local-pool sizes, fallback rate.
- Target: <10–20% CPU overhead for 6–12 regions with lazy indexing; report any
  change in fallback rate, because it can alter the model.

**Acceptance criteria**

- Phase A: at least 2–3× single-run speedup, bit-exact on the golden set.
- Phase B: 5–10× single-run speedup, passing the equivalence programme.
- 1,000 runs in ≤2 hours at 16 workers; peak RSS ≤2 GB.
- All existing tests pass; new benchmarks run in CI.
- The RNG contract and engine version are recorded in every run's provenance.

**Dependencies:** none. Enables Workstreams 3 and 4.

**Effort:** 5–8 h MVP; 1–3 days full.

---

### 6.2 Workstream 2 — Historical evidence ledger and scenario ensemble

**Objective:** make every historical input traceable, uncertain, and
machine-readable, and turn the single `params.yaml` into a reproducible scenario
ensemble.

**Why it matters:** the only real data are historical estimates and one coarse
corpus aggregate. A workbench must treat them as evidence, not as fixed numbers,
and must not fabricate precision.

**Deliverables**

- `src/pasim/evidence/`:
  - Pydantic schema for the evidence ledger;
  - loader/validator;
  - provenance model (source, citation, grade, range, distribution, notes);
  - export to the simulation configuration.
- `src/pasim/scenarios/`:
  - scenario definitions;
  - Latin-hypercube and Sobol sampling over uncertain inputs;
  - deterministic seed derivation;
  - scenario-to-config materialisation.
- A prior predictive report: run a small ensemble and compare outputs to the
  ~85/15 plausibility band and the qualitative regional facts.
- CLI:
  - `pasim evidence validate`
  - `pasim evidence report`
  - `pasim scenario sample --n 100 --seed 42`
  - `pasim scenario run --scenario baseline --n-runs 100`
  - `pasim scenario report`
- Tests: schema validation, deterministic sampling, provenance completeness,
  prior predictive report golden file.

**Acceptance criteria**

- Every input in the baseline configuration has a source and an evidence grade.
- Scenario sampling is reproducible from a seed.
- The prior predictive report states explicitly which real-world constraints are
  satisfied, which are not, and which are underdetermined.
- No manuscript-level dataset is invented or committed.

**Dependencies:** none; feeds Workstreams 3 and 4.

**Effort:** 5–8 h MVP; 1–2 days full.

---

### 6.3 Workstream 3 — Institutional recension models

**Objective:** implement the Alexandrian and Lucianic/Byzantine hypotheses as a
general, configurable family of institutional policies, and compare them against
neutral transmission.

**Why it matters:** the observed corpus contains a substantial minority of
PA-omitting manuscripts (~15% continuous-text). The current neutral omission
regime produces almost none in the sampled corpus, while the insertion regime
produces almost no PA. The literature proposes institutional promotion of the
shorter text in Alexandria and, symmetrically, of the longer text in the
Antiochene/Byzantine sphere. These are structural mechanisms, not parameter
tweaks.

**Deliverables**

- `src/pasim/models/`:
  - `InstitutionalPolicy` schema and registry;
  - policy-centre definitions;
  - mechanism interfaces (`copying_bias`, `demand_multiplier`,
    `exemplar_authority`);
  - policy-aware replay hooks.
- Model registry for H0–H5.
- Region/policy-centre metrics: regional prevalence, majority status,
  time-to-majority, minority share, concentration, monophyly, policy effect.
- Raw vs survival-weighted reporting for every regional metric.
- Extend `results_writer` and the run store to persist these metrics.
- Tests: policy determinism, default-model equivalence, policy effect on a
  controlled toy graph, metric correctness.
- A study comparing H0–H5 on a shared demographic scaffold where possible.

**Acceptance criteria**

- With the policy disabled, results are identical to the matching baseline.
- Enabling a policy produces a measurable, correctly signed effect.
- The H2 and H3 variants are configurable without code changes.
- The report distinguishes production from survival-weighted prevalence.
- Monophyly is computed and reported for the promoted reading.
- No new administrative region is required; policy centres are used.

**Dependencies:** Workstreams 1 and 2 (speed and evidence). Workstream 1 is not
strictly required for correctness, but is required for a useful ensemble.

**Effort:** 8–12 h MVP; 2–3 days full.

---

### 6.4 Workstream 4 — Model discrimination, sensitivity, and experimental design

**Objective:** determine which model variants are distinguishable under the
available evidence, which inputs matter, and which hypothetical observations
would be most decisive.

**Why it matters:** with sparse evidence, the main scientific output is not a
single posterior. It is a statement about robustness, equifinality, and the value
of further evidence.

**Deliverables**

- `src/pasim/inference/` (or `analysis/`):
  - Morris screening;
  - Sobol indices with confidence intervals;
  - common-random-number pairing across model variants;
  - history matching / NROY with implausibility;
  - Gaussian-process emulator for expensive parameter regions;
  - pairwise model discrimination;
  - equifinality matrix;
  - expected-information-gain / experimental-design ranking.
- CLI:
  - `pasim sensitivity --study ...`
  - `pasim history-match --study ...`
  - `pasim discriminate --study ...`
  - `pasim design --study ...`
- Report: which models survive, which pairs need new evidence, and which
  observations would be most decisive.
- Tests: Sobol on an analytic function (e.g., Ishigami), NROY on a synthetic
  simulator, discrimination on known model pairs, determinism.

**Acceptance criteria**

- Sobol indices and Morris rankings are reproducible and carry uncertainty.
- History matching returns an NROY region with a documented implausibility
  threshold.
- The equifinality matrix identifies which H0–H5 pairs are indistinguishable
  under the current constraints.
- The design report ranks at least three hypothetical observations by expected
  information gain.
- No result is presented as a posterior unless a proper observation model and
  summary statistics are defined.

**Dependencies:** Workstreams 2 and 3.

**Effort:** 8–12 h MVP; 2–3 days full.

---

### 6.5 Workstream 5 — Scientific workbench

**Objective:** one command from hypothesis to auditable report, with provenance,
validation, and a queryable store.

**Why it matters:** a scientific workbench is not a collection of scripts. It is
a system in which every number can be traced, every run can be reproduced, and
every claim is bounded by its evidence.

**Deliverables**

- `src/pasim/store/`:
  - SQLite/DuckDB + Parquet store;
  - tables for studies, runs, regimes, metrics, parameters, artefacts,
    provenance;
  - import of existing `results.csv` as historical data, clearly labelled;
  - query API.
- Provenance:
  - git commit and dirty-diff hash;
  - lockfile hash;
  - Python/package versions;
  - platform;
  - seeds and engine version;
  - configuration hash;
  - command line and timestamp.
- `pasim study`:
  - declarative study YAML (model variants × scenarios × replicates);
  - resumable execution;
  - aggregation;
  - report generation.
- `src/pasim/analysis/`:
  - metrics with confidence intervals;
  - effect sizes and convergence diagnostics;
  - plots: violin, time series, regional maps, sensitivity tornado,
    discrimination matrix, posterior/plausibility plots;
  - report templates (Markdown/HTML/Quarto).
- `src/pasim/validation/`:
  - scientific invariants (single autograph, DAG, PA lock, conservation,
    deterministic replay);
  - golden datasets;
  - statistical regression tests;
  - CI workflow.
- Documentation:
  - workbench guide;
  - data dictionary;
  - model card for H0–H5;
  - evidence ledger guide;
  - reproduction instructions.

**Acceptance criteria**

- Fresh clone → `poetry install` → `pasim study studies/pa_institutional_comparison/study.yaml`
  → report, with provenance and diagnostics.
- Re-running the study reproduces identical metrics.
- Store queries return the same numbers as the generated report.
- `pasim validate` runs the scientific invariants and passes.
- Every figure and table in the report is generated, not hand-edited.
- The old `results.csv` and paper draft are stored as historical artefacts and
  labelled superseded.

**Dependencies:** all previous workstreams; can start early with the store and
provenance, but the report cannot be completed until 3 and 4 exist.

**Effort:** 6–10 h MVP; 2–3 days full.

---

## 7. Sequencing and milestones

### Phase 0 — Hygiene (½–1 day, not a workstream)

- Repair the local environment (`poetry install`); the repository `.venv` may be
  incomplete.
- Fix `run_single(..., persistence_level="minimal")` returning `None` while its
  type hint suggests a result object; update `debug_run.py` accordingly.
- Add the benchmark harness skeleton and record the baseline.
- Write the metric contract (§3.2) into the docs.
- Mark the old `results.csv`, `experiment_metadata.json`, and paper draft as
  historical artefacts, not targets.
- Do **not** spend a workstream reconstructing the old paper's numbers.

### Milestone A — Foundations

- Workstream 1 (fast core) reaches Phase A + the region-scaling benchmark.
- Workstream 2 (evidence ledger + scenarios) is usable.
- Exit criterion: a scenario ensemble can run at the new speed with traceable
  inputs.

### Milestone B — Mechanisms and inference

- Workstream 3 (institutional policies, H0–H5) is implemented.
- Workstream 4 (sensitivity, history matching, discrimination) produces a first
  equifinality matrix.
- Exit criterion: a study can compare H0–H5 and state which are distinguishable.

### Milestone C — Workbench and capstone

- Workstream 5 (store, provenance, validation, reports) is complete.
- Capstone study: `studies/pa_institutional_comparison/`.
- Exit criterion: one command produces an auditable report with regional
  prevalence, model comparison, sensitivity, equifinality, and provenance.

### Capstone acceptance

The capstone study must:

1. Run H0–H5 on a shared or explicitly paired demographic scaffold.
2. Report raw and survival-weighted regional prevalence.
3. Report majority status separately from prevalence.
4. Include a prior/evidence ledger and a provenance record.
5. Include Sobol/Morris rankings and an NROY region.
6. Include an equifinality matrix and a ranked list of discriminating
   observations.
7. State clearly which conclusions are robust and which depend on grade-C/D
   assumptions.
8. Be reproducible from a fresh clone in under two hours on a mid-range desktop.

---

## 8. Definition of done

The project reaches "scientific workbench" status when:

- **Engineering:** a fresh clone installs, tests pass, benchmarks pass, and the
  capstone runs reproducibly.
- **Scientific:** every model variant has a written specification; every metric
  has a definition and denominator; every input has a source and evidence grade;
  every claim is accompanied by uncertainty or a robustness statement.
- **Provenance:** every reported number is traceable to a run, a config, a seed,
  an engine version, and a code revision.
- **Performance:** the fast engine is at least 5× faster than the reference
  engine and passes the equivalence programme; the reference engine remains
  available.
- **Communication:** the report generator produces the figures and tables used
  in the paper; no number is hand-copied from a notebook.
- **Honesty:** the project does not claim manuscript-level data it does not have,
  does not present model output as observation, and does not hide equifinality.

---

## 9. Risks, non-goals, and decision log

### Risks

| Risk | Mitigation |
|---|---|
| Performance work changes the model | Two engines; golden hashes; distributional equivalence; versioned RNG contract |
| More regions slow the simulation | Policy centres instead of new regions; lazy spatial index; region-scaling benchmark |
| Sparse evidence leads to false precision | Evidence grades; history matching; robustness statements; no ABC by default |
| Equifinality is mistaken for failure | Report it explicitly; use it to design future evidence |
| Scope creep in the workbench | Milestones; MVP acceptance criteria; no dashboard before the statistical core |
| Stale numbers propagate into papers | Mark old artefacts superseded; generated reports only; provenance checks |
| Data licensing/provenance | No manuscript-level ingestion; cite aggregates; keep the ledger auditable |

### Non-goals

- Ingesting or fabricating a manuscript-level catalogue.
- Full simulation-based inference (SBI/ABC) as a core requirement.
- Adding many administrative regions without a benchmark and a scientific need.
- Changing the default model silently.
- Building a visual dashboard before the metrics, evidence, and provenance are
  sound.
- Claiming historical certainty from a simulation.

### Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-13 | Treat the existing results and paper draft as historical artefacts, not targets | They are outputs of earlier model versions; the project starts from a blank scientific slate |
| 2026-11-13 | Use ~85/15 continuous-text prevalence as a coarse external constraint | It is the only available empirical aggregate; use as a plausibility band, not a likelihood |
| 2026-11-13 | Prefer policy centres over new regions | Geographical granularity for the recension hypotheses without re-parameterising demand, migration, and survival |
| 2026-11-13 | Two-engine performance strategy | Preserve exact reproducibility while allowing a faster, statistically equivalent engine |
| 2026-11-13 | Model family H0–H5 | Separate origin from promotion mechanism; make equifinality explicit |
| 2026-11-13 | History matching first, ABC optional | Sparse evidence does not justify a likelihood by default |

---

## 10. Working protocol for agents and contributors

1. **Read before writing.** Read this roadmap, `project_overview.md`, and the
   modules you will touch.
2. **Design note first.** For changes spanning more than one module, write a
   short design note in the pull request or a `docs/design/` file.
3. **Preserve the default.** New behaviour goes behind configuration or a
   plugin; the default path remains the reference.
4. **Test the science, not just the code.** Add metric tests, invariant tests,
   and reproducibility tests.
5. **Benchmark performance changes.** Report CPU time, wall time, peak RSS, and
   output equivalence.
6. **Record provenance.** Every new input, metric, and output gets a source or a
   definition.
7. **Keep commits small.** One logical change per commit; update the roadmap
   when a decision changes.
8. **Never delete user work or tracked results without explicit confirmation.**
9. **Do not present stale results as current.** Label, archive, or supersede.
10. **When in doubt, prefer an explicit, documented assumption over a hidden
    constant.**

---

## 11. Immediate next steps

1. Complete Phase 0 hygiene and record the baseline benchmark.
2. Implement Workstream 2's evidence-ledger schema and a first `evidence.yaml`
   for the baseline configuration.
3. Implement Workstream 1 Phase A (bit-exact optimisations) and the benchmark
   harness.
4. Implement Workstream 1 Phase B behind `--engine fast`, with the equivalence
   programme.
5. Implement Workstream 3's `InstitutionalPolicy` and the H0–H5 registry.
6. Implement Workstream 4's first sensitivity and history-matching pass.
7. Implement Workstream 5's store, provenance, and report generator.
8. Run the capstone study and use its output as the basis for the next paper
   draft.

---

*This roadmap is a living document. Amend it when the science or the evidence
changes; do not let the code drift away from it in silence.*
