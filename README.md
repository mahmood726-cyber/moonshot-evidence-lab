# Moonshot Evidence Lab

Moonshot Evidence Lab is a starter project for an "autoresearch for evidence synthesis" workflow.

The original `karpathy/autoresearch` pattern is:

- keep the system small
- define a clear experiment surface
- let an agent iterate on that surface
- score changes automatically

This repo ports that pattern to public health evidence synthesis and meta-analysis instead of GPU model training.

## What This Repo Does

- provides a single, explicit agent-tunable surface in `moonshot_evidence_lab/program_surface.py`
- keeps the core data and scoring utilities fixed and reviewable
- supports WHO indicator URL generation and JSON download helpers
- supports an IHME dataset registry for locally staged downloads
- supports PubMed and ClinicalTrials.gov literature discovery connectors
- extracts first-pass effect-size candidates from PubMed abstracts and ClinicalTrials.gov results
- ingests local CSV evidence tables for arm-level or precomputed effect-size datasets
- computes fixed-effect and DerSimonian-Laird random-effects meta-analysis
- adds Paule-Mandel random-effects estimation, Hartung-Knapp intervals, prediction intervals, and richer heterogeneity summaries
- adds advanced diagnostics including Egger regression, leave-one-out analysis, subgroup analysis, cumulative meta-analysis, and multivariable meta-regression
- adds Bayesian random-effects pooling and clustered multilevel random-effects meta-analysis
- adds contrast-based network meta-analysis and multivariate meta-analysis for correlated outcomes
- includes local benchmarks and smoke tests so agents have something objective to optimize against

## Project Layout

```text
moonshot-evidence-lab/
  moonshot_evidence_lab/
    connectors/
      clinicaltrials.py
      ihme.py
      pubmed.py
      who.py
    advanced_meta.py
    autoresearch_loop.py
    benchmarks.py
    cli.py
    evidence.py
    literature.py
    meta_analysis.py
    models.py
    program_surface.py
    review_pipeline.py
    tabular_data.py
  data/
    benchmarks/
    examples/
    processed/
    raw/
    registries/
  tests/
  program.md
  pyproject.toml
```

## Quick Start

Run everything from the project root:

```powershell
python -m moonshot_evidence_lab smoke-test
python -m moonshot_evidence_lab meta-demo
python -m moonshot_evidence_lab literature-benchmark
python -m moonshot_evidence_lab pubmed-url --term "stroke prevention"
python -m moonshot_evidence_lab pubmed-search --term "aspirin" --retmax 3
python -m moonshot_evidence_lab pubmed-extract --pmid 31237833
python -m moonshot_evidence_lab ctgov-url --condition "stroke" --term "aspirin"
python -m moonshot_evidence_lab ctgov-search --condition "stroke" --term "aspirin" --page-size 3
python -m moonshot_evidence_lab ctgov-extract --nct-id NCT00741286
python -m moonshot_evidence_lab review-pipeline --term "aspirin stroke" --condition "stroke" --include-secondary-evidence
python -m moonshot_evidence_lab advanced-meta --input data\examples\stroke_binary_trials.csv --format binary --measure or --subgroup-key subgroup --meta-regression-key follow_up_months
python -m moonshot_evidence_lab advanced-meta --input data\examples\pulsatility_continuous_trials.csv --format continuous --measure md --subgroup-key subgroup --meta-regression-key dose_mg
python -m moonshot_evidence_lab advanced-meta --input data\examples\hypertension_prevalence_surveys.csv --format proportion --measure logit --tau-method pm --subgroup-key region --meta-regression-key mean_age --meta-regression-key urban_share --cumulative-key year
python -m moonshot_evidence_lab advanced-meta --input data\examples\stroke_multivariable_effects.csv --format effects --tau-method pm --meta-regression-key follow_up_months --meta-regression-key mean_age --meta-regression-key baseline_risk --cumulative-key year
python -m moonshot_evidence_lab bayesian-meta --input data\examples\stroke_binary_trials.csv --format binary --measure or
python -m moonshot_evidence_lab multilevel-meta --input data\examples\stroke_clustered_effects.csv --format effects --cluster-key study_id
python -m moonshot_evidence_lab network-meta --input data\examples\stroke_network_contrasts.csv --reference-treatment Placebo
python -m moonshot_evidence_lab multivariate-meta --input data\examples\stroke_multivariate_wide.csv
python -m moonshot_evidence_lab who-url --indicator WHOSIS_000001 --top 5
python -m moonshot_evidence_lab ihme-template
python -m unittest discover -s tests -v
```

## How To Use The Autoresearch Pattern Here

`program.md` is the instruction layer for the agent.

`moonshot_evidence_lab/program_surface.py` is the primary mutable file. The idea is that the agent mostly changes:

- inclusion and exclusion heuristics
- source trust weighting
- model selection logic
- other narrowly scoped synthesis rules

The fixed files do the rest:

- connectors fetch or register data
- evidence utilities normalize study effects
- meta-analysis utilities score pooled results
- the demo benchmark produces a repeatable scorecard

## Data Strategy

WHO:

- use WHO indicator metadata and country-year observations as benchmark context
- keep the WHO base URL configurable because WHO has been transitioning its OData infrastructure

IHME:

- keep a local registry of datasets you have rights to use
- stage downloaded files under `data/raw/ihme/...`
- record provenance in `data/registries/ihme_catalog.template.csv`

Literature:

- use PubMed and ClinicalTrials.gov for discovery, provenance, and screening
- ClinicalTrials.gov searches collect all pages. `--page-size` and `--ctgov-page-size` control records per request, not the total result limit; broad queries may take longer. Invalid pages, repeated continuation tokens, and failed requests raise an error instead of returning a partial search.
- use `NCBI_API_KEY` and `NCBI_EMAIL` environment variables when you want higher-throughput PubMed calls
- extract ratio-based abstract estimates from PubMed when the abstract reports metrics and confidence intervals
- derive count-based and mean-difference candidates from ClinicalTrials.gov results tables when denominators are present
- run `review-pipeline` to search, screen, extract, deduplicate by study and outcome, and pool compatible effects
- convert extracted study arms or precomputed effect sizes into the canonical `StudyEffect` format
- pool them with the provided meta-analysis helpers

Local tabular data:

- place review-ready datasets under `data/examples/` or your own local path and load them with `advanced-meta`
- supported dataset formats are `effects`, `binary`, `continuous`, and `proportion`
- supported binary measures are `or`, `rr`, and `rd`
- supported continuous measures are `md`, `smd`, and `rom`
- supported proportion measures are `logit`, `raw`, and `arcsine`
- keep extra columns such as `year`, `source`, `subgroup`, and numeric moderators in the CSV so diagnostics can use them directly
- use contrast-level CSVs for network meta-analysis with `treatment_a`, `treatment_b`, `effect_size`, and `variance`
- use wide multivariate CSVs for correlated outcomes with `<outcome>__effect`, `<outcome>__variance`, and optional `<outcome1>__<outcome2>__covariance`
- example datasets now include binary trials, continuous outcomes, precomputed log effects, multivariable effect tables, clustered repeated-effects tables, treatment networks, multivariate wide outcome tables, and prevalence surveys

Advanced statistics and maths:

- `advanced-meta` computes pooled estimates with either normal or Hartung-Knapp intervals
- random-effects summaries support both DerSimonian-Laird and Paule-Mandel tau-squared estimation
- heterogeneity summaries include tau-squared, Q, Q p-value, I-squared, H, H-squared, and prediction intervals
- diagnostics include Egger regression and leave-one-out influence analysis
- structured moderators support subgroup analyses and multivariable meta-regression
- cumulative meta-analysis can pool studies in a chosen metadata order such as `year`
- transformed outputs are back-transformed in the final summary so odds ratios, risk ratios, ratio-of-means estimates, and proportions are readable
- `bayesian-meta` fits a normal-normal random-effects model on a tau grid with `half-normal`, `half-cauchy`, or `uniform` priors
- Bayesian outputs include posterior credible intervals, posterior predictive intervals, tau posterior summaries, and posterior probability that the pooled effect is below or above zero
- `multilevel-meta` fits a clustered random-effects model for repeated effects within the same study and reports cluster-level shrinkage estimates
- `network-meta` fits a contrast-based treatment network with weighted least squares, residual heterogeneity, pairwise relative effects, and treatment ranking by P-score
- `multivariate-meta` pools correlated study endpoints jointly with generalized least squares and a diagonal between-study covariance assumption

## Current Status

This is still a scaffold, not a finished autonomous system. It now gives you:

- a clean repo
- a CLI
- first-pass meta-analysis code
- advanced meta-analysis diagnostics and higher-level statistical summaries
- Bayesian pooling and clustered multilevel modeling
- network meta-analysis and multivariate correlated-outcome pooling
- first-pass WHO, IHME, PubMed, and ClinicalTrials.gov connectors
- first-pass effect extraction for PubMed abstracts and ClinicalTrials.gov results
- local CSV ingestion for binary, continuous, proportion, and precomputed effect datasets
- seeded example datasets for stroke, pulsatility, multivariable moderator analyses, clustered repeated-effects analyses, treatment networks, multivariate outcomes, and prevalence analyses
- an agent instruction file

The next serious steps would be to add:

1. abstract and full-text extraction beyond simple heuristic parsing
2. inconsistency diagnostics, node-splitting, and multi-arm correlation handling for richer network meta-analysis
3. full multivariate random-effects covariance estimation instead of the current diagonal assumption
4. benchmark datasets with known "good" syntheses for specific review questions
5. a real agent loop that proposes, runs, scores, and keeps or rejects edits

## References

- Karpathy autoresearch: https://github.com/karpathy/autoresearch
- WHO GHO OData API: https://www.who.int/data/gho/info/gho-odata-api
- WHO Data portal background: https://www.who.int/news-room/feature-stories/detail/who-releases-data.who.int
- NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- PubMed DTD and XML docs: https://www.ncbi.nlm.nih.gov/books/NBK3828/
- ClinicalTrials.gov API: https://clinicaltrials.gov/data-api/api
- ClinicalTrials.gov study data structure: https://clinicaltrials.gov/data-about-studies/study-data-structure
- IHME data transparency: https://www.healthdata.org/Data-tools-practices/data-practices/data-transparency
- GHDx download help: https://ghdx.healthdata.org/help/downloading-data
