# Program

You are working inside Moonshot Evidence Lab, a Karpathy-style autoresearch scaffold for evidence synthesis and meta-analysis.

## North Star

Build an autonomous system that can:

1. gather structured evidence from WHO, IHME, and literature-derived datasets
2. ingest trial and publication metadata from ClinicalTrials.gov and PubMed
3. normalize outcomes into a shared schema
4. decide what evidence should be included
5. pool effect sizes reproducibly
6. generate defensible public-health synthesis outputs

## Core Principle

Prefer changing one narrow experiment surface over changing the entire codebase.

The primary mutable file is:

- `moonshot_evidence_lab/program_surface.py`

Only widen the edit surface if there is a hard blocker.

## Success Conditions

A change is good when it improves one or more of:

- inclusion quality
- source trust calibration
- model-selection logic
- reproducibility
- benchmark score stability

## Constraints

- Do not fake benchmark improvements.
- Do not invent data provenance.
- Do not silently loosen inclusion rules just to increase study count.
- Treat WHO and IHME as reference data lanes, not interchangeable replacements for trial-level evidence.
- Treat PubMed and ClinicalTrials.gov as discovery and provenance lanes until effect sizes are explicitly extracted.
- Prefer deterministic logic over hidden magic.

## Suggested Loop

1. Read the current benchmark result.
2. Form one concrete hypothesis.
3. Edit `moonshot_evidence_lab/program_surface.py`.
4. Run `python -m moonshot_evidence_lab smoke-test`.
5. Keep the change only if the result is more defensible or objectively better.

## High-Value Next Iterations

- tune source trust priors by source type
- add explicit handling for ecological versus trial data
- add small-study sensitivity logic
- add subgroup and leave-one-out analysis
- add ingestion from manual extraction sheets
- add proper benchmark fixtures for known review questions
- connect PubMed and ClinicalTrials.gov discovery directly to extraction workflows
