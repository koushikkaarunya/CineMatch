# Phase 14.1 story-pair review and label provenance

The audit file is `data/audit/phase14_1_story_relevance_pairs.csv`. Its plot text is copied from `data/cleaned/movies.csv`; the evaluation script verifies exact text and IDs.

## Relevance scale

- `0` — the central stories are unrelated. Shared names, vocabulary, cast, genres, or ratings do not make a pair relevant.
- `1` — a broad theme or situation overlaps, but the main conflict and story progression are meaningfully different.
- `2` — the central story, conflict, or character arc is meaningfully related beyond a shared genre or a few words.

Judge the stories described by the plots. Do not use model scores, popularity, ratings, or outside knowledge as evidence.

## Existing labels and provenance

The audit has 22 relevance values in `approved_relevance`. They were first suggested by an AI, then reviewed and approved by the user. Their `label_provenance` and `label_status` are both `user_approved_ai_assisted`. These values are **not independent human judgments**. The separate `proposed_relevance` column preserves the original suggestions.

Three rows (`SR004`, `SR018`, and `SR025`) lack at least one usable plot. They have blank relevance fields and `not_applicable_missing_plot` provenance; do not treat them as negatives.

The evaluation CLI accepts `proposed` and `user_approved_ai_assisted`. It deliberately does not accept `human`, because this audit contains no independently collected human labels. To reproduce the user-approved AI-assisted evaluation:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_1_story_relevance.py --label-source user_approved_ai_assisted
```

If an independent reviewer later labels pairs, store those judgments separately with reviewer identity/process and update the evaluation protocol before reporting them. Do not relabel or overwrite the existing approved values as independent human judgments.

## Limits

The 25 pairs are a small, deliberately selected diagnostic set (22 plot-available pairs across five query groups), not a representative sample. A single user approval of AI-assisted suggestions does not measure inter-rater agreement or independent reviewer reliability. Metrics are descriptive only and cannot establish general model quality.
