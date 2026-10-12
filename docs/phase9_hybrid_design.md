# CineMatch Phase 9 hybrid design

## Signals and provisional weights

Phase 9 fuses seven signal families. The values below are design-time defaults,
not values learned from ratings or optimized against the known examples.

| Signal | Weight | What it contributes |
|---|---:|---|
| Phase 5 metadata | 0.34 | Genre and plot-tag overlap using Jaccard similarity. |
| TF-IDF plot | 0.16 | Weighted word overlap in the plot. |
| Transformer plot | 0.22 | Semantic plot representation from the unchanged Phase 7 artifact. |
| Director | 0.10 | Exact Wikidata person-ID overlap. |
| Writer | 0.07 | Exact Wikidata person-ID overlap. |
| Cast | 0.07 | Exact Wikidata person-ID overlap. |
| Selected crew | 0.04 | Same person in the same selected crew role. |

The Phase 5 metadata score retains its 0.7 genre / 0.3 tag Jaccard weighting
when both fields are present. For a pair with one missing field, weights are
renormalized over the fields present on both movies; a missing field is not
treated as proof of dissimilarity. Genre and tag Jaccard scores remain
individually visible in the result for inspection.

## Score combination

Raw component scores have different meanings, so they are not added directly.
For each query, each available component score is converted to a percentile
rank among candidates for which that component can be compared. The weighted
average of those percentile ranks is multiplied by the fraction of the query's
configured signal weight available for that candidate:

`hybrid_score = weighted_mean(component_rank_percentiles) × evidence_coverage`

This rank-fusion score is an ordering aid, not a probability or a measure of
absolute story relevance. A candidate needs at least two positive component
similarities. Missing signals are omitted from pair scoring and reduce the
available-signal coverage factor. Ties are broken by positive-signal count and
then ascending MovieLens movie ID.

## Credit treatment and explanations

People match by Wikidata QID, not by name strings. Selected crew uses
`role:QID`, so the same person listed in different roles does not count as the
same crew credit. Explanations are generated from actual shared genres, tags,
positive plot scores, and shared credit IDs. Empty or unavailable fields never
produce a “shared” explanation.

## Long-plot experiment

The primary Phase 7 vectors are untouched. The separate experiment uses the
same pinned model, tokenizes the plots, splits inputs longer than the model's
256-token limit into 224-token windows with 48-token overlap, embeds each
window, averages a movie's normalized window vectors, and normalizes the
result. Vectors for short plots are reused from Phase 7. The experimental
artifact is stored separately under `data/models/` and ignored by Git.

The exact model, chunk counts, representative comparisons, and Phase 5/6/7/9
recommendation lists are recorded in
`reports/phase9_hybrid_diagnostics.json`. That report is qualitative; no held-
out rating evaluation or model-weight tuning is performed in this phase.
