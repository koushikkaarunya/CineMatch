# CineMatch Phase 12: Unified Hybrid Recommender

## Scope and architecture

Phase 12 adds a coordinator in `scripts/recommend_hybrid.py`. It calls, rather
than reimplements, the existing Phase 9 `HybridContentRecommender`, Phase 10
`ItemItemCollaborativeRecommender`, and Phase 11 `ColdStartRecommender`.
Movie-to-movie requests remain Phase 9 content requests. Explicit favorites for
a new user go to Phase 11. Known-user requests can combine collaborative and
content candidate rankings.

`build_hybrid_recommender(ratings=...)` passes the same supplied interaction
table to the Phase 10 model and to user-history/seed selection. The coordinator
checks that a supplied Phase 10 model, when it exposes its fitted ratings,
matches that table. This lets a future evaluation pass only training rows and
prevents seed selection from accidentally reading a different, held-out set.
The normal command-line demonstration uses all available MovieLens rows and is
not a held-out evaluation.

## Rank fusion

Phase 9's `hybrid_score` is a content rank-fusion value. Phase 10's
`predicted_rating` is an estimate on the MovieLens 0.5–5.0 scale. The numbers
do not have compatible meanings, so they are not averaged.

For candidate movie `m`, with collaborative rank `r_c` and content rank `r_t`,
when both sources return eligible lists:

`S(m) = 0.5 × (1/r_c(m) + 1/r_t(m))`

A missing candidate rank contributes zero. For example, ranks 2 and 1 produce
`0.5 × (1/2 + 1) = 0.75`. A movie present only at content rank 2 receives
`0.5 × (0 + 1/2) = 0.25` when both sources are available. When only one source
returns candidates, that source's reciprocal rank is used alone; the unavailable
source does not dilute it. Ties break by ascending MovieLens `movie_id`.

Availability means that a source produced at least one eligible unseen
candidate after validation and exclusion of the user's rated movies. A
candidate omitted from an available source receives zero for that source;
this is not treated as a negative rating. Candidate rows retain Phase 10's
`predicted_rating`, `support_count`, and `evidence_weight` separately. The
specified rank-fusion formula does not recalibrate source reliability using
those fields. Phase 13 should test whether this equal source-rank contribution
is useful.

`fusion_score` is only an ordering value. It is not a probability, confidence,
rating prediction, or claim of relevance.

## Personalized history and cold start

For known users, the coordinator reads only that user's rows from the supplied
ratings table. It calculates that user's mean rating, selects movies rated
strictly above the mean, sorts by rating descending then ID ascending, and caps
the seed list at five. This cap is a simple starting choice, not a tuned value.
If all ratings are equal, or the history has only one rating, no positive seed
is inferred. The collaborator is asked for more preferences if neither source
can make a useful list.

The seed IDs go through Phase 11's `recommend_from_favorites`, preserving its
mean-reciprocal-rank aggregation across favorite movies. Its returned list is
then ranked as a content source for Phase 12 fusion. Rated movies are removed
before final ranks are assigned. The coordinator requests a larger candidate
pool (`5 × Top-K + number of seen movies`) to reduce the chance that filtering
seen movies leaves too few results; this is a retrieval safeguard, not a tuned
model parameter.

For new users, selected favorites are delegated to Phase 11. No preferences
returns `needs_preferences`. Movie-to-movie recommendations delegate to Phase
11's Phase 9 content route and remain explicitly distinct from user
personalization. Unknown users and unknown movie IDs remain separate errors or
statuses from known users with inadequate evidence.

The Phase 10 model already enforces its co-rater threshold and similarity
shrinkage. The coordinator does not change those formulas. It treats a
collaborative source as available only when Phase 10 returns candidates; each
candidate keeps its support/evidence fields so the ranking can be inspected.
If no collaborative candidates are available but content candidates are, it
returns a labeled content fallback. The reverse case remains collaborative-only.

## Targeted diagnostics

Run the qualitative experiments with cached local artifacts:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase12_experiments.py
```

The command writes `reports/phase12_targeted_diagnostics.json`. It does not
modify Phase 6–10 models, weights, cleaned data, or cached embeddings.

### TF-IDF character-name sensitivity

The original Phase 6 top five for `Waiting to Exhale` (movie 4) were *Savannah
Smiles* and four Robin Hood/Robin titles. The measured cosine contribution of
the explicitly targeted name unigrams (`Max`, `Robin`, `Savannah`) accounted
for 85–92% of these candidates' cosine scores. For `Grumpier Old Men` (movie
3), the names accounted for 89–99% of the original top-five similarities.
Masking only those diagnostic name tokens changed all five results for both
queries. For `Father of the Bride Part II` (movie 5), the names contributed zero
to its top-five cosine scores and the ranking did not change.

This is a useful ablation showing that name overlap can dominate these known
examples. It is **not** installed as the default TF-IDF behavior: a named
character can genuinely identify a related story, and this targeted list is
not a reliable general-purpose proper-name detector. A controlled test shows
that masking a name can lower similarity for a genuinely matching plot. The
original Phase 6 implementation remains the baseline.

### Transformer unrelated-match investigation

The Phase 7 rankings were reproduced for the same queries. For example,
`Father of the Bride Part II` ranked *When a Stranger Calls* (1979 and 2006)
first and second. As two alternatives, the experiment recorded (a) reciprocal
rank fusion of Phase 7 and Phase 5's metadata list, and (b) the existing Phase 9
content ranker, which combines plot, genres/tags, and available structured
credits using its existing provisional weights.

The Phase 5/7 lists had little overlap among these examples, so their RRF
provided no consensus evidence and did not establish a useful suppression of
unrelated semantic matches. The existing Phase 9 list produced comedy-oriented
results for movie 5, including *First Wives Club* and *Bridesmaids*, and
different comedy/romance candidates for movie 3. This is qualitatively more
consistent with the query metadata in these examples, but Phase 9 also uses
the Transformer signal and its weights remain provisional. This small
comparison does not prove general improvement; no Transformer code or Phase 9
weights were changed.

### Long-plot truncation

The experiment validated and reused the existing chunked embedding cache for
the same model revision and plot fingerprint. The artifact covers 3,400 long
plots using 17,548 windows, 224 wordpieces per window, and 48-wordpiece
overlap; 34 short-plot vectors were copied from the original Phase 7 artifact.
The model's configured sequence limit remains 256, including special tokens.

The diagnostic compared Phase 7 and chunked top-five rankings for a short
plot, `Bed of Roses` (movie 74), and five long plots already selected in Phase
9. The short query vector was identical between artifacts; its top-five
candidate overlap was one because the candidate vectors for long plots had
changed. The five long-plot overlap counts were 1, 1, 0, 3, and 3 out of five.
The chunked strategy preserves more of long plots, but its ranking changes are
mixed and are not evidence that it is generally better. Phase 7 remains the
default; the chunked vectors stay an experiment.

The existing Phase 9 code contains the tested window construction and
normalized-mean aggregation. New tests cover boundaries, overlap, normalization,
invalid means, and valid-cache reuse without encoding. Existing Phase 7 tests
continue to cover missing and blank plots; only usable-plot rows have embedding
vectors.

## Real-data demonstrations

Use offline flags to stay with the cached MiniLM model:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/recommend_hybrid.py user 1 --top-k 5
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/recommend_hybrid.py new-user --favorites 47 3 --top-k 5
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/recommend_hybrid.py new-user --top-k 5
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/recommend_hybrid.py movie 1076 --top-k 3
```

With full local MovieLens data, user 1 selected positive seeds `47, 50, 101,
151, 157`. The top results included *My Fellow Americans* (content rank 1,
fusion 0.50), *Sweet and Lowdown* (collaborative rank 1, predicted rating
4.812, support 13, fusion 0.50), and *Run Silent Run Deep* (collaborative rank
2, predicted rating 4.790, support 51, fusion 0.25). *Garden State* appeared
at collaborative rank 3 and content rank 101, yielding
`0.5 × (1/3 + 1/101) = 0.1716`; its predicted rating remains 4.756 in a
separate column. This is a full-data demonstration, not a held-out prediction.

For explicit new-user favorites 47 and 3, Phase 11 returned *The Game* and
*The Wedding Planner* at mean reciprocal rank 0.50, each supported by one seed.
With no favorites, the status was `needs_preferences` and the list was empty.

For a sparse-evidence fallback demonstration, the same user 1's actual
MovieLens rows for movie 47 (rating 5) and movie 3176 (rating 1) were supplied
as the entire restricted training interaction table. Phase 10 was fitted only
on those two rows, so there were no co-rater similarities. The above-average
seed was movie 47; Phase 11 content results were returned with status
`content_fallback_for_insufficient_collaborative_evidence`. The first result
was *The Game*, content rank 1, fusion 1.0 because content was the only
available source. This deliberately restricted demonstration validates
routing, not recommendation quality for a normal user history.

Movie 1076 (*The Innocents*) has no ratings in the supplied Phase 10 data. The
movie query used Phase 9 content results and reported the missing collaborative
evidence separately. It had usable genres and credits, even though plot-based
signals were absent.

These outputs are examples, not independent evidence of accuracy. MovieLens
source and usage notes remain in `data/README.md` and
`docs/phase8_data_sources.md`; this project does not claim unrestricted
commercial use or redistribution permission.

## Files and tests

Phase 12 adds:

- `scripts/recommend_hybrid.py` — routing, positive-seed selection, candidate
  filtering, and rank fusion.
- `scripts/phase12_experiments.py` — reproducible targeted ablations and
  baseline comparisons.
- `tests/test_hybrid_recommender.py` — controlled integration/fusion tests.
- `tests/test_phase12_experiments.py` — name ablation, RRF, chunk construction,
  aggregation, and cache tests.
- `reports/phase12_targeted_diagnostics.json` — actual before/after results.

Earlier phases were preserved. Phase 9 weights, Phase 10 similarity and
prediction algorithms, Phase 11 routing, cleaned data, and model artifacts
were not changed. Existing Phase 4–11 tests are run alongside new tests.

## What Phase 13 must evaluate

Phase 13 should define repeatable train/validation/test boundaries first. The
Phase 10 model and user histories must be fit only on permitted training
interactions; positive seeds must come from that same history; held-out ratings
must not influence candidate generation, parameters, or tuning. Compare the
hybrid to Phase 5, Phase 9, and Phase 10 baselines with ranking metrics and
appropriate coverage/diversity checks, report results by cold-start and
evidence-availability group, and tune only on validation data. It should
investigate the five-seed cap, equal-source RRF, named-entity mitigation,
Transformer reranking, and chunked embeddings without presenting the current
small diagnostics as general proof.

## Interview questions

1. **Why use rank fusion?** The component scores have different scales and
   meanings, so ordinal ranks provide a simple common basis without pretending
   a content score is a user rating.
2. **How are positive seeds chosen?** From only that user's supplied ratings:
   strictly above their mean, highest first, capped at five.
3. **What happens if a source has no candidates?** It is unavailable and is
   omitted from the score denominator; its absence is not a negative rating.
4. **What does `predicted_rating` mean in the output?** It remains Phase 10's
   estimated MovieLens-scale rating and is not the Phase 12 fusion score.
5. **Did name masking solve TF-IDF's problem?** No. It demonstrated how much
   names influence selected examples, but can remove genuinely useful identity
   information, so it remains an experiment.
6. **Did chunking improve semantic quality?** It preserved more plot text, but
   top-five changes were mixed; this is not evidence of a general improvement.
7. **Why aren't the recommendations called accurate?** Full-data examples and
   small qualitative comparisons are not held-out evaluation. Phase 13 must
   establish that evidence.
