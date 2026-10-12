# CineMatch Phase 11: cold-start handling

## Scope

Phase 11 adds a small coordinator around the existing Phase 9
`HybridContentRecommender.recommend(movie_id, top_k)` and Phase 10
`ItemItemCollaborativeRecommender.recommend(user_id, top_k)` methods. It does not
change either model, their score meanings, the Phase 4 data, or any Phase 5–10
outputs. The enriched movie catalog supplies display titles and whatever
content fields are actually present.

Phase 9's `hybrid_score` is its experimental rank-fusion score, not a
probability. Phase 10's `predicted_rating` is an estimated MovieLens rating.
These values are never added together or relabeled as one another.

## Cases and behavior

### New user without favorites

`recommend_from_favorites([])` returns `needs_preferences` with no
recommendation list. The system asks the person to choose movies they like; it
does not call a generic list personalized.

### New user with favorite movies

`recommend_from_favorites([movie_id, ...])` calls the existing Phase 9
recommender for each unique selected movie. IDs are checked against the actual
catalog. Duplicate seed IDs are ignored in first-seen order. Every selected
movie is removed from the combined results, and duplicate candidate rows are
counted once per seed.

For candidate `m` and `S` unique seeds, each seed contributes the reciprocal of
the candidate's rank when it appears, or zero if it does not:

`aggregate_rank(m) = sum(1 / rank_for_seed(m)) / S`

This mean reciprocal rank is only a simple ranking aggregation; it is not
Phase 9's `hybrid_score`, not a probability, and not an evaluation metric. A
candidate shown by multiple seed lists gets contributions from each. If one
seed produces no candidates, it contributes zero and the result is labeled as
partial evidence. If all seeds produce no usable candidates, the response asks
for other favorites. Ties use seed-list support count, best rank, then ascending
movie ID. The original Phase 9 score and explanation are kept per seed for
inspection, rather than averaged across queries.

### Known user with weak collaborative evidence

`recommend_for_user(user_id, favorite_movie_ids=...)` first distinguishes an
unknown user from a known MovieLens user. If Phase 10 can return collaborative
recommendations, it returns them unchanged and labels the source as Phase 10.
If history has fewer than two ratings, has no variation, or yields no unseen
candidate with retained item-item evidence, the function uses explicitly
provided favorite movie IDs for a Phase 9 content fallback. Without those
favorites, it asks for more preference input. The module does not infer that a
low or average rating means “like,” and it does not invent ratings.

### Known movie without collaborative links

`recommend_for_movie(movie_id)` validates the movie against the catalog and
reports collaborative evidence separately from content results. Phase 10
evidence is considered available when the movie has at least one retained
item-item similarity after Phase 10's existing co-rater threshold. It
distinguishes no collaborative rating data from no qualifying item similarity
and from an ID absent from the catalog. It then requests Phase 9 content
recommendations for that movie; if collaborative links exist, it still labels
the returned list as content-only and does not combine scores.

The response also reports available content fields. Missing plots, genres,
tags, and Wikidata credits stay missing. If no relevant fields are present, the
response reports `no_content_metadata`; if fields exist but Phase 9 returns no
candidates, it reports `no_content_candidates`.

## Existing content limitations remain

Cold-start handling does not fix the weaknesses of the reused content models:

1. TF-IDF can overmatch character names such as Max, Robin, or Savannah rather
   than story meaning.
2. Transformer embeddings can return unrelated semantic matches.
3. The original MiniLM context limit can truncate long plots. Chunking was
   explored in Phase 9, but its comparisons were mixed and did not establish a
   general improvement.

These recommendations are useful as a fallback path, not as proof of quality.
The existing MovieLens usage and attribution notes remain in
[`data/README.md`](../data/README.md) and
[`docs/phase8_data_sources.md`](phase8_data_sources.md); no unrestricted
commercial use or redistribution permission is claimed.

## Evaluation and future integration

Demonstrations using all MovieLens ratings are not held-out evaluations. In
future evaluation, held-out ratings must not enter Phase 10 user histories,
user means, item similarities, or settings selection. Cold-start lists must be
evaluated separately for users or movies with no rating evidence, using only
the preference information available at recommendation time.

Phase 12 may consider content and collaborative signals together only after
their distinct meanings are preserved and an evaluation plan defines how to
compare them. Phase 11 deliberately does not create that hybrid ranker.

## Run

With the project environment, cached Phase 7 model artifact, and project data
available, examples are:

```sh
./.venv/bin/python scripts/recommend_cold_start.py new-user
./.venv/bin/python scripts/recommend_cold_start.py new-user --favorites 47 3 --top-k 5
./.venv/bin/python scripts/recommend_cold_start.py user 6 --favorites 47 --top-k 5
./.venv/bin/python scripts/recommend_cold_start.py movie 1076 --top-k 5
```

The direct functions `recommend_from_favorites`, `recommend_for_user`, and
`recommend_for_movie` can also be used with injected recommender instances.
Tests use small fakes and do not download or load Transformer models.
