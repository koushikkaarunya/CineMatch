# CineMatch Phase 10: item-item collaborative filtering

## Scope and current data

The model reads `data/cleaned/ratings.csv`. Its fitting features are only
`user_id`, `movie_id`, and `rating`; timestamps are retained in the source but
not used. `data/cleaned/movies.csv` is used only to validate movie IDs and show
titles. The enriched catalog is not needed by this model.

The current cleaned files contain 100,836 ratings, 610 users, 9,724 rated
movies, and 9,742 catalog movies. Ratings are complete and fall on the
MovieLens 0.5-to-5.0 half-star scale. There are no repeated user/movie pairs
and every rated movie ID exists in the catalog. The 18 catalog-only movies
have no collaborative interaction evidence.

The original MovieLens release's use conditions still apply. The project notes
and links are in [`data/README.md`](../data/README.md) and
[`docs/phase8_data_sources.md`](phase8_data_sources.md); the files and
recommendations should not be described as unrestricted for commercial use or
redistribution.

## Similarity

For each user, subtract their mean rating from each observed rating:

`residual(u, movie) = rating(u, movie) - mean_rating(u)`

For two movies, use the cosine of the two residual vectors over users who rated
both movies. Missing ratings are absent from the vectors, not zero-valued
ratings. At least three co-raters are required. The score is shrunk according
to overlap size:

`similarity(i, j) = cosine_of_shared_user_residuals × n / (n + 10)`

Here `n` is the number of shared raters. Negative scores are retained because
they can represent opposing rating patterns. SciPy sparse matrices store only
observed ratings and sufficiently supported movie pairs. This avoids building
a dense 610-by-9,742 rating matrix or a dense 9,742-by-9,742 similarity matrix.

## Predicting and ranking for a user

For a candidate movie `i` not yet rated by user `u`, use the user's average as
a baseline and adjust it using their rating deviations on similar movies:

`prediction(u, i) = clip(mean(u) + sum(similarity(i,j) × residual(u,j)) / (1 + sum(abs(similarity(i,j)))), 0.5, 5.0)`

The `1` is the `prediction_shrinkage` starting value: weak total evidence keeps
the estimate closer to the user's average. It was introduced after an initial
demonstration produced several clipped 5.0 estimates; it was not selected using
held-out ratings or accuracy metrics. Negative similarities contribute a
negative adjustment when appropriate. A candidate is eligible only when it
has at least one retained similarity to a movie the user rated. The result is
an estimated MovieLens-scale rating; the list is sorted by estimate, then total
similarity evidence, support count, and movie ID for deterministic ties.

## Data boundary and future evaluation safeguards

The recommender receives the ratings table as an argument. It learns user
means, user histories, co-rater counts, and similarities only from that table.
The demonstration supplies all available interactions, so its outputs are not
held-out predictions and do not establish accuracy.

Before formal evaluation in Phase 13, first divide interactions into training
and held-out sets. Fit similarities and user means only from training rows;
derive each user's recommendation history only from that same allowed history;
never let held-out rows influence minimum-overlap or shrinkage settings, model
selection, or candidate features. Compare against appropriate baselines using
a repeatable split. The whole-data demonstration must not be used to claim
predictive performance.

## Known gaps for Phase 11

- A new user has no user mean or rating history.
- A known user with fewer than two ratings or no variation has too little
  information for mean-centered personalization.
- A catalog movie with no ratings, or with fewer than three co-raters against
  other movies, has no usable collaborative links.
- A user can be known but have no unseen candidate with sufficient links.

Phase 10 reports these cases clearly; it does not fall back to content-based
recommendations or implement cold-start behavior.

## Run

From the project root:

```sh
./.venv/bin/python scripts/recommend_collaborative.py 1 --top-k 10
```

The script rebuilds the small sparse similarity matrix from the supplied
cleaned ratings on each run. It labels the full-data output as a demonstration,
not a held-out evaluation. The SciPy sparse dependency is listed separately in
`requirements-phase10.txt`.
