# Dataset sources and ingestion

The ingestion script downloads source files into `data/raw/` and writes cleaned
tables and a coverage summary into `data/processed/`. Both directories are
ignored by Git so downloaded source material is not accidentally committed.
Phase 4 reads those tables without modifying them and writes normalized output
to `data/cleaned/`, which is also ignored because it is generated.

## Sources

- **MovieLens latest-small**, from GroupLens: movie titles and genres, IMDb ID
  crosswalks, and user ratings with timestamps. The release contains 100,836
  ratings from 610 users across 9,742 movies.
- **MPST: Movie Plot Synopses with Tags**, the `cryptexcode/MPST` release hosted
  on Hugging Face: 14,828 movie records with IMDb IDs, plot synopsis, plot tags,
  and synopsis source. Ingestion keeps only non-empty synopsis rows marked as
  coming from Wikipedia.

MovieLens `links.csv` stores IMDb identifiers as digits without the `tt`
prefix. The script pads them to seven digits and joins them to MPST's `imdb_id`.
The catalog keeps MovieLens `movieId` as `movie_id`; ratings use the same key.
Titles are not used as join keys.

## Outputs

- `movies.csv`: `movie_id`, `imdb_id`, `title`, `release_year`, `genres`,
  `overview`, and `plot_tags`.
- `ratings.csv`: `user_id`, `movie_id`, `rating`, and Unix `timestamp`.
- `ingestion_summary.json`: actual row counts, plot coverage, and duplicate
  handling counts from the run.

All MovieLens movies remain in the catalog when no Wikipedia plot matches; those
rows have a blank plot. The script does not guess missing matches by title.
These sources do not provide cast, director, or a separate keywords field, so
those fields are left out rather than inferred.

## Phase 4 preprocessing outputs

- `data/cleaned/movies.csv`: canonical columns `movie_id`, `imdb_id`, `title`,
  `release_year`, `genres`, `plot`, and `plot_tags`. Phase 2's `overview` field
  is named `plot` here. Missing optional fields remain missing.
- `data/cleaned/ratings.csv`: validated `user_id`, `movie_id`, `rating`, and
  `timestamp` interactions.
- `data/cleaned/preprocessing_summary.json`: row counts, missing-value counts,
  duplicate counts, and rating/catalog validation results.

Preprocessing trims and normalizes whitespace while preserving plot wording,
case, punctuation, Unicode, and paragraph boundaries. Genre and tag labels are
normalized without removing the special MovieLens labels. The script fails on
invalid required IDs, ratings outside the 0.5-to-5.0 half-star scale, ambiguous
duplicate movie IDs or IMDb IDs, or ratings that reference an unknown movie.
It removes only exact duplicate movie rows and exact duplicate rating events;
distinct rating events are retained. Missing optional year, IMDb ID, genre,
plot, and tags are not imputed or used to remove a movie. Director, writers,
cast, selected crew, and poster data remain absent until a reliable source is
integrated in a later phase.

## Usage and attribution

MovieLens terms allow research use subject to attribution and prohibit
commercial or revenue-bearing use without permission from GroupLens. The MPST
dataset card identifies its release as CC BY 4.0 and requests citation of the
associated research papers. Wikipedia-derived plot text may also carry
attribution and share-alike obligations. Keep the source credits and check the
upstream terms before redistribution or commercial use. This repository
excludes raw and processed data by default.

TMDB was not selected as an ML source: its current API terms restrict using
TMDB content in machine-learning or AI applications. Do not assume that the
TMDB 5000 Kaggle files override those restrictions without an applicable
license.

## Planned later: structured cast and crew

Cast and crew are required for the final recommendation system, but they are
intentionally not part of this Phase 2 ingestion. For the later content-based
metadata phase, evaluate Wikidata as a structured enrichment source. Its IMDb
identifier property (P345) can connect records to the catalog's `imdb_id`; film
properties include director (P57), screenwriter (P58), cast member (P161), and
director of photography (P344). Wikidata structured data is CC0. Coverage and
accuracy vary, and cast ordering may not identify the most important actors, so
the later phase must measure coverage and define a clear cast-selection rule.
This is a planned source to validate, not a data dependency in the current
ingestion.

## Run

From the project root, with the project virtual environment active:

```sh
python scripts/ingest_data.py
```

The first run downloads the source files. Later runs reuse them and regenerate
the processed tables.

Run Phase 4 preprocessing from the project root with:

```sh
./.venv/bin/python scripts/preprocess_data.py
```

Run the Phase 4 preprocessing checks with:

```sh
./.venv/bin/python -m unittest discover -s tests -v
```
