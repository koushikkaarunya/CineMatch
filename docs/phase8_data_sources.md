# CineMatch Phase 8 metadata sources and use notes

## Wikidata credits

Phase 8 retrieves structured film and person statements from the Wikidata Query
Service (WDQS). The fields are directors (P57), writers/screenwriters (P58),
cast members (P161), and selected crew: producers (P162), composers (P86),
cinematographers (P344), and film editors (P1040). Person names are labels from
Wikidata; the stable person Wikidata QID is retained beside each label.
The selected crew list focuses on production responsibility and recognizable
craft roles. It intentionally excludes broad, high-volume technical and
supporting credits. Wikidata does not provide a consistently ordered “principal
cast” list here, so `cast` means the available cast statements, without a
claim that the first entries are the stars.

The source is matched to the cleaned MovieLens catalog only through an exact
IMDb title identifier (IMDb `tconst`, Wikidata property P345). A title/year
match is never used. If an IMDb ID maps to more than one Wikidata item, Phase 8
marks the row ambiguous and attaches no credits. This is a structured metadata
source and its data is released under CC0. The raw WDQS response rows are saved
as an ID-fingerprinted snapshot under `data/raw/wikidata/` so the enrichment
can be reproduced without repeating requests.
Wikidata structured data is CC0; attribution is not required, though Wikidata
requests that users consider attribution. The report and source snapshot retain
the source name and retrieval time for traceability.

Wikidata is community maintained. A missing property means “not present in the
retrieved source,” not that the film has no director, writer, cast, or crew.
Names and roles can be incomplete or disputed, so source IDs and per-movie
match audit are retained. Use a descriptive User-Agent, issue small queries,
and stop on WDQS throttling or errors.

## TMDB identifier reference

The MovieLens `links.csv` crosswalk supplies a TMDB ID by the exact MovieLens
`movieId`. Phase 8 keeps this ID only as a provider reference for possible
future work; it does not call TMDB or ingest TMDB metadata. TMDB API terms have
restrictions on using its content with machine-learning applications and on
caching. Do not use TMDB data for CineMatch ML unless the applicable terms or a
written agreement permit that specific use.

## MovieLens source

The source catalog and ratings come from MovieLens latest-small. Its README
describes the dataset as intended for research and provides attribution and
use conditions, including restrictions on commercial/revenue use without
permission. Keep the original dataset separate and review the current README
before redistribution or commercial use. Phase 8 does not modify the ratings.

## IMDb and posters

IMDb’s public datasets are offered under limited personal, noncommercial
conditions and its terms restrict repurposing them into another movie
database. CineMatch therefore does not ingest IMDb datasets; IMDb IDs already
present in MovieLens are used strictly as exact join keys for Wikidata.

No posters are collected in Phase 8. A Wikimedia Commons image may have a
different license per file and requires a rights and attribution check for
each selected image. Poster coverage is consequently zero, rather than
inventing or assuming a reusable image source. Commons reuse guidance confirms
that users must follow the license shown on each file page; that per-file
review has not been done for this catalog.

## Official references reviewed 2026-10-09

- [Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing)
- [Wikidata data access guidance](https://www.wikidata.org/wiki/Wikidata:Data_access/en)
- [Wikidata SPARQL query help](https://www.wikidata.org/wiki/Help:Queries)
- [MovieLens latest-small README](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
- [IMDb data use conditions](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX)
- [IMDb dataset terms](https://www.imdb.com/interfaces/)
- [TMDB API terms](https://www.themoviedb.org/api-terms-of-use)
- [TMDB API FAQ](https://developer.themoviedb.org/docs/faq)
- [Wikimedia Commons reuse guidance](https://commons.wikimedia.org/wiki/Commons:REUSE)
