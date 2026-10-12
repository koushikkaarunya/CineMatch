import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.enrich_metadata import (
    build_enriched_catalog,
    load_or_fetch_wikidata,
    build_sparql_query,
    normalize_imdb_id,
    normalize_tmdb_id,
    parse_wikidata_response,
    WDQS_PARTIAL_PATH,
)


def sample_movies() -> pd.DataFrame:
    return pd.DataFrame([
        {"movie_id": 1, "imdb_id": "tt1234567", "title": "Same Film (2000)",
         "release_year": 2000, "genres": "Drama", "plot": "A careful plot.", "plot_tags": "family"},
        {"movie_id": 2, "imdb_id": pd.NA, "title": "Same Film (2000)",
         "release_year": 2000, "genres": "Comedy", "plot": pd.NA, "plot_tags": pd.NA},
        {"movie_id": 3, "imdb_id": "tt7654321", "title": "Unmatched (2001)",
         "release_year": 2001, "genres": "Action", "plot": "Other plot.", "plot_tags": pd.NA},
    ])


def sample_source_records() -> list[dict[str, str | None]]:
    return [
        {"imdb_id": "tt1234567", "wikidata_id": "Q100", "credit_type": None,
         "person_id": None, "person_name": None},
        {"imdb_id": "tt1234567", "wikidata_id": "Q100", "credit_type": "director",
         "person_id": "Q200", "person_name": " A Director "},
        {"imdb_id": "tt1234567", "wikidata_id": "Q100", "credit_type": "writer",
         "person_id": "Q201", "person_name": "A Writer"},
        {"imdb_id": "tt1234567", "wikidata_id": "Q100", "credit_type": "cast",
         "person_id": "Q202", "person_name": "An Actor"},
        {"imdb_id": "tt1234567", "wikidata_id": "Q100", "credit_type": "producer",
         "person_id": "Q203", "person_name": "A Producer"},
    ]


class MetadataEnrichmentTests(unittest.TestCase):
    def test_identifier_normalization_and_invalid_values(self) -> None:
        self.assertEqual(normalize_imdb_id(" TT123 "), "tt0000123")
        self.assertEqual(normalize_imdb_id("0012345"), "tt0012345")
        self.assertIsNone(normalize_imdb_id(pd.NA))
        self.assertEqual(normalize_tmdb_id(" 00862 "), "862")
        self.assertIsNone(normalize_tmdb_id(pd.NA))
        with self.assertRaises(ValueError):
            normalize_imdb_id("tt12x")
        with self.assertRaises(ValueError):
            normalize_tmdb_id("abc")

    def test_query_uses_exact_ids_and_filters_anonymous_credit_nodes(self) -> None:
        query = build_sparql_query(["tt1234567"])
        self.assertIn('VALUES ?imdb { "tt1234567" }', query)
        self.assertIn('FILTER(REGEX(STR(?person)', query)

    def test_exact_id_match_structured_credits_and_provenance(self) -> None:
        crosswalk = pd.DataFrame({"movie_id": [1, 2, 3], "tmdb_id": ["862", pd.NA, "123"]})
        enriched, audit, summary = build_enriched_catalog(
            sample_movies(), sample_source_records(), crosswalk, "2026-10-09T00:00:00+00:00"
        )
        row = enriched.loc[enriched.movie_id.eq(1)].iloc[0]
        self.assertEqual(row["wikidata_id"], "Q100")
        self.assertEqual(row["metadata_match_method"], "exact_imdb_id")
        self.assertEqual(json.loads(row["directors"])[0], {"wikidata_id": "Q200", "name": "A Director"})
        self.assertEqual(json.loads(row["writers"])[0]["name"], "A Writer")
        self.assertEqual(json.loads(row["cast"])[0]["name"], "An Actor")
        self.assertEqual(json.loads(row["selected_crew"])[0]["role"], "producer")
        self.assertEqual(row["metadata_source"], "Wikidata")
        self.assertEqual(row["tmdb_id_source"], "MovieLens links.csv")
        self.assertEqual(audit.loc[audit.movie_id.eq(1), "metadata_match_status"].iloc[0], "matched")
        self.assertEqual(summary["total_movies"], 3)
        self.assertNotIn("poster_url", enriched.columns)
        self.assertEqual(summary["movies_with_poster_references"], 0)

    def test_missing_records_are_unknown_not_confirmed_empty_credit_lists(self) -> None:
        enriched, audit, summary = build_enriched_catalog(sample_movies(), sample_source_records())
        missing_id = enriched.loc[enriched.movie_id.eq(2)].iloc[0]
        unmatched = enriched.loc[enriched.movie_id.eq(3)].iloc[0]
        self.assertEqual(missing_id["metadata_match_status"], "unmatched_missing_imdb_id")
        self.assertEqual(unmatched["metadata_match_status"], "unmatched_no_wikidata_item")
        self.assertTrue(pd.isna(missing_id["directors"]))
        self.assertTrue(pd.isna(unmatched["cast"]))
        self.assertEqual(summary["unmatched_movies"], 2)
        self.assertEqual(len(audit), 3)

    def test_ambiguous_imdb_matches_are_rejected_without_attaching_credits(self) -> None:
        source = sample_source_records() + [
            {"imdb_id": "tt1234567", "wikidata_id": "Q999", "credit_type": "director",
             "person_id": "Q998", "person_name": "Wrong Director"}
        ]
        enriched, audit, summary = build_enriched_catalog(sample_movies(), source)
        row = enriched.loc[enriched.movie_id.eq(1)].iloc[0]
        self.assertEqual(row["metadata_match_status"], "ambiguous_multiple_wikidata_items")
        self.assertTrue(pd.isna(row["directors"]))
        self.assertEqual(summary["ambiguous_or_rejected_matches"], 1)
        self.assertEqual(json.loads(audit.loc[audit.movie_id.eq(1), "candidate_wikidata_ids"].iloc[0]), ["Q100", "Q999"])

    def test_ambiguous_title_year_is_not_used_as_fallback(self) -> None:
        # Two catalog entries share title/year; without an IMDb ID neither is inferred.
        records = [{"imdb_id": "tt1234567", "wikidata_id": "Q300", "credit_type": "director",
                    "person_id": "Q301", "person_name": "A Director"}]
        enriched, _, _ = build_enriched_catalog(sample_movies(), records)
        missing_id_row = enriched.loc[enriched.movie_id.eq(2)].iloc[0]
        self.assertEqual(missing_id_row["metadata_match_status"], "unmatched_missing_imdb_id")
        self.assertTrue(pd.isna(missing_id_row["wikidata_id"]))

    def test_catalog_imdb_id_shared_by_movies_is_rejected(self) -> None:
        movies = sample_movies()
        movies.loc[movies.movie_id.eq(2), "imdb_id"] = "tt1234567"
        with self.assertRaisesRegex(ValueError, "shared by multiple movies"):
            build_enriched_catalog(movies, sample_source_records())

    def test_duplicate_credit_rows_are_removed_and_person_roles_are_preserved(self) -> None:
        records = sample_source_records()
        records.append(records[1].copy())
        enriched, _, summary = build_enriched_catalog(sample_movies(), records)
        self.assertEqual(summary["duplicate_source_rows_removed"], 1)
        directors = json.loads(enriched.loc[enriched.movie_id.eq(1), "directors"].iloc[0])
        self.assertEqual(len(directors), 1)

    def test_movie_ids_genres_plots_and_ratings_remain_unchanged(self) -> None:
        movies = sample_movies()
        original_movies = movies.copy(deep=True)
        ratings = pd.DataFrame({"user_id": [1], "movie_id": [1], "rating": [4.0], "timestamp": [123]})
        original_ratings = ratings.copy(deep=True)
        enriched, _, summary = build_enriched_catalog(movies, sample_source_records())
        pd.testing.assert_frame_equal(movies, original_movies)
        pd.testing.assert_frame_equal(ratings, original_ratings)
        pd.testing.assert_frame_equal(enriched[original_movies.columns], original_movies)
        self.assertTrue(summary["ratings_used_or_modified"] is False)

    def test_tmdb_id_duplicates_are_preserved_as_references_not_used_as_join_keys(self) -> None:
        movies = sample_movies()
        crosswalk = pd.DataFrame({"movie_id": [1, 2, 3], "tmdb_id": ["862", "862", pd.NA]})
        enriched, _, summary = build_enriched_catalog(movies, [], crosswalk)
        self.assertEqual(enriched["tmdb_id"].tolist()[:2], ["862", "862"])
        self.assertEqual(summary["duplicate_tmdb_id_values"], 1)

    def test_parser_preserves_known_item_without_credits_and_normalizes_qids(self) -> None:
        payload = {"results": {"bindings": [
            {"imdb": {"value": "tt1234567"}, "movie": {"value": "http://www.wikidata.org/entity/Q100"}},
            {"imdb": {"value": "tt1234567"}, "movie": {"value": "http://www.wikidata.org/entity/Q100"},
             "credit_type": {"value": "director"}, "person": {"value": "http://www.wikidata.org/entity/Q200"},
             "personLabel": {"value": "A Director"}},
        ]}}
        rows = parse_wikidata_response(payload)
        self.assertIsNone(rows[0]["person_id"])
        self.assertEqual(rows[1]["person_id"], "Q200")

    def test_matching_does_not_accumulate_cached_source_rows_on_repeat(self) -> None:
        records = sample_source_records()
        first, _, _ = build_enriched_catalog(sample_movies(), records)
        second, _, _ = build_enriched_catalog(sample_movies(), records)
        pd.testing.assert_frame_equal(first, second)

    def test_matching_snapshot_cache_uses_id_fingerprint_and_can_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot_path = Path(directory) / "source.json"
            imdb_ids = ["tt1234567"]
            cached_records = sample_source_records()
            from scripts.enrich_metadata import identifier_fingerprint
            snapshot_path.write_text(json.dumps({
                "catalog_imdb_fingerprint": identifier_fingerprint(imdb_ids),
                "retrieved_at_utc": "cached-time",
                "records": cached_records,
            }), encoding="utf-8")
            with patch("scripts.enrich_metadata.fetch_wikidata_records") as fetch:
                records, retrieved, reused = load_or_fetch_wikidata(imdb_ids, snapshot_path)
                self.assertEqual(records, cached_records)
                self.assertEqual(retrieved, "cached-time")
                self.assertTrue(reused)
                fetch.assert_not_called()

                fresh_records = sample_source_records()[:1]
                fetch.return_value = fresh_records
                records, retrieved, reused = load_or_fetch_wikidata(
                    ["tt7654321"], snapshot_path
                )
                self.assertEqual(records, fresh_records)
                self.assertFalse(reused)
                fetch.assert_called_once_with(
                    ["tt7654321"], progress_path=WDQS_PARTIAL_PATH,
                )

                fetch.reset_mock()
                fetch.return_value = cached_records
                _, _, reused = load_or_fetch_wikidata(imdb_ids, snapshot_path, refresh=True)
                self.assertFalse(reused)
                fetch.assert_called_once_with(
                    imdb_ids, progress_path=WDQS_PARTIAL_PATH,
                )

    def test_fetch_resumes_only_after_completed_batches(self) -> None:
        from scripts.enrich_metadata import fetch_wikidata_records, identifier_fingerprint

        with tempfile.TemporaryDirectory() as directory:
            progress_path = Path(directory) / "partial.json"
            ids = ["tt0000001", "tt0000002", "tt0000003"]
            first_batch = [{"imdb_id": ids[0], "wikidata_id": "Q1", "credit_type": None,
                            "person_id": None, "person_name": None}]
            last_batch = [{"imdb_id": ids[2], "wikidata_id": "Q3", "credit_type": None,
                           "person_id": None, "person_name": None}]
            progress_path.write_text(json.dumps({
                "catalog_imdb_fingerprint": identifier_fingerprint(ids),
                "batch_size": 2,
                "completed_batches": 1,
                "records": first_batch,
            }), encoding="utf-8")

            output = io.StringIO()
            with patch("scripts.enrich_metadata.fetch_wikidata_batch", return_value=last_batch) as fetch:
                with redirect_stdout(output):
                    records = fetch_wikidata_records(
                        ids, batch_size=2, pause_seconds=0, progress_path=progress_path
                    )
            self.assertEqual(records, first_batch + last_batch)
            fetch.assert_called_once_with([ids[2]])
            # Progress output describes the mocked request; the test never calls WDQS.
            self.assertIn("Resuming from completed Wikidata batch 1/2", output.getvalue())
            self.assertIn("Querying Wikidata batch 2/2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
