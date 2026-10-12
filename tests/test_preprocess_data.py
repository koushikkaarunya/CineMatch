import unittest

import pandas as pd

from scripts.preprocess_data import (
    clean_genres,
    clean_plot,
    clean_plot_tags,
    normalize_imdb_id,
    normalize_positive_integer_ids,
    prepare_movies,
    prepare_ratings,
    validate_release_year,
)


class PreprocessingTests(unittest.TestCase):
    def test_id_normalization_and_invalid_ids(self) -> None:
        normalized = normalize_positive_integer_ids(pd.Series(["12", "13.0"]), "movie_id")
        self.assertEqual(normalized.tolist(), [12, 13])
        with self.assertRaises(ValueError):
            normalize_positive_integer_ids(pd.Series(["0", "bad"]), "movie_id")
        self.assertEqual(normalize_imdb_id(" TT123 "), "tt0000123")
        self.assertEqual(normalize_imdb_id("0012345"), "tt0012345")
        self.assertTrue(pd.isna(normalize_imdb_id(pd.NA)))
        with self.assertRaises(ValueError):
            normalize_imdb_id("tt12x")

    def test_text_cleanup_preserves_plot_content_and_paragraphs(self) -> None:
        plot = "  A  HERO!\r\n\r\n  Second\tparagraph.  "
        self.assertEqual(clean_plot(plot), "A HERO!\n\nSecond paragraph.")
        self.assertTrue(pd.isna(clean_plot(" \n ")))
        self.assertEqual(clean_genres(" Drama |Comedy|drama||"), "Drama|Comedy")
        self.assertTrue(pd.isna(clean_genres(" || ")))
        self.assertEqual(clean_plot_tags(" Dramatic,  revenge,DRAMATIC,, REVENGE "),
                         "dramatic, revenge")
        self.assertTrue(pd.isna(clean_plot_tags(" , ")))

    def test_release_year_keeps_missing_and_counts_malformed(self) -> None:
        values = pd.Series(["1999", pd.NA, "not-a-year"])
        year, invalid_count = validate_release_year(values)
        self.assertEqual(year.iloc[0], 1999)
        self.assertTrue(pd.isna(year.iloc[1]))
        self.assertTrue(pd.isna(year.iloc[2]))
        self.assertEqual(invalid_count, 1)

    def test_movie_cleanup_removes_exact_rows_but_rejects_conflicting_ids(self) -> None:
        row = {
            "movie_id": "7", "imdb_id": "TT1234567", "title": " Film (2001) ",
            "release_year": "2001", "genres": " Drama |Drama", "overview": " Plot! ",
            "plot_tags": " Suspenseful, suspenseful ",
        }
        movies, counts = prepare_movies(pd.DataFrame([row, row.copy()]))
        self.assertEqual(len(movies), 1)
        self.assertEqual(movies.loc[0, "plot"], "Plot!")
        self.assertEqual(movies.loc[0, "genres"], "Drama")
        self.assertEqual(movies.loc[0, "plot_tags"], "suspenseful")
        self.assertEqual(counts["exact_duplicate_movie_rows_removed"], 1)

        conflicting = row.copy()
        conflicting["title"] = "Different title"
        with self.assertRaises(ValueError):
            prepare_movies(pd.DataFrame([row, conflicting]))

    def test_rating_cleanup_preserves_distinct_events_and_checks_scale_and_catalog(self) -> None:
        rows = [
            {"user_id": 1, "movie_id": 7, "rating": 4.0, "timestamp": 100},
            {"user_id": 1, "movie_id": 7, "rating": 4.0, "timestamp": 100},
            {"user_id": 1, "movie_id": 7, "rating": 4.5, "timestamp": 200},
        ]
        ratings, counts = prepare_ratings(pd.DataFrame(rows), {7})
        self.assertEqual(len(ratings), 2)
        self.assertEqual(counts["exact_duplicate_rating_events_removed"], 1)

        invalid_rating = pd.DataFrame([
            {"user_id": 1, "movie_id": 7, "rating": 4.2, "timestamp": 100}
        ])
        with self.assertRaises(ValueError):
            prepare_ratings(invalid_rating, {7})

        orphan = pd.DataFrame([
            {"user_id": 1, "movie_id": 8, "rating": 4.0, "timestamp": 100}
        ])
        with self.assertRaises(ValueError):
            prepare_ratings(orphan, {7})


if __name__ == "__main__":
    unittest.main()
