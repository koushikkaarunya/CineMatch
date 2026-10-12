"""Phase 8: enrich the cleaned catalog with Wikidata credits and MovieLens IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIES_PATH = PROJECT_ROOT / "data" / "cleaned" / "movies.csv"
RATINGS_PATH = PROJECT_ROOT / "data" / "cleaned" / "ratings.csv"
LINKS_PATH = PROJECT_ROOT / "data" / "raw" / "movielens" / "ml-latest-small" / "links.csv"
RAW_WIKIDATA_PATH = PROJECT_ROOT / "data" / "raw" / "wikidata" / "movie_credits_snapshot.json"
ENRICHED_DIR = PROJECT_ROOT / "data" / "enriched"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
REPORTS_DIR = PROJECT_ROOT / "reports"
WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
WDQS_BATCH_SIZE = 200
WDQS_PAUSE_SECONDS = 1.0
WDQS_PARTIAL_PATH = RAW_WIKIDATA_PATH.with_suffix(".partial.json")
MOVIE_COLUMNS = ["movie_id", "imdb_id", "title", "release_year", "genres", "plot", "plot_tags"]

# These are common credits that could add useful context later. We intentionally
# avoid importing every Wikidata property that happens to describe a contributor.
CREDIT_PROPERTIES = {
    "director": "P57",
    "writer": "P58",
    "cast": "P161",
    "producer": "P162",
    "composer": "P86",
    "cinematographer": "P344",
    "film_editor": "P1040",
}
PERSON_FIELDS = ("director", "writer", "cast")
CREW_FIELDS = ("producer", "composer", "cinematographer", "film_editor")


def normalize_imdb_id(value: object) -> str | None:
    """Normalize an IMDb tconst while retaining missing values as None."""
    if pd.isna(value) or not str(value).strip():
        return None
    identifier = str(value).strip().lower()
    digits = identifier[2:] if identifier.startswith("tt") else identifier
    if not digits.isdigit() or len(digits) > 12:
        raise ValueError(f"Invalid IMDb ID: {value!r}")
    return "tt" + digits.zfill(7)


def normalize_tmdb_id(value: object) -> str | None:
    """Normalize the MovieLens TMDB crosswalk ID as a nullable digit string."""
    if pd.isna(value) or not str(value).strip():
        return None
    identifier = str(value).strip()
    if not identifier.isdigit() or int(identifier) <= 0:
        raise ValueError(f"Invalid TMDB ID in MovieLens links.csv: {value!r}")
    return str(int(identifier))


def clean_person_name(value: object) -> str | None:
    """Normalize spacing and Unicode form without inventing or translating names."""
    if pd.isna(value):
        return None
    name = " ".join(unicodedata.normalize("NFC", str(value)).split())
    return name or None


def _same_column_values(left: pd.Series, right: pd.Series) -> bool:
    """Compare preserved data values while treating CSV-style nulls alike."""
    if len(left) != len(right):
        return False
    null_marker = "<CINEMATCH_NULL>"
    left_values = left.astype("object").where(left.notna(), null_marker).astype(str).tolist()
    right_values = right.astype("object").where(right.notna(), null_marker).astype(str).tolist()
    return left_values == right_values


def identifier_fingerprint(imdb_ids: list[str]) -> str:
    """Hash the sorted IDs so cached source rows match this catalog exactly."""
    return hashlib.sha256("\n".join(sorted(imdb_ids)).encode("utf-8")).hexdigest()


def build_sparql_query(imdb_ids: list[str]) -> str:
    """Build one exact-IMDb-ID query for IDs in a controlled batch."""
    values = " ".join(json.dumps(identifier) for identifier in imdb_ids)
    credit_branches = []
    for role, property_id in CREDIT_PROPERTIES.items():
        credit_branches.append(
            f"{{ ?movie wdt:{property_id} ?person . BIND(\"{role}\" AS ?credit_type) }}"
        )
    union = " UNION ".join(credit_branches)
    return f"""\
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?imdb ?movie ?credit_type ?person ?personLabel WHERE {{
  VALUES ?imdb {{ {values} }}
  ?movie wdt:P345 ?imdb .
  OPTIONAL {{
    {union}
    # Keep only identified Wikidata person items; anonymous blank nodes cannot
    # be represented as reliable people or joined by a stable identifier.
    FILTER(REGEX(STR(?person), "^http://www.wikidata.org/entity/Q[0-9]+$"))
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\". }}
  }}
}}
"""


def _parse_binding(row: dict, name: str) -> str | None:
    value = row.get(name)
    return value.get("value") if value else None


def parse_wikidata_response(payload: dict) -> list[dict[str, str | None]]:
    """Convert WDQS JSON bindings to flat, auditable source rows."""
    records = []
    for row in payload.get("results", {}).get("bindings", []):
        imdb_id = normalize_imdb_id(_parse_binding(row, "imdb"))
        movie_uri = _parse_binding(row, "movie")
        movie_id = movie_uri.rsplit("/", 1)[-1] if movie_uri else None
        person_uri = _parse_binding(row, "person")
        person_id = person_uri.rsplit("/", 1)[-1] if person_uri else None
        if person_id is not None and (not person_id.startswith("Q") or not person_id[1:].isdigit()):
            raise ValueError(f"Unexpected Wikidata person identifier: {person_uri!r}")
        records.append({
            "imdb_id": imdb_id,
            "wikidata_id": movie_id,
            "credit_type": _parse_binding(row, "credit_type"),
            "person_id": person_id,
            "person_name": clean_person_name(_parse_binding(row, "personLabel")),
        })
    return records


def fetch_wikidata_batch(imdb_ids: list[str], timeout_seconds: int = 60) -> list[dict[str, str | None]]:
    """Fetch one narrow ID batch using the public WDQS endpoint without a key."""
    query = build_sparql_query(imdb_ids)
    request = Request(
        WDQS_ENDPOINT,
        data=urlencode({"query": query, "format": "json"}).encode("utf-8"),
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "CineMatch-Phase8/1.0 (educational metadata enrichment)",
            # Avoid implicit compression because urllib does not decompress it.
            "Accept-Encoding": "identity",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            # WDQS can emit raw control characters inside labels. JSON's strict
            # mode rejects those; clean_person_name collapses them as whitespace.
            payload = json.loads(response.read().decode("utf-8"), strict=False)
    except HTTPError as error:
        retry_after = error.headers.get("Retry-After", "not provided")
        raise RuntimeError(
            f"Wikidata returned HTTP {error.code} (Retry-After: {retry_after}). "
            "Wait before retrying; no fallback source was used."
        ) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Wikidata request failed: {error}. Check network access and rerun."
        ) from error
    return parse_wikidata_response(payload)


def fetch_wikidata_records(
    imdb_ids: list[str], batch_size: int = WDQS_BATCH_SIZE,
    pause_seconds: float = WDQS_PAUSE_SECONDS,
    progress_path: Path | None = None,
) -> list[dict[str, str | None]]:
    """Query small batches and checkpoint completed batches for safe resumption."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    records: list[dict[str, str | None]] = []
    batches = [imdb_ids[index:index + batch_size] for index in range(0, len(imdb_ids), batch_size)]
    first_batch = 0
    if progress_path and progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if (progress.get("catalog_imdb_fingerprint") == identifier_fingerprint(imdb_ids)
                    and progress.get("batch_size") == batch_size):
                first_batch = int(progress.get("completed_batches", 0))
                records = progress.get("records", [])
                if not 0 <= first_batch <= len(batches):
                    first_batch, records = 0, []
                if first_batch:
                    print(f"Resuming from completed Wikidata batch {first_batch}/{len(batches)}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            first_batch, records = 0, []

    for batch_index in range(first_batch, len(batches)):
        batch_number = batch_index + 1
        batch = batches[batch_index]
        print(f"Querying Wikidata batch {batch_number}/{len(batches)} ({len(batch)} IMDb IDs)")
        records.extend(fetch_wikidata_batch(batch))
        if progress_path:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress = {
                "catalog_imdb_fingerprint": identifier_fingerprint(imdb_ids),
                "batch_size": batch_size,
                "completed_batches": batch_number,
                "records": records,
            }
            temporary_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(progress, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary_path.replace(progress_path)
        if batch_number < len(batches):
            time.sleep(pause_seconds)
    return records


def load_or_fetch_wikidata(
    imdb_ids: list[str], snapshot_path: Path = RAW_WIKIDATA_PATH,
    refresh: bool = False,
) -> tuple[list[dict[str, str | None]], str, bool]:
    """Reuse an exact-ID snapshot or fetch and atomically save a fresh one."""
    fingerprint = identifier_fingerprint(imdb_ids)
    if snapshot_path.exists() and not refresh:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("catalog_imdb_fingerprint") == fingerprint:
            return snapshot["records"], snapshot["retrieved_at_utc"], True

    if refresh:
        WDQS_PARTIAL_PATH.unlink(missing_ok=True)
    records = fetch_wikidata_records(imdb_ids, progress_path=WDQS_PARTIAL_PATH)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snapshot = {
        "source": "Wikidata Query Service",
        "endpoint": WDQS_ENDPOINT,
        "retrieved_at_utc": retrieved_at,
        "catalog_imdb_fingerprint": fingerprint,
        "imdb_ids_requested": len(imdb_ids),
        "records": records,
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(snapshot_path)
    WDQS_PARTIAL_PATH.unlink(missing_ok=True)
    return records, retrieved_at, False


def load_tmdb_crosswalk(path: Path = LINKS_PATH) -> pd.DataFrame:
    """Read MovieLens links by its stable movieId key and normalize TMDB IDs."""
    links = pd.read_csv(path, usecols=["movieId", "tmdbId"], dtype={"tmdbId": "string"})
    links = links.rename(columns={"movieId": "movie_id", "tmdbId": "tmdb_id"})
    if links["movie_id"].duplicated().any():
        raise ValueError("MovieLens links.csv contains duplicate movieId values")
    links["movie_id"] = pd.to_numeric(links["movie_id"], errors="raise").astype("int64")
    links["tmdb_id"] = links["tmdb_id"].map(normalize_tmdb_id).astype("string")
    return links


def build_enriched_catalog(
    movies: pd.DataFrame,
    source_records: list[dict[str, str | None]],
    tmdb_crosswalk: pd.DataFrame | None = None,
    retrieved_at_utc: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Attach only exact IMDb-ID Wikidata matches and produce a per-movie audit."""
    missing_columns = set(MOVIE_COLUMNS) - set(movies.columns)
    if missing_columns:
        raise ValueError(f"Cleaned catalog is missing columns: {sorted(missing_columns)}")
    if movies["movie_id"].duplicated().any():
        raise ValueError("Cleaned catalog contains duplicate movie_id values")

    original = movies[MOVIE_COLUMNS].copy().reset_index(drop=True)
    original["movie_id"] = pd.to_numeric(original["movie_id"], errors="raise").astype("int64")
    original["imdb_id"] = original["imdb_id"].map(normalize_imdb_id).astype("string")
    duplicate_imdb_ids = original.loc[
        original["imdb_id"].notna() & original["imdb_id"].duplicated(keep=False), "imdb_id"
    ].unique()
    if len(duplicate_imdb_ids):
        raise ValueError(
            "Cleaned catalog has IMDb IDs shared by multiple movies: "
            + ", ".join(sorted(duplicate_imdb_ids.astype(str)))
        )
    catalog_ids = set(original["movie_id"])
    if tmdb_crosswalk is not None:
        if set(["movie_id", "tmdb_id"]) - set(tmdb_crosswalk.columns):
            raise ValueError("TMDB crosswalk must contain movie_id and tmdb_id")
        if tmdb_crosswalk["movie_id"].duplicated().any():
            raise ValueError("TMDB crosswalk has multiple rows for one movie_id")
        unknown_ids = set(tmdb_crosswalk["movie_id"]) - catalog_ids
        if unknown_ids:
            raise ValueError("TMDB crosswalk contains movie IDs missing from the cleaned catalog")
        original = original.merge(
            tmdb_crosswalk[["movie_id", "tmdb_id"]], on="movie_id", how="left",
            validate="one_to_one", sort=False,
        )
    else:
        original["tmdb_id"] = pd.Series(pd.NA, index=original.index, dtype="string")

    # Collapse exact duplicate source rows for list creation, but count them for the report.
    normalized_records: list[dict[str, str | None]] = []
    for row in source_records:
        imdb_id = normalize_imdb_id(row.get("imdb_id"))
        wikidata_id = row.get("wikidata_id")
        credit_type = row.get("credit_type")
        person_id = row.get("person_id")
        person_name = clean_person_name(row.get("person_name"))
        if wikidata_id and (not str(wikidata_id).startswith("Q") or not str(wikidata_id)[1:].isdigit()):
            raise ValueError(f"Invalid Wikidata movie identifier: {wikidata_id!r}")
        if credit_type is not None and credit_type not in CREDIT_PROPERTIES:
            raise ValueError(f"Unexpected credit type: {credit_type!r}")
        normalized_records.append({
            "imdb_id": imdb_id, "wikidata_id": wikidata_id,
            "credit_type": credit_type, "person_id": person_id,
            "person_name": person_name,
        })
    unique_records = list({tuple(sorted(row.items())): row for row in normalized_records}.values())
    duplicate_source_rows_removed = len(normalized_records) - len(unique_records)

    qids_by_imdb: dict[str, set[str]] = {}
    credits_by_imdb: dict[str, list[dict[str, str | None]]] = {}
    for row in unique_records:
        imdb_id = row["imdb_id"]
        wikidata_id = row["wikidata_id"]
        if not imdb_id or not wikidata_id:
            continue
        qids_by_imdb.setdefault(imdb_id, set()).add(wikidata_id)
        if row["credit_type"] and row["person_id"]:
            credits_by_imdb.setdefault(imdb_id, []).append(row)

    enriched_rows = []
    audit_rows = []
    credit_label_conflicts = 0
    for movie in original.to_dict(orient="records"):
        imdb_id = normalize_imdb_id(movie["imdb_id"])
        qids = sorted(qids_by_imdb.get(imdb_id, set())) if imdb_id else []
        if not imdb_id:
            status, method = "unmatched_missing_imdb_id", pd.NA
        elif not qids:
            status, method = "unmatched_no_wikidata_item", pd.NA
        elif len(qids) > 1:
            status, method = "ambiguous_multiple_wikidata_items", pd.NA
        else:
            status, method = "matched", "exact_imdb_id"

        enriched = dict(movie)
        enriched.update({
            "wikidata_id": qids[0] if status == "matched" else pd.NA,
            "directors": pd.NA,
            "writers": pd.NA,
            "cast": pd.NA,
            "selected_crew": pd.NA,
            "tmdb_id_source": "MovieLens links.csv" if pd.notna(movie["tmdb_id"]) else pd.NA,
            "metadata_source": "Wikidata" if status == "matched" else pd.NA,
            "metadata_match_method": method,
            "metadata_match_status": status,
            "metadata_retrieved_at_utc": retrieved_at_utc or pd.NA,
        })

        if status == "matched":
            movie_credits = [
                row for row in credits_by_imdb.get(imdb_id, []) if row["wikidata_id"] == qids[0]
            ]
            people_by_role: dict[str, dict[str, set[str]]] = {}
            for credit in movie_credits:
                role = str(credit["credit_type"])
                person_id = str(credit["person_id"])
                names = people_by_role.setdefault(role, {}).setdefault(person_id, set())
                if credit["person_name"]:
                    names.add(str(credit["person_name"]))

            def people_for(role: str) -> list[dict[str, str | None]]:
                nonlocal credit_label_conflicts
                people = []
                for person_id in sorted(people_by_role.get(role, {})):
                    names = sorted(people_by_role[role][person_id])
                    if len(names) > 1:
                        credit_label_conflicts += 1
                    people.append({
                        "wikidata_id": person_id,
                        "name": names[0] if names else None,
                    })
                return people

            for role in PERSON_FIELDS:
                people = people_for(role)
                if people:
                    enriched["cast" if role == "cast" else ("writers" if role == "writer" else "directors")] = json.dumps(
                        people, ensure_ascii=False, separators=(",", ":")
                    )

            selected_crew = []
            for role in CREW_FIELDS:
                for person in people_for(role):
                    selected_crew.append({**person, "role": role})
            if selected_crew:
                enriched["selected_crew"] = json.dumps(
                    selected_crew, ensure_ascii=False, separators=(",", ":")
                )
        enriched_rows.append(enriched)
        audit_rows.append({
            "movie_id": int(movie["movie_id"]),
            "imdb_id": imdb_id or pd.NA,
            "metadata_match_status": status,
            "metadata_match_method": method,
            "candidate_wikidata_ids": json.dumps(qids, separators=(",", ":")) if qids else "[]",
            "source_credit_rows": len(credits_by_imdb.get(imdb_id, [])) if imdb_id else 0,
        })

    enriched_catalog = pd.DataFrame(enriched_rows)
    audit = pd.DataFrame(audit_rows)
    if len(enriched_catalog) != len(movies) or not enriched_catalog["movie_id"].equals(original["movie_id"]):
        raise ValueError("Enrichment changed catalog size or movie ID order")
    if not all(
        _same_column_values(enriched_catalog[column], original[column])
        for column in MOVIE_COLUMNS
    ):
        raise ValueError("Enrichment changed an existing cleaned movie field")

    tmdb_counts = enriched_catalog["tmdb_id"].dropna().value_counts()
    source_status = audit["metadata_match_status"]
    summary = {
        "phase": "Phase 8: Movie Metadata Enrichment",
        "total_movies": int(len(enriched_catalog)),
        "movies_with_usable_plot": int(enriched_catalog["plot"].fillna("").astype(str).str.strip().ne("").sum()),
        "movies_without_usable_plot": int(enriched_catalog["plot"].fillna("").astype(str).str.strip().eq("").sum()),
        "movies_with_tmdb_id_from_movielens_crosswalk": int(enriched_catalog["tmdb_id"].notna().sum()),
        "unique_tmdb_ids": int(enriched_catalog["tmdb_id"].nunique()),
        "duplicate_tmdb_id_values": int((tmdb_counts > 1).sum()),
        "successful_exact_imdb_identifier_matches": int(source_status.eq("matched").sum()),
        "unmatched_movies": int(source_status.str.startswith("unmatched_").sum()),
        "movies_missing_imdb_id": int(source_status.eq("unmatched_missing_imdb_id").sum()),
        "ambiguous_or_rejected_matches": int(source_status.eq("ambiguous_multiple_wikidata_items").sum()),
        "movies_with_directors": int(enriched_catalog["directors"].notna().sum()),
        "movies_with_writers": int(enriched_catalog["writers"].notna().sum()),
        "movies_with_cast": int(enriched_catalog["cast"].notna().sum()),
        "movies_with_selected_crew": int(enriched_catalog["selected_crew"].notna().sum()),
        "movies_with_poster_references": 0,
        "source_rows_received": int(len(source_records)),
        "duplicate_source_rows_removed": int(duplicate_source_rows_removed),
        "credit_label_conflicts": int(credit_label_conflicts),
        "title_year_fallback_matches": 0,
        "title_year_fallback_policy": "not used; ambiguous title/year records are left unmatched",
        "ratings_used_or_modified": False,
        "original_movie_fields_preserved": True,
        "credit_coverage_percent": {
            field: round(100 * int(enriched_catalog[field].notna().sum()) / len(enriched_catalog), 2)
            for field in ("directors", "writers", "cast", "selected_crew")
        },
    }
    return enriched_catalog, audit, summary


def sha256_file(path: Path) -> str:
    """Hash a file to verify that a read-only source such as ratings was untouched."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich the movie catalog from Wikidata credits.")
    parser.add_argument(
        "--refresh-source", action="store_true",
        help="query Wikidata again instead of reusing the saved raw snapshot",
    )
    args = parser.parse_args()

    if not MOVIES_PATH.exists() or not RATINGS_PATH.exists() or not LINKS_PATH.exists():
        parser.error("Expected cleaned catalog, cleaned ratings, and raw MovieLens links.csv files")

    movies = pd.read_csv(MOVIES_PATH, dtype={"imdb_id": "string"})
    if set(MOVIE_COLUMNS) - set(movies.columns):
        parser.error("The cleaned movie catalog does not have the expected Phase 4 columns")
    imdb_ids = sorted({value for value in movies["imdb_id"].map(normalize_imdb_id) if value})
    ratings_hash_before = sha256_file(RATINGS_PATH)
    movies_hash_before = sha256_file(MOVIES_PATH)

    try:
        source_records, retrieved_at, used_snapshot = load_or_fetch_wikidata(
            imdb_ids, refresh=args.refresh_source
        )
        tmdb_crosswalk = load_tmdb_crosswalk()
        enriched, audit, summary = build_enriched_catalog(
            movies, source_records, tmdb_crosswalk, retrieved_at
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))

    summary.update({
        "source": "Wikidata Query Service",
        "source_endpoint": WDQS_ENDPOINT,
        "source_snapshot_path": str(RAW_WIKIDATA_PATH),
        "source_snapshot_reused": bool(used_snapshot),
        "source_retrieved_at_utc": retrieved_at,
        "movie_source_path": str(MOVIES_PATH),
        "tmdb_id_source": "MovieLens ml-latest-small links.csv, exact movieId join",
        "poster_source_status": (
            "not collected: TMDB API terms prohibit use in ML applications; "
            "Wikimedia Commons image rights require per-file review"
        ),
        "ratings_file_sha256_before": ratings_hash_before,
    })

    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(ENRICHED_DIR / "movies.csv", index=False)
    audit.to_csv(AUDIT_DIR / "phase8_match_audit.csv", index=False)
    (REPORTS_DIR / "phase8_metadata_coverage.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary["cleaned_movie_file_unchanged"] = sha256_file(MOVIES_PATH) == movies_hash_before
    summary["cleaned_ratings_file_unchanged"] = sha256_file(RATINGS_PATH) == ratings_hash_before
    if not summary["cleaned_movie_file_unchanged"] or not summary["cleaned_ratings_file_unchanged"]:
        raise RuntimeError("A Phase 4 source file changed during enrichment")
    (REPORTS_DIR / "phase8_metadata_coverage.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Enriched catalog: {ENRICHED_DIR / 'movies.csv'}")
    print(f"Match audit: {AUDIT_DIR / 'phase8_match_audit.csv'}")


if __name__ == "__main__":
    main()
