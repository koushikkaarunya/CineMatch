"""Phase 14.1: audit story relevance without changing production recommenders.

The fixed pair labels include AI suggestions approved by the user. They are
not independent human judgments, and the CLI names that provenance explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase9_diagnostics import CHUNKED_EMBEDDINGS_PATH, EMBEDDINGS_PATH
from scripts.recommend_hybrid_content import HybridContentRecommender, load_enriched_movies
from scripts.recommend_tfidf import TfidfMovieRecommender, load_movies
from scripts.recommend_transformer import MODEL_ID, MODEL_REVISION, TransformerMovieRecommender

PAIRS_PATH = PROJECT_ROOT / "data" / "audit" / "phase14_1_story_relevance_pairs.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "phase14_1_story_relevance.json"
REPORT_MD = PROJECT_ROOT / "reports" / "phase14_1_story_relevance.md"
TOP_K = 5
RETRIEVAL_K = 100
MODEL_NAMES = ("phase6_tfidf", "phase7_transformer", "phase9_content", "chunked_transformer")
PAIR_COLUMNS = [
    "pair_id", "query_movie_id", "query_title", "candidate_movie_id", "candidate_title",
    "query_plot", "candidate_plot", "proposed_relevance", "approved_relevance",
    "label_status", "label_provenance", "proposed_justification", "reviewer_notes",
]


def _usable_plot(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_pairs(pairs: pd.DataFrame, catalog: pd.DataFrame) -> None:
    """Check IDs, copied plot text, unique pairs, and honest label states."""
    missing = set(PAIR_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"Story pair file is missing columns: {sorted(missing)}")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("Story pair file contains duplicate pair_id values")
    if pairs.duplicated(["query_movie_id", "candidate_movie_id"]).any():
        raise ValueError("Story pair file contains a repeated query/candidate pair")
    by_id = catalog.set_index("movie_id", drop=False)
    for row in pairs.itertuples(index=False):
        query_id, candidate_id = int(row.query_movie_id), int(row.candidate_movie_id)
        if query_id == candidate_id:
            raise ValueError(f"Pair {row.pair_id} uses the query as its own candidate")
        if query_id not in by_id.index or candidate_id not in by_id.index:
            raise ValueError(f"Pair {row.pair_id} refers to an unknown movie ID")
        for field, movie_id in (("query", query_id), ("candidate", candidate_id)):
            expected = by_id.loc[movie_id, "plot"]
            actual = getattr(row, f"{field}_plot")
            expected_text = "" if pd.isna(expected) else str(expected)
            actual_text = "" if pd.isna(actual) else str(actual)
            if expected_text != actual_text:
                raise ValueError(f"Pair {row.pair_id} has altered or fabricated {field} plot text")
        proposed = row.proposed_relevance
        status = str(row.label_status)
        plots_available = _usable_plot(getattr(row, "query_plot")) and _usable_plot(
            getattr(row, "candidate_plot")
        )
        provenance = str(row.label_provenance)
        approved = row.approved_relevance
        if plots_available:
            if pd.isna(proposed) or int(proposed) not in (0, 1, 2):
                raise ValueError(f"Pair {row.pair_id} needs a proposed 0/1/2 label")
            if status == "user_approved_ai_assisted":
                if provenance != status or pd.isna(approved) or int(approved) not in (0, 1, 2):
                    raise ValueError(f"Pair {row.pair_id} has invalid user-approved AI-assisted labels")
            elif status == "ai_proposed":
                if provenance != status or not pd.isna(approved):
                    raise ValueError(f"Pair {row.pair_id} has inconsistent AI-proposed label provenance")
            elif status == "human_judged":
                raise ValueError(
                    f"Pair {row.pair_id} uses legacy human_judged status; record the actual label provenance"
                )
            else:
                raise ValueError(f"Pair {row.pair_id} has an invalid judgment status")
        elif (not pd.isna(proposed) or not pd.isna(approved)
              or status != "unjudgeable_missing_plot"
              or provenance != "not_applicable_missing_plot"):
            raise ValueError(f"Pair {row.pair_id} has a missing plot and must be unlabelled")


def _label_column_for_source(label_source: str) -> str:
    """Map explicit audit provenance names to the corresponding label column."""
    columns = {
        "proposed": "proposed_relevance",
        "user_approved_ai_assisted": "approved_relevance",
    }
    if label_source not in columns:
        raise ValueError(
            "label_source must be 'proposed' or 'user_approved_ai_assisted'; "
            "independent human judgments are not present in this audit"
        )
    return columns[label_source]


def _ndcg(relevances: list[int], order: list[int], k: int) -> float:
    """Compute graded NDCG using gain 2**relevance - 1."""
    ranked = [relevances[index] for index in order[:k]]
    ideal = sorted(relevances, reverse=True)[:k]

    def dcg(values: list[int]) -> float:
        return sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(values))

    ideal_dcg = dcg(ideal)
    return dcg(ranked) / ideal_dcg if ideal_dcg else float("nan")


def ranking_metrics(frame: pd.DataFrame, score_column: str, label_column: str,
                    *, candidate_mask: pd.Series | None = None) -> dict[str, object]:
    """Summarize per-query pair ranking; relevance 2 is the strict-positive cutoff."""
    use = frame.loc[frame[label_column].notna()].copy()
    if candidate_mask is not None:
        use = use.loc[candidate_mask.reindex(use.index).fillna(False)]
    if use.empty:
        return {"judged_pairs": 0, "query_groups": 0, "metrics": None}

    query_results = []
    pairwise_correct: list[float] = []
    unrelated_top_k = 0
    plausible_pushed_down = 0
    groups = 0
    for _, group in use.groupby("query_movie_id", sort=True):
        group = group.sort_values([score_column, "candidate_movie_id"], ascending=[False, True],
                                  kind="mergesort").reset_index(drop=True)
        labels = group[label_column].astype(int).tolist()
        scores = group[score_column].astype(float).tolist()
        relevance_order = sorted(range(len(labels)), key=lambda i: (-scores[i], int(group.iloc[i].candidate_movie_id)))
        query_results.append(_ndcg(labels, relevance_order, TOP_K))
        groups += 1
        strict_relevant = [i for i, label in enumerate(labels) if label == 2]
        strict_irrelevant = [i for i, label in enumerate(labels) if label == 0]
        pairs_for_query = []
        for relevant in strict_relevant:
            for irrelevant in strict_irrelevant:
                pairs_for_query.append(
                    1.0 if scores[relevant] > scores[irrelevant]
                    else 0.5 if scores[relevant] == scores[irrelevant] else 0.0
                )
        pairwise_correct.extend(pairs_for_query)
        top_rows = group.head(TOP_K)
        unrelated_top_k += int(top_rows[label_column].eq(0).sum())
        top_ids = set(top_rows.candidate_movie_id.astype(int))
        plausible_pushed_down += sum(
            int(row.candidate_movie_id) not in top_ids
            for row in group.itertuples(index=False)
            if int(getattr(row, label_column)) >= 1
        )
    finite_ndcg = [value for value in query_results if np.isfinite(value)]
    return {
        "judged_pairs": int(len(use)),
        "query_groups": groups,
        "queries_with_grade_2": int(use.groupby("query_movie_id")[label_column].apply(lambda s: s.eq(2).any()).sum()),
        "relevance_grade_counts": {str(i): int(use[label_column].eq(i).sum()) for i in (0, 1, 2)},
        "pairwise_accuracy_grade2_vs_grade0": float(np.mean(pairwise_correct)) if pairwise_correct else None,
        "ndcg_at_5": float(np.mean(finite_ndcg)) if finite_ndcg else None,
        "clearly_unrelated_in_top5": unrelated_top_k,
        "grade1_or_2_pairs_outside_top5": plausible_pushed_down,
    }


def _validated_chunk_cache(transformer: TransformerMovieRecommender) -> tuple[np.ndarray, dict[str, object]]:
    """Load the Phase 14 cache only when model, fingerprint, IDs, and vectors match."""
    if not CHUNKED_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing experimental chunked cache: {CHUNKED_EMBEDDINGS_PATH}")
    with np.load(CHUNKED_EMBEDDINGS_PATH, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"].item()))
        ids = artifact["movie_ids"].astype(np.int64)
        vectors = artifact["embeddings"].astype(np.float32)
    expected_ids = transformer.plot_movies["movie_id"].to_numpy(dtype=np.int64)
    expected = {
        "model_id": transformer.model_id,
        "model_revision": transformer.model_revision,
        "plot_fingerprint": transformer.fingerprint,
        "chunk_size": 224,
        "overlap": 48,
        "embedding_dimension": 384,
        "normalized": True,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Chunked cache metadata does not match the pinned Phase 7 model and text")
    if not np.array_equal(ids, expected_ids) or vectors.shape != transformer.embeddings.shape:
        raise ValueError("Chunked cache movie ID order or vector shape is misaligned")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError("Chunked cache contains invalid or non-normalized vectors")
    return vectors, metadata


def _plot_length_diagnostics(transformer: TransformerMovieRecommender,
                             max_sequence_length: int) -> dict[str, object]:
    """Count actual pinned-tokenizer input lengths without running the model."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Phase 7's existing transformers dependency is required for token-length diagnostics") from error
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    token_limit = max_sequence_length - special_tokens
    counts = []
    by_movie_id = {}
    for movie_id, plot in zip(transformer.plot_movies.movie_id, transformer.plot_movies["plot"]):
        # This is a full-text token count, not model inference. Suppress only
        # the tokenizer's generic 512-token advisory; the chunked model path
        # separately enforces its actual 256-token input limit.
        token_count = len(tokenizer.encode(
            str(plot), add_special_tokens=False, verbose=False,
        ))
        counts.append(token_count)
        by_movie_id[int(movie_id)] = token_count + special_tokens
    examples = {}
    for movie_id in (3, 4, 5, 1445, 4893, 43904):
        if movie_id in by_movie_id:
            examples[str(movie_id)] = {
                "wordpieces_including_special_tokens": by_movie_id[movie_id],
                "exceeds_model_input_limit": by_movie_id[movie_id] > max_sequence_length,
            }
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sentence_transformer_max_sequence_length": max_sequence_length,
        "usable_plots_measured": len(counts),
        "plots_over_input_limit": int(sum(count > token_limit for count in counts)),
        "plots_within_input_limit": int(sum(count <= token_limit for count in counts)),
        "sample_wordpiece_counts": examples,
        "note": "Counts are from the pinned tokenizer on original full plots, before Sentence Transformer truncation.",
    }


def _top_records(results: pd.DataFrame, score_column: str, *, limit: int = 5) -> list[dict[str, object]]:
    rows = []
    for row in results.head(limit).itertuples(index=False):
        score = getattr(row, score_column)
        score_available = score is not None and not pd.isna(score)
        rows.append({
            "movie_id": int(row.movie_id), "title": str(row.title),
            "score": float(score) if score_available else None,
            "plot_score_available": bool(score_available) if score_column == "chunked_similarity" else None,
        })
    return rows


def rerank_retrieved_candidates(
    retrieved: pd.DataFrame,
    plot_scores: dict[int, float],
    *,
    query_movie_id: int,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Reorder a fixed Phase 9 list by plot evidence; never add candidates."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    required = {"movie_id", "hybrid_score"}
    if not required.issubset(retrieved.columns):
        raise ValueError(f"Retrieved candidates need columns: {sorted(required)}")
    shortlist = retrieved.drop_duplicates("movie_id", keep="first").copy()
    shortlist = shortlist.loc[shortlist.movie_id.astype(int).ne(int(query_movie_id))]
    shortlist["chunked_similarity"] = shortlist.movie_id.map(
        lambda candidate_id: plot_scores.get(int(candidate_id), np.nan)
    )
    return shortlist.sort_values(
        ["chunked_similarity", "hybrid_score", "movie_id"],
        ascending=[False, False, True], kind="mergesort", na_position="last",
    ).head(top_k).reset_index(drop=True)


def _make_markdown(report: dict[str, object]) -> str:
    lines = [
        "# CineMatch Phase 14.1 — Story-Level Relevance Audit", "",
        f"> Label source: `{report['label_source']}`. These are AI-proposed labels approved by the user, not independent human judgments. This selected-pair pilot is descriptive, not a general accuracy estimate.", "",
        "## Audit dataset", "",
        f"- Pairs: {report['dataset']['total_pairs']}; labelable with both plots: {report['dataset']['plot_available_pairs']}; missing-plot exclusions: {report['dataset']['missing_plot_pairs']}",
        f"- AI-proposed label counts: {report['dataset']['proposed_label_counts']}; user-approved AI-assisted labels: {report['dataset']['approved_label_counts']}",
        f"- Queries: {report['dataset']['query_groups']}; plots span {report['dataset']['genres_seen']} distinct catalog genres.",
        "- The CSV records `approved_relevance` with `label_provenance=user_approved_ai_assisted`; three missing-plot rows remain unjudgeable.", "",
        "## Pair ranking results", "",
        "Metrics use grade 2 as strict relevance for pairwise accuracy against grade 0. NDCG uses grades 0–2. Models are scored on the same judged rows when required plot inputs exist. The top-five counts apply only to the deliberately selected pair candidates for each query, not the full catalog recommendation lists.", "",
        "| Model | Pairs | Queries | Pairwise accuracy (2 vs 0) | NDCG@5 | Unrelated in top 5 | Plausible pairs below top 5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["model_metrics"].items():
        lines.append(
            f"| {name} | {values.get('judged_pairs', 0)} | {values.get('query_groups', 0)} | "
            f"{_fmt(values.get('pairwise_accuracy_grade2_vs_grade0'))} | {_fmt(values.get('ndcg_at_5'))} | "
            f"{values.get('clearly_unrelated_in_top5', '—')} | {values.get('grade1_or_2_pairs_outside_top5', '—')} |"
        )
    rerank = report["reranker"]
    lines.extend([
        "", "## Experimental Phase 9 shortlist reranking", "",
        f"The reranker first retrieves up to {RETRIEVAL_K} Phase 9 candidates, then reorders only those candidates by the cached chunked Transformer cosine. Original Phase 9 results remain unchanged.",
        f"Candidate-pair availability: {rerank['candidate_pair_coverage']}/{rerank['labelable_pairs']} labeled pairs were present in the Phase 9 shortlist; this conditional result does not measure retrieval misses.",
        f"On those same retrieved audit pairs, Phase 9 NDCG@5={_fmt(rerank['phase9_metrics_on_same_retrieved_pairs'].get('ndcg_at_5'))}; chunked rerank NDCG@5={_fmt(rerank['metrics_on_retrieved_labeled_pairs'].get('ndcg_at_5'))}. Pairwise accuracy is not computable because no query group has both a grade-2 pair and a grade-0 pair after retrieval.",
        f"Among the selected candidate rows only, Phase 9 placed {rerank['phase9_metrics_on_same_retrieved_pairs'].get('clearly_unrelated_in_top5', 0)} grade-0 pairs in the top five; chunked reranking placed {rerank['metrics_on_retrieved_labeled_pairs'].get('clearly_unrelated_in_top5', 0)}. These are not counts from the full recommendation catalog.",
        f"Scoring/ranking time: {report['runtime']['pair_scoring_seconds']:.3f}s; peak process RSS: {report['runtime']['peak_rss_mib']:.1f} MiB; cache size: {report['runtime']['chunked_cache_bytes'] / (1024 * 1024):.2f} MiB; cache reused: {report['runtime']['chunked_cache_reused']}.",
        f"Pinned-tokenizer check: {report['transformer_input_lengths']['plots_over_input_limit']}/{report['transformer_input_lengths']['usable_plots_measured']} usable plots exceed the {report['transformer_input_lengths']['sentence_transformer_max_sequence_length']}-wordpiece model limit.",
        "", "| Query | Plot / rerank | Phase 9 original top 5 | Chunked reranked top 5 |", "|---|---|---|---|",
    ])
    for audit in report["qualitative_audit"]:
        before = "; ".join(r["title"] for r in audit["phase9_top5"])
        after = "; ".join(r["title"] for r in audit["chunked_reranked_top5"])
        plot_status = "available; rerank applied" if audit["chunked_rerank_applied"] else "missing; Phase 9 metadata list retained"
        lines.append(f"| {audit['query_title']} | {plot_status} | {before} | {after} |")
    lines.extend([
        "", "## Findings and limitations", "",
        *[f"- {item}" for item in report["findings"]],
        "", "## Reproduction", "",
        "```sh", "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_1_story_relevance.py", "```", "",
        "Evaluate the user-approved AI-assisted labels:", "", "```sh",
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_1_story_relevance.py --label-source user_approved_ai_assisted", "```", "",
    ])
    return "\n".join(lines)


def _fmt(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def run_audit(*, label_source: str = "proposed") -> dict[str, object]:
    """Score fixed pairs with unchanged models and an isolated chunked reranker."""
    started = time.perf_counter()
    clean = load_movies()
    enriched = load_enriched_movies()
    pairs = pd.read_csv(PAIRS_PATH, dtype={
        "query_title": "string", "candidate_title": "string", "query_plot": "string",
        "candidate_plot": "string", "label_status": "string", "proposed_justification": "string",
        "reviewer_notes": "string",
    })
    validate_pairs(pairs, clean)
    label_column = _label_column_for_source(label_source)
    plot_available_rows = pairs.proposed_relevance.notna()
    if pairs.loc[plot_available_rows, label_column].isna().any():
        raise ValueError(f"Label source {label_source!r} is missing a value for a plot-available pair")
    pairs[label_column] = pd.to_numeric(pairs[label_column], errors="coerce")
    valid = pairs.loc[pairs["proposed_relevance"].notna()].copy()

    tfidf = TfidfMovieRecommender(clean)
    transformer = TransformerMovieRecommender(clean, cache_path=EMBEDDINGS_PATH)
    hybrid = HybridContentRecommender(enriched, tfidf_model=tfidf, transformer_model=transformer)
    chunked_vectors, chunk_metadata = _validated_chunk_cache(transformer)
    plot_lengths = _plot_length_diagnostics(
        transformer, int(chunk_metadata["max_sequence_length"]),
    )
    tfidf_row = tfidf.movie_id_to_row
    transformer_row = transformer.movie_id_to_row
    enriched_position = {int(movie_id): i for i, movie_id in enumerate(hybrid.catalog.movie_id)}

    query_ids = sorted(set(pairs.query_movie_id.astype(int)))
    tfidf_rankings: dict[int, dict[int, float]] = {}
    transformer_rankings: dict[int, dict[int, float]] = {}
    hybrid_rankings: dict[int, dict[int, float]] = {}
    hybrid_explanations: dict[int, dict[int, str]] = {}
    chunked_rankings: dict[int, dict[int, float]] = {}
    hybrid_top100: dict[int, list[int]] = {}
    hybrid_top5: dict[int, list[dict[str, object]]] = {}
    reranked_top5: dict[int, list[dict[str, object]]] = {}
    score_start = time.perf_counter()
    for query_id in query_ids:
        if query_id in tfidf_row:
            qrow = tfidf_row[query_id]
            tfidf_scores = cosine_similarity(tfidf.tfidf_matrix[qrow], tfidf.tfidf_matrix).ravel()
            tfidf_rankings[query_id] = {
                int(movie_id): float(tfidf_scores[row])
                for movie_id, row in tfidf_row.items() if int(movie_id) != query_id
            }
        else:
            tfidf_rankings[query_id] = {}
        if query_id in transformer_row:
            qrow = transformer_row[query_id]
            transformer_scores = transformer.embeddings @ transformer.embeddings[qrow]
            transformer_rankings[query_id] = {
                int(movie_id): float(transformer_scores[row])
                for movie_id, row in transformer_row.items() if int(movie_id) != query_id
            }
            chunk_scores = chunked_vectors @ chunked_vectors[qrow]
            chunked_rankings[query_id] = {
                int(movie_id): float(chunk_scores[row])
                for movie_id, row in transformer_row.items() if int(movie_id) != query_id
            }
        else:
            transformer_rankings[query_id] = {}
            chunked_rankings[query_id] = {}

        full_hybrid = hybrid.recommend(query_id, top_k=len(hybrid.catalog))
        hybrid_rankings[query_id] = {
            int(row.movie_id): float(row.hybrid_score)
            for row in full_hybrid.itertuples(index=False)
        }
        hybrid_explanations[query_id] = {
            int(row.movie_id): str(row.explanation)
            for row in full_hybrid.itertuples(index=False)
        }
        retrieved = full_hybrid.head(RETRIEVAL_K)
        hybrid_top100[query_id] = retrieved.movie_id.astype(int).tolist()
        hybrid_top5[query_id] = _top_records(full_hybrid, "hybrid_score")
        qposition = enriched_position[query_id]
        if query_id in transformer_row:
            reranked = rerank_retrieved_candidates(
                retrieved, chunked_rankings[query_id], query_movie_id=query_id,
            )
            reranked_top5[query_id] = _top_records(reranked, "chunked_similarity")
        else:
            # A plotless query has no direct plot-to-plot score; retain its
            # metadata-based Phase 9 list without claiming a rerank occurred.
            reranked_top5[query_id] = _top_records(retrieved, "hybrid_score")

    score_seconds = time.perf_counter() - score_start
    scored = valid.copy()
    scored["phase6_tfidf"] = [tfidf_rankings.get(int(q), {}).get(int(c), np.nan)
                               for q, c in zip(scored.query_movie_id, scored.candidate_movie_id)]
    scored["phase7_transformer"] = [transformer_rankings.get(int(q), {}).get(int(c), np.nan)
                                    for q, c in zip(scored.query_movie_id, scored.candidate_movie_id)]
    scored["phase9_content"] = [hybrid_rankings.get(int(q), {}).get(int(c), 0.0)
                                 for q, c in zip(scored.query_movie_id, scored.candidate_movie_id)]
    scored["chunked_transformer"] = [chunked_rankings.get(int(q), {}).get(int(c), np.nan)
                                      for q, c in zip(scored.query_movie_id, scored.candidate_movie_id)]
    scored["in_phase9_top100"] = [int(c) in set(hybrid_top100.get(int(q), []))
                                  for q, c in zip(scored.query_movie_id, scored.candidate_movie_id)]

    model_metrics = {}
    model_coverage = {}
    for name in MODEL_NAMES:
        available = scored.loc[scored[name].notna()].copy()
        model_metrics[name] = ranking_metrics(available, name, label_column)
        model_coverage[name] = {
            "available_labeled_pairs": int(len(available)),
            "total_labeled_pairs": int(len(scored)),
            "available_queries": int(available.query_movie_id.nunique()),
        }

    rerank_labeled = scored.loc[scored.in_phase9_top100].copy()
    reranker_metrics = ranking_metrics(
        rerank_labeled, "chunked_transformer", label_column
    )
    original_shortlist_metrics = ranking_metrics(
        rerank_labeled, "phase9_content", label_column
    )
    qualitative = []
    audit_query_ids = (1, 3, 4, 5, 1445)
    for query_id in audit_query_ids:
        if query_id not in hybrid.catalog_ids:
            continue
        qrow = hybrid.catalog.loc[hybrid.catalog.movie_id.eq(query_id)].iloc[0]
        original_rows = hybrid_top5.get(query_id, [])
        reranked_rows = reranked_top5.get(query_id, [])
        qualitative.append({
            "query_movie_id": query_id,
            "query_title": str(qrow.title),
            "query_plot_available": _usable_plot(qrow["plot"]),
            "query_plot_characters": len(str(qrow["plot"])) if _usable_plot(qrow["plot"]) else 0,
            "chunked_rerank_applied": query_id in transformer_row,
            "phase9_top5": original_rows,
            "chunked_reranked_top5": reranked_rows,
            "top5_overlap": len({r["movie_id"] for r in original_rows} & {r["movie_id"] for r in reranked_rows}),
        })

    missing_pair_rows = pairs.loc[
        pairs.query_plot.fillna("").str.strip().eq("") | pairs.candidate_plot.fillna("").str.strip().eq("")
    ]
    genres_seen = set()
    genre_column = enriched.set_index("movie_id")["genres"]
    for movie_id in set(pairs.query_movie_id.astype(int)) | set(pairs.candidate_movie_id.astype(int)):
        value = genre_column.loc[movie_id]
        if not pd.isna(value):
            genres_seen.update(g for g in str(value).split("|") if g and g != "(no genres listed)")

    scored_lookup = {str(row.pair_id): row for row in scored.itertuples(index=False)}
    pair_results = []
    for source in pairs.itertuples(index=False):
        row = scored_lookup.get(str(source.pair_id))
        pair_results.append({
            "pair_id": str(source.pair_id),
            "query_movie_id": int(source.query_movie_id),
            "query_title": str(source.query_title),
            "candidate_movie_id": int(source.candidate_movie_id),
            "candidate_title": str(source.candidate_title),
            "proposed_relevance": None if pd.isna(source.proposed_relevance) else int(source.proposed_relevance),
            "approved_relevance": None if pd.isna(source.approved_relevance) else int(source.approved_relevance),
            "label_status": str(source.label_status),
            "label_provenance": str(source.label_provenance),
            "proposed_justification": str(source.proposed_justification),
            "phase9_explanation": hybrid_explanations.get(int(source.query_movie_id), {}).get(
                int(source.candidate_movie_id)
            ),
            "scores": {
                name: None if row is None or pd.isna(getattr(row, name)) else float(getattr(row, name))
                for name in MODEL_NAMES
            },
            "in_phase9_top100": bool(row.in_phase9_top100) if row is not None else False,
        })
    known_collision_pairs = {"SR001": ("Max",), "SR005": ("Savannah",),
                             "SR006": ("Robin",), "SR007": ("Robin",)}
    collision_diagnostics = []
    for item in pair_results:
        names = known_collision_pairs.get(item["pair_id"], ())
        source_pair = pairs.loc[pairs.pair_id.eq(item["pair_id"])].iloc[0]
        query_text = "" if pd.isna(source_pair.query_plot) else str(source_pair.query_plot)
        candidate_text = "" if pd.isna(source_pair.candidate_plot) else str(source_pair.candidate_plot)
        shared_names = [name for name in names if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query_text, re.I)
                        and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", candidate_text, re.I)]
        if names:
            collision_diagnostics.append({
                "pair_id": item["pair_id"], "shared_name_tokens": shared_names,
                "query_occurrences": {name: len(re.findall(rf"(?<!\w){re.escape(name)}(?!\w)", query_text, re.I)) for name in names},
                "candidate_occurrences": {name: len(re.findall(rf"(?<!\w){re.escape(name)}(?!\w)", candidate_text, re.I)) for name in names},
                "tfidf_cosine": item["scores"]["phase6_tfidf"],
            })
    exact_duplicate_plot_pairs = []
    for left, right in (("SR013", "SR014"),):
        first = pairs.loc[pairs.pair_id.eq(left), "candidate_plot"].iloc[0]
        second = pairs.loc[pairs.pair_id.eq(right), "candidate_plot"].iloc[0]
        exact_duplicate_plot_pairs.append({"pair_ids": [left, right], "plots_identical": str(first) == str(second)})
    fixed_label_lookup = {
        (item["query_movie_id"], item["candidate_movie_id"]): (
            item["approved_relevance"] if label_source == "user_approved_ai_assisted" else item["proposed_relevance"]
        )
        for item in pair_results
    }
    recommendation_audit = []
    for audit in qualitative:
        query_id = audit["query_movie_id"]
        comparison = {"query_movie_id": query_id, "query_title": audit["query_title"]}
        for label, key in (("phase9", "phase9_top5"), ("chunked_rerank", "chunked_reranked_top5")):
            reviewed = []
            for rank, candidate in enumerate(audit[key], start=1):
                relevance = fixed_label_lookup.get((query_id, candidate["movie_id"]))
                if relevance is not None:
                    reviewed.append({"rank": rank, "movie_id": candidate["movie_id"],
                                     "title": candidate["title"], "relevance": relevance})
            comparison[label] = reviewed
            comparison[f"{label}_grade0_count"] = sum(item["relevance"] == 0 for item in reviewed)
        recommendation_audit.append(comparison)
    observed_rerank_changes = []
    for pair in pair_results:
        label = pair["approved_relevance"] if label_source == "user_approved_ai_assisted" else pair["proposed_relevance"]
        if label is None:
            continue
        query_id = pair["query_movie_id"]
        candidate_id = pair["candidate_movie_id"]
        original_ids = [row["movie_id"] for row in hybrid_top5.get(query_id, [])]
        reranked_ids = [row["movie_id"] for row in reranked_top5.get(query_id, [])]
        old_rank = original_ids.index(candidate_id) + 1 if candidate_id in original_ids else None
        new_rank = reranked_ids.index(candidate_id) + 1 if candidate_id in reranked_ids else None
        if old_rank is not None:
            observed_rerank_changes.append({
                "pair_id": pair["pair_id"], "audit_relevance": label,
                "phase9_rank": old_rank, "chunked_rerank_rank": new_rank,
                "pushed_below_top5": new_rank is None and label >= 1,
                "unrelated_still_in_top5": label == 0 and new_rank is not None,
            })

    report = {
        "phase": "Phase 14.1: Story-Level Relevance Improvement",
        "label_source": label_source,
        "label_interpretation": (
            "AI-generated proposal labels; not independent human judgments"
            if label_source == "proposed"
            else "AI-generated relevance suggestions reviewed and approved by the user; not independent human judgments"
        ),
        "label_provenance": label_source,
        "dataset": {
            "path": str(PAIRS_PATH.relative_to(PROJECT_ROOT)),
            "total_pairs": int(len(pairs)),
            "plot_available_pairs": int(len(valid)),
            "missing_plot_pairs": int(len(missing_pair_rows)),
            "query_groups": int(valid.query_movie_id.nunique()),
            "genres_seen": len(genres_seen),
            "proposed_label_counts": {str(i): int(pd.to_numeric(pairs.proposed_relevance, errors="coerce").eq(i).sum()) for i in (0, 1, 2)},
            "approved_label_counts": {
                str(i): int(pd.to_numeric(pairs.approved_relevance, errors="coerce").eq(i).sum())
                for i in (0, 1, 2)
            },
            "approved_label_pairs": int(pairs.approved_relevance.notna().sum()),
            "label_status_counts": {str(k): int(v) for k, v in pairs.label_status.value_counts(dropna=False).items()},
            "missing_plot_pair_ids": missing_pair_rows.pair_id.astype(str).tolist(),
        },
        "model_coverage": model_coverage,
        "model_metrics": model_metrics,
        "reranker": {
            "method": f"Phase 9 top {RETRIEVAL_K} candidate retrieval, then cached chunked Transformer cosine reranking",
            "candidate_pair_coverage": int(scored.loc[scored.in_phase9_top100, label_column].notna().sum()),
            "labelable_pairs": int(scored[label_column].notna().sum()),
            "metrics_on_retrieved_labeled_pairs": reranker_metrics,
            "phase9_metrics_on_same_retrieved_pairs": original_shortlist_metrics,
            "top5_unrelated_in_fixed_pair_audit": reranker_metrics.get("clearly_unrelated_in_top5"),
            "plausible_pairs_outside_top5_in_retrieved_audit": reranker_metrics.get("grade1_or_2_pairs_outside_top5"),
            "recommendation_changes_for_labeled_pairs_that_were_in_phase9_top5": observed_rerank_changes,
            "labeled_plausible_phase9_top5_pairs_pushed_below_top5": sum(
                item["pushed_below_top5"] for item in observed_rerank_changes
            ),
            "labeled_grade0_phase9_top5_pairs_still_in_reranked_top5": sum(
                item["unrelated_still_in_top5"] for item in observed_rerank_changes
            ),
            "note": "Conditional on Phase 9 retrieval; omitted candidates are not recovered by reranking. Original rank lists are untouched.",
        },
        "pair_results": pair_results,
        "targeted_text_diagnostics": {
            "known_name_collision_pairs": collision_diagnostics,
            "duplicate_candidate_plot_checks": exact_duplicate_plot_pairs,
            "plot_length_chars_by_pair": {
                str(row.pair_id): {
                    "query": len(str(row.query_plot)) if _usable_plot(row.query_plot) else None,
                    "candidate": len(str(row.candidate_plot)) if _usable_plot(row.candidate_plot) else None,
                }
                for row in pairs.itertuples(index=False)
            },
        },
        "qualitative_audit": qualitative,
        "recommendation_audit_labeled_overlap_only": recommendation_audit,
        "runtime": {
            "pair_scoring_seconds": score_seconds,
            "total_runtime_seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
            "chunked_cache_path": str(CHUNKED_EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)),
            "chunked_cache_bytes": CHUNKED_EMBEDDINGS_PATH.stat().st_size,
            "chunked_cache_reused": True,
            "chunk_count": chunk_metadata.get("chunk_count"),
            "long_plot_count": chunk_metadata.get("long_plot_count"),
            "chunk_size_wordpieces": chunk_metadata.get("chunk_size"),
            "overlap_wordpieces": chunk_metadata.get("overlap"),
        },
        "transformer_input_lengths": plot_lengths,
        "findings": [
            "The prior errors are heterogeneous: the Phase 13 audit shows Max overlap for Grumpier Old Men → Pi and Robin/Savannah name collisions for Waiting to Exhale; genres, generic relationship themes, the Phase 9 rank-fusion design, truncation, and source-plot quality are separate plausible contributors.",
            "The Father of the Bride Part II → When a Stranger Calls results are unrelated by the supplied plot texts. The two When a Stranger Calls records have identical plot text, so their mutual remake similarity does not validate their match to the query.",
            "Phase 14 found name masking unchanged versus Phase 9 on rating metrics; that is not evidence about story relevance and is not repeated as a proposed solution here.",
            "The Phase 7 model is capped at 256 wordpieces. The pinned Phase 14 chunk cache records 3,400 long plots; truncation is a credible coverage limitation, but chunk averaging can erase order and no semantic ground truth yet shows it fixes relevance.",
            "The pair set is small and selected to probe known cases. Labels used for the approved-label run were AI-generated suggestions reviewed and approved by the user; there are no independent human judgments. Metrics are descriptive and should not be generalized.",
            "No suitable pairwise model was already cached. An MS MARCO cross-encoder is designed for query-to-passage ranking and an STS cross-encoder for sentence similarity, not full plot-to-plot story relevance; no model or dependency was downloaded.",
            "No production weights, recommendation code, Phase 13/14 reports, or evaluation manifests were changed. No Phase 13 test partition data was read for this experiment.",
        ],
        "artifacts": {
            "chunked_model_cache": "validated by pinned model ID/revision, plot fingerprint, ordered IDs, vector shape and unit norms",
            "rating_features_used": False,
            "fabricated_plots_or_labels": False,
            "production_changes": False,
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    REPORT_MD.write_text(_make_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CineMatch content recommenders on fixed story pairs.")
    parser.add_argument(
        "--label-source", choices=("proposed", "user_approved_ai_assisted"),
        default="proposed",
        help="Choose AI proposals or user-approved AI-assisted labels; independent human labels are not available.",
    )
    args = parser.parse_args()
    report = run_audit(label_source=args.label_source)
    print(json.dumps({
        "report_json": str(REPORT_JSON),
        "report_markdown": str(REPORT_MD),
        "label_source": report["label_source"],
        "dataset": report["dataset"],
        "model_metrics": report["model_metrics"],
        "reranker": report["reranker"],
        "runtime": report["runtime"],
    }, indent=2))


if __name__ == "__main__":
    main()
