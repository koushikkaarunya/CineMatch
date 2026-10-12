"""Phase 14.2 diagnostics for story-level relevance; production models stay unchanged."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase14_1_story_relevance import (
    _ndcg,
    _validated_chunk_cache,
    _usable_plot,
)
from scripts.phase9_diagnostics import (
    CHUNKED_EMBEDDINGS_PATH,
    _tokenizer_from_model,
    chunk_token_ids,
    normalized_mean,
)
from scripts.recommend_hybrid_content import HybridContentRecommender, load_enriched_movies
from scripts.recommend_tfidf import TfidfMovieRecommender, load_movies
from scripts.recommend_transformer import (
    EMBEDDINGS_PATH,
    MODEL_ID,
    MODEL_REVISION,
    TransformerMovieRecommender,
    load_sentence_model,
)

PAIRS_PATH = PROJECT_ROOT / "data" / "audit" / "phase14_2_story_relevance_pairs.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "phase14_2_story_relevance.json"
REPORT_MD = PROJECT_ROOT / "reports" / "phase14_2_story_relevance.md"
NAME_DISCOUNT = 0.80
STORY_GATE_FLOOR = 0.20
TOP_K = 5
SPLITS = ("development", "locked_audit")
SCORE_METHODS = (
    "phase9_baseline",
    "tfidf_plot_only",
    "transformer_plot_only",
    "name_discounted_tfidf",
    "phase9_story_gate",
    "chunked_mean",
    "chunked_max_pair",
    "chunked_top2_mean",
)
PERSON_NAME_BEFORE_ACTOR = re.compile(
    r"\b([A-Z][a-z]+(?:[’'-][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+){0,2})\s+\([^()]{1,100}\)"
)
CAPITALIZED_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")
SENTENCE_START_WORDS = {"a", "an", "and", "as", "at", "but", "for", "he", "her", "his", "in", "it", "later", "meanwhile", "she", "the", "they", "this", "when", "while"}


def validate_pairs(pairs: pd.DataFrame, catalog: pd.DataFrame) -> None:
    """Validate group isolation, source plots, label values, and AI provenance."""
    required = {
        "pair_id", "split", "query_movie_id", "query_title", "candidate_movie_id",
        "candidate_title", "query_plot", "candidate_plot", "provisional_relevance",
        "suggestion_status", "label_provenance", "provisional_rationale",
    }
    if required - set(pairs.columns):
        raise ValueError(f"Pair file is missing columns: {sorted(required - set(pairs.columns))}")
    if pairs.pair_id.duplicated().any() or pairs.duplicated(["query_movie_id", "candidate_movie_id"]).any():
        raise ValueError("Audit pair IDs and query/candidate combinations must be unique")
    if not set(pairs.split).issubset(SPLITS):
        raise ValueError(f"Split values must be one of {SPLITS}")
    split_counts = pairs.groupby("query_movie_id").split.nunique()
    if split_counts.gt(1).any():
        raise ValueError("A query group appears in both development and locked audit splits")

    by_id = catalog.set_index("movie_id", drop=False)
    for row in pairs.itertuples(index=False):
        query_id = int(row.query_movie_id)
        candidate_id = int(row.candidate_movie_id)
        if query_id == candidate_id or query_id not in by_id.index or candidate_id not in by_id.index:
            raise ValueError(f"Pair {row.pair_id} has a self-pair or unknown movie ID")
        for field, movie_id in (("query_plot", query_id), ("candidate_plot", candidate_id)):
            source = by_id.loc[movie_id, "plot"]
            source_text = "" if pd.isna(source) else str(source)
            if str(getattr(row, field)) != source_text:
                raise ValueError(f"Pair {row.pair_id} does not contain the catalog's exact {field}")
            if not _usable_plot(source_text):
                raise ValueError(f"Pair {row.pair_id} has no usable plot")
        label = row.provisional_relevance
        if row.suggestion_status == "ai_assisted_suggestion":
            if pd.isna(label) or int(label) not in (0, 1, 2):
                raise ValueError(f"Pair {row.pair_id} needs a provisional 0/1/2 label")
            if row.label_provenance != "ai_assisted_suggestion":
                raise ValueError(f"Pair {row.pair_id} has unclear AI-assisted label provenance")
        elif row.suggestion_status == "needs_review":
            if not pd.isna(label) or row.label_provenance != "ai_assisted_needs_review":
                raise ValueError(f"Pair {row.pair_id} must remain unlabeled and flagged for review")
        else:
            raise ValueError(f"Pair {row.pair_id} has an invalid suggestion status")


def likely_person_name_terms(plot: str) -> set[str]:
    """Return a conservative, explainable set of likely person-name tokens.

    Names attached to actor parentheticals are strong cues. Multiword
    capitalized phrases add coverage for plots such as ``Max Cohen``. This is
    deliberately a noisy heuristic, not named-entity recognition.
    """
    phrases = [match.group(1) for match in PERSON_NAME_BEFORE_ACTOR.finditer(plot)]
    for match in CAPITALIZED_PHRASE.finditer(plot):
        phrase = match.group(0)
        first = phrase.split()[0].lower()
        if first not in SENTENCE_START_WORDS:
            phrases.append(phrase)
    terms = set()
    for phrase in phrases:
        terms.update(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", phrase.lower()))
    return terms


def name_discounted_score(
    tfidf_model: TfidfMovieRecommender,
    query_id: int,
    candidate_id: int,
    query_plot: str,
    candidate_plot: str,
    *,
    discount: float = NAME_DISCOUNT,
) -> tuple[float, float, list[str]]:
    """Discount only the TF-IDF contribution of likely shared person-name unigrams."""
    if not 0 <= discount <= 1:
        raise ValueError("discount must be between 0 and 1")
    qrow = tfidf_model.movie_id_to_row[query_id]
    crow = tfidf_model.movie_id_to_row[candidate_id]
    query_vector = tfidf_model.tfidf_matrix[qrow]
    candidate_vector = tfidf_model.tfidf_matrix[crow]
    base_score = float(query_vector.multiply(candidate_vector).sum())
    shared_names = sorted(likely_person_name_terms(query_plot) & likely_person_name_terms(candidate_plot))
    vocabulary = tfidf_model.vectorizer.vocabulary_
    columns = [vocabulary[name] for name in shared_names if name in vocabulary]
    name_contribution = 0.0
    if columns:
        name_contribution = float(
            query_vector[:, columns].multiply(candidate_vector[:, columns]).sum()
        )
    adjusted = max(0.0, base_score - discount * name_contribution)
    return adjusted, name_contribution, shared_names


def chunk_pair_scores(query_chunks: np.ndarray, candidate_chunks: np.ndarray) -> tuple[float, float]:
    """Score chunk pairs by max interaction and mean of the two strongest matches."""
    q = np.asarray(query_chunks, dtype=np.float32)
    c = np.asarray(candidate_chunks, dtype=np.float32)
    if q.ndim != 2 or c.ndim != 2 or not len(q) or not len(c) or q.shape[1] != c.shape[1]:
        raise ValueError("Both chunk arrays must be non-empty matrices with matching dimensions")
    similarities = q @ c.T
    flattened = np.sort(similarities.ravel())[::-1]
    return float(flattened[0]), float(flattened[:2].mean())


def _rank_percentiles(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = pd.Series(scores, dtype="float64")
    if len(values) == 1:
        return {int(values.index[0]): 1.0}
    ranks = values.rank(method="average", ascending=True)
    return {int(movie_id): float((rank - 1) / (len(values) - 1))
            for movie_id, rank in ranks.items()}


def _token_chunks(tokenizer, plot: str, *, model_limit: int, requested_size: int = 224,
                  overlap: int = 48) -> tuple[list[list[int]], int]:
    """Tokenize without the model-forward advisory, then window within its cap."""
    ids = tokenizer.encode(plot, add_special_tokens=False, verbose=False)
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    safe_size = min(requested_size, model_limit - special_tokens)
    if safe_size < 1:
        raise ValueError("The model limit leaves no room for plot tokens")
    return chunk_token_ids(ids, chunk_size=safe_size, overlap=min(overlap, safe_size - 1)), len(ids)


def _encode_audit_chunks(
    plot_movies: pd.DataFrame,
    movie_ids: set[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, object]]:
    """Encode only the audit plots' windows using the already cached Phase 7 model."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = load_sentence_model()
    model_limit = int(model.max_seq_length)
    if model_limit != 256:
        raise ValueError(f"Expected the pinned Phase 7 input cap of 256, found {model_limit}")
    tokenizer = _tokenizer_from_model(model)
    by_id = plot_movies.set_index("movie_id")
    chunk_texts: list[str] = []
    ranges: dict[int, tuple[int, int]] = {}
    plot_wordpieces: dict[int, int] = {}
    for movie_id in sorted(movie_ids):
        plot = str(by_id.loc[movie_id, "plot"])
        windows, token_count = _token_chunks(tokenizer, plot, model_limit=model_limit)
        plot_wordpieces[movie_id] = token_count + int(tokenizer.num_special_tokens_to_add(pair=False))
        start = len(chunk_texts)
        chunk_texts.extend(tokenizer.decode(window, skip_special_tokens=True) for window in windows)
        ranges[movie_id] = (start, len(chunk_texts))
    if not chunk_texts:
        raise ValueError("No audit plot chunks were generated")
    vectors = model.encode(
        chunk_texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 384 or not np.isfinite(vectors).all():
        raise ValueError("The cached Phase 7 model returned invalid chunk embeddings")
    chunks_by_id = {movie_id: vectors[start:end] for movie_id, (start, end) in ranges.items()}
    means_by_id = {movie_id: normalized_mean(chunks) for movie_id, chunks in chunks_by_id.items()}
    return chunks_by_id, means_by_id, {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_input_limit": model_limit,
        "chunk_size_wordpieces": 224,
        "overlap_wordpieces": 48,
        "encoded_chunk_count": len(chunk_texts),
        "unique_plot_count": len(chunks_by_id),
        "long_plot_count": sum(count > model_limit for count in plot_wordpieces.values()),
        "plot_wordpieces_including_special_tokens": {str(k): int(v) for k, v in plot_wordpieces.items()},
    }


def _metric_summary(frame: pd.DataFrame, score_column: str) -> dict[str, object]:
    """Rank only explicitly labeled audit candidates; never infer labels for others."""
    frame = frame.loc[frame.provisional_relevance.notna()].copy()
    if frame.empty:
        return {
            "labeled_pairs": 0,
            "query_groups": 0,
            "relevance_counts": {str(i): 0 for i in (0, 1, 2)},
            "pairwise_accuracy_grade2_vs_grade0": None,
            "ndcg_at_5": None,
            "per_query_ndcg_at_5": {},
            "unrelated_in_top5_of_labeled_candidates": 0,
            "plausible_labeled_pairs_below_top5": 0,
            "pairwise_comparisons": 0,
        }
    results = []
    pairwise = []
    unrelated_top5 = 0
    plausible_outside = 0
    group_ndcg = {}
    for query_id, group in frame.groupby("query_movie_id", sort=True):
        ordered = group.sort_values([score_column, "candidate_movie_id"], ascending=[False, True], kind="mergesort")
        labels = ordered.provisional_relevance.astype(int).tolist()
        candidate_order = list(range(len(ordered)))
        value = _ndcg(labels, candidate_order, TOP_K)
        group_ndcg[str(int(query_id))] = None if not np.isfinite(value) else float(value)
        if np.isfinite(value):
            results.append(value)
        top = ordered.head(TOP_K)
        unrelated_top5 += int(top.provisional_relevance.eq(0).sum())
        top_ids = set(top.candidate_movie_id.astype(int))
        plausible_outside += sum(
            int(row.candidate_movie_id) not in top_ids
            for row in group.itertuples(index=False)
            if int(row.provisional_relevance) >= 1
        )
        relevant_scores = ordered.loc[ordered.provisional_relevance.eq(2), score_column].to_numpy()
        irrelevant_scores = ordered.loc[ordered.provisional_relevance.eq(0), score_column].to_numpy()
        for positive in relevant_scores:
            for negative in irrelevant_scores:
                pairwise.append(1.0 if positive > negative else 0.5 if positive == negative else 0.0)
    return {
        "labeled_pairs": int(len(frame)),
        "query_groups": int(frame.query_movie_id.nunique()),
        "relevance_counts": {str(i): int(frame.provisional_relevance.eq(i).sum()) for i in (0, 1, 2)},
        "pairwise_accuracy_grade2_vs_grade0": float(np.mean(pairwise)) if pairwise else None,
        "ndcg_at_5": float(np.mean(results)) if results else None,
        "per_query_ndcg_at_5": group_ndcg,
        "unrelated_in_top5_of_labeled_candidates": unrelated_top5,
        "plausible_labeled_pairs_below_top5": plausible_outside,
        "pairwise_comparisons": len(pairwise),
    }


def _ranked_examples(frame: pd.DataFrame, methods: tuple[str, ...] = SCORE_METHODS) -> list[dict[str, object]]:
    frame = frame.loc[frame.provisional_relevance.notna()].copy()
    examples = []
    for query_id, group in frame.groupby("query_movie_id", sort=True):
        item = {
            "split": str(group.split.iloc[0]),
            "query_movie_id": int(query_id),
            "query_title": str(group.query_title.iloc[0]),
            "labeled_candidates": int(len(group)),
            "rankings": {},
        }
        for method in methods:
            ordered = group.sort_values([method, "candidate_movie_id"], ascending=[False, True], kind="mergesort")
            item["rankings"][method] = [
                {"candidate_movie_id": int(row.candidate_movie_id),
                 "candidate_title": str(row.candidate_title),
                 "relevance": int(row.provisional_relevance),
                 "score": float(getattr(row, method))}
                for row in ordered.head(TOP_K).itertuples(index=False)
            ]
        examples.append(item)
    return examples


def _make_markdown(report: dict[str, object]) -> str:
    lines = [
        "# CineMatch Phase 14.2 — Targeted Story-Relevance Experiments", "",
        "> All labels are AI-assisted suggestions from the supplied plots, not independent human judgments. Metrics describe selected audit candidates and do not establish general performance.", "",
        "## Audit data and split", "",
        f"- Pairs: {report['dataset']['total_pairs']}; usable provisional labels: {report['dataset']['usable_labeled_pairs']}; flagged for review: {report['dataset']['needs_review_pairs']}; missing plots: {report['dataset']['missing_plot_pairs']}",
        f"- Development: {report['dataset']['split_groups']['development']} query groups / {report['dataset']['split_pairs']['development']} pairs; locked audit: {report['dataset']['split_groups']['locked_audit']} query groups / {report['dataset']['split_pairs']['locked_audit']} pairs.",
        f"- Label counts among usable pairs: {report['dataset']['label_counts']}. Provenance: `{report['dataset']['label_provenance']}`.",
        "- Groups are kept intact. The locked subset was not used to choose or tune experiment parameters. Its examples were selected as a targeted diagnostic set, not sampled randomly.",
        "- Metrics rank only explicitly labeled candidates in this file. Review-flagged rows and candidates outside the file are unjudged, not negatives; top-five counts are within this selected labeled-candidate pool.", "",
        "## Methods", "",
        "- **Phase 9 baseline:** unchanged production hybrid ranking and default weights.",
        "- **Plot-only checks:** unchanged TF-IDF and Phase 7 Transformer plot similarities.",
        f"- **Name-discounted TF-IDF:** subtracts {NAME_DISCOUNT:.2f} of the shared TF-IDF contribution from heuristic likely-person-name unigrams only. The heuristic can mistake places or other proper nouns for people.",
        f"- **Story-gated Phase 9:** multiplies the unchanged Phase 9 score by `{STORY_GATE_FLOOR:.2f} + (1 - {STORY_GATE_FLOOR:.2f}) × max(TF-IDF rank percentile, chunked-plot rank percentile)`. This is a fixed diagnostic rule, not a production weight change.",
        "- **Chunk aggregation:** compares the existing normalized mean-document chunk vector with maximum chunk-pair cosine and the mean of the strongest two chunk-pair cosines. Maximum matching may reward one incidental segment; top-two may still ignore narrative order.", "",
        "## Development results", "",
        "| Method | Labeled pairs | Groups | Pairwise accuracy (2 vs 0) | NDCG@5 | Unrelated in audited top 5 | Plausible below top 5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in report["metrics"]["development"].items():
        lines.append(
            f"| {name} | {metric['labeled_pairs']} | {metric['query_groups']} | "
            f"{_fmt(metric['pairwise_accuracy_grade2_vs_grade0'])} | {_fmt(metric['ndcg_at_5'])} | "
            f"{metric['unrelated_in_top5_of_labeled_candidates']} | {metric['plausible_labeled_pairs_below_top5']} |"
        )
    lines.extend([
        "", "## Locked audit results", "",
        "| Method | Labeled pairs | Groups | Pairwise accuracy (2 vs 0) | NDCG@5 | Unrelated in audited top 5 | Plausible below top 5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, metric in report["metrics"]["locked_audit"].items():
        lines.append(
            f"| {name} | {metric['labeled_pairs']} | {metric['query_groups']} | "
            f"{_fmt(metric['pairwise_accuracy_grade2_vs_grade0'])} | {_fmt(metric['ndcg_at_5'])} | "
            f"{metric['unrelated_in_top5_of_labeled_candidates']} | {metric['plausible_labeled_pairs_below_top5']} |"
        )
    lines.extend(["", "## Interpretation", "", *[f"- {finding}" for finding in report["findings"]]])
    lines.extend(["", "## Per-query top-five audit examples", ""])
    for query in report["per_query_examples"]:
        lines.append(f"### {query['split']}: {query['query_title']} (movie_id={query['query_movie_id']})")
        for method in SCORE_METHODS:
            titles = "; ".join(
                f"{row['candidate_title']} [grade {row['relevance']}]"
                for row in query["rankings"][method]
            )
            lines.append(f"- **{method}:** {titles}")
        lines.append("")
    lines.extend([
        "## Reproduction", "", "```sh",
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_2_story_relevance.py",
        "```", "",
    ])
    return "\n".join(lines)


def _fmt(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def run_audit() -> dict[str, object]:
    started = time.perf_counter()
    clean = load_movies()
    enriched = load_enriched_movies()
    pairs = pd.read_csv(PAIRS_PATH)
    validate_pairs(pairs, clean)
    usable = pairs.loc[pairs.suggestion_status.eq("ai_assisted_suggestion")].copy()
    needs_review = pairs.loc[pairs.suggestion_status.eq("needs_review")].copy()
    usable["provisional_relevance"] = pd.to_numeric(usable.provisional_relevance, errors="raise").astype(int)
    tfidf = TfidfMovieRecommender(clean)
    transformer = TransformerMovieRecommender(clean, cache_path=EMBEDDINGS_PATH)
    hybrid = HybridContentRecommender(enriched, tfidf_model=tfidf, transformer_model=transformer)
    chunked_vectors, chunk_metadata = _validated_chunk_cache(transformer)

    movie_by_id = clean.set_index("movie_id")
    query_ids = set(pairs.query_movie_id.astype(int))
    candidate_ids = set(pairs.candidate_movie_id.astype(int))
    audit_movie_ids = query_ids | candidate_ids
    plot_movies = clean.loc[
        clean.movie_id.isin(audit_movie_ids)
        & clean["plot"].fillna("").str.strip().ne("")
    ].copy()
    chunk_vectors, chunk_means, chunk_details = _encode_audit_chunks(plot_movies, audit_movie_ids)

    rows = []
    query_support: dict[int, dict[int, float]] = {}
    query_phase9: dict[int, dict[int, float]] = {}
    query_plot_scores: dict[int, dict[str, dict[int, float]]] = {}
    for query_id in sorted(query_ids):
        q_tf_row = tfidf.movie_id_to_row[query_id]
        tfidf_full = cosine_similarity(tfidf.tfidf_matrix[q_tf_row], tfidf.tfidf_matrix).ravel()
        tfidf_scores = {int(movie_id): float(tfidf_full[row])
                        for movie_id, row in tfidf.movie_id_to_row.items() if int(movie_id) != query_id}
        q_tr_row = transformer.movie_id_to_row[query_id]
        transformer_full = transformer.embeddings @ transformer.embeddings[q_tr_row]
        transformer_scores = {int(movie_id): float(transformer_full[row])
                              for movie_id, row in transformer.movie_id_to_row.items() if int(movie_id) != query_id}
        q_chunk_row = transformer.movie_id_to_row[query_id]
        chunked_full = chunked_vectors @ chunked_vectors[q_chunk_row]
        chunked_scores = {int(movie_id): float(chunked_full[row])
                          for movie_id, row in transformer.movie_id_to_row.items() if int(movie_id) != query_id}
        tfidf_percentiles = _rank_percentiles(tfidf_scores)
        chunk_percentiles = _rank_percentiles(chunked_scores)
        query_support[query_id] = {
            movie_id: max(tfidf_percentiles.get(movie_id, 0.0), chunk_percentiles.get(movie_id, 0.0))
            for movie_id in set(tfidf_percentiles) | set(chunk_percentiles)
        }
        phase9_results = hybrid.recommend(query_id, top_k=len(enriched))
        query_phase9[query_id] = {
            int(row.movie_id): float(row.hybrid_score) for row in phase9_results.itertuples(index=False)
        }
        query_plot_scores[query_id] = {
            "tfidf": tfidf_scores,
            "transformer": transformer_scores,
            "chunked_mean": chunked_scores,
        }

    for pair in usable.itertuples(index=False):
        query_id = int(pair.query_movie_id)
        candidate_id = int(pair.candidate_movie_id)
        adjusted_tfidf, name_contribution, shared_names = name_discounted_score(
            tfidf, query_id, candidate_id, str(pair.query_plot), str(pair.candidate_plot),
        )
        max_chunk, top2_chunk = chunk_pair_scores(
            chunk_vectors[query_id], chunk_vectors[candidate_id],
        )
        base_phase9 = query_phase9[query_id].get(candidate_id, 0.0)
        support = query_support[query_id].get(candidate_id, 0.0)
        scores = {
            "phase9_baseline": base_phase9,
            "tfidf_plot_only": query_plot_scores[query_id]["tfidf"].get(candidate_id, 0.0),
            "transformer_plot_only": query_plot_scores[query_id]["transformer"].get(candidate_id, 0.0),
            "name_discounted_tfidf": adjusted_tfidf,
            "phase9_story_gate": base_phase9 * (STORY_GATE_FLOOR + (1 - STORY_GATE_FLOOR) * support),
            "chunked_mean": query_plot_scores[query_id]["chunked_mean"].get(candidate_id, 0.0),
            "chunked_max_pair": max_chunk,
            "chunked_top2_mean": top2_chunk,
        }
        rows.append({
            "pair_id": pair.pair_id,
            "split": pair.split,
            "query_movie_id": query_id,
            "query_title": pair.query_title,
            "candidate_movie_id": candidate_id,
            "candidate_title": pair.candidate_title,
            "provisional_relevance": int(pair.provisional_relevance),
            "shared_likely_name_terms": shared_names,
            "tfidf_shared_name_contribution": name_contribution,
            "text_support_percentile": support,
            "phase9_recommendation_coverage": candidate_id in query_phase9[query_id],
            **scores,
        })
    scored = pd.DataFrame(rows)
    metrics = {}
    for split in SPLITS:
        subset = scored.loc[scored.split.eq(split)].copy()
        metrics[split] = {
            method: _metric_summary(subset, method) for method in SCORE_METHODS
        }
    paired_differences = {}
    for method in SCORE_METHODS:
        paired_differences[method] = {
            split: {
                str(qid): None if value is None or metrics[split]["phase9_baseline"]["per_query_ndcg_at_5"].get(str(qid)) is None
                else float(value - metrics[split]["phase9_baseline"]["per_query_ndcg_at_5"][str(qid)])
                for qid, value in metrics[split][method]["per_query_ndcg_at_5"].items()
            }
            for split in SPLITS
        }

    per_query = _ranked_examples(scored)
    name_checks = []
    for pair_id in ("P142-001", "P142-057", "P142-049", "P142-073"):
        found = scored.loc[scored.pair_id.eq(pair_id)]
        if not found.empty:
            row = found.iloc[0]
            name_checks.append({
                "pair_id": pair_id, "query_title": row.query_title, "candidate_title": row.candidate_title,
                "relevance": int(row.provisional_relevance), "shared_likely_name_terms": row.shared_likely_name_terms,
                "tfidf_shared_name_contribution": float(row.tfidf_shared_name_contribution),
                "tfidf_plot_only": float(row.tfidf_plot_only),
                "name_discounted_tfidf": float(row.name_discounted_tfidf),
            })
    coverage_by_split = {}
    for split in SPLITS:
        split_rows = scored.loc[scored.split.eq(split)]
        coverage_by_split[split] = {
            "returned_labeled_candidates": int(split_rows.phase9_recommendation_coverage.sum()),
            "labeled_candidate_pairs": int(len(split_rows)),
            "fraction": float(split_rows.phase9_recommendation_coverage.mean()) if len(split_rows) else None,
        }
    dev = metrics["development"]
    locked = metrics["locked_audit"]
    dynamic_findings = [
        (
            "On development groups, name-discounted TF-IDF changed NDCG@5 from "
            f"{dev['tfidf_plot_only']['ndcg_at_5']:.3f} to "
            f"{dev['name_discounted_tfidf']['ndcg_at_5']:.3f} and unrelated audited top-five entries "
            f"from {dev['tfidf_plot_only']['unrelated_in_top5_of_labeled_candidates']} to "
            f"{dev['name_discounted_tfidf']['unrelated_in_top5_of_labeled_candidates']}; it still ranked "
            "unrelated candidates highly, so the heuristic did not remove the failure mode."
        ),
        (
            "The fixed story gate changed development NDCG@5 from "
            f"{dev['phase9_baseline']['ndcg_at_5']:.3f} to {dev['phase9_story_gate']['ndcg_at_5']:.3f}; "
            "on the locked groups it changed from "
            f"{locked['phase9_baseline']['ndcg_at_5']:.3f} to {locked['phase9_story_gate']['ndcg_at_5']:.3f}. "
            "This small, selected audit does not establish a robust gate benefit."
        ),
        (
            "Maximum chunk-pair cosine versus normalized-mean chunk scoring changed development NDCG@5 "
            f"from {dev['chunked_mean']['ndcg_at_5']:.3f} to {dev['chunked_max_pair']['ndcg_at_5']:.3f} "
            f"and locked NDCG@5 from {locked['chunked_mean']['ndcg_at_5']:.3f} to "
            f"{locked['chunked_max_pair']['ndcg_at_5']:.3f}; locked unrelated audited top-five entries "
            f"rose from {locked['chunked_mean']['unrelated_in_top5_of_labeled_candidates']} to "
            f"{locked['chunked_max_pair']['unrelated_in_top5_of_labeled_candidates']}."
        ),
    ]
    report = {
        "phase": "Phase 14.2: Targeted Story-Relevance Experiments",
        "label_provenance": "ai_assisted_suggestion",
        "label_interpretation": "AI-assisted plot-based suggestions; not independent human judgments",
        "dataset": {
            "path": str(PAIRS_PATH.relative_to(PROJECT_ROOT)),
            "total_pairs": int(len(pairs)),
            "usable_labeled_pairs": int(len(usable)),
            "needs_review_pairs": int(len(needs_review)),
            "missing_plot_pairs": 0,
            "label_provenance": "ai_assisted_suggestion; uncertain pairs are explicitly flagged ai_assisted_needs_review",
            "label_counts": {str(i): int(usable.provisional_relevance.eq(i).sum()) for i in (0, 1, 2)},
            "needs_review_pair_ids": needs_review.pair_id.astype(str).tolist(),
            "split_groups": {split: int(usable.loc[usable.split.eq(split), "query_movie_id"].nunique()) for split in SPLITS},
            "split_pairs": {split: int(len(usable.loc[usable.split.eq(split)])) for split in SPLITS},
            "split_total_pairs_including_review": {split: int(pairs.split.eq(split).sum()) for split in SPLITS},
            "unique_query_groups": int(pairs.query_movie_id.nunique()),
            "review_rows_excluded_from_metrics": True,
        },
        "approaches": {
            "phase9_story_gate": {"fixed_floor": STORY_GATE_FLOOR, "uses_test_partition": False},
            "name_discounted_tfidf": {"shared_likely_person_name_discount": NAME_DISCOUNT, "heuristic": "capitalized multiword phrases and person name preceding actor parentheses"},
            "chunk_aggregation": {"mean": "existing normalized mean-document chunk vector", "max": "maximum pairwise chunk cosine", "top2_mean": "mean of two strongest pairwise chunk cosines"},
        },
        "metrics": metrics,
        "paired_per_query_ndcg_differences_vs_phase9": paired_differences,
        "per_query_examples": per_query,
        "name_overlap_diagnostics": name_checks,
        "coverage": {
            "phase9_returned_labeled_candidates": int(scored.phase9_recommendation_coverage.sum()),
            "labeled_candidate_pairs": int(len(scored)),
            "phase9_labeled_candidate_coverage_by_split": coverage_by_split,
            "phase9_unreturned_candidates_are_bottom_ranked": True,
            "candidate_pairs_outside_audit_are_unjudged": True,
        },
        "long_plot_processing": chunk_details,
        "existing_chunk_cache": {
            "path": str(CHUNKED_EMBEDDINGS_PATH.relative_to(PROJECT_ROOT)),
            "reused": True,
            "long_plot_count_full_catalog": chunk_metadata.get("long_plot_count"),
            "chunk_count_full_catalog": chunk_metadata.get("chunk_count"),
        },
        "runtime_seconds": time.perf_counter() - started,
        "findings": [
            *dynamic_findings,
            "The dataset is a selected diagnostic set assembled from known model outputs and plot comparisons; neither its metrics nor its locked groups estimate broad catalog performance.",
            "Uncertain pairs are excluded, and unjudged candidates are not treated as irrelevant. Metrics rank only the explicitly labeled pairs inside this audit file.",
            "Name attenuation changes only likely shared name-token contributions. Capitalization is an imperfect proxy; it may suppress meaningful people, places, or sequel continuity. Keep name diagnostics as an experiment, not a production text cleaner.",
            "The story gate leaves Phase 9 weights untouched but adds a fixed plot-support factor. It can demote a genuinely related story when both plot models rank it poorly, and the factor has not been tuned on locked groups.",
            "Maximum chunk matching may be fooled by one incidental passage; top-two aggregation reduces single-window domination but still loses chronology. Chunk scores do not prove semantic relevance.",
            "The Phase 9 reranker comparison is conditional on its candidate retrieval and the selected labels. No approach is declared a winner from these limited examples.",
            "No Phase 13 test partition, ratings, Phase 13/14 reports, Phase 14.1 files, production recommendation logic, or default weights were used or changed.",
        ],
        "artifacts": {
            "new_model_downloaded": False,
            "new_dependency_added": False,
            "production_recommender_changed": False,
            "phase14_1_artifacts_changed": False,
            "phase13_14_artifacts_changed": False,
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    REPORT_MD.write_text(_make_markdown(report))
    scored.to_csv(PROJECT_ROOT / "reports" / "phase14_2_story_relevance_scores.csv", index=False)
    return report


def main() -> None:
    report = run_audit()
    print(json.dumps({
        "report_json": str(REPORT_JSON),
        "report_markdown": str(REPORT_MD),
        "scores_csv": str(PROJECT_ROOT / "reports" / "phase14_2_story_relevance_scores.csv"),
        "dataset": report["dataset"],
        "development_metrics": report["metrics"]["development"],
        "locked_audit_metrics": report["metrics"]["locked_audit"],
        "long_plot_processing": report["long_plot_processing"],
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
