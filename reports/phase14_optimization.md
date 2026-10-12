# CineMatch Phase 14 — Evidence-Based Optimization

## Validation boundary

Nested temporal holdout from Phase 13 training only: 64,081 train and 16,338 validation interactions across 610 users. Phase 13 test overlap: 0. Boundary ties: 132. The Phase 13 test report and manifest were not used for tuning.

## Ranking validation

Relevance is a validation rating ≥4.0. All models use the same validation users and candidate universe (catalog excluding each user's Phase 14 training-rated movies). Recall/NDCG omit users with no relevant validation item; precision includes every eligible user.

| Model | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | Users with recs / eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_popularity | 0.0646 | 0.0449 | 0.0759 | 0.0552 | 0.0750 | 0.0822 | 610/610 (100.0%) |
| phase9_content_hybrid | 0.0170 | 0.0106 | 0.0208 | 0.0151 | 0.0168 | 0.0220 | 608/610 (99.7%) |
| phase12_content_only_route | 0.0170 | 0.0106 | 0.0208 | 0.0151 | 0.0168 | 0.0220 | 608/610 (99.7%) |
| phase10_collaborative | 0.0079 | 0.0051 | 0.0094 | 0.0067 | 0.0084 | 0.0100 | 608/610 (99.7%) |
| phase12_unified_hybrid | 0.0170 | 0.0104 | 0.0201 | 0.0125 | 0.0157 | 0.0197 | 608/610 (99.7%) |
| phase12_weighted_content_75 | 0.0177 | 0.0110 | 0.0215 | 0.0143 | 0.0170 | 0.0217 | 608/610 (99.7%) |
| phase9_name_masked_diagnostic | 0.0170 | 0.0106 | 0.0208 | 0.0151 | 0.0168 | 0.0220 | 608/610 (99.7%) |
| phase9_chunked_diagnostic | 0.0161 | 0.0104 | 0.0201 | 0.0152 | 0.0171 | 0.0221 | 608/610 (99.7%) |

## Phase 10 rating prediction

| Model | N | Coverage | MAE | RMSE |
|---|---:|---:|---:|---:|
| global_mean | 16338 | 100.0% | 0.8102 | 1.0340 |
| user_mean | 16338 | 100.0% | 0.7410 | 0.9552 |
| phase10_collaborative | 13542 | 82.9% | 0.6708 | 0.8828 |

## Paired differences at K=10

Differences are first model minus baseline; intervals bootstrap users.

- `phase12_weighted_content_75_minus_phase9_content_hybrid_precision@10`: -0.0008, 95% interval [-0.0025, 0.0008] (n=610).
- `phase12_weighted_content_75_minus_phase9_content_hybrid_recall@10`: 0.0002, 95% interval [-0.0023, 0.0029] (n=593).
- `phase12_weighted_content_75_minus_phase9_content_hybrid_ndcg@10`: -0.0003, 95% interval [-0.0022, 0.0017] (n=593).
- `phase12_weighted_content_75_minus_phase12_unified_hybrid_precision@10`: 0.0018, 95% interval [-0.0000, 0.0038] (n=610).
- `phase12_weighted_content_75_minus_phase12_unified_hybrid_recall@10`: 0.0013, 95% interval [-0.0017, 0.0049] (n=593).
- `phase12_weighted_content_75_minus_phase12_unified_hybrid_ndcg@10`: 0.0019, 95% interval [-0.0007, 0.0046] (n=593).
- `phase12_unified_hybrid_minus_phase9_content_hybrid_precision@10`: -0.0026, 95% interval [-0.0052, -0.0002] (n=610).
- `phase12_unified_hybrid_minus_phase9_content_hybrid_recall@10`: -0.0011, 95% interval [-0.0056, 0.0030] (n=593).
- `phase12_unified_hybrid_minus_phase9_content_hybrid_ndcg@10`: -0.0023, 95% interval [-0.0059, 0.0019] (n=593).
- `phase9_name_masked_diagnostic_minus_phase9_content_hybrid_precision@10`: 0.0000, 95% interval [0.0000, 0.0000] (n=610).
- `phase9_name_masked_diagnostic_minus_phase9_content_hybrid_recall@10`: 0.0000, 95% interval [0.0000, 0.0000] (n=593).
- `phase9_name_masked_diagnostic_minus_phase9_content_hybrid_ndcg@10`: 0.0000, 95% interval [0.0000, 0.0000] (n=593).
- `phase9_chunked_diagnostic_minus_phase9_content_hybrid_precision@10`: 0.0002, 95% interval [-0.0005, 0.0008] (n=610).
- `phase9_chunked_diagnostic_minus_phase9_content_hybrid_recall@10`: 0.0003, 95% interval [-0.0001, 0.0011] (n=593).
- `phase9_chunked_diagnostic_minus_phase9_content_hybrid_ndcg@10`: 0.0002, 95% interval [-0.0007, 0.0011] (n=593).
- `phase12_weighted_content_75_minus_global_popularity_precision@10`: -0.0410, 95% interval [-0.0490, -0.0331] (n=610).
- `phase12_weighted_content_75_minus_global_popularity_recall@10`: -0.0581, 95% interval [-0.0735, -0.0448] (n=593).
- `phase12_weighted_content_75_minus_global_popularity_ndcg@10`: -0.0605, 95% interval [-0.0747, -0.0478] (n=593).

## Sparse history and source diagnostics

Sparse users: 136; weighted-fusion coverage 98.5%; P@10 0.0088; R@10 0.0294; NDCG@10 0.0252.
Both sources returned a list for 608/610 users; both were empty for 2. Their top-10 lists shared 0.04 movies on average (mean Jaccard 0.0022; overlap for 25 users). Relevant validation hits: content 92, collaborative 41, equal fusion 76; equal fusion omitted 40 relevant items that were in content's top 10.
Equal-fusion scores tied within top 10 for 608 users; 26 had a tie at rank 10/11. The existing ascending movie_id tie-break is preserved. Phase 12 top-10 source rows: {'collaborative': 2944, 'content': 2831, 'collaborative, content': 305}.
Popularity ordering used training counts/means and covered all 9742 catalog IDs; 2602 had no training ratings and were placed last. 39/100 top-100 titles had training mean ≥4, compared with 1699/7140 rated catalog titles overall. Exposure effects remain unresolved.
Usable plots were present for 3434/9742 catalog titles; the JSON report also breaks plot/genre/tag/credit coverage down for positive training seeds, relevant validation items, and users with/without a plot-bearing seed. Missing credit values mean unavailable/unretrieved, not confirmed absence.

## Diagnostic variant comparisons

- **phase9_name_masked_diagnostic**: P@10 0.0151, R@10 0.0168, NDCG@10 0.0220, coverage 99.7%.
  Paired NDCG@10 difference vs Phase 9: 0.0000, 95% interval [0.0000, 0.0000].
- **phase9_chunked_diagnostic**: P@10 0.0152, R@10 0.0171, NDCG@10 0.0221, coverage 99.7%.
  Paired NDCG@10 difference vs Phase 9: 0.0002, 95% interval [-0.0007, 0.0011].
Chunked Transformer validation reused the pinned cache (17,548 windows for 3,400 long plots; 224 wordpieces with 48 overlap). Original/chunked artifact sizes: 4.88/4.89 MB. Phase 9 seed-ranking time: 34.09s baseline vs 33.18s chunked; total Phase 14 run: 299.41s. Embedding generation time was not measured in this run because the cache already existed.

## Qualitative audit

The JSON report contains full deterministic query rows and evidence. These are diagnostics, not human semantic relevance judgments.

- **Grumpier Old Men (1995)** (movie_id 3, plot available: True): Phase 9 top result: Wedding Planner, The (2001); name-masked diagnostic: Wedding Planner, The (2001); chunked diagnostic: Two Much (1995).
- **Waiting to Exhale (1995)** (movie_id 4, plot available: True): Phase 9 top result: Hope Floats (1998); name-masked diagnostic: Hope Floats (1998); chunked diagnostic: Hope Floats (1998).
- **Father of the Bride Part II (1995)** (movie_id 5, plot available: True): Phase 9 top result: First Wives Club, The (1996); name-masked diagnostic: First Wives Club, The (1996); chunked diagnostic: Movie 43 (2013).
- **McHale's Navy (1997)** (movie_id 1445, plot available: True): Phase 9 top result: Russians Are Coming, the Russians Are Coming, The (1966); name-masked diagnostic: Russians Are Coming, the Russians Are Coming, The (1966); chunked diagnostic: Russians Are Coming, the Russians Are Coming, The (1966).
- **Toy Story (1995)** (movie_id 1, plot available: False): Phase 9 top result: Toy Story 2 (1999); name-masked diagnostic: Toy Story 2 (1999); chunked diagnostic: Toy Story 2 (1999).
- **Godfather, The (1972)** (movie_id 858, plot available: False): Phase 9 top result: Godfather: Part II, The (1974); name-masked diagnostic: Godfather: Part II, The (1974); chunked diagnostic: Godfather: Part II, The (1974).

## Interpretation and limitations

- No semantic ground-truth labels or human judgments are available; rating relevance is not story similarity.
- MovieLens is positive-unlabeled: an unobserved candidate is not necessarily disliked.
- Name masking is a diagnostic using previously identified names, not a general entity-resolution solution and not a production change.
- Chunked vectors use a verified local cache; the stored artifact size and validation cost are recorded in JSON. One validation split cannot establish general improvement.
- Popularity counts, means, and ordering are fitted only on Phase 14 training ratings. Exposure bias cannot be distinguished from broad appeal in this dataset.
- No Phase 13 test outcomes were used to choose the experimental fusion or content diagnostic variants. Production defaults remain unchanged.

## Reproducibility

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_optimization.py`

The Phase 13 test report and manifest are preserved. Phase 14 writes only its own report and validation manifest.
