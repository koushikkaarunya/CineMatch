# CineMatch Phase 14.1 — Story-Level Relevance Audit

> Label source: `user_approved_ai_assisted`. These are AI-proposed labels approved by the user, not independent human judgments. This selected-pair pilot is descriptive, not a general accuracy estimate.

## Audit dataset

- Pairs: 25; labelable with both plots: 22; missing-plot exclusions: 3
- AI-proposed label counts: {'0': 11, '1': 9, '2': 2}; user-approved AI-assisted labels: {'0': 11, '1': 9, '2': 2}
- Queries: 5; plots span 14 distinct catalog genres.
- The CSV records `approved_relevance` with `label_provenance=user_approved_ai_assisted`; three missing-plot rows remain unjudgeable.

## Pair ranking results

Metrics use grade 2 as strict relevance for pairwise accuracy against grade 0. NDCG uses grades 0–2. Models are scored on the same judged rows when required plot inputs exist. The top-five counts apply only to the deliberately selected pair candidates for each query, not the full catalog recommendation lists.

| Model | Pairs | Queries | Pairwise accuracy (2 vs 0) | NDCG@5 | Unrelated in top 5 | Plausible pairs below top 5 |
|---|---:|---:|---:|---:|---:|---:|
| phase6_tfidf | 22 | 5 | 0.200 | 0.669 | 11 | 3 |
| phase7_transformer | 22 | 5 | 1.000 | 0.822 | 9 | 1 |
| phase9_content | 22 | 5 | 1.000 | 0.882 | 8 | 0 |
| chunked_transformer | 22 | 5 | 0.800 | 0.842 | 8 | 0 |

## Experimental Phase 9 shortlist reranking

The reranker first retrieves up to 100 Phase 9 candidates, then reorders only those candidates by the cached chunked Transformer cosine. Original Phase 9 results remain unchanged.
Candidate-pair availability: 12/22 labeled pairs were present in the Phase 9 shortlist; this conditional result does not measure retrieval misses.
On those same retrieved audit pairs, Phase 9 NDCG@5=0.878; chunked rerank NDCG@5=0.890. Pairwise accuracy is not computable because no query group has both a grade-2 pair and a grade-0 pair after retrieval.
Among the selected candidate rows only, Phase 9 placed 2 grade-0 pairs in the top five; chunked reranking placed 2. These are not counts from the full recommendation catalog.
Scoring/ranking time: 6.121s; peak process RSS: 695.2 MiB; cache size: 4.66 MiB; cache reused: True.
Pinned-tokenizer check: 3400/3434 usable plots exceed the 256-wordpiece model limit.

| Query | Plot / rerank | Phase 9 original top 5 | Chunked reranked top 5 |
|---|---|---|---|
| Toy Story (1995) | missing; Phase 9 metadata list retained | Toy Story 2 (1999); Bug's Life, A (1998); Cars (2006); Luxo Jr. (1986); Monsters, Inc. (2001) | Toy Story 2 (1999); Bug's Life, A (1998); Cars (2006); Luxo Jr. (1986); Monsters, Inc. (2001) |
| Grumpier Old Men (1995) | available; rerank applied | Wedding Planner, The (2001); Cousins (1989); John Tucker Must Die (2006); Good Luck Chuck (2007); What Happens in Vegas... (2008) | My Giant (1998); Money Pit, The (1986); Say It Isn't So (2001); Blame It on Rio (1984); John Tucker Must Die (2006) |
| Waiting to Exhale (1995) | available; rerank applied | Hope Floats (1998); I Think I Love My Wife (2007); Coyote Ugly (2000); Hannah and Her Sisters (1986); Manhattan (1979) | Hope Floats (1998); How to Deal (2003); Sex and the City (2008); Now, Voyager (1942); In the Land of Women (2007) |
| Father of the Bride Part II (1995) | available; rerank applied | First Wives Club, The (1996); Bridesmaids (2011); Grown Ups 2 (2013); Brady Bunch Movie, The (1995); Friends with Kids (2011) | How Do You Know (2010); Object of My Affection, The (1998); Send Me No Flowers (1964); Georgy Girl (1966); Flirting With Disaster (1996) |
| McHale's Navy (1997) | available; rerank applied | Russians Are Coming, the Russians Are Coming, The (1966); Guy X (2005); Catch-22 (1970); Stripes (1981); Objective, Burma! (1945) | Thin Red Line, The (1998); Stripes (1981); Uncommon Valor (1983); Russians Are Coming, the Russians Are Coming, The (1966); Police Academy 2: Their First Assignment (1985) |

## Findings and limitations

- The prior errors are heterogeneous: the Phase 13 audit shows Max overlap for Grumpier Old Men → Pi and Robin/Savannah name collisions for Waiting to Exhale; genres, generic relationship themes, the Phase 9 rank-fusion design, truncation, and source-plot quality are separate plausible contributors.
- The Father of the Bride Part II → When a Stranger Calls results are unrelated by the supplied plot texts. The two When a Stranger Calls records have identical plot text, so their mutual remake similarity does not validate their match to the query.
- Phase 14 found name masking unchanged versus Phase 9 on rating metrics; that is not evidence about story relevance and is not repeated as a proposed solution here.
- The Phase 7 model is capped at 256 wordpieces. The pinned Phase 14 chunk cache records 3,400 long plots; truncation is a credible coverage limitation, but chunk averaging can erase order and no semantic ground truth yet shows it fixes relevance.
- The pair set is small and selected to probe known cases. Labels used for the approved-label run were AI-generated suggestions reviewed and approved by the user; there are no independent human judgments. Metrics are descriptive and should not be generalized.
- No suitable pairwise model was already cached. An MS MARCO cross-encoder is designed for query-to-passage ranking and an STS cross-encoder for sentence similarity, not full plot-to-plot story relevance; no model or dependency was downloaded.
- No production weights, recommendation code, Phase 13/14 reports, or evaluation manifests were changed. No Phase 13 test partition data was read for this experiment.

## Reproduction

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_1_story_relevance.py
```

Evaluate the user-approved AI-assisted labels:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_1_story_relevance.py --label-source user_approved_ai_assisted
```
