# CineMatch Phase 13 — Leakage-Safe Evaluation

## Split and leakage controls

Per-user temporal holdout: final 20% by timestamp, with MovieLens `movie_id` as deterministic tie-break. The minimum training history was 5 ratings. This yielded 610 users, 80,419 training interactions, and 20,417 test interactions. 130 users had an equal-timestamp tie at the boundary; their within-second order is unknowable.

The split manifest records keys, timestamps, order, and partition, not rating values. Phase 10, Phase 11, and Phase 12 receive training rows only. Movie content metadata is joined by `movie_id`; no row-position joins are used. Test ratings are used only after recommendation generation to calculate outcomes.

## Rating prediction

Phase 10 exposes ranked predicted ratings for candidates with sufficient training-only item-similarity evidence. Content and rank-fusion outputs are not rating predictions and are marked unsupported for this task.

| Model | N predictions | Coverage | MAE | RMSE |
|---|---:|---:|---:|---:|
| Global mean (training only) | 20,417 | 100.0% | 0.8367 | 1.0684 |
| User mean (training only) | 20,417 | 100.0% | 0.7495 | 0.9661 |
| Phase 10 collaborative | 16,692 | 81.8% | 0.6793 | 0.8923 |

Paired CF minus user-mean comparison on 16,692 common predictions: MAE difference -0.0689, user-bootstrap 95% interval [-0.07792714893096278, -0.060226172907754456]; RMSE difference -0.0715, interval [-0.08170437361078933, -0.06142798404654595]. Negative favors CF. This does not measure plot relevance.

## Personalized top-K ranking

Relevance is a held-out rating ≥ 4.0, selected before examining results. Candidates are the full movie catalog minus each user's training-rated movies. This is a positive-unlabeled benchmark: missing ratings are unknown, not explicit dislikes; unobserved candidates can still be counted as non-hits by the conventional precision denominator.

| Model | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | Users with recs / eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_popularity | 0.0692 | 0.0307 | 0.0799 | 0.0562 | 0.0488 | 0.0754 | 610/610 |
| phase5_metadata | 0.0108 | 0.0057 | 0.0112 | 0.0089 | 0.0081 | 0.0109 | 609/610 |
| phase6_tfidf | 0.0043 | 0.0021 | 0.0053 | 0.0030 | 0.0025 | 0.0045 | 436/610 |
| phase7_transformer | 0.0030 | 0.0016 | 0.0033 | 0.0026 | 0.0020 | 0.0033 | 436/610 |
| phase9_content_hybrid | 0.0269 | 0.0129 | 0.0312 | 0.0205 | 0.0175 | 0.0279 | 609/610 |
| phase10_collaborative | 0.0075 | 0.0035 | 0.0071 | 0.0069 | 0.0070 | 0.0080 | 609/610 |
| phase12_unified_hybrid | 0.0200 | 0.0085 | 0.0240 | 0.0172 | 0.0164 | 0.0232 | 609/610 |

`Precision@K` always divides by K (short lists leave unfilled slots as misses). Recall and NDCG exclude users with no relevant held-out movie and report that exclusion count. User-bootstrap intervals resample users, not individual ratings.

## History groups

| Training-history group | Users | Phase 12 recommendation coverage | Phase 12 P@10 | R@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| sparse_5_to_20 | 85 | 98.8% | 0.0071 | 0.0165 | 0.0165 |
| moderate_21_to_50 | 199 | 100.0% | 0.0116 | 0.0240 | 0.0172 |
| substantial_over_50 | 326 | 100.0% | 0.0233 | 0.0120 | 0.0284 |

## Qualitative relevance audit

These fixed cases were selected without looking at held-out ratings. The table shows the first three results for the TF-IDF, Transformer, Phase 9, and Phase 12 favorite routes. Detailed top-five rows include each result's plot excerpt and signal evidence in the JSON report. Missing plot results are reported as unavailable rather than imputed.

### Grumpier Old Men (1995) (movie_id=3; plot present)

| Model | Top results (rank — title; plot availability; score/evidence) |
|---|---|
| phase6_tfidf | 1 — Pi (1998) (plot yes; score=0.296; Max Cohen is the story's protagonist and unreliable narrator. Unemployed and living in a d)<br>2 — Mighty, The (1998) (plot yes; score=0.274; Kevin (Kieran Culkin) is a 13-year-old boy who has Mucopolysaccharidosis IV (or Morquio sy)<br>3 — Bent (1997) (plot yes; score=0.269; Max (Clive Owen) is a promiscuous gay man living in 1930s Berlin. He is at odds with his w) |
| phase7_transformer | 1 — Apartment, The (Appartement, L') (1996) (plot yes; score=0.527; Max (Vincent Cassel) is a former bohemian and an amateur writer who gets a job in New York)<br>2 — Humanoids from the Deep (1980) (plot yes; score=0.525; Anglers from the fishing village of Noyo, California catch what appears to be a monster. T)<br>3 — Hands Off the Loot (Touchez pas au grisbi) (1954) (plot yes; score=0.521; The entire film takes place over the course of three days. Max, a decent and principled ga) |
| phase9_content_hybrid | 1 — Wedding Planner, The (2001) (plot yes; score=0.858; shared genres: comedy, romance; plot word overlap (TF-IDF cosine 0.019); plot semantic match (Transformer cosine 0.394); shared cast member: Kevin Pollak; After planning and coordinating another successful wedding ceremony, San Francisco wedding)<br>2 — Cousins (1989) (plot yes; score=0.848; shared genres: comedy, romance; shared plot tags: revenge; plot word overlap (TF-IDF cosine 0.138); plot semantic match (Transformer cosine 0.425); Larry Kozinski (Danson) and Maria Hardy (Rossellini) meet at the marriage of Larry's uncle)<br>3 — John Tucker Must Die (2006) (plot yes; score=0.847; shared genres: comedy, romance; shared plot tags: revenge; plot word overlap (TF-IDF cosine 0.128); plot semantic match (Transformer cosine 0.422); The movie begins with Kate (Brittany Snow) discussing her mother Lori's (Jenny McCarthy) s) |
| phase12_new_user_favorite_route | 1 — Wedding Planner, The (2001) (plot yes; After planning and coordinating another successful wedding ceremony, San Francisco wedding)<br>2 — Cousins (1989) (plot yes; Larry Kozinski (Danson) and Maria Hardy (Rossellini) meet at the marriage of Larry's uncle)<br>3 — John Tucker Must Die (2006) (plot yes; The movie begins with Kate (Brittany Snow) discussing her mother Lori's (Jenny McCarthy) s) |

### Waiting to Exhale (1995) (movie_id=4; plot present)

| Model | Top results (rank — title; plot availability; score/evidence) |
|---|---|
| phase6_tfidf | 1 — Savannah Smiles (1982) (plot yes; score=0.162; The story begins at a prison work site where Boots "Bootsie" McGaffey (Donovan Scott) atte)<br>2 — Robin Hood: Prince of Thieves (1991) (plot yes; score=0.149; Robin of Locksley – an English nobleman who joined Richard the Lionheart, King of England )<br>3 — Robin and Marian (1976) (plot yes; score=0.142; An aging Robin Hood (Sean Connery) is a trusted captain fighting for King Richard the Lion) |
| phase7_transformer | 1 — How to Deal (2003) (plot yes; score=0.567; Halley Marie Martin (Mandy Moore) is a 17-year-old high school student who is disillusione)<br>2 — Beaches (1988) (plot yes; score=0.513; The story of two friends from different backgrounds, whose friendship spans more than 35 y)<br>3 — Maid to Order (1987) (plot yes; score=0.511; Jessie Montgomery (Ally Sheedy) is a spoiled rich girl in her mid 20's whose hard partying) |
| phase9_content_hybrid | 1 — Hope Floats (1998) (plot yes; score=0.905; shared genres: comedy, drama, romance; shared plot tags: revenge; plot word overlap (TF-IDF cosine 0.025); plot semantic match (Transformer cosine 0.452); shared director: Forest Whitaker; Birdee Pruitt (Sandra Bullock) is a Chicago housewife who is invited onto the Toni Post ta)<br>2 — I Think I Love My Wife (2007) (plot yes; score=0.872; shared genres: comedy, drama, romance; plot word overlap (TF-IDF cosine 0.029); plot semantic match (Transformer cosine 0.444); shared cast member: Wendell Pierce; Richard Cooper (Chris Rock) is a happily married and professionally successful man. He is )<br>3 — Coyote Ugly (2000) (plot yes; score=0.844; shared genres: comedy, drama, romance; plot word overlap (TF-IDF cosine 0.044); plot semantic match (Transformer cosine 0.438); Violet Sanford leaves her hometown of South Amboy, New Jersey, her father Bill, and her be) |
| phase12_new_user_favorite_route | 1 — Hope Floats (1998) (plot yes; Birdee Pruitt (Sandra Bullock) is a Chicago housewife who is invited onto the Toni Post ta)<br>2 — I Think I Love My Wife (2007) (plot yes; Richard Cooper (Chris Rock) is a happily married and professionally successful man. He is )<br>3 — Coyote Ugly (2000) (plot yes; Violet Sanford leaves her hometown of South Amboy, New Jersey, her father Bill, and her be) |

### Father of the Bride Part II (1995) (movie_id=5; plot present)

| Model | Top results (rank — title; plot availability; score/evidence) |
|---|---|
| phase6_tfidf | 1 — Object of My Affection, The (1998) (plot yes; score=0.573; Social worker Nina Borowski (Jennifer Aniston) is a bright young woman living in a cozy Br)<br>2 — How Do You Know (2010) (plot yes; score=0.340; Softball player Lisa Jorgenson is devastated when she is left off the Team USA roster. Uns)<br>3 — Phenomenon (1996) (plot yes; score=0.309; George Malley is a kind but average auto mechanic in a small town in Northern California. ) |
| phase7_transformer | 1 — When a Stranger Calls (1979) (plot yes; score=0.552; On one side of town, a babysitter and the kids she is babysitting are brutally murdered. T)<br>2 — When a Stranger Calls (2006) (plot yes; score=0.552; On one side of town, a babysitter and the kids she is babysitting are brutally murdered. T)<br>3 — House Arrest (1996) (plot yes; score=0.541; The film is told through the perspective of Grover and TJ through the literary device know) |
| phase9_content_hybrid | 1 — First Wives Club, The (1996) (plot yes; score=0.829; shared genres: comedy; plot word overlap (TF-IDF cosine 0.166); plot semantic match (Transformer cosine 0.374); shared cast member: Diane Keaton; At Middlebury College in 1969, four young friends, Annie MacDuggan, Elise Elliot, Brenda M)<br>2 — Bridesmaids (2011) (plot yes; score=0.825; shared genres: comedy; shared plot tags: comedy; plot word overlap (TF-IDF cosine 0.281); plot semantic match (Transformer cosine 0.455); Annie Walker (Kristen Wiig) is a single woman in her late 30s. Following the failure of he)<br>3 — Grown Ups 2 (2013) (plot yes; score=0.824; shared genres: comedy; shared plot tags: comedy; plot word overlap (TF-IDF cosine 0.034); plot semantic match (Transformer cosine 0.474); Three years after the events of the first film, Lenny Feder has relocated his family back ) |
| phase12_new_user_favorite_route | 1 — First Wives Club, The (1996) (plot yes; At Middlebury College in 1969, four young friends, Annie MacDuggan, Elise Elliot, Brenda M)<br>2 — Bridesmaids (2011) (plot yes; Annie Walker (Kristen Wiig) is a single woman in her late 30s. Following the failure of he)<br>3 — Grown Ups 2 (2013) (plot yes; Three years after the events of the first film, Lenny Feder has relocated his family back ) |

### Toy Story (1995) (movie_id=1; plot missing)

| Model | Top results (rank — title; plot availability; score/evidence) |
|---|---|
| phase6_tfidf | Unavailable: Movie 1 has no usable plot for TF-IDF comparison |
| phase7_transformer | Unavailable: Movie 1 has no usable plot for semantic comparison |
| phase9_content_hybrid | 1 — Toy Story 2 (1999) (plot no; score=0.999; shared genres: adventure, animation, children, comedy, fantasy; shared director: John Lasseter; shared writer: Andrew Stanton; shared selected crew: Lee Unkrich, Randy Newman; )<br>2 — Bug's Life, A (1998) (plot no; score=0.996; shared genres: adventure, animation, children, comedy; shared director: John Lasseter; shared writer: Andrew Stanton; shared selected crew: Lee Unkrich, Randy Newman; )<br>3 — Cars (2006) (plot no; score=0.987; shared genres: animation, children, comedy; shared director: John Lasseter; shared writer: Joe Ranft, John Lasseter; shared selected crew: Randy Newman; ) |
| phase12_new_user_favorite_route | 1 — Toy Story 2 (1999) (plot no; )<br>2 — Bug's Life, A (1998) (plot no; )<br>3 — Cars (2006) (plot no; ) |

### Godfather, The (1972) (movie_id=858; plot missing)

| Model | Top results (rank — title; plot availability; score/evidence) |
|---|---|
| phase6_tfidf | Unavailable: Movie 858 has no usable plot for TF-IDF comparison |
| phase7_transformer | Unavailable: Movie 858 has no usable plot for semantic comparison |
| phase9_content_hybrid | 1 — Godfather: Part II, The (1974) (plot no; score=0.996; shared genres: crime, drama; shared director: Francis Ford Coppola; shared writer: Francis Ford Coppola, Mario Puzo; shared cast member: Abe Vigoda, Al Martino, Al Pacino; shared selected crew: Gordon Willis, Nino Rota, Peter Zinner; )<br>2 — Godfather: Part III, The (1990) (plot no; score=0.942; shared genres: crime, drama; shared director: Francis Ford Coppola; shared writer: Francis Ford Coppola, Mario Puzo; shared cast member: Al Martino, Al Pacino, Carmine Coppola; shared selected crew: Gordon Willis, Nino Rota; )<br>3 — Rainmaker, The (1997) (plot yes; score=0.851; shared genres: drama; shared director: Francis Ford Coppola; shared writer: Francis Ford Coppola; Rudy Baylor (Matt Damon) was brought up in a slum. His father, who was a severe alcoholic,) |
| phase12_new_user_favorite_route | 1 — Godfather: Part II, The (1974) (plot no; )<br>2 — Godfather: Part III, The (1990) (plot no; )<br>3 — Rainmaker, The (1997) (plot yes; Rudy Baylor (Matt Damon) was brought up in a slum. His father, who was a severe alcoholic,) |

### McHale's Navy (1997) (movie_id=1445; plot present)

| Model | Top results (rank — title; plot availability; score/evidence) |
|---|---|
| phase6_tfidf | 1 — Down Periscope (1996) (plot yes; score=0.081; Lt. Commander Thomas Dodge (Kelsey Grammer) is about to be passed over a third time for hi)<br>2 — A-Team, The (2010) (plot yes; score=0.080; The A-Team is a naturally episodic show, with few overarching stories, except the characte)<br>3 — Thin Red Line, The (1998) (plot yes; score=0.075; United States Army Private Witt goes AWOL from his unit and lives among the carefree Melan) |
| phase7_transformer | 1 — Thin Red Line, The (1998) (plot yes; score=0.433; United States Army Private Witt goes AWOL from his unit and lives among the carefree Melan)<br>2 — Six Days Seven Nights (1998) (plot yes; score=0.425; Robin Monroe (Anne Heche) is a New York journalist who works for Dazzle, a fashion magazin)<br>3 — Teahouse of the August Moon, The (1956) (plot yes; score=0.420; Misfit Captain Fisby (Glenn Ford) is sent to Americanize the village of Tobiki on Okinawa,) |
| phase9_content_hybrid | 1 — Russians Are Coming, the Russians Are Coming, The (1966) (plot yes; score=0.884; shared genres: comedy, war; plot word overlap (TF-IDF cosine 0.051); plot semantic match (Transformer cosine 0.384); A Russian submarine called Спрут ("Octopus") draws too close to the New England coast one )<br>2 — Guy X (2005) (plot yes; score=0.878; shared genres: comedy, war; plot word overlap (TF-IDF cosine 0.025); plot semantic match (Transformer cosine 0.417); It is 1979, four years after the end of US involvement in the Vietnam War. US Army Corpora)<br>3 — Catch-22 (1970) (plot yes; score=0.877; shared genres: comedy, war; plot word overlap (TF-IDF cosine 0.030); plot semantic match (Transformer cosine 0.347); The plotline follows the airmen of the 256th Squadron while in action over Italy, and thei) |
| phase12_new_user_favorite_route | 1 — Russians Are Coming, the Russians Are Coming, The (1966) (plot yes; A Russian submarine called Спрут ("Octopus") draws too close to the New England coast one )<br>2 — Guy X (2005) (plot yes; It is 1979, four years after the end of US involvement in the Vietnam War. US Army Corpora)<br>3 — Catch-22 (1970) (plot yes; The plotline follows the airmen of the 256th Squadron while in action over Italy, and thei) |

**Name sensitivity:** targeted names accounted for these shares of the Phase 6 top-five cosine scores:
- Query 3: 98%, 89%, 99%, 94%, 98%; top-five overlap after the targeted-name ablation was 0/5.
- Query 4: 92%, 89%, 85%, 88%, 90%; top-five overlap after the targeted-name ablation was 0/5.
- Query 5: 0%, 0%, 0%, 0%, 0%; top-five overlap after the targeted-name ablation was 5/5.

**Long-plot experiment:** One deterministic comparison is diagnostic only. Chunked embeddings remain an experiment; the Phase 7 model and default remain unchanged.

**Observed Phase 12 vs Phase 9:** P@10 difference -0.0033, R@10 difference -0.0011, and NDCG@10 difference -0.0047. NDCG user-bootstrap 95% interval: [-0.007650810346503333, -0.0017846776908289831]. This is held-out rating relevance, not a semantic plot-similarity judgment.

## Limitations and interpretation

- Historical ratings are sparse and reflect what users chose to rate; ranking metrics do not establish story-level similarity.
- The global popularity baseline may exploit dataset exposure patterns; MovieLens observations are not a randomized preference sample.
- Equal timestamps occur frequently. Movie ID makes the holdout deterministic but cannot reconstruct order within that timestamp.
- Existing content recommenders return ranked lists; evaluation uses their APIs and removes training-seen movies. Phase 12's retrieval and fallback behavior remains unchanged.
- Rating prediction is only defined for Phase 10's supported candidate scores. Unavailable predictions are reported as uncovered, not silently imputed.
- The Phase 9 weights, Phase 10 algorithm, Phase 11 routing, Phase 12 fusion, and Phase 7 embedding default were not tuned or modified.
- The user-bootstrap intervals describe variation across these users, not uncertainty from a new population or a new data collection process.

## Run and artifacts

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/evaluate_recommenders.py
```

The JSON report contains full metrics, per-group coverage, source/fallback behavior, detailed audit rows, missing-data counts, and baseline-preservation checks. The split manifest contains interaction keys and partition labels only.
