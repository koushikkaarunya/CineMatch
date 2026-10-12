# CineMatch Phase 14.2 — Targeted Story-Relevance Experiments

> All labels are AI-assisted suggestions from the supplied plots, not independent human judgments. Metrics describe selected audit candidates and do not establish general performance.

## Audit data and split

- Pairs: 80; usable provisional labels: 73; flagged for review: 7; missing plots: 0
- Development: 6 query groups / 45 pairs; locked audit: 4 query groups / 28 pairs.
- Label counts among usable pairs: {'0': 17, '1': 41, '2': 15}. Provenance: `ai_assisted_suggestion; uncertain pairs are explicitly flagged ai_assisted_needs_review`.
- Groups are kept intact. The locked subset was not used to choose or tune experiment parameters. Its examples were selected as a targeted diagnostic set, not sampled randomly.
- Metrics rank only explicitly labeled candidates in this file. Review-flagged rows and candidates outside the file are unjudged, not negatives; top-five counts are within this selected labeled-candidate pool.

## Methods

- **Phase 9 baseline:** unchanged production hybrid ranking and default weights.
- **Plot-only checks:** unchanged TF-IDF and Phase 7 Transformer plot similarities.
- **Name-discounted TF-IDF:** subtracts 0.80 of the shared TF-IDF contribution from heuristic likely-person-name unigrams only. The heuristic can mistake places or other proper nouns for people.
- **Story-gated Phase 9:** multiplies the unchanged Phase 9 score by `0.20 + (1 - 0.20) × max(TF-IDF rank percentile, chunked-plot rank percentile)`. This is a fixed diagnostic rule, not a production weight change.
- **Chunk aggregation:** compares the existing normalized mean-document chunk vector with maximum chunk-pair cosine and the mean of the strongest two chunk-pair cosines. Maximum matching may reward one incidental segment; top-two may still ignore narrative order.

## Development results

| Method | Labeled pairs | Groups | Pairwise accuracy (2 vs 0) | NDCG@5 | Unrelated in audited top 5 | Plausible below top 5 |
|---|---:|---:|---:|---:|---:|---:|
| phase9_baseline | 45 | 6 | 0.938 | 0.849 | 2 | 5 |
| tfidf_plot_only | 45 | 6 | 0.125 | 0.568 | 9 | 12 |
| transformer_plot_only | 45 | 6 | 0.562 | 0.712 | 6 | 9 |
| name_discounted_tfidf | 45 | 6 | 0.188 | 0.618 | 8 | 11 |
| phase9_story_gate | 45 | 6 | 0.938 | 0.844 | 2 | 5 |
| chunked_mean | 45 | 6 | 0.625 | 0.664 | 6 | 9 |
| chunked_max_pair | 45 | 6 | 0.688 | 0.732 | 3 | 6 |
| chunked_top2_mean | 45 | 6 | 0.625 | 0.719 | 3 | 6 |

## Locked audit results

| Method | Labeled pairs | Groups | Pairwise accuracy (2 vs 0) | NDCG@5 | Unrelated in audited top 5 | Plausible below top 5 |
|---|---:|---:|---:|---:|---:|---:|
| phase9_baseline | 28 | 4 | 1.000 | 0.812 | 1 | 4 |
| tfidf_plot_only | 28 | 4 | 0.600 | 0.732 | 3 | 6 |
| transformer_plot_only | 28 | 4 | 1.000 | 0.852 | 3 | 6 |
| name_discounted_tfidf | 28 | 4 | 0.600 | 0.774 | 3 | 6 |
| phase9_story_gate | 28 | 4 | 1.000 | 0.837 | 1 | 4 |
| chunked_mean | 28 | 4 | 1.000 | 0.809 | 3 | 6 |
| chunked_max_pair | 28 | 4 | 1.000 | 0.827 | 4 | 7 |
| chunked_top2_mean | 28 | 4 | 1.000 | 0.827 | 4 | 7 |

## Interpretation

- On development groups, name-discounted TF-IDF changed NDCG@5 from 0.568 to 0.618 and unrelated audited top-five entries from 9 to 8; it still ranked unrelated candidates highly, so the heuristic did not remove the failure mode.
- The fixed story gate changed development NDCG@5 from 0.849 to 0.844; on the locked groups it changed from 0.812 to 0.837. This small, selected audit does not establish a robust gate benefit.
- Maximum chunk-pair cosine versus normalized-mean chunk scoring changed development NDCG@5 from 0.664 to 0.732 and locked NDCG@5 from 0.809 to 0.827; locked unrelated audited top-five entries rose from 3 to 4.
- The dataset is a selected diagnostic set assembled from known model outputs and plot comparisons; neither its metrics nor its locked groups estimate broad catalog performance.
- Uncertain pairs are excluded, and unjudged candidates are not treated as irrelevant. Metrics rank only the explicitly labeled pairs inside this audit file.
- Name attenuation changes only likely shared name-token contributions. Capitalization is an imperfect proxy; it may suppress meaningful people, places, or sequel continuity. Keep name diagnostics as an experiment, not a production text cleaner.
- The story gate leaves Phase 9 weights untouched but adds a fixed plot-support factor. It can demote a genuinely related story when both plot models rank it poorly, and the factor has not been tuned on locked groups.
- Maximum chunk matching may be fooled by one incidental passage; top-two aggregation reduces single-window domination but still loses chronology. Chunk scores do not prove semantic relevance.
- The Phase 9 reranker comparison is conditional on its candidate retrieval and the selected labels. No approach is declared a winner from these limited examples.
- No Phase 13 test partition, ratings, Phase 13/14 reports, Phase 14.1 files, production recommendation logic, or default weights were used or changed.

## Per-query top-five audit examples

### development: Grumpier Old Men (1995) (movie_id=3)
- **phase9_baseline:** Wedding Planner, The (2001) [grade 0]; Cousins (1989) [grade 1]; Good Luck Chuck (2007) [grade 1]; Wedding Crashers (2005) [grade 1]; Notting Hill (1999) [grade 1]
- **tfidf_plot_only:** Pi (1998) [grade 0]; My Giant (1998) [grade 0]; Cousins (1989) [grade 1]; Wedding Crashers (2005) [grade 1]; Notting Hill (1999) [grade 1]
- **transformer_plot_only:** Good Luck Chuck (2007) [grade 1]; Cousins (1989) [grade 1]; Wedding Planner, The (2001) [grade 0]; Pi (1998) [grade 0]; Notting Hill (1999) [grade 1]
- **name_discounted_tfidf:** Pi (1998) [grade 0]; My Giant (1998) [grade 0]; Notting Hill (1999) [grade 1]; Good Luck Chuck (2007) [grade 1]; Cousins (1989) [grade 1]
- **phase9_story_gate:** Wedding Planner, The (2001) [grade 0]; Cousins (1989) [grade 1]; Wedding Crashers (2005) [grade 1]; Good Luck Chuck (2007) [grade 1]; Notting Hill (1999) [grade 1]
- **chunked_mean:** My Giant (1998) [grade 0]; Cousins (1989) [grade 1]; Wedding Planner, The (2001) [grade 0]; Wedding Crashers (2005) [grade 1]; Notting Hill (1999) [grade 1]
- **chunked_max_pair:** My Giant (1998) [grade 0]; Notting Hill (1999) [grade 1]; Wedding Crashers (2005) [grade 1]; Wedding Planner, The (2001) [grade 0]; Cousins (1989) [grade 1]
- **chunked_top2_mean:** My Giant (1998) [grade 0]; Wedding Planner, The (2001) [grade 0]; Notting Hill (1999) [grade 1]; Wedding Crashers (2005) [grade 1]; Cousins (1989) [grade 1]

### development: Waiting to Exhale (1995) (movie_id=4)
- **phase9_baseline:** Hope Floats (1998) [grade 1]; Coyote Ugly (2000) [grade 1]; Boys on the Side (1995) [grade 2]; Beaches (1988) [grade 2]; Joy Luck Club, The (1993) [grade 1]
- **tfidf_plot_only:** Savannah Smiles (1982) [grade 0]; Robin Hood: Prince of Thieves (1991) [grade 0]; Robin and Marian (1976) [grade 0]; Boys on the Side (1995) [grade 2]; Coyote Ugly (2000) [grade 1]
- **transformer_plot_only:** Beaches (1988) [grade 2]; Savannah Smiles (1982) [grade 0]; Hope Floats (1998) [grade 1]; Coyote Ugly (2000) [grade 1]; Boys on the Side (1995) [grade 2]
- **name_discounted_tfidf:** Savannah Smiles (1982) [grade 0]; Coyote Ugly (2000) [grade 1]; Robin and Marian (1976) [grade 0]; Robin Hood: Prince of Thieves (1991) [grade 0]; Beaches (1988) [grade 2]
- **phase9_story_gate:** Hope Floats (1998) [grade 1]; Coyote Ugly (2000) [grade 1]; Boys on the Side (1995) [grade 2]; Beaches (1988) [grade 2]; Joy Luck Club, The (1993) [grade 1]
- **chunked_mean:** Hope Floats (1998) [grade 1]; Savannah Smiles (1982) [grade 0]; Beaches (1988) [grade 2]; Coyote Ugly (2000) [grade 1]; Joy Luck Club, The (1993) [grade 1]
- **chunked_max_pair:** Hope Floats (1998) [grade 1]; Savannah Smiles (1982) [grade 0]; Joy Luck Club, The (1993) [grade 1]; Beaches (1988) [grade 2]; Coyote Ugly (2000) [grade 1]
- **chunked_top2_mean:** Hope Floats (1998) [grade 1]; Savannah Smiles (1982) [grade 0]; Beaches (1988) [grade 2]; Coyote Ugly (2000) [grade 1]; Joy Luck Club, The (1993) [grade 1]

### development: Father of the Bride Part II (1995) (movie_id=5)
- **phase9_baseline:** First Wives Club, The (1996) [grade 1]; Bridesmaids (2011) [grade 1]; Friends with Kids (2011) [grade 1]; Object of My Affection, The (1998) [grade 1]; House Arrest (1996) [grade 1]
- **tfidf_plot_only:** Object of My Affection, The (1998) [grade 1]; Bridesmaids (2011) [grade 1]; First Wives Club, The (1996) [grade 1]; House Arrest (1996) [grade 1]; Friends with Kids (2011) [grade 1]
- **transformer_plot_only:** When a Stranger Calls (1979) [grade 0]; When a Stranger Calls (2006) [grade 0]; House Arrest (1996) [grade 1]; Object of My Affection, The (1998) [grade 1]; Friends with Kids (2011) [grade 1]
- **name_discounted_tfidf:** Object of My Affection, The (1998) [grade 1]; Bridesmaids (2011) [grade 1]; First Wives Club, The (1996) [grade 1]; House Arrest (1996) [grade 1]; Friends with Kids (2011) [grade 1]
- **phase9_story_gate:** Bridesmaids (2011) [grade 1]; First Wives Club, The (1996) [grade 1]; Friends with Kids (2011) [grade 1]; Object of My Affection, The (1998) [grade 1]; House Arrest (1996) [grade 1]
- **chunked_mean:** Object of My Affection, The (1998) [grade 1]; House Arrest (1996) [grade 1]; Friends with Kids (2011) [grade 1]; When a Stranger Calls (2006) [grade 0]; When a Stranger Calls (1979) [grade 0]
- **chunked_max_pair:** Object of My Affection, The (1998) [grade 1]; House Arrest (1996) [grade 1]; First Wives Club, The (1996) [grade 1]; Bridesmaids (2011) [grade 1]; Friends with Kids (2011) [grade 1]
- **chunked_top2_mean:** Object of My Affection, The (1998) [grade 1]; House Arrest (1996) [grade 1]; First Wives Club, The (1996) [grade 1]; Friends with Kids (2011) [grade 1]; Bridesmaids (2011) [grade 1]

### locked_audit: Balto (1995) (movie_id=13)
- **phase9_baseline:** American Tail: Fievel Goes West, An (1991) [grade 1]; Journey of Natty Gann, The (1985) [grade 1]; Brother Bear (2003) [grade 1]; Snow Walker, The (2003) [grade 2]; Babe (1995) [grade 1]
- **tfidf_plot_only:** 13 Going on 30 (2004) [grade 0]; Spy Hard (1996) [grade 0]; Brother Bear (2003) [grade 1]; Journey of Natty Gann, The (1985) [grade 1]; American Tail: Fievel Goes West, An (1991) [grade 1]
- **transformer_plot_only:** Brother Bear (2003) [grade 1]; Snow Walker, The (2003) [grade 2]; Babe (1995) [grade 1]; Journey of Natty Gann, The (1985) [grade 1]; American Tail: Fievel Goes West, An (1991) [grade 1]
- **name_discounted_tfidf:** 13 Going on 30 (2004) [grade 0]; Spy Hard (1996) [grade 0]; Brother Bear (2003) [grade 1]; Journey of Natty Gann, The (1985) [grade 1]; American Tail: Fievel Goes West, An (1991) [grade 1]
- **phase9_story_gate:** Journey of Natty Gann, The (1985) [grade 1]; Brother Bear (2003) [grade 1]; American Tail: Fievel Goes West, An (1991) [grade 1]; Snow Walker, The (2003) [grade 2]; Babe (1995) [grade 1]
- **chunked_mean:** Journey of Natty Gann, The (1985) [grade 1]; Snow Walker, The (2003) [grade 2]; Babe (1995) [grade 1]; Brother Bear (2003) [grade 1]; American Tail: Fievel Goes West, An (1991) [grade 1]
- **chunked_max_pair:** Snow Walker, The (2003) [grade 2]; Journey of Natty Gann, The (1985) [grade 1]; Brother Bear (2003) [grade 1]; Babe (1995) [grade 1]; 13 Going on 30 (2004) [grade 0]
- **chunked_top2_mean:** Snow Walker, The (2003) [grade 2]; Journey of Natty Gann, The (1985) [grade 1]; Brother Bear (2003) [grade 1]; Babe (1995) [grade 1]; 13 Going on 30 (2004) [grade 0]

### development: Money Train (1995) (movie_id=20)
- **phase9_baseline:** Ice Harvest, The (2005) [grade 1]; Taking of Pelham 1 2 3, The (2009) [grade 2]; Hard Rain (1998) [grade 2]; Mean Streets (1973) [grade 1]; Suicide Kings (1997) [grade 1]
- **tfidf_plot_only:** Ice Harvest, The (2005) [grade 1]; Suicide Kings (1997) [grade 1]; Me, Myself & Irene (2000) [grade 0]; Mean Streets (1973) [grade 1]; Hard Rain (1998) [grade 2]
- **transformer_plot_only:** Suicide Kings (1997) [grade 1]; Mean Streets (1973) [grade 1]; Trapped in Paradise (1994) [grade 2]; Ice Harvest, The (2005) [grade 1]; Taking of Pelham 1 2 3, The (2009) [grade 2]
- **name_discounted_tfidf:** Ice Harvest, The (2005) [grade 1]; Suicide Kings (1997) [grade 1]; Me, Myself & Irene (2000) [grade 0]; Mean Streets (1973) [grade 1]; Hard Rain (1998) [grade 2]
- **phase9_story_gate:** Ice Harvest, The (2005) [grade 1]; Taking of Pelham 1 2 3, The (2009) [grade 2]; Hard Rain (1998) [grade 2]; Mean Streets (1973) [grade 1]; Suicide Kings (1997) [grade 1]
- **chunked_mean:** Suicide Kings (1997) [grade 1]; Ice Harvest, The (2005) [grade 1]; Trapped in Paradise (1994) [grade 2]; Mean Streets (1973) [grade 1]; Taking of Pelham 1 2 3, The (2009) [grade 2]
- **chunked_max_pair:** Suicide Kings (1997) [grade 1]; Ice Harvest, The (2005) [grade 1]; Trapped in Paradise (1994) [grade 2]; Mean Streets (1973) [grade 1]; Hard Rain (1998) [grade 2]
- **chunked_top2_mean:** Suicide Kings (1997) [grade 1]; Ice Harvest, The (2005) [grade 1]; Mean Streets (1973) [grade 1]; Hard Rain (1998) [grade 2]; Trapped in Paradise (1994) [grade 2]

### development: Copycat (1995) (movie_id=22)
- **phase9_baseline:** Sea of Love (1989) [grade 2]; From Hell (2001) [grade 2]; Identity (2003) [grade 2]; Mindhunters (2004) [grade 2]; Born to Kill (1947) [grade 1]
- **tfidf_plot_only:** Raising Helen (2004) [grade 0]; Born to Kill (1947) [grade 1]; Candyman: Farewell to the Flesh (1995) [grade 1]; Sea of Love (1989) [grade 2]; Peeping Tom (1960) [grade 2]
- **transformer_plot_only:** Born to Kill (1947) [grade 1]; Raising Helen (2004) [grade 0]; From Hell (2001) [grade 2]; Sea of Love (1989) [grade 2]; Identity (2003) [grade 2]
- **name_discounted_tfidf:** Peeping Tom (1960) [grade 2]; Raising Helen (2004) [grade 0]; Born to Kill (1947) [grade 1]; Sea of Love (1989) [grade 2]; Candyman: Farewell to the Flesh (1995) [grade 1]
- **phase9_story_gate:** Sea of Love (1989) [grade 2]; Mindhunters (2004) [grade 2]; Born to Kill (1947) [grade 1]; Identity (2003) [grade 2]; From Hell (2001) [grade 2]
- **chunked_mean:** Born to Kill (1947) [grade 1]; Sea of Love (1989) [grade 2]; Raising Helen (2004) [grade 0]; Candyman: Farewell to the Flesh (1995) [grade 1]; Mindhunters (2004) [grade 2]
- **chunked_max_pair:** Candyman: Farewell to the Flesh (1995) [grade 1]; Born to Kill (1947) [grade 1]; Sea of Love (1989) [grade 2]; Identity (2003) [grade 2]; Mindhunters (2004) [grade 2]
- **chunked_top2_mean:** Candyman: Farewell to the Flesh (1995) [grade 1]; Born to Kill (1947) [grade 1]; Sea of Love (1989) [grade 2]; Peeping Tom (1960) [grade 2]; Identity (2003) [grade 2]

### locked_audit: City of Lost Children, The (Cité des enfants perdus, La) (1995) (movie_id=29)
- **phase9_baseline:** Delicatessen (1991) [grade 1]; Starcrash (a.k.a. Star Crash) (1978) [grade 0]; Dernier Combat, Le (Last Battle, The) (1983) [grade 1]; Postman, The (1997) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]
- **tfidf_plot_only:** Postman, The (1997) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]; Metropolis (1927) [grade 1]; Delicatessen (1991) [grade 1]; Ink (2009) [grade 1]
- **transformer_plot_only:** Dernier Combat, Le (Last Battle, The) (1983) [grade 1]; Starcrash (a.k.a. Star Crash) (1978) [grade 0]; Ink (2009) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]; Postman, The (1997) [grade 1]
- **name_discounted_tfidf:** Postman, The (1997) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]; Metropolis (1927) [grade 1]; Dernier Combat, Le (Last Battle, The) (1983) [grade 1]; Ink (2009) [grade 1]
- **phase9_story_gate:** Delicatessen (1991) [grade 1]; Starcrash (a.k.a. Star Crash) (1978) [grade 0]; Postman, The (1997) [grade 1]; Ink (2009) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]
- **chunked_mean:** Starcrash (a.k.a. Star Crash) (1978) [grade 0]; Ink (2009) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]; Metropolis (1927) [grade 1]; Dernier Combat, Le (Last Battle, The) (1983) [grade 1]
- **chunked_max_pair:** Ink (2009) [grade 1]; Starcrash (a.k.a. Star Crash) (1978) [grade 0]; Dernier Combat, Le (Last Battle, The) (1983) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]; Metropolis (1927) [grade 1]
- **chunked_top2_mean:** Ink (2009) [grade 1]; Starcrash (a.k.a. Star Crash) (1978) [grade 0]; Dernier Combat, Le (Last Battle, The) (1983) [grade 1]; Lemony Snicket's A Series of Unfortunate Events (2004) [grade 1]; Metropolis (1927) [grade 1]

### locked_audit: Richard III (1995) (movie_id=41)
- **phase9_baseline:** Henry V (1989) [grade 1]; Andrei Rublev (Andrey Rublyov) (1969) [grade 1]; Libertine, The (2004) [grade 1]; Elizabeth (1998) [grade 1]; Breaker Morant (1980) [grade 1]
- **tfidf_plot_only:** Adventures of Robin Hood, The (1938) [grade 0]; Man for All Seasons, A (1966) [grade 1]; Elizabeth (1998) [grade 1]; Henry V (1989) [grade 1]; Libertine, The (2004) [grade 1]
- **transformer_plot_only:** Man for All Seasons, A (1966) [grade 1]; Libertine, The (2004) [grade 1]; Adventures of Robin Hood, The (1938) [grade 0]; Henry V (1989) [grade 1]; Andrei Rublev (Andrey Rublyov) (1969) [grade 1]
- **name_discounted_tfidf:** Henry V (1989) [grade 1]; Man for All Seasons, A (1966) [grade 1]; Adventures of Robin Hood, The (1938) [grade 0]; Elizabeth (1998) [grade 1]; Andrei Rublev (Andrey Rublyov) (1969) [grade 1]
- **phase9_story_gate:** Henry V (1989) [grade 1]; Libertine, The (2004) [grade 1]; Elizabeth (1998) [grade 1]; Man for All Seasons, A (1966) [grade 1]; Andrei Rublev (Andrey Rublyov) (1969) [grade 1]
- **chunked_mean:** Man for All Seasons, A (1966) [grade 1]; Adventures of Robin Hood, The (1938) [grade 0]; Elizabeth (1998) [grade 1]; Henry V (1989) [grade 1]; Libertine, The (2004) [grade 1]
- **chunked_max_pair:** Adventures of Robin Hood, The (1938) [grade 0]; Henry V (1989) [grade 1]; Elizabeth (1998) [grade 1]; Man for All Seasons, A (1966) [grade 1]; Breaker Morant (1980) [grade 1]
- **chunked_top2_mean:** Adventures of Robin Hood, The (1938) [grade 0]; Elizabeth (1998) [grade 1]; Man for All Seasons, A (1966) [grade 1]; Henry V (1989) [grade 1]; Libertine, The (2004) [grade 1]

### development: Mortal Kombat (1995) (movie_id=44)
- **phase9_baseline:** Mortal Kombat: Annihilation (1997) [grade 2]; Thor: The Dark World (2013) [grade 1]; 13th Warrior, The (1999) [grade 1]; Journey to the West: Conquering the Demons (Daai wa sai you chi Chui mo chun kei) (2013) [grade 1]; Halloweentown High (2004) [grade 0]
- **tfidf_plot_only:** Mortal Kombat: Annihilation (1997) [grade 2]; Dead Pool, The (1988) [grade 0]; Halloweentown High (2004) [grade 0]; Kiss of the Dragon (2001) [grade 1]; 13th Warrior, The (1999) [grade 1]
- **transformer_plot_only:** Mortal Kombat: Annihilation (1997) [grade 2]; Thor: The Dark World (2013) [grade 1]; Ninja Scroll (Jûbei ninpûchô) (1995) [grade 1]; Journey to the West: Conquering the Demons (Daai wa sai you chi Chui mo chun kei) (2013) [grade 1]; 13th Warrior, The (1999) [grade 1]
- **name_discounted_tfidf:** Mortal Kombat: Annihilation (1997) [grade 2]; Halloweentown High (2004) [grade 0]; 13th Warrior, The (1999) [grade 1]; Thor: The Dark World (2013) [grade 1]; Journey to the West: Conquering the Demons (Daai wa sai you chi Chui mo chun kei) (2013) [grade 1]
- **phase9_story_gate:** Mortal Kombat: Annihilation (1997) [grade 2]; Thor: The Dark World (2013) [grade 1]; 13th Warrior, The (1999) [grade 1]; Journey to the West: Conquering the Demons (Daai wa sai you chi Chui mo chun kei) (2013) [grade 1]; Halloweentown High (2004) [grade 0]
- **chunked_mean:** Mortal Kombat: Annihilation (1997) [grade 2]; Ninja Scroll (Jûbei ninpûchô) (1995) [grade 1]; 13th Warrior, The (1999) [grade 1]; Journey to the West: Conquering the Demons (Daai wa sai you chi Chui mo chun kei) (2013) [grade 1]; Thor: The Dark World (2013) [grade 1]
- **chunked_max_pair:** Mortal Kombat: Annihilation (1997) [grade 2]; Ninja Scroll (Jûbei ninpûchô) (1995) [grade 1]; Thor: The Dark World (2013) [grade 1]; Journey to the West: Conquering the Demons (Daai wa sai you chi Chui mo chun kei) (2013) [grade 1]; 13th Warrior, The (1999) [grade 1]
- **chunked_top2_mean:** Mortal Kombat: Annihilation (1997) [grade 2]; Ninja Scroll (Jûbei ninpûchô) (1995) [grade 1]; Thor: The Dark World (2013) [grade 1]; Journey to the West: Conquering the Demons (Daai wa sai you chi Chui mo chun kei) (2013) [grade 1]; 13th Warrior, The (1999) [grade 1]

### locked_audit: Misérables, Les (1995) (movie_id=73)
- **phase9_baseline:** Joyeux Noël (Merry Christmas) (2005) [grade 1]; Misérables, Les (1998) [grade 2]; Hunchback of Notre Dame, The (1939) [grade 1]; I Am David (2003) [grade 1]; Misérables, Les (2000) [grade 2]
- **tfidf_plot_only:** Misérables, Les (1998) [grade 2]; Misérables, Les (2012) [grade 2]; Misérables, Les (2000) [grade 2]; I Am David (2003) [grade 1]; Joyeux Noël (Merry Christmas) (2005) [grade 1]
- **transformer_plot_only:** Misérables, Les (1998) [grade 2]; Misérables, Les (2012) [grade 2]; Misérables, Les (2000) [grade 2]; 13 Tzameti (2005) [grade 0]; I Am David (2003) [grade 1]
- **name_discounted_tfidf:** Misérables, Les (1998) [grade 2]; Misérables, Les (2012) [grade 2]; Misérables, Les (2000) [grade 2]; I Am David (2003) [grade 1]; Joyeux Noël (Merry Christmas) (2005) [grade 1]
- **phase9_story_gate:** Misérables, Les (1998) [grade 2]; Joyeux Noël (Merry Christmas) (2005) [grade 1]; Hunchback of Notre Dame, The (1939) [grade 1]; I Am David (2003) [grade 1]; Misérables, Les (2000) [grade 2]
- **chunked_mean:** Misérables, Les (2012) [grade 2]; Misérables, Les (2000) [grade 2]; Misérables, Les (1998) [grade 2]; 13 Tzameti (2005) [grade 0]; Hunchback of Notre Dame, The (1939) [grade 1]
- **chunked_max_pair:** Misérables, Les (1998) [grade 2]; Misérables, Les (2012) [grade 2]; Misérables, Les (2000) [grade 2]; 13 Tzameti (2005) [grade 0]; Joyeux Noël (Merry Christmas) (2005) [grade 1]
- **chunked_top2_mean:** Misérables, Les (1998) [grade 2]; Misérables, Les (2012) [grade 2]; Misérables, Les (2000) [grade 2]; 13 Tzameti (2005) [grade 0]; Joyeux Noël (Merry Christmas) (2005) [grade 1]

## Reproduction

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./.venv/bin/python scripts/phase14_2_story_relevance.py
```
