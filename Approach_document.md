# Amazon ML Challenge 2026 — Approach

**Team:** sunshines  
**Members:** Pavni Malhotra, Pranjal Samaiya, Surya Dev Gupta  
**College:** Lakshmi Narain College of Technology Excellence (LNCTE), Bhopal  
**Public leaderboard:** 0.870631 on the previous file. Latest holdout is 0.899.

## Problem

Each Source 1 business must be linked to the Source 2 and Source 3 records that refer to the same real-world entity. There is no shared identifier. Names and addresses are noisy (abbreviations, typos, word order, transliteration). Training countries are the US and India. The test set also contains France, so country is treated as an open set of labels: each distinct string is its own block, and no country is one-hot encoded or dropped. The metric is macro F0.5, which weights precision above recall. A correct empty prediction on a singleton scores 1.0.

## Method

**Blocking.** Records are compared only inside the same country. Within a country, Source 2 and Source 3 are indexed by normalized blocking keys: full name with legal suffixes removed, sorted name tokens, a consonant skeleton, individual name and address tokens, a short prefix of long tokens, an address-number signature (street number, ZIP, or PIN), and combinations of a name token with an address token or number. A key is stored only when it is not extremely common, so a city name does not create a block of hundreds of thousands of businesses. Each Source 1 record keeps the highest-IDF candidates (about 50) plus records from its rarer address and name tokens that still share distinctive words. That list is `candidate_pairs.tsv`. It is the exact set the model scores.

**Features.** Both strings are lowercased, punctuation-stripped, accent-folded, and lightly transliterated from Devanagari. Features are RapidFuzz ratio, token-set, token-sort, and partial ratio on the name and on the address, plus token Jaccard, containment, digit-signature overlap, exact normalized name, length ratios, and a shared 4-character prefix.

**Model.** LightGBM (MIT license), 500 trees, learning rate 0.05, 63 leaves. XGBoost 3.2.0 and CatBoost 1.2.10 (both Apache 2.0) were trained on the same pairs. LightGBM was best. The threshold is chosen on a held-out 0.8% of Source 1 entities (`crc32(id) % 1000 < 8`) to maximize macro F0.5. Training uses the next 2% and downsamples negatives to four per positive. After scoring, each Source 2 or Source 3 id is given to only one Source 1 entity, the one with the highest probability. In the training labels, all 7,638,365 match ids obey that rule.

## Experiments

| Idea | Holdout F0.5 | Notes |
| --- | --- | --- |
| Top-40 IDF candidates, threshold 0.84 | 0.857 | First full test file |
| Wider blocks, top 80, threshold 0.82 | 0.882 | India link recall 0.80, US 0.92 |
| Same blocks, one match-id per owner, threshold 0.85 | 0.888 | Public score 0.871 |
| Rarer address tokens forced into the candidate list, threshold 0.90 | **0.899** | Latest test file. LightGBM beat XGBoost 0.893 and CatBoost 0.889 |
| LightGBM / XGBoost / CatBoost on one candidate set | 0.888 / 0.884 / 0.879 | Tree library is not the gap |
| Sorted neighborhood of 12 neighbors | 0.24 | Names that differ in the first characters are not adjacent |
| Similarity rule when the true record is already in hand | 0.993 | Upper bound. 97.8% of true pairs pass; random businesses almost never do |

India pairs share an address token 95% of the time, but the rarest shared token is often too common to survive a short candidate list (median document frequency about 200, 80th percentile about 4,300). That is why recall, not the choice of booster, limits the score.

## Conclusion

The submitted system is country blocking, a capped inverted index, LightGBM on string-similarity features, and a one-owner assignment. Holdout macro F0.5 is 0.899 at threshold 0.90. The previous public leaderboard score, 0.870631, is from the 0.888 file. False merges of businesses that share a street are controlled by the high threshold. Missed matches are mostly India records whose only overlap is a frequent locality word or a name that does not share a Latin token. No external business database, registry, or geocoder is used.
