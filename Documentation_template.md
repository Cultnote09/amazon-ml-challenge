# ML Challenge 2026: Business Entity Resolution

**Team Name:** sunshines  
**Team Members:** Pavni Malhotra, Pranjal Samaiya, Surya Dev Gupta  
**College:** Lakshmi Narain College of Technology Excellence (LNCTE), Bhopal  
**Submission Date:** 2026-10-02

---

## 1. Executive Summary

Each Source 1 business is matched to Source 2 and Source 3 records from the same country using a blocking index and a LightGBM classifier. Country is treated as an open set of strings: every distinct label, including France, is its own block, and no country is one-hot encoded or dropped. The decision threshold is chosen on a held-out slice of training entities to maximize macro F0.5, which keeps false merges down.

---

## 2. Methodology

### 2.1 Problem Analysis

Training covers the US and India only; the test set adds France. True matches always share a country label, so country equality is a hard block, and the set of labels is discovered from the data.

About 5.6% of Source 1 entities are singletons. The rest have between 1 and 11 matches, usually in both Source 2 and Source 3 (about 7.6 million true links). Names are noisy: legal-suffix variants, punctuation, word-order swaps, character substitutions (0/O, 1/l), accents, Devanagari transliteration, and trade names that do not overlap the legal name. Token Jaccard on the name is 0 for about 15% of true links, while address token overlap is usually still present, and address digit groups (street number, ZIP, PIN) overlap on about 80% of true pairs.

Because F0.5 weights precision twice as heavily as recall, a wrong merge costs more than a missed link. A correct empty prediction on a singleton scores 1.0, so the threshold is kept high.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted classifier

**Core Innovation:** A multi-key inverted index (normalized name, address tokens, house/PIN/ZIP signature, and name–address combinations) with document-frequency caps, followed by a precision-oriented LightGBM threshold. Normalization includes accent folding, a small Devanagari transliterator, legal-suffix removal, and letter/digit confusable fixes. No external business, registry, or geocoding data is used.

---

## 3. Candidate Generation (Blocking)

Records are partitioned by the raw country string. Within a country, Source 2 and Source 3 are indexed by blocking keys, and a key is stored only when its document frequency is below a per-family cap, so a city name cannot form a block of hundreds of thousands of businesses. Each Source 1 record scores the surviving postings by IDF and keeps the top 50, plus up to 30 more from a pool of 300 whose name and address token-set similarities sum to at least 160. The three rarest address or name tokens (document frequency 2 to 1500) are then scanned, and up to 20 further records that share distinctive words are added. This list is exactly what the model scores and exactly what is written to `candidate_pairs.tsv`. After scoring, each Source 2 or Source 3 id is assigned only to the highest-scoring Source 1 entity, because every training match id belongs to exactly one Source 1 record.

- **Blocking keys used:** concatenated core name, sorted-token concatenation, consonant skeleton, up to five name tokens, 5-character prefix of long name tokens, character bigrams of the longest name token, up to four address tokens, 6-character prefix of long address tokens, bigrams of long address tokens, address-number signature (length 2–6), name–address token combinations, and longest name token plus number signature.
- **Candidate pairs generated:** IDF top 50, plus up to 30 high-similarity records, plus up to 20 rare-token records. On the latest holdout this averaged about 69 candidates in India and about 63 in the US.
- **How true matches were kept:** Several redundant keys let a pair survive a name typo if the address overlaps, or an address typo if the name overlaps. IDF ranking prefers rare shared tokens, and the similarity and rare-token passes recover true matches that would lose the IDF top-50 contest to common locality words. Held-out link recall on this candidate set was 0.953 for US entities and 0.862 for India entities.

---

## 4. Matching Model

**Features used:**

- **Name:** RapidFuzz ratio, token-set, token-sort and partial ratio; token Jaccard; token containment; exact normalized name; sorted-token equality; length ratio; shared 4-character prefix.
- **Address:** the same four RapidFuzz ratios; token Jaccard; token containment; digit-signature Jaccard; length ratio.
- **Country:** not a model feature. It is only the block key, so a new label such as France uses the same string features.

**Model type:** LightGBM binary classifier (MIT license, far below the 8 billion parameter limit): `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=40`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`. Training pairs come from Source 1 entities with `crc32(id) % 1000` in `[8, 28)`, with negatives downsampled to four per positive. The threshold is not trained; it is swept on entities with `crc32(id) % 1000 < 8`.

**Threshold selection:** Maximize macro F0.5 on that holdout after assigning each Source 2 or Source 3 id to only the highest-scoring Source 1 entity. The selected threshold is 0.90, giving a validation macro F0.5 of 0.899.

---

## 5. Results & Error Analysis

- **F0.5 score (macro):** 0.899 on 17,644 held-out Source 1 entities (threshold 0.90), after one-owner assignment. Blocking recall is 0.953 (US) and 0.862 (India). The earlier candidate list, before rare address and name tokens were forced in, scored 0.888 on holdout (threshold 0.85) and 0.870631 on the public leaderboard. On the final pairs LightGBM (0.899) beat XGBoost (0.893) and CatBoost (0.889).
- **Common false positives:** Different businesses that share a rare-looking name token and a city or road token. The high threshold and the one-owner rule remove most of these. Singletons are protected by the same threshold, since a predicted match on a true singleton scores 0.
- **Common false negatives:** India pairs whose names share no Latin token (Devanagari vs English, or a trade name vs the legal name) and whose only remaining overlap is a frequent locality word. The document-frequency cap drops those keys, so the true record never enters the candidate list.

---

## 6. Conclusion

The pipeline is a country-partitioned inverted index plus a LightGBM ranker tuned for macro F0.5. Held-out F0.5 is 0.899 at threshold 0.90, and each Source 2 or Source 3 id goes to only one Source 1 entity. The main remaining errors are India pairs that share neither a distinctive name token nor a distinctive address token after normalization.

---

## Appendix

### A. Code Artefacts

Runnable code is in `code/business_entity_resolution/`.

- `src/run_pipeline.py`: end-to-end entry point
- `requirements.txt`: numpy 2.1.3, scipy 1.15.2, lightgbm 4.7.0, xgboost 3.2.0, catboost 1.2.10, rapidfuzz 3.14.6
- `README.md`: exact command line

From `code/business_entity_resolution`:

```bash
pip install -r requirements.txt
python src/run_pipeline.py --data-root ../../dataset --work-dir ../../work --out-dir ../../output --stage all --topk 50
```

`--stage train` writes `work/model.pkl` and `work/val_metrics.txt`. `--stage predict` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. A normalized cache under `work/cache/` is reused across runs.

### B. Additional Results

Holdout threshold sweep (macro F0.5):

| Threshold | F0.5 | Precision | Recall |
| --- | --- | --- | --- |
| 0.55 | 0.847 | 0.867 | 0.856 |
| 0.71 | 0.878 | 0.915 | 0.829 |
| 0.79 | 0.885 | 0.928 | 0.816 |
| 0.87 | 0.888 | 0.939 | 0.796 |
| 0.95 | 0.876 | 0.940 | 0.757 |

Strongest features by LightGBM split count: name ratio, name token-sort ratio, address token-set ratio, name token-set ratio, and name partial ratio.
