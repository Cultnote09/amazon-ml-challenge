# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** student_team  
**Team Members:** solo  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

Each Source 1 business is matched to Source 2 and Source 3 records from the same country label using a blocking index and a LightGBM classifier. Country is treated as an open set of strings: every distinct label, including France, is its own block, and no country is one-hot encoded or dropped. The decision threshold is chosen on a held-out slice of training entities to maximize macro F0.5, which keeps false merges down.

---

## 2. Methodology

### 2.1 Problem Analysis

Training covers the US and India only. The test set adds France. True training matches always share a country label, so country equality is a hard block, while the set of labels is discovered from the data rather than fixed in advance.

About 5.6% of Source 1 entities are singletons. The rest have between 1 and 11 matches, usually in both Source 2 and Source 3 (about 7.6 million true links). Names are noisy: legal-suffix variants, punctuation, word-order swaps, character substitutions (0/O, 1/l), accents, transliteration into Devanagari, and occasional trade names or domain strings that do not overlap the legal name. Addresses are more stable. On a sample of true pairs, token Jaccard on the name is 0 for about 15% of links, while address token overlap is usually still present. Digit groups in the address (street numbers, ZIP, PIN) overlap on about 80% of true pairs.

Because F0.5 weights precision twice as heavily as recall, a wrong merge costs more than a missed link. A correct empty prediction on a singleton scores 1.0, so the threshold is kept high.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted classifier

**Core Innovation:** Multi-key inverted index (normalized name, address tokens, house/PIN/ZIP signature, and name–address combos) with document-frequency caps, then a precision-oriented LightGBM threshold. Normalization includes accent folding, a small Devanagari transliterator, legal-suffix removal, and letter/digit confusable fixes inside mixed tokens. No external business, registry, or geocoding data is used.

---

## 3. Candidate Generation (Blocking)

Records are partitioned by the raw country string. Within a country, Source 2 and Source 3 are indexed by blocking keys. A key is stored only when its document frequency is between 1 and a per-family cap (name tokens 1500, address tokens 2500, number signatures 1200, name–address combos 2500, full normalized-name concatenations 4000). At query time each Source 1 record scores postings by IDF, keeps the top 50, and also keeps up to 30 further records from a pool of 300 whose name and address token-set similarities sum to at least 160. That list is exactly what the model scores and exactly what is written to `candidate_pairs.tsv`. After scoring, each Source 2 or Source 3 id is assigned to only the highest-scoring Source 1 entity, because every training match id belongs to exactly one Source 1 record.

- **Blocking keys used:** concatenated core name, sorted-token concatenation, consonant skeleton, up to five name tokens (length ≥ 3, legal suffixes removed), a 5-character prefix of long name tokens, up to four address tokens, a 6-character prefix of long address tokens, a signature of address numbers of length 2–6, combos of the longest name tokens with the longest address tokens, and the longest name token plus the number signature.
- **Candidate pairs generated:** about 50 candidates per Source 1 entity on the training holdout (India 52.4, US 49.7).
- **How you ensured true matches were not lost:** Several redundant keys, so a pair can survive a name typo if the address still overlaps, or an address typo if the name still overlaps. IDF ranking prefers rare shared tokens. The extra similarity pass pulls back high string-similarity pairs that IDF rank alone would cut. Held-out blocking recall was 0.938 on US entities and 0.805 on India entities.

---

## 4. Matching Model

**Features used:**

- Name features: RapidFuzz ratio, token set ratio, token sort ratio, partial ratio, token Jaccard, token containment, exact normalized name, sorted-token equality, length ratio, shared 4-character prefix
- Address features: the same four RapidFuzz ratios, token Jaccard, token containment, digit-signature Jaccard, length ratio
- Other: country is not a model feature. It is only the block key, so a new label such as France uses the same string features.

**Model type:** LightGBM binary classifier (MIT license, far below 8 billion parameters). `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=40`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`. Training pairs are the blocked candidates of Source 1 entities with `crc32(id) % 1000` in `[8, 28)`. Negatives are downsampled to four per positive. The decision threshold is not trained; it is swept on entities with `crc32(id) % 1000 < 8`.

**Threshold selection method:** Maximize macro F0.5 on that holdout, after assigning each Source 2 or Source 3 id to only the highest-scoring Source 1 entity. The selected threshold is 0.90 (validation macro F0.5 = 0.899). Neighboring points are precision 0.928 and recall 0.816 at 0.79, and precision 0.939 and recall 0.796 at 0.87. Ties prefer the higher threshold.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.899 on 17,644 held-out Source 1 entities (threshold 0.90), after one-owner assignment. Blocking recall on this run is 0.953 (US) and 0.863 (India).
- **Common false positives (wrong merges):** Different businesses that share a rare-looking name token and a city or road token. The high threshold and the one-owner rule remove most of these. Singletons are protected by the same threshold: an empty prediction scores 1.0, and any predicted match on a true singleton scores 0.
- **Common false negatives (missed matches):** India pairs whose names do not share a Latin token (Devanagari vs English, or a trade name versus the legal name) and whose remaining overlap is only a high-frequency locality word. Those keys are dropped by the document-frequency cap, so the true record never enters the candidate list.

---

## 6. Conclusion

The pipeline is a country-partitioned inverted index plus a LightGBM ranker tuned for macro F0.5. Held-out F0.5 is 0.899 at threshold 0.90. Each Source 2 or Source 3 id is given to only one Source 1 entity. The main remaining errors are India pairs that share neither a distinctive name token nor a distinctive address token after normalization.

---

## Appendix

### A. Code Artefacts

Runnable code is in `code/business_entity_resolution/`.

- `src/run_pipeline.py` — end-to-end entry point
- `requirements.txt` — numpy 2.1.3, scipy 1.15.2, lightgbm 4.7.0, rapidfuzz 3.14.6
- `README.md` — exact command line

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
| 0.85 | 0.888 | — | — |
| 0.87 | 0.888 | 0.939 | 0.796 |
| 0.95 | 0.876 | 0.940 | 0.757 |

Strongest features by LightGBM split count: name ratio, name token-sort ratio, address token-set ratio, name token-set ratio, and name partial ratio.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
