# Amazon ML Challenge 2026 — Business Entity Resolution

Team **sunshines** took part in the Amazon ML Challenge 2026 (Unstop). The task was to match each Source 1 business to the Source 2 and Source 3 records that refer to the same real-world entity, with no shared identifier.

**College:** Lakshmi Narain College of Technology Excellence (LNCTE), Bhopal  
**Members:** Pavni Malhotra, Pranjal Samaiya, Surya Dev Gupta  
**Challenge window:** 25 September 2026, 12:00 AM IST to 27 September 2026, 11:59 PM IST  
**Metric:** macro F0.5 (precision weighted more heavily than recall)

## Result

| | |
| --- | --- |
| Public leaderboard score | **0.870631** |
| Rank when that score was recorded | 1774 |
| Recorded at | 26 September 2026, 03:09 AM IST |
| Model behind that submission | LightGBM, threshold 0.85, holdout macro F0.5 0.888 |
| Later holdout (same metric, not the public score) | **0.899** at threshold 0.90 |

The 0.899 figure is from a held-out slice of the training labels (`crc32(id) % 1000 < 8`, 17,644 Source 1 entities), after each Source 2 or Source 3 id was assigned to only one Source 1 entity. On that same candidate set, LightGBM (0.899) beat XGBoost (0.893) and CatBoost (0.889). The public score 0.870631 is from the earlier file.

The test prediction covers all 1,732,544 Source 1 rows. Both output files passed the official validator. They are not in this repository: `candidate_pairs.tsv` is about 1.5 GB, which is over GitHub’s file size limit.

## What the system does

Records are compared only inside the same country. Country labels are discovered from the data, so France (present in test, absent from training) is its own block and is not one-hot encoded.

Inside a country, Source 2 and Source 3 are indexed by normalized name, address, number, and combination keys. Very common keys, such as a city name, are dropped. Each Source 1 record keeps the highest-IDF candidates, plus records that share a rare address or name token. A LightGBM classifier scores those pairs on RapidFuzz and token-overlap features. The decision threshold is chosen to maximize macro F0.5.

No external business database, registry, or geocoder is used. LightGBM is MIT-licensed and far below the 8 billion parameter limit.

## Holdout experiments

| Idea | Holdout F0.5 |
| --- | --- |
| Top-40 IDF candidates, threshold 0.84 | 0.857 |
| Wider blocks, top 80, threshold 0.82 | 0.882 |
| One match-id per owner, threshold 0.85 | 0.888 |
| Rare address and name tokens added, threshold 0.90 | 0.899 |
| XGBoost on that last candidate set | 0.893 |
| CatBoost on that last candidate set | 0.889 |

## Repository

| Path | What it is |
| --- | --- |
| `code/business_entity_resolution/` | Runnable pipeline, pinned dependencies, and run instructions |
| `Documentation_template.md` | Methodology write-up submitted for the challenge |
| `Approach_document.md` | Short approach note |

From `code/business_entity_resolution`:

```bash
pip install -r requirements.txt
python src/run_pipeline.py --data-root ..\..\dataset --work-dir ..\..\work --out-dir ..\..\output --stage all --topk 50
```

The full write-up of blocking, features, and the model is in [Documentation_template.md](Documentation_template.md).
