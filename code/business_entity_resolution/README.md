# Business entity resolution

Matches every Source 1 business to zero or more Source 2 and Source 3 records.
Country is an open set of labels. Each distinct label, including France, is its own block.
Country is not one-hot encoded and no record is dropped because its country was absent from training.

The files to submit are already built:

- `../../output/matching_results.tsv` — leaderboard file. Holdout macro F0.5 = 0.899 at threshold 0.90.
- `../../output/candidate_pairs.tsv` — the candidates the model scored.

Both files passed `utils/validate_submission.py` (1,732,544 Source 1 rows).

## Reproduce

From this directory:

```bash
pip install -r requirements.txt
python src/run_pipeline.py --data-root ..\..\dataset --work-dir ..\..\work --out-dir ..\..\output --stage all --topk 50
```

On Linux or macOS, replace `^` line breaks with `\`.

- `--stage train` writes `work/model.pkl` and `work/val_metrics.txt`.
- `--stage predict` writes the two TSV files in `--out-dir`.
- The first run builds a normalized cache under `work/cache/`. Later runs reuse it.

The holdout is Source 1 ids with `crc32(id) % 1000 < 8`. Training pairs use `8 <= crc32(id) % 1000 < 28`.

The matcher is LightGBM (MIT). Dependencies are pinned in `requirements.txt`.
No external business data, registry, or geocoder is used.

Expect several hours and about 8 GB of RAM.

## Validate

From `student_resource/`:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
