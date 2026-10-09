# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** sunshines  
**Members:** Pavni Malhotra, Pranjal Samaiya, Surya Dev Gupta  
**College:** Lakshmi Narain College of Technology Excellence (LNCTE), Bhopal

| | |
| --- | --- |
| Public leaderboard score | 0.870631 |
| Rank | 1774 |
| Recorded | 26 September 2026, 03:09 AM IST |
| Holdout macro F0.5 | 0.899 |

The public score is the leaderboard result. The holdout score is from local validation on 17,644 training entities and was not the number shown on the public board.

## What we got wrong

- The leaderboard file was the earlier run (holdout 0.888, threshold 0.85). The later run, holdout 0.899, was not the public score.
- Many true India matches only share a common locality word. Those keys were dropped so the candidate list stayed short, and the classifier never saw the true record. That capped recall before any model could help.
- Cutting the candidate list further lost more true matches. Keeping the rare-token matches raised recall and made the candidate file about 1.5 GB.
- Sorted-neighborhood blocking scored 0.24. Businesses whose names differ in the first characters are not neighbors, so the true record was outside the window.
- XGBoost (0.893) and CatBoost (0.889) did not beat LightGBM on the same pairs. Changing the tree library did not recover matches that blocking had already dropped.
- Blocking keys used Python’s process-local `hash()`, so a new run does not rebuild the exact same candidate ids.
