# Results on the frozen 60-item exam

final = (code score on cut items + blur test) / 2. Code checks and the judge (Qwen2.5-14B blur test) are described in the design. All runs on free Kaggle T4s.

| Model | Code (cut items) | Blur | Final | Status acc |
|---|---|---|---|---|
| Qwen2.5-3B untrained | 0.02 (code score) | – | – | – |
| 3B v1 (28 domains) | – | – | 0.07 | – |
| 3B v2 (212 domains, repair) | 0.62 | 0.43 | 0.50 | 0.97 |
| 3B v3 (+ real docs + rewrites) | 0.65 | 0.35 | 0.50 | 1.00 |
| **3B v4 (+ text-grounded summaries)** | **0.65** | **0.75** | **0.70** | **1.00** |
| Qwen2.5-7B untrained + repair | – | 0.58 | 0.59 | – |
| Qwen2.5-32B untrained + repair (teacher) | – | 0.65 | 0.62 | – |

v4 synthetic held-out test (unseen domains): final 0.74, blur 0.63, relations kept 0.72.

What moved the score: the code repair step (v1 to v2), domain variety from 178 teacher-written packs, and replacing generic pack summaries in the training targets with summaries grounded in each text (v3 to v4: blur 0.35 to 0.75).
