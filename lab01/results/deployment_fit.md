| model | tier | memory_budget | latency_budget_ms | size_budget | peak_rss_mb | latency_ms | size_kb | memory_ok | memory_ok_smallest_device | latency_ok | size_ok | fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | Cloud | >= 1024 MB | 100 | 500 MB | 144.3 | 0.2372 | 1.03 | yes | yes | yes | yes | yes |
| logistic_regression | Edge | 256-1024 MB | 50 | 50 MB | 144.3 | 0.2372 | 1.03 | yes | yes | yes | yes | yes |
| logistic_regression | Mobile | 64-256 MB | 20 | 10 MB | 144.3 | 0.2372 | 1.03 | yes | no | yes | yes | yes |
| logistic_regression | TinyML | <= 256 KB | 10 | 100 KB | 144.3 | 0.2372 | 1.03 | no | no | yes | yes | no |
| random_forest | Cloud | >= 1024 MB | 100 | 500 MB | 144.7 | 12.2585 | 284.07 | yes | yes | yes | yes | yes |
| random_forest | Edge | 256-1024 MB | 50 | 50 MB | 144.7 | 12.2585 | 284.07 | yes | yes | yes | yes | yes |
| random_forest | Mobile | 64-256 MB | 20 | 10 MB | 144.7 | 12.2585 | 284.07 | yes | no | yes | yes | yes |
| random_forest | TinyML | <= 256 KB | 10 | 100 KB | 144.7 | 12.2585 | 284.07 | no | no | no | no | no |
