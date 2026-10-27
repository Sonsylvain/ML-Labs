# Validation report

Rows checked: 48842 (none dropped). Total rule violations: 0. Rows with missing values ('?'): 3620. Exact duplicate rows: 52.

| Check | Column | Rule | Observed | Violations |
|---|---|---|---|---:|
| type | age | int | int64 | 0 |
| type | fnlwgt | int | int64 | 0 |
| type | education-num | int | int64 | 0 |
| type | capital-gain | int | int64 | 0 |
| type | capital-loss | int | int64 | 0 |
| type | hours-per-week | int | int64 | 0 |
| type | workclass | category | category | 0 |
| type | education | category | category | 0 |
| type | marital-status | category | category | 0 |
| type | occupation | category | category | 0 |
| type | relationship | category | category | 0 |
| type | race | category | category | 0 |
| type | sex | category | category | 0 |
| type | native-country | category | category | 0 |
| type | income | category | category | 0 |
| columns | unexpected columns | none | none | 0 |
| range | age | 17-90 | min 17, max 90 | 0 |
| range | hours-per-week | 1-99 | min 1, max 99 | 0 |
| range | capital-gain | 0-99999 | min 0, max 99999 | 0 |
| range | capital-loss | 0-4356 | min 0, max 4356 | 0 |
| category | workclass | 9 allowed values incl. '?' | 9 distinct | 0 |
| category | education | 17 allowed values incl. '?' | 16 distinct | 0 |
| category | marital-status | 8 allowed values incl. '?' | 7 distinct | 0 |
| category | occupation | 15 allowed values incl. '?' | 15 distinct | 0 |
| category | relationship | 7 allowed values incl. '?' | 6 distinct | 0 |
| category | race | 6 allowed values incl. '?' | 5 distinct | 0 |
| category | sex | 3 allowed values incl. '?' | 2 distinct | 0 |
| category | native-country | 42 allowed values incl. '?' | 42 distinct | 0 |
| category | income | 3 allowed values incl. '?' | 2 distinct | 0 |
| missing | workclass | '?' allowed | 2799 rows (5.73%) | 0 |
| missing | education | '?' allowed | 0 rows (0.00%) | 0 |
| missing | marital-status | '?' allowed | 0 rows (0.00%) | 0 |
| missing | occupation | '?' allowed | 2809 rows (5.75%) | 0 |
| missing | relationship | '?' allowed | 0 rows (0.00%) | 0 |
| missing | race | '?' allowed | 0 rows (0.00%) | 0 |
| missing | sex | '?' allowed | 0 rows (0.00%) | 0 |
| missing | native-country | '?' allowed | 857 rows (1.75%) | 0 |
| missing | income | '?' allowed | 0 rows (0.00%) | 0 |
| missing | age | no NaN | 0 rows (0.00%) | 0 |
| missing | fnlwgt | no NaN | 0 rows (0.00%) | 0 |
| missing | education-num | no NaN | 0 rows (0.00%) | 0 |
| missing | capital-gain | no NaN | 0 rows (0.00%) | 0 |
| missing | capital-loss | no NaN | 0 rows (0.00%) | 0 |
| missing | hours-per-week | no NaN | 0 rows (0.00%) | 0 |
| duplicates | exact duplicate rows | reported | 52 rows | 0 |
