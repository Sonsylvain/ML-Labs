# Data card: UCI Adult (Census Income)

## Source
- **Origin:** extracted by Barry Becker from the 1994 US Census Bureau database. Donated to the UCI Machine Learning Repository in 1996 by Ronny Kohavi and Barry Becker.
- **Version used:** OpenML dataset 1590, `fetch_openml(name='adult', version=2, as_frame=True)`, https://www.openml.org/d/1590. This version merges the original `adult.data` and `adult.test` files, and the trailing "." on the test labels is removed.
- **Raw content hash:** `02a7bbc4…11a2` (full value in `results/run_config.yaml`), computed at ingestion and checked again at the end of the pipeline.

## License
- UCI lists the dataset under **Creative Commons Attribution 4.0 (CC BY 4.0)**. OpenML lists it as "Public".
- Use requires citing: Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Size
| | Rows | Columns |
|---|---:|---:|
| Raw | 48,842 | 15 (14 features + `income`) |
| Train after split / after dedup | 34,189 / 34,165 | |
| Test (untouched) | 14,653 | |
| Processed file (`data/adult/adult_processed.csv`) | 48,818 | 19 (split, raw_row, 14 cleaned features, 2 derived, income) |

Feature types: 6 integer (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical. The target is `income` (`<=50K` / `>50K`).
Missing values (`?`): workclass 2,799 rows (5.73%), occupation 2,809 rows (5.75%), native-country 857 rows (1.75%). A total of 3,620 rows have at least one missing value.

## Known biases
- **Class imbalance:** 23.93% of rows are `>50K`. Accuracy alone is misleading, so F1 is also reported.
- **Sex:** 66.85% of rows are male. 30.38% of men are `>50K`, against 10.93% of women. A model trained on this data learns that gap.
- **Race:** 85.50% of rows are White, and the other groups are small (Amer-Indian-Eskimo 0.96%, Other 0.83%). Error rates for small groups are estimated on very few test rows.
- **Geography:** 89.74% of rows are United-States. Many native-country values occur fewer than 100 times.
- **Time:** the incomes are 1994 dollars and are not adjusted for inflation. The $50K threshold and the occupation and labour-market patterns are from 1994 and do not describe today's population.
- **Sampling:** the extraction kept only adults (age > 16) with hours-per-week > 0 and a few other filters. `fnlwgt` is a census sampling weight, not a personal attribute.
- **Label noise:** 5 groups (10 rows) have identical features but conflicting income labels (`results/duplicate_conflicts.csv`).

## Privacy concerns
- The data contains no direct identifiers (no names, addresses or IDs), and the Census Bureau anonymised it before release.
- It contains sensitive attributes: race, sex, marital status, relationship and native country. Using them to make decisions about people (credit, hiring) raises fairness and legal issues.
- Combinations of quasi-identifiers (age + race + sex + native-country + occupation) are unique for some rows, especially in the small groups, and linkage re-identification is possible in principle. This pipeline publishes no row-level outputs beyond the audit samples, and the processed file stays in the git-ignored `data/` folder.

## Intended use
Teaching and benchmarking tabular classification and data-pipeline practice. It must not be used for real decisions about individuals.
