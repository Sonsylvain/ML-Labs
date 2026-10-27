"""Task 7: write the processed dataset, hash it with SHA256 and store the hash with the config YAML."""

import hashlib
import json

import pandas as pd
import yaml

from common import CONFIG, DATA_DIR, RESULTS_DIR, TARGET, add_derived, fetch_raw, frame_sha256, prepare

if __name__ == "__main__":
    p = prepare()
    parts = []
    for name in ("train", "test"):
        X = add_derived(p[f"X_{name}"])
        X[TARGET] = p[f"y_{name}"].values
        X.insert(0, "split", name)
        parts.append(X)
    processed = pd.concat(parts).rename_axis("raw_row").reset_index()
    path = DATA_DIR / "adult_processed.csv"
    processed.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    # The raw data must be exactly what ingestion saw
    ingested = json.loads((RESULTS_DIR / "ingestion_log.json").read_text())
    assert frame_sha256(fetch_raw()) == ingested["raw_sha256"], "raw data changed"

    run = {"config": CONFIG,
           "raw": {"rows": ingested["rows"], "columns": ingested["columns"], "sha256": ingested["raw_sha256"]},
           "processed": {"file": "data/adult/adult_processed.csv", "rows": len(processed),
                         "columns": processed.shape[1], "sha256": digest},
           "preprocessor": "lab04/results/preprocessor.joblib"}
    (RESULTS_DIR / "run_config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))
    print(f"processed {processed.shape}, sha256 {digest}")
