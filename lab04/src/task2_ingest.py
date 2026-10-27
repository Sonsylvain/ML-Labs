"""Task 2: ingest the raw Adult data without modifying it and log its shape."""

import json

from common import CONFIG, RESULTS_DIR, fetch_raw, frame_sha256

if __name__ == "__main__":
    raw = fetch_raw()
    rows, cols = raw.shape
    log = {"source": CONFIG["dataset"]["source"], "rows": rows, "columns": cols,
           "column_names": list(raw.columns), "raw_sha256": frame_sha256(raw)}
    print(f"Ingested {rows} rows x {cols} columns")
    assert (rows, cols) == (CONFIG["dataset"]["expected_rows"], CONFIG["dataset"]["expected_columns"])
    (RESULTS_DIR / "ingestion_log.json").write_text(json.dumps(log, indent=2))
