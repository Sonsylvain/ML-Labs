# ML Systems Engineering — Labs

Individual lab repository for the course based on Reddi, *Introduction to Machine Learning Systems*.

## Environment

Python **3.11.8**, all packages pinned in `requirements.txt`.

```bash
python3.11 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Known conflict in the pinned set:** `mlflow==2.14.1` declares `pyarrow<16`, but the
course pins `pyarrow==16.1.0`, so a plain `pip install -r requirements.txt` refuses to resolve.
To keep every version exactly as pinned, install with an override:

```bash
uv pip install -r requirements.txt --override overrides.txt
```

(`overrides.txt` contains `pyarrow==16.1.0`.)

## Layout

```
requirements.txt   pinned versions
overrides.txt      resolver override for the mlflow/pyarrow conflict
lab01/             Environment and first system measurements
  report.md
  src/
  results/
data/              downloaded datasets (git-ignored)
```

## Running a lab

```bash
python lab01/src/run_all.py
```
