# Self-Learning Framework for Industrial Fault Prediction (URECA EEE26210)

Partial discharge (PD) classification from raw sensor waveforms.
This README covers the **data layer only**: environment setup, data storage and the ingestion pipeline. Modelling is not started.

---

## 1. Data

| Item | Value |
|---|---|
| Format | `.txt`, one value per line, no header, 16,384 lines |
| Units | Volts |
| Sampling rate | 125 MS/s (8 ns per sample), ~131 µs per file |
| Filename | Contains a 14-digit capture timestamp `YYYYMMDDHHMMSS` (e.g. `PA20260901105717.txt`) |
| Label | Taken from the class folder name |
| Classes | `partial_discharge`, `switching_noise`, `transient_noise`, `spark_lighter` |

First dataset ("Onsite data", as of 1 Oct 2026): 167 files = 13 PD, 139 switching noise, 7 transient noise, 8 spark lighter. Severe class imbalance. Spark lighter files clip at ±1.0 V. See the EDA summary for details.

---

## 2. Architecture

Code and data are kept separate.

- **Code**: this git repo, stored locally (outside OneDrive), managed with `uv`.
- **Data**: a folder on the school OneDrive (`URECA_project_root`), found through the `URECA_DATA_ROOT` environment variable. No paths are hard-coded.

```
URECA_project_root/            (OneDrive, set to "Always keep on this device")
  drop/            new batches arrive here (zip or folder with class subfolders)
  processed/       <label>/<file>.txt   after successful registration
  rejected/        <batch>/<label>/<file>.txt   files that failed validation
  db/ureca.sqlite  metadata tables
  cache/           rebuildable arrays (not used yet)
  models/          saved models (not used yet)
```

Waveform values are never stored in the database. The database stores metadata and a relative path to each file.

---

## 3. Setup (new machine)

### 3.1 Prerequisites
- Python 3.14+ (pinned in `.python-version`; uv downloads it if missing) and [uv](https://docs.astral.sh/uv/)
- Git, VS Code
- OneDrive desktop app signed in to the **school account**. Do not enable folder backup.

### 3.2 Clone and create the environment
Keep the repo **outside OneDrive**.

```powershell
git clone https://github.com/tthymept/Self-Learning-Framework-for-Industrial-Fault-Prediction.git
cd Self-Learning-Framework-for-Industrial-Fault-Prediction
uv sync
```

First-time creation of this project was done with:

```powershell
uv init
uv venv
uv add numpy pandas torch pyyaml
```

In VS Code: open the repo folder, then `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv`.

Verify:
```powershell
uv run python -c "import torch, pandas, sqlite3; print(torch.__version__, pandas.__version__)"
```

### 3.3 Data root
Create `URECA_project_root` inside the school OneDrive folder, right-click it → **Always keep on this device** (otherwise cloud-only placeholders break Python reads).

Set the environment variable once per machine (use your own path), then **reopen VS Code and terminals**:

```powershell
setx URECA_DATA_ROOT "C:\Users\<you>\OneDrive - Nanyang Technological University\URECA_project_root"
```

`setx` stores the value in the Windows user profile (registry, `HKEY_CURRENT_USER\Environment`). It affects only newly started programs and is per user, per PC. It is not stored in git.

Create the subfolders and the database:
```powershell
uv run python -c "from ureca.config import ensure_dirs; print(ensure_dirs())"
uv run python -c "from ureca.db import init_db; init_db()"
```

### 3.4 `.gitignore`
Commit `pyproject.toml` and `uv.lock` (they define the environment). Ignore:
```
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info
.venv
*.sqlite
*.npy
config.local.*
```

---

## 4. Repository layout

The git repo is the `ureca-pd/` folder (that is where `.git` lives). Everything above it is local only and not in git.

```
URECA/                          local workspace, NOT in git
  ureca-pd/                     <- the git repo (code only)
    pyproject.toml, uv.lock     environment definition (commit both)
    .python-version, .gitignore
    README.md
    run_ingest.bat              optional double-click launcher (edit the path inside)
    src/ureca/                  the Python package (`import ureca`)
      __init__.py
      config.py                 data root lookup, creates the folder structure
      db.py                     SQLite schema and connection helpers
      classes.py                folder-name -> class label matching
      ingest.py                 the ingestion pipeline
    notebooks/                  analysis notebooks (read-only use of the data)
      01_eda_onsite_data.ipynb  exploratory analysis of each class
    .venv/                      created by `uv sync`, ignored by git
  data_original/                untouched copy of the data as first received
    Onsite data/
  admin/                        emails, scope documents, other paperwork
```

The live data (drop/processed/db) is **not** in either place. It is in `URECA_project_root` on OneDrive (section 2).

The package uses the `src/` layout: `uv sync` installs `ureca` into `.venv` in editable mode, so `uv run python -m ureca.ingest` works and code edits apply immediately. New modules (for example `dataset.py`, `split.py`) go in `src/ureca/`.

---

## 5. Database (SQLite)

SQLite is a single file read and written through Python's built-in `sqlite3`. No server and no install. Pandas can read it directly. Optional viewers: VS Code extension "SQLite Viewer" or DB Browser for SQLite.

**`batches`**: one row per dropped batch.
`batch_id, name (drop folder/zip name), ingested_at, n_ok, n_rejected, notes`

**`files`**: one row per valid waveform.
| Column | Meaning |
|---|---|
| `file_id` | Auto-increment key |
| `batch_id` | Link to `batches` |
| `raw_path` | Path relative to `processed/` (e.g. `partial_discharge/PA2026....txt`) |
| `label` | Class (NULL allowed for future unlabelled data) |
| `source` | `labelled_training` now; room for real-world data later |
| `timestamp` | ISO time parsed from the filename |
| `sha256` | Content hash, UNIQUE, used for duplicate detection |
| `n_samples` | 16384 |
| `peak_value` | Max \|deviation from median\|, QA only, not a model input |
| `split` | `train` / `val` / `test`. Empty until assigned |
| `status` | `ok` (an `excluded` value is available for bad captures) |

**`rejections`**: files that failed validation (`batch_id, filename, reason, rejected_at`). Rejected files do not enter `files`.

Indexes on `label` and `batch_id` speed up common filters as the data grows.

**Sharing:** only one person (the pipeline owner) writes. Others read a synced or copied `ureca.sqlite`. If several people ever need to write, or the lab provides a hosted server, migrate to PostgreSQL. The table design carries over.

---

## 6. Ingestion workflow

1. Drop a **zip or a folder** into `drop/`. The folder (or zip) should contain one subfolder per class. Nothing runs automatically.
2. Run:
   ```powershell
   uv run python -m ureca.ingest
   ```
   or double-click `run_ingest.bat` (make a desktop shortcut for convenience).
3. The script lists each class folder, the label it maps to and the file count. Type `y` + Enter to process. Anything else cancels and leaves `drop/` untouched. Use `-y` to skip the prompt.

Options: `uv run python -m ureca.ingest "<batch folder name>"` processes one batch only.

### What the script does
1. Extracts any `.zip` in `drop/` into `drop/<zip name>/`. A single wrapper folder is flattened. A browser `(1)` suffix is stripped.
2. Resolves each class folder name to a label (section 7). If any folder is unrecognised, the **whole batch is skipped** before anything is changed.
3. For every `.txt` file:
   - Validates: exactly 16,384 numeric values, no NaN/inf, max amplitude ≤ 1.5 V, a parseable 14-digit timestamp, not a duplicate (sha256), no same-name file with different content.
   - On success: copies to `processed/<label>/`, inserts the row, then deletes the original. On any crash, re-running is safe because of the hash check.
   - On failure: copies to `rejected/<batch>/<label>/`, logs the reason in `rejections`, removes the original.
4. Updates the batch counts and prints class counts and rejection reasons.
5. Removes emptied folders (best effort) and deletes the zip only once its folder is fully ingested.

The script works on a **temporary local copy** of the database and copies it back to OneDrive at the end, so OneDrive never syncs a half-written file.

---

## 7. Class label matching (`src/ureca/classes.py`)

Folder names vary (e.g. `Data_Partial discharge`). Names are lowercased, punctuation is stripped, and keywords are matched:

| Label | Keywords |
|---|---|
| `partial_discharge` | `partial`, `pd` (whole word) |
| `switching_noise` | `switch` |
| `transient_noise` | `transient` |
| `spark_lighter` | `spark`, `lighter` |

A folder matching zero or several classes is **never guessed**. The batch is skipped and the folder is listed. Rename it or add a keyword in `CLASS_KEYWORDS`. New classes are added deliberately in that file, so a typo cannot create a fifth class.

---

## 8. Common tasks

**Check what is in the database**
```powershell
uv run python -c "from ureca.db import connect; c=connect(); print(c.execute('select label,count(*) from files group by label').fetchall())"
```

**Redo everything from scratch** (wipes all batches)
1. Delete `db/ureca.sqlite`.
2. Delete the class folders inside `processed/` and `rejected/`.
3. Put the original zip back into `drop/`.
4. Run the ingest.

**Troubleshooting**
| Problem | Cause / fix |
|---|---|
| `ImportError: cannot import name 'resolve_label'` | `src/ureca/classes.py` empty or unsaved. Save the file. |
| `PermissionError ... rmdir` on a `drop/` folder | OneDrive briefly locks empty folders. Cleanup is now non-fatal. Delete the leftover folder manually. |
| `RuntimeError: Set URECA_DATA_ROOT` | Variable not set, or terminal/VS Code not restarted after `setx`. |
| Python cannot read data files | Files are cloud-only. Set the folder to "Always keep on this device". |
| Batch skipped, unrecognised class folders | Rename the folder or add keywords in `classes.py`. |

---

## 9. Design decisions

- **SQLite over hosted DB (Supabase etc.):** school restrictions are unknown, the data is small (~11 MB as float32) and there is a single writer. A file works on any device with no credentials.
- **SQLite over local PostgreSQL:** a local server is not reachable by teammates without network and firewall changes.
- **Metadata table, raw files kept as `.txt`:** the database stays tiny and the raw data stays human-readable.
- **Code on GitHub, data on OneDrive:** keeps git light. The venv stays out of OneDrive (thousands of small files sync badly).
- **Zip or folder accepted:** a zip is a single file for OneDrive to sync, and it matches how data arrives.
- **Safe ordering (copy, register, delete) plus a hash check:** re-running never creates duplicates.
- **Strict class matching, no LLM:** a wrong guess would silently mislabel training data.
- **Confirmation prompt:** prevents processing a wrongly dropped file.

---

## 10. Status and next steps

Done: environment, git, data root, schema, ingestion with validation, class matching, confirmation prompt, first batch (167 files).

Not done yet:
1. **Split assignment:** fill `split` once, stratified by class, keeping consecutive-second files together to avoid leakage. New batches get a split on ingest without changing existing rows.
2. **PyTorch `Dataset`:** read the table, load waveforms, return `(tensor[1, 16384], label_id)`, with a label map and class weights for the imbalance.
3. Optional: `.npy` cache for faster loading, unlabelled real-world ingestion (`source`, NULL `label`, `predictions` table), model registry in `models/`.

Open question for the supervisors: does the lab or school have a shared database server?
