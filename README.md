# Self-Learning Framework for Industrial Fault Prediction (URECA EEE26210)

Partial discharge (PD) classification from raw sensor waveforms.
This README covers the **data layer**: environment setup, data storage, the ingestion pipeline, and how files are split into cross-validation folds without data leakage (section 9). The models themselves are explained in notebooks 03 (baseline) and 04 (CNN).

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
      split.py                  recording sessions and cross-validation folds (section 9)
      data.py                   file index, waveform cache, PyTorch Dataset, samplers
      evaluate.py               out-of-fold predictions and scores
      features.py               hand-made features for the baseline
      cnn.py                    1-D CNN, trained and scored with the folds
      plotting.py               shared plot style for the notebooks
    notebooks/                  analysis notebooks (read-only use of the data)
      01_eda_onsite_data.ipynb  exploratory analysis of each class
      02_setup_folds_and_dataloader_explained.ipynb   sessions, folds, data loader
      03_baseline_model.ipynb   feature-based baseline, incl. the leakage check
      04_cnn_model.ipynb        1-D CNN, compared with the baseline
    .venv/                      created by `uv sync`, ignored by git
  data_original/                untouched copy of the data as first received
    Onsite data/
  admin/                        emails, scope documents, other paperwork
```

The live data (drop/processed/db) is **not** in either place. It is in `URECA_project_root` on OneDrive (section 2).

The package uses the `src/` layout: `uv sync` installs `ureca` into `.venv` in editable mode, so `uv run python -m ureca.ingest` works and code edits apply immediately. New modules go in `src/ureca/`.

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
| `split` | `train` / `val` / `test`. Not used: evaluation uses `fold` instead |
| `session_id` | Recording session, `<label>@<first timestamp>` (section 9) |
| `fold` | Cross-validation fold `0`–`2`. A whole session shares one fold (section 9) |
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
5. Assigns sessions and folds to the new files (section 9). Existing files keep their fold.
6. Removes emptied folders (best effort) and deletes the zip only once its folder is fully ingested.

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

## 9. Cross-validation folds and data leakage (`src/ureca/split.py`)

### The problem
*Data leakage* means the validation files contain information the model already saw in training, so the score looks better than the model really is.

Here it would come from how the data was recorded. Switching noise was captured about once per second for a minute, so files recorded seconds apart are **near-copies**. Each session also has its own signature (trigger level, amplitude). With an ordinary random split, near-copies from the same burst land in both training and validation. The model can then recognise the session instead of learning the waveform shape, and the test becomes partly a memory test.

### The fix: split by session, not by file
1. **Sessions:** files of the same class recorded less than `SESSION_GAP_S = 60` seconds apart share a `session_id`.
2. **Folds:** whole sessions are assigned to folds, so a session is never split between training and validation. Large sessions are placed first, each in the fold with the fewest files of its class, to keep the classes balanced across folds.
3. **Stable:** sessions that already have a fold keep it when new batches arrive, so earlier results stay comparable.

There are `N_FOLDS = 3` folds because switching noise has only 3 sessions, and each fold needs at least one of each class. Every score is out-of-fold: each file is predicted by a model trained on the other two folds (`evaluate.cv_predict`).

```powershell
uv run python -m ureca.split --dry-run   # show the assignment without writing
uv run python -m ureca.split             # (re)assign and print files/sessions per class and fold
```

### How much it matters (notebook 03, section 6)
Macro-F1 for the same models, scored with the session folds and with a random 3-fold split (mean of 5 shuffles):

| Model | Session split (honest) | Random split (leaky) |
|---|---|---|
| Logistic regression | 0.533 | 0.937 |
| Random forest | 0.763 | 0.916 |

The models are the same; only the split changed. **Always report the session-split score**, and when comparing with published PD results, check how they split their data.

### Other leaks to avoid
- **Normalisation:** with `normalize="global"`, compute the scale from the training folds only, never from the validation files.
- **Early stopping:** the CNN trains for a fixed number of epochs. Stopping where the validation score peaks would let the validation files choose the model.
- **Any other preprocessing** fitted to data (scalers, feature selection, thresholds) is fitted inside each fold on the training part only.

---

## 10. Design decisions

- **SQLite over hosted DB (Supabase etc.):** school restrictions are unknown, the data is small (~11 MB as float32) and there is a single writer. A file works on any device with no credentials.
- **SQLite over local PostgreSQL:** a local server is not reachable by teammates without network and firewall changes.
- **Metadata table, raw files kept as `.txt`:** the database stays tiny and the raw data stays human-readable.
- **Code on GitHub, data on OneDrive:** keeps git light. The venv stays out of OneDrive (thousands of small files sync badly).
- **Zip or folder accepted:** a zip is a single file for OneDrive to sync, and it matches how data arrives.
- **Safe ordering (copy, register, delete) plus a hash check:** re-running never creates duplicates.
- **Strict class matching, no LLM:** a wrong guess would silently mislabel training data.
- **Confirmation prompt:** prevents processing a wrongly dropped file.
- **Session-based folds instead of a fixed train/val/test split:** near-copies within a session would leak across a random split (section 9). With only 3 switching-noise sessions, cross-validation uses all the data, and a separate test set is not possible yet.

---

## 11. Status and next steps

Done: environment, git, data root, schema, ingestion with validation, class matching, confirmation prompt, first batch (167 files), EDA, session-based folds, PyTorch `Dataset` and samplers, waveform cache, baseline model, 1-D CNN.

Not done yet:
1. **Held-out test set:** once more sessions are recorded, keep some whole sessions aside as a final test set, never used for model choices.
2. Optional: unlabelled real-world ingestion (`source`, NULL `label`, `predictions` table), model registry in `models/`.

Open question for the supervisors: does the lab or school have a shared database server?
