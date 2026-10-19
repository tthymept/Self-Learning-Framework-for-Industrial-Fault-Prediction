import sqlite3
from .config import ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id     INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    ingested_at  TEXT NOT NULL,
    n_ok         INTEGER DEFAULT 0,
    n_rejected   INTEGER DEFAULT 0,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS files (
    file_id      INTEGER PRIMARY KEY,
    batch_id     INTEGER NOT NULL REFERENCES batches(batch_id),
    raw_path     TEXT NOT NULL,          -- relative to data root
    label        TEXT,                   -- NULL allowed (future unlabelled data)
    source       TEXT NOT NULL DEFAULT 'labelled_training',
    timestamp    TEXT,                   -- ISO, parsed from filename
    sha256       TEXT NOT NULL UNIQUE,   -- duplicate detection
    n_samples    INTEGER NOT NULL,
    peak_value   REAL,                   -- QA only, not a model input
    split        TEXT CHECK (split IN ('train','val','test')),
    status       TEXT NOT NULL DEFAULT 'ok'
);

CREATE TABLE IF NOT EXISTS rejections (
    id           INTEGER PRIMARY KEY,
    batch_id     INTEGER NOT NULL REFERENCES batches(batch_id),
    filename     TEXT NOT NULL,
    reason       TEXT NOT NULL,
    rejected_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_label ON files(label);
CREATE INDEX IF NOT EXISTS idx_files_batch ON files(batch_id);
"""

def db_path():
    return ensure_dirs()["db"] / "ureca.sqlite"

def connect(path=None):
    conn = sqlite3.connect(path or db_path())
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(path=None):
    with connect(path) as conn:
        conn.executescript(SCHEMA)