"""`uv run python -m ureca.ingest`: ingest the batches in drop/."""
import argparse

from .pipeline import main

ap = argparse.ArgumentParser()
ap.add_argument("batches", nargs="*")
ap.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
a = ap.parse_args()
main(a.batches or None, a.yes)
