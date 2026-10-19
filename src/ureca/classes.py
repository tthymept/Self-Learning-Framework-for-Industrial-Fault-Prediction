# to match class name to the table
# new classes to be added here

import re

# canonical label -> keyword patterns (matched on lowercase, punctuation-stripped name)
CLASS_KEYWORDS = {
    "partial_discharge": [r"partial", r"\bpd\b"],
    "switching_noise": [r"switch"],
    "transient_noise": [r"transient"],
    "spark_lighter": [r"spark", r"lighter"],
}


def resolve_label(folder_name: str) -> str | None:
    s = re.sub(r"[^a-z0-9]+", " ", folder_name.lower()).strip()
    hits = [
        label
        for label, pats in CLASS_KEYWORDS.items()
        if any(re.search(p, s) for p in pats)
    ]
    return hits[0] if len(hits) == 1 else None