#!/usr/bin/env python3
"""Check documentation assumptions that can be validated without large data."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "README.md",
    "docker-compose.yml",
    "fantasia/constants.yaml",
    "config/prott5_test.yaml",
    "config/prott5_full.yaml",
    "data_sample/sample.fasta",
    "scripts/filter_raw_results_by_identity.py",
    "scripts/merge_raw_results.py",
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        fail("documented repository files are missing: " + ", ".join(missing))

    for relative in ("config/prott5_test.yaml", "config/prott5_full.yaml"):
        with (ROOT / relative).open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, dict):
            fail(f"{relative} does not contain a YAML mapping")

    sources = [
        ROOT / "README.md",
        *sorted((ROOT / "docs" / "source").rglob("*.rst")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    forbidden = {
        r"(?i)\bcafa(?:\d+)?\b": "CAFA analysis remains in user documentation",
        r"FANTASIA\.png": "the removed documentation logo is still referenced",
        r"--cutoffs\b": "unsupported --cutoffs option is documented",
        r"(?<![\w-])--k(?:\s|=)": "unsupported --k option is documented",
    }
    for pattern, message in forbidden.items():
        if re.search(pattern, text):
            fail(message)

    required_phrases = (
        "mkdir -p data lookup/{logs,experiments,embeddings}",
        "--limit_per_entry",
        "--threshold 0.90",
        "data_sample/sample.fasta",
    )
    for phrase in required_phrases:
        if phrase not in text:
            fail(f"required documented example is absent: {phrase}")

    print("Documentation assumptions validated.")


if __name__ == "__main__":
    main()
