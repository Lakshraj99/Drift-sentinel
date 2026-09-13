import csv
import hashlib
from pathlib import Path

import pytest

from driftsentinel.recovery import validate_insects_file


def _write_fixture(path: Path) -> dict:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["f1", "f2", "Class"])
        writer.writerows([[0.1, 0.2, 0], [0.3, 0.4, 1]])
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": 2,
        "features": 2,
        "classes": 2,
    }


def test_insects_provenance_validation_checks_hash_and_metadata(tmp_path: Path):
    path = tmp_path / "insects.csv"
    provenance = _write_fixture(path)
    validate_insects_file(path, provenance)
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_insects_file(path, {**provenance, "rows": 3})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_insects_file(path, {**provenance, "sha256": "0" * 64})
