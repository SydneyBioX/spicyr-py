"""src/core is a verbatim copy of spicyR's C++ core (tools/sync_core.sh); it must never be edited here."""

import hashlib
from pathlib import Path

CORE = Path(__file__).parents[1] / "src" / "core"


def test_core_matches_checksums():
    lines = [l for l in (CORE / "CHECKSUMS").read_text().splitlines() if l and not l.startswith("#")]
    listed = {}
    for l in lines:
        digest, path = l.split(maxsplit=1)
        listed[path] = digest
    on_disk = {str(p.relative_to(CORE)) for p in CORE.rglob("*") if p.is_file() and p.name != "CHECKSUMS"}
    assert on_disk == set(listed), "files added to or removed from src/core: resync from spicyR"
    for path, digest in listed.items():
        assert hashlib.sha256((CORE / path).read_bytes()).hexdigest() == digest, f"{path} differs from spicyR's core"
