from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

EPSS_COMMIT = "862f9bbdf887ef60fd45f9d2fbb89bfaca2f8a48"
KEV_COMMIT = "b82bd290510b1f553dafc6a0d996e6c38305bc66"
ISSUE_CUTOFFS = (
    "2025-04-30", "2025-05-31", "2025-06-30", "2025-07-31",
    "2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30",
    "2025-12-31", "2026-01-28", "2026-04-30",
)
LAGS = (1, 7, 30)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def required_epss_dates() -> tuple[date, ...]:
    values: set[date] = set()
    for cutoff_s in ISSUE_CUTOFFS:
        cutoff = date.fromisoformat(cutoff_s)
        values.add(cutoff)
        for lag in LAGS:
            values.add(cutoff - timedelta(days=lag))
    return tuple(sorted(values))


def fetch(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Topologica-FALCON-public-data-transport/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, output.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    if output.stat().st_size == 0:
        raise RuntimeError(f"empty download: {url}")


def validate_epss(path: Path, expected_date: date) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        first = fh.readline().strip()
        if not first.startswith("#"):
            raise ValueError(f"EPSS metadata comment missing: {path}")
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["cve", "epss", "percentile"]:
            raise ValueError(f"Unexpected EPSS columns in {path}: {reader.fieldnames}")
        rows = 0
        min_score = 1.0
        max_score = 0.0
        seen: set[str] = set()
        for row in reader:
            cve = row["cve"]
            if not cve.startswith("CVE-"):
                raise ValueError(f"Invalid CVE identifier in {path}: {cve}")
            if cve in seen:
                raise ValueError(f"Duplicate CVE in {path}: {cve}")
            seen.add(cve)
            score = float(row["epss"])
            percentile = float(row["percentile"])
            if not (0.0 <= score <= 1.0 and 0.0 <= percentile <= 1.0):
                raise ValueError(f"EPSS value out of range in {path}: {row}")
            min_score = min(min_score, score)
            max_score = max(max_score, score)
            rows += 1
    if rows < 100_000:
        raise ValueError(f"EPSS snapshot unexpectedly small: {path}, rows={rows}")
    expected_text = expected_date.isoformat()
    if expected_text not in first:
        raise ValueError(f"EPSS metadata date mismatch for {path}: expected {expected_text}, got {first}")
    return {"rows": rows, "metadata": first, "min_epss": min_score, "max_epss": max_score}


def validate_kev(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"cveID", "dateAdded"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required KEV columns: {reader.fieldnames}")
        rows = 0
        seen: set[str] = set()
        newest: date | None = None
        for row in reader:
            cve = row["cveID"]
            if cve in seen:
                raise ValueError(f"Duplicate KEV CVE: {cve}")
            seen.add(cve)
            added = date.fromisoformat(row["dateAdded"])
            newest = added if newest is None or added > newest else newest
            rows += 1
    if rows < 1_000:
        raise ValueError(f"KEV catalog unexpectedly small: rows={rows}")
    if newest is None or newest < date(2026, 7, 29):
        raise ValueError(f"KEV catalog is not mature through final holdout horizon: newest={newest}")
    return {"rows": rows, "newest_date_added": newest.isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for snapshot_date in required_epss_dates():
        name = f"epss_scores-{snapshot_date.isoformat()}.csv.gz"
        rel = Path(str(snapshot_date.year)) / name
        url = f"https://raw.githubusercontent.com/empiricalsec/epss_scores/{EPSS_COMMIT}/{rel.as_posix()}"
        path = out / "epss" / name
        fetch(url, path)
        validation = validate_epss(path, snapshot_date)
        records.append({
            "source": "EPSS",
            "source_commit": EPSS_COMMIT,
            "snapshot_date": snapshot_date.isoformat(),
            "url": url,
            "path": str(path.relative_to(out)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **validation,
        })

    kev_url = (
        "https://raw.githubusercontent.com/cisagov/kev-data/"
        f"{KEV_COMMIT}/known_exploited_vulnerabilities.csv"
    )
    kev_path = out / "cisa" / "known_exploited_vulnerabilities.csv"
    fetch(kev_url, kev_path)
    kev_validation = validate_kev(kev_path)
    records.append({
        "source": "CISA_KEV",
        "source_commit": KEV_COMMIT,
        "url": kev_url,
        "path": str(kev_path.relative_to(out)),
        "bytes": kev_path.stat().st_size,
        "sha256": sha256_file(kev_path),
        **kev_validation,
    })

    manifest = {
        "transport_schema_version": 1,
        "epss_commit": EPSS_COMMIT,
        "kev_commit": KEV_COMMIT,
        "issue_cutoffs": list(ISSUE_CUTOFFS),
        "lags_days": list(LAGS),
        "epss_snapshot_count": len(required_epss_dates()),
        "records": records,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "epss_snapshots": len(required_epss_dates()),
        "files": len(records),
        "manifest_sha256": sha256_file(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
