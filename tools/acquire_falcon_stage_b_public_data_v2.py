from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
import zipfile

EPSS_COMMIT = "862f9bbdf887ef60fd45f9d2fbb89bfaca2f8a48"
KEV_COMMIT = "b82bd290510b1f553dafc6a0d996e6c38305bc66"
CVE_REPO = "CVEProject/cvelistV5"
ISSUE_CUTOFFS = (
    "2025-04-30", "2025-05-31", "2025-06-30", "2025-07-31",
    "2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30",
    "2025-12-31", "2026-01-28", "2026-04-30", "2026-05-11",
)
LAGS = (1, 7, 30)


def request_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Topologica-FALCON-public-data-transport/2.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Topologica-FALCON-public-data-transport/2.0"})
    with urllib.request.urlopen(req, timeout=900) as response, output.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1024 * 1024)
    if output.stat().st_size == 0:
        raise RuntimeError(f"empty download: {url}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def epss_url(day: date) -> str:
    rel = f"{day.year}/epss_scores-{day.isoformat()}.csv.gz"
    return f"https://raw.githubusercontent.com/empiricalsec/epss_scores/{EPSS_COMMIT}/{rel}"


def acquire_epss(day: date, target: Path) -> dict[str, object]:
    download(epss_url(day), target)
    rows = 0
    seen: set[str] = set()
    with gzip.open(target, "rt", encoding="utf-8", newline="") as fh:
        metadata = fh.readline().strip()
        if day.isoformat() not in metadata:
            raise ValueError(f"EPSS metadata date mismatch for {day}: {metadata}")
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["cve", "epss", "percentile"]:
            raise ValueError(f"unexpected EPSS schema: {reader.fieldnames}")
        for row in reader:
            cve = row["cve"]
            if cve in seen:
                raise ValueError(f"duplicate EPSS CVE {cve} on {day}")
            seen.add(cve)
            score = float(row["epss"])
            percentile = float(row["percentile"])
            if not (0.0 <= score <= 1.0 and 0.0 <= percentile <= 1.0):
                raise ValueError(f"EPSS value out of range {cve} on {day}")
            rows += 1
    if rows < 100_000:
        raise ValueError(f"EPSS snapshot unexpectedly small for {day}: {rows}")
    return {
        "date": day.isoformat(), "url": epss_url(day), "sha256": sha256_file(target),
        "bytes": target.stat().st_size, "rows": rows, "metadata": metadata,
    }


def load_epss_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        fh.readline()
        for row in csv.DictReader(fh):
            ids.add(row["cve"])
    return ids


def acquire_kev(target: Path) -> tuple[dict[str, date], dict[str, object]]:
    url = f"https://raw.githubusercontent.com/cisagov/kev-data/{KEV_COMMIT}/known_exploited_vulnerabilities.csv"
    download(url, target)
    values: dict[str, date] = {}
    with target.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not {"cveID", "dateAdded"}.issubset(reader.fieldnames or []):
            raise ValueError(f"unexpected KEV schema: {reader.fieldnames}")
        for row in reader:
            cve = row["cveID"]
            if cve in values:
                raise ValueError(f"duplicate KEV CVE {cve}")
            values[cve] = date.fromisoformat(row["dateAdded"])
    newest = max(values.values())
    if newest < date(2026, 8, 9):
        raise ValueError(f"KEV snapshot not mature through Stage B horizon: {newest}")
    return values, {
        "url": url, "sha256": sha256_file(target), "bytes": target.stat().st_size,
        "rows": len(values), "newest_date_added": newest.isoformat(),
    }


def release_assets(day: date) -> dict[str, object]:
    tag = f"cve_{day.isoformat()}_0000Z"
    api = f"https://api.github.com/repos/{CVE_REPO}/releases/tags/{urllib.parse.quote(tag)}"
    release = json.loads(request_bytes(api))
    assets = release.get("assets", [])
    baseline = [a for a in assets if "_all_CVEs_at_midnight.zip" in str(a.get("name", ""))]
    delta = [a for a in assets if f"{day.isoformat()}_delta_CVEs_at_0000Z.zip" in str(a.get("name", ""))]
    if len(baseline) != 1 or len(delta) != 1:
        raise RuntimeError(
            f"expected one prior-midnight baseline and one 0000Z delta for {tag}; "
            f"found {[a.get('name') for a in assets]}"
        )
    b, d = baseline[0], delta[0]
    return {
        "tag": tag,
        "release_id": release.get("id"),
        "published_at": release.get("published_at"),
        "target_commitish": release.get("target_commitish"),
        "baseline_name": b["name"],
        "baseline_url": b["browser_download_url"],
        "baseline_size": b.get("size"),
        "delta_name": d["name"],
        "delta_url": d["browser_download_url"],
        "delta_size": d.get("size"),
    }


def first_english_description(cna: dict) -> str:
    texts: list[str] = []
    for item in cna.get("descriptions") or []:
        if str(item.get("lang", "")).lower().startswith("en") and item.get("value"):
            value = str(item["value"]).strip()
            if value:
                texts.append(value)
    return "\n".join(texts)


def compact_record(record: dict) -> dict[str, object] | None:
    meta = record.get("cveMetadata") or {}
    cve_id = meta.get("cveId")
    if not cve_id:
        return None
    containers = record.get("containers") or {}
    cna = containers.get("cna") or {}
    affected = cna.get("affected") or []
    vendors = sorted({str(x.get("vendor", "")).strip() for x in affected if str(x.get("vendor", "")).strip()})
    products = sorted({str(x.get("product", "")).strip() for x in affected if str(x.get("product", "")).strip()})
    problem_texts: list[str] = []
    for block in cna.get("problemTypes") or []:
        for desc in block.get("descriptions") or []:
            value = str(desc.get("description", "")).strip()
            cwe = str(desc.get("cweId", "")).strip()
            if cwe or value:
                problem_texts.append(" ".join(x for x in (cwe, value) if x))
    metrics: list[dict[str, object]] = []
    for block in cna.get("metrics") or []:
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if not isinstance(value, dict) or not key.lower().startswith("cvss"):
                continue
            metrics.append({
                "schema": key,
                "baseScore": value.get("baseScore"),
                "baseSeverity": value.get("baseSeverity"),
                "vectorString": value.get("vectorString"),
            })
    references = cna.get("references") or []
    hosts: set[str] = set()
    for ref in references:
        url = str(ref.get("url", ""))
        try:
            host = urllib.parse.urlparse(url).hostname
        except Exception:
            host = None
        if host:
            hosts.add(host.lower())
    provider = cna.get("providerMetadata") or {}
    return {
        "cve_id": str(cve_id),
        "state": meta.get("state"),
        "date_published": meta.get("datePublished"),
        "date_updated": meta.get("dateUpdated"),
        "assigner_short_name": provider.get("shortName"),
        "description_en": first_english_description(cna),
        "vendors": vendors,
        "products": products,
        "problem_types": sorted(set(problem_texts)),
        "metrics": metrics,
        "reference_hosts": sorted(hosts),
        "reference_count": len(references),
        "has_adp": bool(containers.get("adp")),
    }


def overlay_archive(archive: Path, eligible: set[str], records: dict[str, dict[str, object]]) -> int:
    changed = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".json"):
                continue
            cve_id = Path(info.filename).stem
            if cve_id not in eligible:
                continue
            try:
                raw = json.loads(zf.read(info))
            except Exception as exc:
                raise RuntimeError(f"failed parsing {info.filename} in {archive}: {exc}") from exc
            compact = compact_record(raw)
            if compact is not None:
                records[cve_id] = compact
                changed += 1
    return changed


def reconstruct_context(day: date, eligible: set[str], output: Path, work: Path) -> dict[str, object]:
    rel = release_assets(day)
    baseline_path = work / str(rel["baseline_name"])
    delta_path = work / str(rel["delta_name"])
    download(str(rel["baseline_url"]), baseline_path)
    baseline_sha = sha256_file(baseline_path)
    baseline_bytes = baseline_path.stat().st_size
    records: dict[str, dict[str, object]] = {}
    baseline_matches = overlay_archive(baseline_path, eligible, records)
    baseline_path.unlink()

    download(str(rel["delta_url"]), delta_path)
    delta_sha = sha256_file(delta_path)
    delta_bytes = delta_path.stat().st_size
    delta_matches = overlay_archive(delta_path, eligible, records)
    delta_path.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    published = 0
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as fh:
        for cve_id in sorted(records):
            record = records[cve_id]
            if record.get("state") == "PUBLISHED":
                published += 1
            fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
    if len(records) < 100_000:
        raise ValueError(f"historical CVE context unexpectedly sparse for {day}: {len(records)}")
    return {
        **rel,
        "issue_date": day.isoformat(),
        "eligible_epss_rows": len(eligible),
        "baseline_matching_records": baseline_matches,
        "delta_matching_records": delta_matches,
        "compact_rows": len(records),
        "published_rows": published,
        "baseline_sha256": baseline_sha,
        "baseline_bytes": baseline_bytes,
        "delta_sha256": delta_sha,
        "delta_bytes": delta_bytes,
        "compact_path": str(output.name),
        "compact_sha256": sha256_file(output),
        "compact_bytes": output.stat().st_size,
        "reconstruction": "prior-midnight all-CVEs baseline overlaid by same release 0000Z delta",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)

    kev_path = out / "cisa" / "known_exploited_vulnerabilities.csv"
    kev, kev_record = acquire_kev(kev_path)

    needed_epss: set[date] = set()
    for value in ISSUE_CUTOFFS:
        d = date.fromisoformat(value)
        needed_epss.add(d)
        for lag in LAGS:
            needed_epss.add(d - timedelta(days=lag))
    epss_records: list[dict[str, object]] = []
    for d in sorted(needed_epss):
        path = out / "epss" / f"epss_scores-{d.isoformat()}.csv.gz"
        epss_records.append(acquire_epss(d, path))

    context_records: list[dict[str, object]] = []
    for value in ISSUE_CUTOFFS:
        d = date.fromisoformat(value)
        epss_path = out / "epss" / f"epss_scores-{d.isoformat()}.csv.gz"
        ids = load_epss_ids(epss_path)
        eligible = {cve for cve in ids if kev.get(cve, date.max) > d}
        output = out / "cve_context" / f"cve_context-{d.isoformat()}.jsonl.gz"
        context_records.append(reconstruct_context(d, eligible, output, work))
        print(json.dumps({"completed_context": d.isoformat(), "rows": context_records[-1]["compact_rows"], "bytes": context_records[-1]["compact_bytes"]}, sort_keys=True), flush=True)

    shutil.rmtree(work)
    manifest = {
        "schema_version": 2,
        "epss_commit": EPSS_COMMIT,
        "kev_commit": KEV_COMMIT,
        "cve_repository": CVE_REPO,
        "issue_cutoffs": list(ISSUE_CUTOFFS),
        "lags_days": list(LAGS),
        "epss_records": epss_records,
        "kev_record": kev_record,
        "cve_context_records": context_records,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "epss_files": len(epss_records),
        "context_files": len(context_records),
        "manifest_sha256": sha256_file(manifest_path),
        "total_compact_context_bytes": sum(int(x["compact_bytes"]) for x in context_records),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
