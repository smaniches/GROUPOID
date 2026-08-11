from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import gzip
import hashlib
import io
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
            "User-Agent": "Topologica-FALCON-public-data-transport/1.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Topologica-FALCON-public-data-transport/1.0"})
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


def load_epss_ids(day: date, work: Path) -> tuple[set[str], dict[str, object]]:
    path = work / f"epss_scores-{day.isoformat()}.csv.gz"
    download(epss_url(day), path)
    ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        metadata = fh.readline().strip()
        if day.isoformat() not in metadata:
            raise ValueError(f"EPSS metadata date mismatch for {day}: {metadata}")
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["cve", "epss", "percentile"]:
            raise ValueError(f"unexpected EPSS schema: {reader.fieldnames}")
        for row in reader:
            ids.add(row["cve"])
    if len(ids) < 100_000:
        raise ValueError(f"EPSS snapshot unexpectedly small for {day}: {len(ids)}")
    record = {"date": day.isoformat(), "url": epss_url(day), "sha256": sha256_file(path), "rows": len(ids)}
    return ids, record


def load_kev(work: Path) -> tuple[dict[str, date], dict[str, object]]:
    url = f"https://raw.githubusercontent.com/cisagov/kev-data/{KEV_COMMIT}/known_exploited_vulnerabilities.csv"
    path = work / "known_exploited_vulnerabilities.csv"
    download(url, path)
    values: dict[str, date] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            values[row["cveID"]] = date.fromisoformat(row["dateAdded"])
    if len(values) < 1000:
        raise ValueError("KEV snapshot unexpectedly small")
    return values, {"url": url, "sha256": sha256_file(path), "rows": len(values), "newest_date_added": max(values.values()).isoformat()}


def release_for(day: date) -> dict:
    tag = f"cve_{day.isoformat()}_0000Z"
    api = f"https://api.github.com/repos/{CVE_REPO}/releases/tags/{urllib.parse.quote(tag)}"
    release = json.loads(request_bytes(api))
    assets = release.get("assets", [])
    candidates = [a for a in assets if str(a.get("name", "")).endswith("_all_CVEs_at_midnight.zip")]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one midnight baseline asset for {tag}; found {[a.get('name') for a in assets]}")
    asset = candidates[0]
    return {
        "tag": tag,
        "release_id": release.get("id"),
        "published_at": release.get("published_at"),
        "asset_name": asset["name"],
        "asset_url": asset["browser_download_url"],
        "asset_size": asset.get("size"),
        "target_commitish": release.get("target_commitish"),
    }


def first_english_description(cna: dict) -> str:
    descriptions = cna.get("descriptions") or []
    texts = []
    for item in descriptions:
        if str(item.get("lang", "")).lower().startswith("en") and item.get("value"):
            texts.append(str(item["value"]).strip())
    return "\n".join(x for x in texts if x)


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


def process_baseline(day: date, eligible: set[str], output: Path, work: Path) -> dict[str, object]:
    rel = release_for(day)
    archive = work / rel["asset_name"]
    download(str(rel["asset_url"]), archive)
    archive_sha = sha256_file(archive)
    emitted = 0
    published = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf, gzip.open(output, "wt", encoding="utf-8") as out:
        for info in zf.infolist():
            name = info.filename
            if not name.lower().endswith(".json") or "/cve-" not in name.lower():
                continue
            cve_id = Path(name).stem
            if cve_id not in eligible:
                continue
            try:
                record = json.loads(zf.read(info))
            except Exception as exc:
                raise RuntimeError(f"failed parsing {name} in {archive}: {exc}") from exc
            compact = compact_record(record)
            if compact is None:
                continue
            emitted += 1
            if compact.get("state") == "PUBLISHED":
                published += 1
            out.write(json.dumps(compact, separators=(",", ":"), ensure_ascii=False) + "\n")
    archive_bytes = archive.stat().st_size
    archive.unlink()
    if emitted < 100_000:
        raise ValueError(f"historical CVE context unexpectedly sparse for {day}: emitted={emitted}")
    return {
        **rel,
        "issue_date": day.isoformat(),
        "eligible_epss_rows": len(eligible),
        "compact_rows": emitted,
        "published_rows": published,
        "release_archive_sha256": archive_sha,
        "release_archive_bytes": archive_bytes,
        "compact_path": output.name,
        "compact_sha256": sha256_file(output),
        "compact_bytes": output.stat().st_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    kev, kev_record = load_kev(work)
    epss_records: list[dict[str, object]] = []
    context_records: list[dict[str, object]] = []
    needed_epss: set[date] = set()
    for value in ISSUE_CUTOFFS:
        d = date.fromisoformat(value)
        needed_epss.add(d)
        for lag in LAGS:
            needed_epss.add(d - timedelta(days=lag))
    for d in sorted(needed_epss):
        _, record = load_epss_ids(d, work)
        epss_records.append(record)
        (out / "epss").mkdir(exist_ok=True)
        shutil.move(str(work / f"epss_scores-{d.isoformat()}.csv.gz"), str(out / "epss" / f"epss_scores-{d.isoformat()}.csv.gz"))
    for value in ISSUE_CUTOFFS:
        d = date.fromisoformat(value)
        epss_path = out / "epss" / f"epss_scores-{d.isoformat()}.csv.gz"
        ids: set[str] = set()
        with gzip.open(epss_path, "rt", encoding="utf-8", newline="") as fh:
            fh.readline()
            for row in csv.DictReader(fh):
                cve = row["cve"]
                if kev.get(cve, date.max) > d:
                    ids.add(cve)
        context_path = out / "cve_context" / f"cve_context-{d.isoformat()}.jsonl.gz"
        context_records.append(process_baseline(d, ids, context_path, work))
    shutil.rmtree(work)
    manifest = {
        "schema_version": 1,
        "epss_commit": EPSS_COMMIT,
        "kev_commit": KEV_COMMIT,
        "cve_repository": CVE_REPO,
        "issue_cutoffs": list(ISSUE_CUTOFFS),
        "lags_days": list(LAGS),
        "epss_records": epss_records,
        "kev_record": kev_record,
        "cve_context_records": context_records,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "epss_files": len(epss_records),
        "context_files": len(context_records),
        "manifest_sha256": sha256_file(path),
        "total_compact_context_bytes": sum(int(x["compact_bytes"]) for x in context_records),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
