from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path

import acquire_falcon_stage_b_public_data_v2 as v2


def acquire_kev(target: Path):
    url = f"https://raw.githubusercontent.com/cisagov/kev-data/{v2.KEV_COMMIT}/known_exploited_vulnerabilities.csv"
    v2.download(url, target)
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
    commit_api = f"https://api.github.com/repos/cisagov/kev-data/commits/{v2.KEV_COMMIT}"
    commit = json.loads(v2.request_bytes(commit_api))
    committed_at_text = commit["commit"]["committer"]["date"]
    committed_at = datetime.fromisoformat(committed_at_text.replace("Z", "+00:00")).astimezone(timezone.utc)
    horizon_end = date(2026, 8, 9)
    if committed_at.date() <= horizon_end:
        raise ValueError(f"KEV source commit does not postdate final label horizon: {committed_at_text}")
    return values, {
        "url": url,
        "sha256": v2.sha256_file(target),
        "bytes": target.stat().st_size,
        "rows": len(values),
        "newest_date_added": max(values.values()).isoformat(),
        "source_commit": v2.KEV_COMMIT,
        "source_commit_committed_at": committed_at_text,
        "label_horizon_maturity_basis": "source commit timestamp postdates 2026-08-09 horizon end; max dateAdded is not used as a maturity proxy",
    }


v2.acquire_kev = acquire_kev

if __name__ == "__main__":
    v2.main()
