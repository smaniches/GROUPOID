#!/usr/bin/env bash
# Installs CPU-only torch from the PyTorch index with retries.
#
# That index (download.pytorch.org/whl/cpu, Cloudflare R2-backed as
# download-r2.pytorch.org) intermittently drops the TLS handshake
# (SSLV3_ALERT_HANDSHAKE_FAILURE) from GitHub-hosted runners. pip's own
# built-in retries reuse the same connection pool within one process and
# do not help here; a fresh `pip install` invocation gets a fresh TLS
# connection and has in practice succeeded where the retries within a
# single invocation did not. Five attempts with linear backoff (10s, 20s,
# 30s, 40s) before giving up.
set -euo pipefail

max_attempts=5
attempt=1
until pip install torch --index-url https://download.pytorch.org/whl/cpu; do
  if [ "${attempt}" -ge "${max_attempts}" ]; then
    echo "::error::torch install failed after ${max_attempts} attempts" >&2
    exit 1
  fi
  sleep_seconds=$((attempt * 10))
  echo "torch install attempt ${attempt}/${max_attempts} failed; retrying in ${sleep_seconds}s" >&2
  sleep "${sleep_seconds}"
  attempt=$((attempt + 1))
done
