# Security Policy

## Scope

GROUPOID is a research prototype. It is not designed for deployment in
security-sensitive environments.

## Reporting a vulnerability

If you discover a security issue, please report it by emailing
santiago.maniches@gmail.com. Do not open a public issue.

## Supported versions

Only the latest commit on `main` is supported. There are no stable
releases.

## Known limitations

- No differential privacy is implemented despite optional dependencies
  being listed. Do not rely on this software for privacy guarantees.
- Input validation is minimal. The library assumes trusted inputs from
  the caller.
- The `torch` runtime dependency is flagged by `pip-audit` for advisory
  CVE-2025-3000. OSV expresses its affected range as a Git/commit range whose
  listed versions run up to v2.6.0, with no published `fixed` release. Because
  that upper bound is a source commit rather than a resolvable version,
  `pip-audit` conservatively flags the installed `torch` (currently 2.12.0, past
  the listed v2.6.0 affected range) instead of clearing it. CI tracks the
  advisory via a documented `--ignore-vuln`; there is no fixed release to
  upgrade to, and the ignore is revisited when upstream publishes one. The
  upstream report concerns `torch.jit.script` memory corruption, not a general
  claim about every `torch.load` call. GROUPOID does not call `torch.jit.script`.
  This exception is not a statement that arbitrary PyTorch inputs are safe.
- The CI-pinned `setuptools` (held to `>=78.1.1,<82` because `torch` requires
  `setuptools<82`) is flagged by `pip-audit` for advisory PYSEC-2026-3447
  (CVE-2026-59890): prior to setuptools 83.0.0, `MANIFEST.in` exclude,
  `global-exclude`, `recursive-exclude`, and `prune` directives could be
  bypassed on macOS APFS/HFS+ by NFD-normalized file names, packing excluded
  files into source distributions. The fixed release (83.0.0) is outside the
  window torch permits, and the affected code path requires building an sdist
  on a macOS filesystem — all GROUPOID CI and release builds run on
  `ubuntu-latest`. CI tracks the advisory via a documented `--ignore-vuln`;
  the ignore is revisited once torch allows `setuptools>=83`.

## FileLock compatibility constraint (2026-10-23)

The fresh PyTorch environment resolved FileLock 3.32.3, affected by CVE-2026-112940. The package now requires 3.20.3, which predates the affected SQLite locking feature and supports Python 3.10. This is a dependency constraint, not an advisory suppression. Revisit when the supported PyTorch range permits patched FileLock >=4.2.1. Reachability of the vulnerable SQLite lock path from GROUPOID was not established. Existing scanner exceptions above are separate and are not blanket security approval.

Advisory: https://github.com/tox-dev/filelock/security/advisories/GHSA-v4x8-3wr5-3wq8
