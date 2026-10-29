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
- The pinned CI security job separately documents and permits the legacy
  `torch` CVE-2025-3000 exception for code paths GROUPOID does not exercise.
  `pip-audit` also cannot fully inspect the `torch` CPU wheel with a local
  `+cpu` version identifier, so a passing job does not certify PyTorch.
- A fresh October 29 security run demonstrated that torch 2.15.0+cpu installs
  `filelock 4.2.4` and declares `setuptools>=77.0.3`. The old project CI
  pin `setuptools<82` was not imposed by PyTorch and was removed. Security CI
  now requires setuptools 84.1.0 or newer to address known advisories.
- `filelock>=4.2.1` addresses the October 2026 file-lock advisories on
  Python 3.11+. That patched release line requires Python 3.11 or newer.
  Python 3.10 became upstream end-of-life on 2026-10-01, and the package
  currently still declares 3.10 support. Its transitive filelock dependency
  cannot be described as fully patched. Do not use the Python 3.10 runtime
  in untrusted or multi-user environments; a supported-Python-only policy
  remains a separate compatibility and release decision.
