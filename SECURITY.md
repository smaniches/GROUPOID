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
- The older dependency closure required `torch` despite no implemented code
  path importing it. This pulled in vulnerable `filelock` and constrained
  `setuptools` below patched releases. The current source dependency closure
  removes that unused requirement and the corresponding CI installer. CI
  verifies the actual installed environment using `pip check` and a hard
  `pip-audit` gate without CVE/PYSEC ignore flags. This does not retroactively
  change any published package version, and it does not establish that other
  environments or future dependency updates are vulnerability-free.
