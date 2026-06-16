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
  advisory concerns crafted-checkpoint / JIT deserialization (`torch.load` and
  related paths), which GROUPOID's own code does not exercise.
