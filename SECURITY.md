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
