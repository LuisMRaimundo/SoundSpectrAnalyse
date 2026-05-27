# Security and provenance notes

This document describes the trust model of the `SoundSpectrAnalyse` installer
subsystem and the cryptographic verification performed on downloaded artefacts.
It is intended for users who will cite analyses produced by this tool in
published research, for thesis examiners, and for future maintainers.

## 1. What the installer downloads

On first run, `installers/common/bootstrap.py` downloads three categories of
external artefact over HTTPS:

1. **A Python Build Standalone (PBS) tarball** from
   `github.com/astral-sh/python-build-standalone` — the official source for
   portable CPython builds.
2. **A Windows embeddable Python ZIP** from `www.python.org` — the official
   source for CPython.
3. **The PyPA `get-pip.py` bootstrap script** from `bootstrap.pypa.io` — the
   official source for `pip` installation.

The installer then runs `pip install -r requirements.txt` to install the
project's runtime dependencies from PyPI.

## 2. What is cryptographically verified

| Artefact | Verification mechanism | Authority |
|---|---|---|
| PBS tarball | SHA-256, fetched from the upstream `.sha256` file co-located with each release asset (e.g. `<tarball>.sha256`), compared against the local digest after download | astral-sh/python-build-standalone release |
| Windows embeddable Python ZIP | SHA-256, hardcoded in `installers/common/config.py` as `WIN_EMBED_ZIP_SHA256`, sourced from the python.org release for the pinned `PYTHON_VERSION` | python.org |
| get-pip.py | SHA-256, hardcoded in `installers/common/config.py` as `GET_PIP_PY_SHA256`. The hash was computed locally by the project maintainer at the time of pinning (see `GET_PIP_PY_PINNED_AT`), from a download of `https://bootstrap.pypa.io/pip/get-pip.py`. The URL is pinned to `/pip/get-pip.py`, but PyPA may update the file at this path over time; the pinned hash therefore freezes the artefact to a specific point in time chosen by the maintainer. Self-certified by the project maintainer, not upstream-certified. | Project maintainer (artefact-of-record from PyPA at `GET_PIP_PY_PINNED_AT`) |

A mismatch at any of these three points causes the installer to abort with an
explicit error and refuse to execute the affected artefact.

## 3. What is NOT cryptographically verified

The following remain trust-on-HTTPS:

1. **PyPI package downloads** triggered by `pip install -r requirements.txt`.
   The project's `requirements.txt` does not use `--hash=sha256:...` pinning.
   Full pip hash-pinning is a separate, larger change that has not been
   performed; see §4.
2. **The repository contents themselves**, beyond what GitHub already provides
   over HTTPS. A specific Git revision is not pinned by the installer at
   install time, and a `RUNTIME_PROVENANCE.txt` file is not written.

## 4. Roadmap for stronger reproducibility

For users who require archival-grade reproducibility (e.g. when an analysis
will be cited in a published article or doctoral thesis), the following
additional measures should be performed in a separate task:

1. Regenerate `requirements.txt` with full transitive hash pinning, for example
   using `pip-compile --generate-hashes` (`pip-tools`) or `uv pip compile
   --generate-hashes`. Commit the regenerated file and add
   `--require-hashes` to the `pip install` invocation in `bootstrap.py`.
2. At install time, record the Git revision installed (commit SHA + tag, if
   any) into a `RUNTIME_PROVENANCE.txt` file beside the runtime, and have the
   GUI display the runtime revision in an "About" panel.
3. Tag the repository (`v0.x.y-installer-hardened`) and direct cited analyses
   to use a specific tag rather than `main`.

## 5. Reporting a security concern

Please report security concerns through a private channel (a GitHub Security
Advisory on `LuisMRaimundo/SoundSpectrAnalyse`, or direct email to the
maintainer). Do not open a public issue for unpatched vulnerabilities.

## 6. Versions pinned at the time of writing

| Constant | Value |
|---|---|
| `PYTHON_VERSION` | `3.11.9` |
| `PBS_TAG` | `20240415` |
| `GET_PIP_PY_PINNED_AT` | `2026-05-27` |

Changing any of these constants requires updating the corresponding hash
constants in the same commit.
