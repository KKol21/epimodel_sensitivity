# Development and releases

Use Python 3.10 or 3.11 with the current dependency versions. The CI matrix covers
Linux on both Python versions and Windows on Python 3.11. Runtime, test, notebook,
documentation and development dependencies are declared in `pyproject.toml`.

## Local checks

From the repository root, preferably inside a virtual environment:

```sh
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip check
python -m black --check emsa emsa_examples tests scripts setup.py
python -m flake8 emsa emsa_examples tests scripts setup.py --ignore E501,W503,F401
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

`python -m build` builds the source distribution, then builds the wheel from that
source distribution. The wheel contains the `emsa` library and license. The source
distribution also includes the example configurations/data, tests, notebook and
installed-package smoke script. Build outputs and local results are ignored by Git.

## Installed-wheel check

Create a **new** virtual environment, install CPU PyTorch and the wheel from `dist/`,
then run `scripts/smoke_installed.py` with that environment's Python from a directory
outside this checkout. The script verifies the installed import location, license
metadata, a trajectory and an in-memory seeded sensitivity analysis. It uses no
example files or development dependencies. Run `python -m pip check` there too.

The `wheel-smoke` CI job does this on Linux with Python 3.10 and 3.11, using the exact
wheel produced by the build job. Tests and lint must pass before building; both
wheel checks must pass before publishing.

## Publish a version

1. Update `project.version` in `pyproject.toml`, the version in `.bumpversion.cfg`,
   and `release` in `docs/source/conf.py`. The existing bumpversion configuration
   points at these metadata locations. Use a version not already published on PyPI.
2. Review and commit the release changes, merge them into `main`, and let CI pass.
3. Push a matching tag, for example `v1.0.4` for package version `1.0.4`.
4. The tag starts fresh checks on that exact commit. A mismatched tag/version fails.
   Publishing downloads the tested build artifact without rebuilding or changing
   the source commit.

PyPI must have a [Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
configured for this repository, workflow `cicd.yml`, and environment `pypi`.
Repository environment approval rules, if configured, still apply. Branch pushes,
pull requests and manual check runs do not publish packages. Commit-message markers
such as `[bump]` and `[publish]` no longer trigger releases.

## Numerical checks

The test suite compares SIR, SEIHR, contact and vaccination trajectories with SciPy
integration of explicit compartment equations, and checks that Euler error decreases
as its step is halved. The vaccination reference uses the normal ninety-day campaign
and integrates its start/end intervals separately. These are representative checks,
not a guarantee for every user-supplied model or parameter range.

Euler's one-day default is retained for compatibility. Choose the step size for your
model by checking convergence; coarse steps can have substantial trajectory error
even when populations stay nonnegative. The example accuracy checks use Dopri5 with
a quarter-day maximum step and tighter tolerances. For time-dependent interventions,
choose a time resolution appropriate to the intervention duration and verify it.
