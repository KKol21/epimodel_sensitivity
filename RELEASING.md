# Releasing EMSA

Merge into `main` with `[bump] [publish]` in the merge commit message, for example:

```text
Finalize EMSA [bump] [publish]
```

CI runs the checks, bumps the patch version with `bumpversion`, builds and tests
the package, then commits the version changes, pushes the `vX.Y.Z` tag and publishes
the tested files to PyPI. All version references update automatically.

For a minor or major release, run `bumpversion minor` or `bumpversion major` on
`main`, then push the commit and tag with `git push origin main --follow-tags`.
Tag pushes run the same checks and publish without another bump.

One-time setup: configure PyPI's Trusted Publisher for this repository, workflow
`cicd.yml` and environment `pypi`. Normal commits, PRs and manual CI runs do not publish.
