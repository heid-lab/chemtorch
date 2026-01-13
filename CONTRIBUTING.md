# Contributing to ChemTorch

Thanks for your interest in improving ChemTorch! This document explains how to get set up, propose changes, and collaborate effectively with the core team.
Please review this guide before submitting an issue or pull request.

## Ways to contribute
- Report bugs and unexpected behavior using the issue templates.
- Improve documentation: tutorials, reference pages, diagrams, or docstrings.
- Enhance coverage: add datasets, benchmarks, reaction representations, or model components.
- Optimize workflows: improve logging, performance, reproducibility, extensibility.

<!-- We welcome contributions from both research and industry users. -->

## Code of conduct
ChemTorch follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold these standards. Please report unacceptable behavior to the maintainers.

## Before you start
- Search existing [issues](https://github.com/heid-lab/chemtorch/issues) and discussions to avoid duplicates.
- For significant changes (new modules, major refactors), open an issue or discussion first so we can align on scope and design.
- Branch from the latest `main` (we follow a trunk-based development workflow: https://trunkbaseddevelopment.com/).

### Branch naming policy
Use lowercase, kebab-cased branch names with a short, meaningful prefix:

Allowed prefixes:
- `feature/` — new functionality or capabilities (e.g., `feature/reaction-core-splitter`).
- `bugfix/` — fixes for reported defects (e.g., `bugfix/dataloader-deadlock`).
- `hotfix/` — urgent, targeted fixes (often for a quick PATCH release).
- `refactor/` — internal restructuring without user-visible changes.
- `docs/` — documentation-only changes (guides, API docs, tutorials).
- `chore/` — repo maintenance (formatting, lint config, non-functional cleanup).
- `ci/` — CI, workflows, or build-automation changes.
- `tests/` — add/modify tests.
- `perf/` — performance optimizations.
- `build/` — packaging or build-system updates.

Exceptions: automated branches like `dependabot/**` and short-lived `release/**` freeze branches.

A CI job validates branch names on pull requests; nonconforming names will fail the check.

### Commit message policy
We prefix the commit message with a [git commit emoji](https://gist.github.com/parmentf/035de27d6ed1dce0b36a) to capture at a glance what the commit is about.


## Development environment
Follow the [official ChemTorch installation instructions](https://heid-lab.github.io/chemtorch/getting_started/quick_start.html) and see the detailed installation instructions for installing additional developer dependencies.

### Optional: dataset checkout
To run the quick-start experiments locally, clone the [reaction database](https://github.com/heid-lab/reaction_database) into `data/`:
```bash
git clone https://github.com/heid-lab/reaction_database.git data
```

## Running tests
- Unit tests: `uv run pytest`
- Integration tests (may require datasets/GPU): `uv run pytest -m integration`
- Extended or benchmarking suites: `uv run pytest -m extended`

Refer to the [Integrity & Reproducibility guide](https://heid-lab.github.io/chemtorch/advanced_guide/integration_tests.html) for setting up dataset integrity checks and long-running regression tests. Please run the relevant suites before opening a pull request.

## Documentation changes
- Build docs locally with `uv run make -C docs html`. Generated pages live in `docs/build/html`.
- Keep the landing page and README in sync when adding or renaming tutorials or examples.
- For figures or diagrams, place source assets under `docs/source/_static/` when possible.

## Pull request checklist
- Keep changes focused; unrelated changes should be in separate PRs.
- Give the PR a clear but concise title. The title will appear in the automaitcally generated release notes.
- Provide a short summary of the problem, the solution, and any follow-up work.
- Add suitable label(s). To keep release notes accurate, pull requests must include at least one category label (e.g., `feature`, `bug`, `documentation`, `maintenance`, `tests`, `ci`) and add `breaking-change` when applicable.
- Update or add tests covering new behavior and ensure ALL tests pass.
- Update configuration defaults where relevant.
- Update/add documentation if applicable.

Maintainers may request additional context, benchmarks, or documentation updates before merging.

### Review routing
GitHub automatically requests maintainers for changes in core paths. If your PR touches unfamiliar areas, add additional reviewers for faster feedback.

### Merging policy
Merges into `main` are performed by maintainers. "Rebase and merge" is disabled on `main`; maintainers choose between squash and merge commits per the internal policy (see `MAINTAINERS.md`). As a contributor, you only need to prepare your branch:

Contributor workflow before merging:
1. Fetch latest `main`: `git fetch origin main && git checkout main && git pull`.
2. Rebase your branch: `git checkout feature/xyz && git rebase main` (resolve conflicts, amend tests).
3. Push your branch updates:
	 - If you rebased locally, update the remote branch history with a safe force push to your feature branch: `git push --force-with-lease`. (This applies to your branch only; force push to `main` is disabled for all contributors.)
	 - Prefer to avoid force pushes? Merge `main` into your branch instead of rebasing: `git merge main && git push`. This keeps history non-rewritten and is fine since the PR will be squashed or merged on `main`.
4. Ensure CI passes; reviewer decides squash vs merge commit.

Note on squashed PRs: include a clear PR title & description (used in release notes) since intermediate commit messages are discarded.

Large refactor tip: consider temporary `release/` or `refactor/` branch only if coordination across many PRs is needed; otherwise keep trunk small and frequent.

## Need help?
- Open a documentation improvement issue if instructions are unclear.
- Start a GitHub Discussion for architectural proposals or research collaborations.
- Reach out via the contact details listed in the README for questions that cannot be handled publicly.

We appreciate your time and contributions—thank you for helping make ChemTorch better!
