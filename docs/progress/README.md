# Progress documentation (`docs/progress/`)

After completing a larger plan phase, an entry is created here.
While the plan under `docs/plan/` captures the **intent**, the progress
documentation captures the **result**: what was actually implemented,
which files were affected, which checks ran, what deviated, and what
remains open.

Since AI agents do not commit, progress entries initially contain
`Commits: <to be added after review>`. The reviewer adds commit hashes
after review and commit as needed.

## Naming convention

When a GitHub issue exists, the issue number is used:

```text
docs/progress/<github-issue>_<phase>-<short-name>.md
```

- `<github-issue>` = same GitHub issue number as in the plan.
- `<phase>` = number or key of the plan phase.
- `<short-name>` = short slug, usually the same as for the plan.
- Example: issue `#17`, phase 2 →
  `docs/progress/17_2-<feature-slug>.md`.

When no GitHub issue exists yet but an existing plan or roadmap section
is affected, the plan key is used:

```text
docs/progress/<plan-key>_<phase>-<short-name>.md
```

- `<plan-key>` = plan file name in lowercase without `.md`, e.g.
  `renderer`, `network`, `tooling`.
- `<phase>` = phase, sub-phase, or section from the plan.
- Example: `docs/plan/RENDERER.md`, phase 4 clustering →
  `docs/progress/renderer_4-clustering.md`.

For larger sub-phases the phase can be extended:

```text
docs/progress/renderer_4-1-cluster-binning.md
docs/progress/renderer_4-2-cluster-shading.md
```

## Workflow

1. Implement the phase from the plan.
2. Run the appropriate checks / smoke tests, as far as possible.
3. Copy the progress entry from [`_TEMPLATE.md`](_TEMPLATE.md) and
   name it: use the issue number if present; otherwise use plan key
   and phase.
4. Document the implementation, deviations, verification, open points,
   and next phase.
5. Set the plan checkbox and update the status if needed.
6. Do not commit. List the changes and propose an English commit message
   for review.

## Purpose

Progress documentation makes later technical decisions traceable —
for humans and for new AI sessions that should not work from chat
memory.