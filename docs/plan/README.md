# Plans (`docs/plan/`)

For larger features, bugs with unclear code path, and architecture
changes, a plan document lives here. The plan captures the **intent**:
goal, acceptance criteria, affected files, decisions, phases, tests,
and risks.

When an existing long-term roadmap plan already covers the affected
section, no artificial new issue needs to be created. Progress entries
can then be named via plan key and phase, e.g.
`docs/progress/<plan-key>_<phase>-<short-name>.md`.

Small trivial changes like typos or obvious one-line fixes do not need
a plan.

## Naming convention

```text
docs/plan/<github-issue>-<short-name>.md
```

- `<github-issue>` = issue number from
  `https://github.com/<org>/<repo>/issues`, e.g. `17`.
- `<short-name>` = short, descriptive slug in `kebab-case`, preferably
  an English or established project term.
- Example: issue `#17` "<Feature title>" →
  `docs/plan/17-<feature-slug>.md`.

The optional Git branch starts with the same issue number:
`17-<feature-slug>`.

## Workflow

1. Open or select a GitHub issue and note the issue number.
2. Copy the plan from [`_TEMPLATE.md`](_TEMPLATE.md) and name it.
3. Align goal, acceptance criteria, and phases.
4. Implement phase by phase — per larger phase: implement, verify,
   prepare progress documentation, and set the plan checkbox.
5. Do not commit. Only prepare changes for review and propose an
   English commit message matching the existing Git history.

## Status values

`planned` → `in progress` → `review` → `done`

## When a plan is required

| Change | Plan needed? |
|---|---|
| Typo / small doc fix | no |
| Obvious small code fix | usually no |
| New public API / new module / new data structure | yes |
| Changes to performance-, security-, or compatibility-relevant paths | yes |
| Data format / pipeline change with migration effort | yes |
| Bug without clear code path | yes |