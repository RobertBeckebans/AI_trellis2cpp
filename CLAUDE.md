# CLAUDE.md

Project **trellis2.cpp** — C++/ggml port of the TRELLIS.2
Image-to-3D pipeline. The full guide for architecture, conventions,
commands, planning process, pitfalls, and review workflow lives in
[`AGENTS.md`](AGENTS.md) — please read it first and in full.

The most important rules in brief:

- AI agents **never commit themselves** and never push.
- Changes are only prepared so the reviewer can review them against the
  current Git state.
- For larger features, bugs with unclear code path, and architecture
  changes, first create a plan under `docs/plan/` or use the existing
  roadmap plan [`docs/PLAN.md`](docs/PLAN.md).
- Plans remain organized in phases. After completing larger phases,
  prepare a result doc under `docs/progress/`: with the GitHub issue
  number if present, otherwise with the plan key and phase, e.g.
  `plan_texture-stage.md`.
- Documentation in English; code comments, identifiers, Git logs, and
  proposed commit messages in English.
- The primary build is `cmake-ninja-win64-rocm.bat` (Windows, HIP/ROCm,
  Radeon AI PRO R9700 / `gfx1201`) — assume that backend unless the task
  says otherwise. It wipes `build/` first, so only run it on request.
- This is a downstream fork: the pipeline is upstream work by rms80 and
  Richard Palethorpe; this fork adds the Windows/HIP, Vulkan and DLL
  paths. Check `git log` before attributing a design decision.
- Formatting exclusively via `./format_code.sh` or `format_code.bat`
  with clang-format **18.1.8** — `ggml/` and `third_party/` are never
  reformatted.
- Numerical parity is binding: a stage change that worsens a tap in
  [`docs/VERIFICATION.md`](docs/VERIFICATION.md) is a bug. A `SKIP`
  (return code 77, missing fixtures) is not a passing test.
- Every change to `src/trellis2_capi.h` bumps `T2_CAPI_ABI_VERSION` and must
  be propagated in `server/engine.go`.
- License MIT. No code adoption from license-incompatible third-party
  projects (including the TRELLIS reference itself); behavior and
  formats must only be implemented in a license-safe manner under our own
  design. The CGAL path (GPL) stays encapsulated behind `TRELLIS2_CGAL`.
- No recursive delete commands without prior approval — `ggufs/`,
  `models/`, `dumps/`, `generations/`, `assets/` are gitignored and not
  recoverable.