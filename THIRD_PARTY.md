# Third-Party Attributions

trellis2.cpp is MIT (see [`LICENSE`](LICENSE)). This file lists every
external component the project copies, links, or reproduces, and under
which terms.

Three categories, kept apart on purpose:

1. **Vendored source** — foreign code physically present in this
   repository under `third_party/`.
2. **External build dependencies** — resolved at configure time (vcpkg
   or distro package), never copied into the repository.
3. **Reproduced behaviour** — no foreign source was adopted; only the
   observable behaviour, data layout, or numerical result was
   reimplemented under our own design, per the license rules in
   [`AGENTS.md`](AGENTS.md).

---

## 1. Vendored source (`third_party/`)

All of these are MIT or public domain and are compiled into
`libtrellis2` unconditionally. Each upstream license notice travels
inside the vendored file itself; the files are never reformatted
(`format_code.sh` excludes `third_party/`).

### xatlas

- **Project:** <https://github.com/jpcy/xatlas>
- **License:** MIT License, Copyright (c) 2018-2020 Jonathan Young
- **Vendored:** `third_party/xatlas/xatlas.{h,cpp}` — complete, unmodified.
- **Used by:** `mesh_export.cpp` — UV unwrap for the projected/print
  bake, and the opt-in ordinary atlas path behind `T2GLB_XATLAS`.
- **Notice:** MIT text at the top of `xatlas.h`.

### meshoptimizer

- **Project:** <https://github.com/zeux/meshoptimizer>
- **License:** MIT License, Copyright (c) 2016-2026 Arseny Kapoulkine
- **Version:** 1.2 (`MESHOPTIMIZER_VERSION 1020`)
- **Vendored:** a **subset**, not the whole library —
  `meshoptimizer.h`, `allocator.cpp`, `indexgenerator.cpp`,
  `simplifier.cpp`, `vfetchoptimizer.cpp`.
- **Used by:** `mesh_export.cpp` — index generation, vertex-fetch
  optimisation, and decimation on the export path.
- **Notice:** MIT text at the end of `meshoptimizer.h`.

### stb

- **Project:** <https://github.com/nothings/stb> — Sean Barrett
- **License:** dual-licensed, public domain (Unlicense) **or** MIT
- **Vendored:** `stb_image.h` (v2.30), `stb_image_write.h` (v1.16).
- **Used by:** image decode on the untrusted upload path
  (`trellis2_preprocess_rgba`, the fuzz targets) and debug image dumps.
- **Notice:** license block at the end of each header.

---

## 2. Submodule

### ggml

- **Project:** <https://github.com/ggml-org/ggml>, tracked through the
  fork <https://github.com/RobertBeckebans/ggml>
- **License:** MIT License, Copyright (c) 2023-2024 The ggml authors
- **Integration:** Git submodule at `ggml/`, built via
  `add_subdirectory(ggml)`. Never reformatted. A submodule pointer
  update is a deliberate, separately reviewed decision — see
  [`AGENTS.md`](AGENTS.md).
- **Alternative:** `-DTRELLIS2_USE_EXTERNAL_GGML=ON` consumes a system
  ggml through `find_package` instead.

---

## 3. External build dependencies

Not present in this repository. Resolved at configure time — on Windows
through vcpkg (`-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`,
as in `cmake-ninja-win64-rocm-cgal.bat`), on Linux through distro
packages or the containers in `docker/`.

### CGAL — optional, **copyleft**

- **Project:** <https://www.cgal.org/>
- **License:** CGAL is split-licensed. The **3D Alpha Wrapping**
  package used here is **GPL-3.0-or-later**, or available under a
  commercial CGAL license.
- **Used by:** `print_remesh.cpp` — `alpha_wrap()` and, currently, the
  closest-surface `project_pbr()` AABB tree.
- **Encapsulation:** strictly behind `TRELLIS2_CGAL` /
  `TRELLIS2_USE_CGAL=1`. No CGAL header or type appears in
  `trellis2.h`, `trellis2_capi.h`, `mesh_export.h`, or in
  unconditionally compiled code. A build without CGAL retains no CGAL
  dependency and reports the feature as unavailable at runtime.
- **Consequence:** binaries built with `TRELLIS2_CGAL=ON` **and** CGAL
  detected are subject to GPL-3.0-or-later. Default builds are not.

### Transitive dependencies of the CGAL path

Pulled in by the vcpkg `cgal` port, therefore only present in a CGAL
build:

- **Boost** — Boost Software License 1.0 (permissive).
- **GMP** — dual LGPL-3.0-or-later / GPL-2.0-or-later.
- **MPFR** — LGPL-3.0-or-later.

The LGPL components are consumed as unmodified shared libraries.

### Eigen — planned, optional

- **Project:** <https://gitlab.com/libeigen/eigen>
- **License:** **MPL-2.0** (some bundled files under BSD or other
  MPL2-compatible terms; see `COPYING.README` upstream). Confirmed by
  vcpkg, which reports `MPL-2.0` for the `eigen3` port.
- **Version:** 5.0.1 is the known-good version — it is what upstream
  AutoRemesher bundles and what `vcpkg install eigen3` currently
  resolves to.
- **Planned use:** required by the AutoRemesher quad stage
  (see [`docs/plan/autoremesher-quad-remesh.md`](docs/plan/autoremesher-quad-remesh.md)).
- **Deliberately not vendored.** Obtained via `vcpkg install eigen3`
  (or a distro package) and consumed through
  `find_package(Eigen3 3.4...<6 CONFIG)` → `Eigen3::Eigen`, so no MPL2
  file enters this MIT repository. A version *range* is required:
  Eigen's `Eigen3ConfigVersion.cmake` treats a single requested version
  as an exact component match, so `find_package(Eigen3 3.4)` fails
  against 5.0.1.
- **Obligation:** Eigen is used unmodified. MPL-2.0 is file-level
  copyleft and explicitly permits combination with a differently
  licensed Larger Work (§3.3); the remaining duty is to retain this
  notice and keep the source obtainable, which the project URL above
  discharges.

---

## 4. Reproduced behaviour — no source adopted

Per [`AGENTS.md`](AGENTS.md): behaviour, tensor layouts,
hyperparameters, and numerical results are reproduced; source text is
not.

### TRELLIS.2 reference pipeline

- **Relationship:** trellis2.cpp is a from-scratch C++/ggml
  implementation. Stage structure, tensor layouts, hyperparameters and
  numerical results are reproduced so that the taps in
  [`docs/VERIFICATION.md`](docs/VERIFICATION.md) match the PyTorch
  reference — **no reference source is adopted**, including where its
  license would be incompatible with MIT.
- **Weights** are separate artefacts with their own terms; see the
  model links and license notes in [`README.md`](README.md). The DINOv3
  weights in particular are under the DINOv3 License, not MIT.

### Pillow — Lanczos resampling

- **Project:** <https://python-pillow.org/>
- **Relationship:** `trellis2_preprocess_rgba()` reproduces PIL's
  Lanczos-512 resampling **byte-exactly**, as an independent
  implementation of the documented filter and its coefficient
  generation. No Pillow source is present in this repository.

---

## Planned additions

The following are proposed but **not yet imported**. They become real
entries above only once the corresponding phase is implemented and
reviewed — see
[`docs/plan/autoremesher-quad-remesh.md`](docs/plan/autoremesher-quad-remesh.md).

| Component | License | Route | Purpose |
|---|---|---|---|
| [tinybvh](https://github.com/jbikker/tinybvh) — Jacco Bikker | MIT | vendored, `third_party/tinybvh/` | BVH builder for the CGAL-free closest-surface PBR transfer |
| [AutoRemesher](https://github.com/huxingyi/autoremesher) core — Dust3D Project / Jeremy HU | MIT (since 1.0.0) | vendored, `third_party/autoremesher/` | quad remeshing for the mid-poly export path |
| isotropicremesher — Jeremy HU | MIT | vendored with the above | required by the AutoRemesher core |
| [Eigen](https://eigen.tuxfamily.org/) | MPL-2.0 | external, vcpkg / distro | sparse solvers inside AutoRemesher — see §3 |
