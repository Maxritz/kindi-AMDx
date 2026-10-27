# Shard B note — RPW=8 GEMV + rmsfold-norm-fusion evaluation

## Files created
- `shaders/g5ptq_bytenr_rpw8.comp` — copy of `shaders/g5ptq_bytenr.comp` with the
  `RPW>=4` pattern extended to `RPW>=8` (`r4..r7`, `k4..k7`, `acc4..acc7`,
  `b54..b57`, `sc4..sc7`, `p4..p7`). With `-DRPW<8` the new blocks vanish and
  the shader is behaviourally identical to bytenr. Defaults to RPW=8 when
  compiled without -D, so build.ps1's generic shaders\*.comp loop emits a
  usable g5ptq_bytenr_rpw8.comp.spv on its own (I compiled and then deleted the
  local artifact to keep the tree clean; rebuild regenerates it).
- This file (`SHARDB_NOTE.md`).

Untouched per constraints: `shaders/g5ptq_bytenr.comp`, `shaders/rmsfold.comp`,
`kindi_vk_decode.cpp`, `build.ps1`.

## Compile log
```
glslangValidator.exe -V --target-env vulkan1.1 "-DRPW=8" \
  -o shaders\g5ptq_bytenr_rpw8.comp.spv shaders\g5ptq_bytenr_rpw8.comp
exit=0
```
Output: `g5ptq_bytenr_rpw8.comp.spv`, 21508 bytes
(reference: bytenr RPW=4 compiles to 14484 bytes; the ~7 KB delta is the
unrolled second row-quartet — expected, no new bindings/push layout).

## P1: RPW=8 shared-memory math (feasible)
`shared float sh[NSUB][RPW][32]` with NSUB=8 (WGS=256, subgroup pinned to 32):
- RPW=8: 8*8*32 = 2048 floats = **8192 B (8.0 KiB)**
- `tlf[1280]` = 1280 floats = **5120 B (5.0 KiB)**
- Total static shared ≈ **13312 B ≈ 13.0 KiB << 48 KiB budget** (27% used).

Shared memory is NOT the blocker. (Note: the reduction already uses
`subgroupAdd`, so `sh` is likely dead-code-eliminated and the live footprint
is just the 5 KiB `tlf` — either way far under budget.)

Register/live-value pressure is the real A/B question for the host:
- RPW=4 live set: acc0..3, b50..b53, sc0..sc3, p0..p3 (~16 scalars).
- RPW=8 live set doubles to ~28 scalars (acc4..7, b54..57, sc4..7, p4..7)
  plus `xv` and loop state, all held across the `ntb`-iteration weight loop.
- Expected effect: halves workgroup count per GEMV (`per = wg*RPW` doubles,
  dispatches halve), quarters x-traffic per MAC, but raises per-thread VGPR
  footprint, which can reduce EU occupancy on the Meteor Lake iGPU. Bit-identity
  holds by construction (each row gets exactly its single-row FMA sequence; only
  the sharing of each `x` load widens), so this is safe to A/B on correctness —
  gate on measured tok/s, not on theory.

Host wiring hint (Shard C, not done here): pipeline select mirrors
`gemv()`'s `g5rpw` branch — `Pipe* P = (g5rpw==8) ? &pg5nr8 : ...`, grid
`(rows + wg*8 - 1)/(wg*8)`, same 3 bindings + 8 B push. No shader-side change
needed beyond `-DRPW=8` at compile time.

## P0-eval: fuse rmsfold norm into GEMV — verdict: NOT FEASIBLE without a full rewrite
One-line reason: **the norm result is a global reduction consumed by every GEMV
workgroup, so "fusing" it means either recomputing it in each workgroup (more
work than the dispatch it saves) or broadcasting it through global memory
(which is just the existing dispatch + barrier under another name).**

Detail:
- `rmsfold.comp`: 512-thread workgroups (`local_size_x=512`), grid `C/512`;
  per-chunk RMS over the full C vector (512-lane strided accumulation →
  `wsum_d` XOR reduction → `red[16]` → second XOR stage → `bscale`), then
  weight/scale multiply, then a 9-stage FWHT with a `barrier()` per stage on
  `xs[4096]` (cross-lane butterfly, `j<256` active).
- `g5ptq_bytenr.comp`: 256-thread workgroups (`WGS=256`, NSUB=8 subgroups),
  grid `R/per`; row-parallel GEMV with lane-strided weight fetch and a single
  `barrier()` (for the `tlf` trit table); reduction via `subgroupAdd`.
- The geometries are incompatible: FWHT needs 512 cooperating lanes and 9
  barriers *before any GEMV input is valid*, while GEMV needs its input
  (`xp`) replicated to *all* workgroups. A fused kernel serialises into
  norm-phase (most lanes idle during the tail of the reduction + all 9 FWHT
  stages) then GEMV-phase — and must still publish `xp` globally for the
  sibling projections, because the FFN consumer is `gemv3n` (gate+up in ONE
  launch): fusing the norm into only one sibling leaves the other unfused,
  and fusing it into the joint launch duplicates the whole norm+FWHT inside
  an already register-heavy kernel.
- The cheaper, already-proven lever is the existing sibling fusion
  (`gemv3n` gate+up in one launch, `gemv3` on the 4-trit path), which removes
  the inter-GEMV barriers without touching the norm's reduction geometry.

Recommendation: do NOT pursue norm-into-GEMV fusion. If the 1 norm dispatch
per FFN must go, the viable direction is norm-into-`smfold` (both are
C/element-parallel over the same vector, same grid shape) — still a rewrite,
but at least the parallelism matches. Out of scope for this shard.
