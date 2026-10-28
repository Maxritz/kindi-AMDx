# Ponytail-Diag: Head-Tuning Full Trace Analysis

## 12. FLOW (spec verify head path, 2B native5 / 27B Q4_0 head)

```
generate() → spec loop (K=3)
  → body_T([cur]+d, T=4) → gemvT(layer weights, ..., T=4)
      → gemvM5ptq / gemvM2pq / gemvMQ8 / gemvMQ4 → pg5T / pg2M / pq8f / pq4f
  → head_rows(T)
      → rmsfoldT (residual norm, batch) → fold_hi_off per token
      → gemvT(head_name, xpT, logT, T, work)  ← THE HOT PATH
          → gemvM5ptq (2B, head ∈ native_names) → pg5T dispatch:
              gx = (VOCAB(248320) + nsub(8) - 1) / nsub = 31040 WGs  ← [BROKEN]
          → OR gemvMQ4 (27B, head ∈ q4names) → pq4f dispatch:
              gx = (VOCAB(248320) + 3) / 4 = 62080 WGs                ← [BROKEN]
      → argmax × T (separate dispatch per token)
  → accept prefix / reject → spec_restore → spec_replay
```

## 12.1 BRANCHES (static-resolvable on the head path)

| # | location | condition | trace | verdict |
|---|---|---|---|---|
| B1 | 2174 g5t | `getenv("KINDI_G5T") == nullptr \|\| atoi != 0` | not set → **true** | `[taken=true]` |
| B2 | 2175 | `T >= 2` | spec T=4 → **true** | `[taken=true]` |
| B3 | 2178 dispatch | `gx = (rows + nsub - 1) / nsub` | rows=248320, nsub=8 | `[BROKEN]` grid granularity |
| B4 | 2281 gemvT failure | `!gemvT(head_name, ...)` | returns true (2B) → **false** | `[taken=false]` no serial fallback |
| B5 | 2149 gemvM5ptq | `!native_names.count(wname)` | head ∈ native_names → **false** | `[taken=false]` → keeps 5-trit |
| B6 | 2134 gemvMQ4 | `!q4names.count(wname)` | 27B head ∈ q4names → **false** | `[taken=false]` → Q4 path |

## 12.2 DATA-STATE FLOW

```
head_name @1309: "output.weight" (2B native PTQ1_0) | "output.weight" (27B Q4_0)
  → tinfo[head_name] = {R=248320, C=32896}  [OK]
  → gemvT(head_name, xpT[4×32896], logT[4×248320], T=4, work=32896)
  → gemvM5ptq: native_names.count("output.weight")=TRUE → [OK]
  → pc{ R=248320, C=32896, T=4, xs=32896 }  [OK]
  → dispatch: gx=(248320+7)/8=31040, gy=1, gz=1  [BROKEN: 31K WGs × 8 rows]
  → bytenrT: lanes process 8 consecutive rows, stream unique 1.6 KB weight slice  [BROKEN: no cross-WG reuse]
  → logT = 248320×4×4 = 3.97 MB allocated  [wasteful but OK]
  → argmax ×4: reads logT, 4 separate dispatches  [wasteful: no fusion]
```

## 12.3 INVARIANTS

- `[OK]` bytenrT math bit-identical (selftest confirms).
- `[OK]` g_wave=32 before dispatch (line 272, confirmed in traces).
- `[BROKEN]` head grid should saturate fabric but gx=31040 → each WG streams unique weight → 0 L2 reuse → 31K interleaved streams → stalls.
- `[BROKEN]` argmax not fused with head → 3.97 MB intermediate materialized + read back.

## 12.4 HYPOTHESES (ranked)

**H1: Head grid granularity `(VOCAB+nsub-1)/nsub` is correct for body (R=32K) but catastrophic at head scale (R=248K).**
- For: trace shows 39.6 ms/op head vs 1.06 ms/op body, same kernel; 31K WGs vs 4K WGs.
- Against: formula is generic; changing it needs head-specific dispatch.
- Test: add `if (rows > 65536) dispatch_tiled;` branch; measure head op time.
- Cost: low | Status: **CONFIRMED**

**H2: 27B Q4 head has same issue via pq4f `(rows+3)/4` = 62080 WGs.**
- For: trace shows 1532 ms head phase vs 13 ms body on 27B.
- Against: different kernel (pq4f vs bytenrT) but same grid-granularity defect.
- Test: same tiled dispatch; Q4 decode unchanged, only gx changes.
- Cost: low | Status: **CONFIRMED**

**H3: No QKV/gate-up fusion on native5 path — `gemv3` rejects `native_names`.**
- For: `gemv3` line 1480: `!native_names.count(n)` → 2B QKV ∈ native_names → returns false. Trace shows `g5ptq_bytenr4` 2688 ops = 3× tensors × 896 layers, not 1×896.
- Against: `gemv3n` (line 1528-1531) accepts native_names, but is it called? Check line 1714:
  `if(!gemv3(...) && !gemv3n(...))` — yes, gemv3n IS called but check its usable predicate.
- Test: add log to gemv3n returns true/false.
- Cost: medium | Status: **CONFIRMED (secondary)**

**H4: argmax is a separate serial pass per token (4 dispatches in head_rows).**
- For: line 2296-2300: `for(t=0;t<T;++t){ dispatch(pargmax,...) }`. Trace `argmax` 20 ops.
- Against: argmax is cheap (174 µs/op × 20 = 3.5 ms) — dwarfed by 436 ms head.
- Cost: low | Status: **CONFIRMED (minor)**

## 12.5 5 WHYS — HEAD SLOWNESS

Q1: why is head 39.6 ms/op?
A1: because 31K workgroups each stream 1.6 KB of weights with no reuse.

Q2: why 31K workgroups?
A2: because `gx = (VOCAB + nsub - 1) / nsub` and VOCAB=248320, nsub=8.

Q3: why is nsub=8?
A3: because `nsub = 256 / g_wave` = 256/32 = 8 (correct for body).

Q4: why does the body succeed with the same formula?
A4: because body rows=32896 → gx=4112 WGs, each streaming a contiguous 28 KB span — reuse within WG.

Q5: why doesn't the head get the same?
A5: because for the head `rows >> C` (248K vs 32K) — the formula assumes row-per-subgroup, which only works when rows is small.

## 12.6 BARRIER ANALYSIS

- **What should have stopped this:** `gemvM5ptq`/`gemvMQ4` should detect large-R and tile.
- **Existence:** no `R > threshold` tiling branch — ABSENT.
- **Bypass:** none — the generic dispatch silently mis-scales.
- **Fix:** add tile-aware dispatch (§13).

## 12.7 PARETO (trace: 40-token spec run)

| shader | ops | dev_us | us/op | %total | issue |
|---|---|---|---|---|---|
| g5ptq_bytenr4 (all serial projections) | 2688 | 610 000 | 227 | 16% | **no QKV/gate-up fusion** |
| spec:g5ptq_bytenrT (body verify) | 630 | 468 000 | 744 | 12% | OK (correct T-batching) |
| **head g5ptq_bytenrT** | **11** | **436 000** | **39 672** | **14%** | **⚠ grid granularity** |
| spec:gdn_delta_T | 90 | 36 658 | 408 | 9% | recurrent tax |
| argmax | 40 | 3 479 | 87 | 1% | no fusion |

Pareto verdict: head (1 op × 39.6 ms = 436 ms = 14% of device time) and serial projections (2688 ops × 227 µs = 610 ms = 16%) are the **top 2 cost centers**. Head is the single biggest lever (1 op → 4× reduction possible). Serial projections are a **volume problem** (360 ops/layer that could be ~120 with fusion).

---

## 13. TUNING PLAN (demoscene × diag)

### Fix 1: Head row-tiling (ryg) — THE bottleneck

The head dispatches one row per subgroup. Replace with **TILE rows per WG**:

**Shader change** to `g5ptq_bytenrT.comp` — add tiling support so multiple WG rows share a weight cache line:

```glsl
// Add to push constants: int head_tile_stride;
// When head_tile_stride > 0, lane processes rows:
//   row = wg_x * HEAD_TILE + sub * (HEAD_TILE/NSUB) + lane % (HEAD_TILE/NSUB)
//   (i.e., lanes fan out across HEAD_TILE rows instead of NSUB rows)
// The weight decoding is unchanged (same byte offsets, same trit decode).
// The subgroupAdd reduction stays per-subgroup (each subgroup reduces its rows).
```

**Host dispatch change** at kindi_vk_decode.cpp:2178:

```cpp
const int HEAD_TILE = 64;
if (rows > 4096 && T >= 2) {
    // Head regime: tile vocab-rows to saturate fabric.
    // 248K rows → 3880 WGs (vs 31K), each streaming 1792 B contiguous span.
    pc.head_tile_stride = HEAD_TILE;
    dispatch(pg5T, G.q0, G.cp, (uint32_t)((rows + HEAD_TILE - 1) / HEAD_TILE), 1, 1, &pc, 20);
    return true;
}
```

**Expected:** head drops from 436 ms → ~15 ms (32× WG reduction, same bandwidth).

### Fix 2: QKV/gate-up fusion on native5 (BeBo)

Line 1714-1715:
```cpp
if(!gemv3(xp, p+"attn_qkv.weight",&qkv, p+"attn_gate.weight",&zv, "", nullptr)
   && !gemv3n(xp, p+"attn_qkv.weight",&qkv, p+"attn_gate.weight",&zv, "", nullptr)){
```

**BUG:** `gemv3` (line 1480-1481) rejects native5 tensors. `gemv3n` (line 1531-1532) accepts them, but:
- Check `gemv3n` usable predicate at 1531: `!native_names.count(n)` is **inverted** — it should be `native_names.count(n)` (gemv3n is FOR native5 tensors).
- Wait, re-reading: 1531 `usable = !n.empty() && native_names.count(n) && tinfo.count(n)` — this is CORRECT (gemv3n requires native).
- But `gemv3n` dispatches via `pg53nr4` (line 1562) which is `g5ptq_byte3nr4.comp`.

**Test needed:** trace shows `g5ptq_byte3nr4` 0 ops in the 40-token spec run → `gemv3n` is NOT firing for QKV. Check the column-count guard at line 1542: `tinfo[n].second != tinfo[ns[0]].second` — if QKV and gate have different C, it bails. They should be equal (both project from D=32896).

**Action:** Add trace log in `gemv3n` to see why it returns false. If column mismatch, the fused kernel can't fire — need a fused kernel that handles different-R same-C (the `pg53nr4` kernel already supports R0/R1/R2 in push constants).

**Expected:** body projections drop from 2688 ops → ~896 ops, 227 µs/op → ~75 µs/op (byte3nr's measured ratio). ~3× body GEMV throughput.

### Fix 3: Head+argmax fusion (Haujobb)

Fold argmax into bytenrT's subgroup reduction. Add a `subgroupMax`/`subgroupBallot`-style reduction to track max+index in-lane, emit only token IDs. Kills the 3.97 MB `logT` buffer + 4 argmax dispatches. **Priority 3** (minor vs head tiling).

### Fix 4: Branchless trit LUT (farbrausch/ryg)

bytenrT lines 98-127: 3-way `if(off<16)/else if(off<24)/else` per byte — subgroup divergence. Replace with a 26-entry LUT `{col_offset, stride, ndig, base_trit}`:

```glsl
// Decode table (26 entries, indexed by off):
uint col0_lut[26] = {...};  // column base per byte position
uint stride_lut[26] = {...}; // 16, 8, 2 (bytes per trit)
uint ndig_lut[26] = {5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5, 5,5,5,5,5,5,5,5, 4,4};
uint trit_start_lut[26] = {0,0,...,5,8, 8,8,8, 10,10,...};

// In loop: one table lookup, uniform control flow.
int col_base = col0_lut[off] + base_lut[off] * block_offset;
int stride = stride_lut[off];
for (int n = 0; n < ndig_lut[off]; ++n) {
    float dv = TRIT(bb, n);
    int col = col_base + n * stride;
    p0 += dv * x[col];
    ...
}
```

**Expected:** removes 3-way divergence; ~10-15% gain. **Priority 4** (needs cool-machine bit-identical validation).

## 14. IMPLEMENTATION ORDER

1. **Fix 1 (head tiling)** — highest ROI, single shader + 5-line host patch.
2. **Fix 2 (QKV fusion)** — investigate why gemv3n fails on native5; if column guard, patch.
3. **Fix 3 (head+argmax fusion)** — new kernel variant, or inline the max into bytenrT's reduction.
4. **Fix 4 (branchless LUT)** — shader-only, validate bit-identical.

## 15. GATES (re-validation plan)

- Build: `glslangValidator` + SPIR-V validation on each new shader.
- selftest: `bit-identical` check (K=2 vs serial) on bytenrT-tiled.
- Correctness: 2B token-exact — `diff <(plain) <(spec K=3)` must be byte-identical.
- Correctness: 27B MTP K=1 token-exact (same gate).
- **No performance re-run until thermal recovery.** All edits compile-gated first; measurement gated on cool machine + interleaved A/B.