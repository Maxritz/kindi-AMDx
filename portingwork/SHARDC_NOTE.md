# Shard C — host-side dispatch overhead (Intel iGPU): measurement + verdicts

Status: **analysis only, gain < 1%. One comment-only edit, zero behavior change.**

## 1. Measure first: host binding is NOT the bottleneck

`trace.h` `record()` measures `host_us` with QueryPerformanceCounter around the
`bindbufs()` + `vkCmdBind*` + `vkCmdDispatch` host path of each op
(`kindi_vk_decode.cpp:455,469-472`); `dump()` prints
`host total: %.2f ms across %zu ops (%.2f us/op)`.

Measured baselines already in the repo (RX 580 traces; the host path is identical
code on Intel, and on Intel the dev/host ratio is *more* skewed since device time
per op is higher on the iGPU):

| trace | device | host | host share | host/op |
|---|---|---|---|---|
| `trace-spec-k1.json` (27B spec, 13109 ops) | 9718.8 ms | 9.91 ms | **0.10%** | 0.76 us |
| PERFORMANCE.md 2B n=40 (17852 ops) | 418.1 ms | 12.44 ms | **0.7% of 1700 ms device+wait** | 0.70 us |
| EXPERIMENTS-FINDINGS.md §2 (27B 20 tok) | 4707 ms | 14 ms | **0.3% of 6100 ms wall** | ~0.5 us |
| Instrumentation floor (256 empty ops) | — | 0.09–0.11 us/op | — | — |

Descriptor allocation + `vkUpdateDescriptorSets` is only a *fraction* of that
~0.5–0.8 us/op host time (which also includes `bindbufs` vector allocs,
`vkCmdBindPipeline`, `vkCmdPushConstants`, and trace accounting). Even a perfect
descriptor path (zero-cost binds) removes at most ~half of host time:

- 2B case: ~6 ms of 1700 ms device+wait = **0.35%**.
- 27B spec case: ~5 ms of ~11000 ms wall (device 9.7 s) = **< 0.05%**.

For the task's cited numbers (host 34 ms vs device 531 ms for 5 tokens), host is
6% of device — and descriptor work is at most ~1/3 of that host time, so the
ceiling for *any* descriptor optimization is ~2% of host ≈ **~0.1% of wall**.
Verdict: host binding is confirmed not the bottleneck; host-side P1 work is
rejected on measured math, not on difficulty.

## 2. Verdicts

### P1a. VK_EXT_descriptor_buffer — DO NOT IMPLEMENT

Available on Intel (`intelL14-laptop-gpu-spec.md:133` lists
`VK_EXT_descriptor_buffer`), but rejected on the math above (<0.1% wall ceiling)
plus invasiveness:

- `mkpipe()` (`kindi_vk_decode.cpp:304-316`) builds a per-pipeline
  `VkDescriptorSetLayout` + `VkPool` + pre-allocated `P.dset`. Descriptor buffers
  require `VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT` on every
  layout, a `VkDescriptorBufferBindingInfoEXT` per bind, rewriting `bindbufs()`
  *and* the `dispatch()` set-selection logic (`kindi_vk_decode.cpp:459`) for
  every pipeline at creation time. There is no opt-in usage — the flag day
  touches all ~30 `mkpipe` call sites' layout contract and every bind path.
  That is exactly the "invasive pipeline-layout change" the task names as the
  do-not-implement trigger.
- Expected gain: <0.1% wall (see §1). Risk: layout-flag mismatch across 30+
  pipelines is a silent-UB class bug (wrong data, still-valid Vulkan).

### P1b. VK_KHR_push_descriptor — DO NOT IMPLEMENT

Available (`intelL14-laptop-gpu-spec.md:130`), same ceiling (<0.1% wall).
Same invasiveness: push descriptors require
`VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR` on the layout and
replace `bind+update+bindset` with `vkCmdPushDescriptorSetKHR` — again a
per-pipeline layout-flag change plus a rewrite of `bindbufs()`/`dispatch()`.
Additionally, push-descriptor writes go through the command buffer, so the
per-dispatch `std::vector` allocs in `bindbufs()` (the actual dominant host
cost) remain. Rejected: same ceiling, still invasive, keeps the largest host
cost intact.

Cheaper host win if host ever matters (not now): reuse static thread-local
`VkDescriptorBufferInfo`/`VkWriteDescriptorSet` vectors in `bindbufs()` instead
of allocating two vectors per dispatch. Not done — host is 0.1–0.7%, and even
halving it is noise.

### P2a. Timeline semaphores — DO NOT IMPLEMENT (~0 gain)

Available (`intelL14-laptop-gpu-spec.md:130` lists
`VK_KHR_timeline_semaphore`), but the engine cannot use what they offer:

- The engine is **single compute queue, synchronous**: `dispatch()` accumulates
  into one command buffer on `G.q0` and `batch_flush()`
  (`kindi_vk_decode.cpp:396-415`) ends + `vkQueueSubmit(q0, fence)` +
  `vkWaitForFences`. `G.q1` is fetched (`:240`) but never used anywhere else
  (2 grep hits total: declaration + `vkGetDeviceQueue`). Host already overlaps
  *nothing* with device — but there is nothing to overlap: batching means the
  host records 446–936 ops per submit at ~0.7 us/op (~0.3–0.6 ms per batch)
  while the GPU then executes the batch for tens of ms. The host is idle during
  `vkWaitForFences`, but making the wait a timeline wait does not create
  concurrency — there is no second queue's work, no host compute, and no
  next-batch recording (the same `g_bcb` is being executed) to overlap with.
- Timeline semaphores pay off when **two queues** (compute + transfer, or
  two computes) need ordered handoff without host round-trips. The transfer
  queue `G.qt` is used only by `upload()` at load time, never during decode.
  Promoting `q1` to a real second compute queue would require partitioning the
  op stream with cross-queue ownership barriers — a correctness redesign of
  `batch_barrier()`/`batch_flush()`, not a semaphore swap — for a host that is
  already <1% of wall.
- Verdict: switching the fence to a timeline semaphore while staying
  single-queue changes the wait primitive, not the schedule. Expected gain ≈ 0.

### P2b. KINDI_BARRIER default — verified best, keep `readonly` reasoning as-is

Measured (`EXPERIMENTS-FINDINGS.md §2.1`, `PERFORMANCE.md:445-449`): all three
scopes cost the same within noise (memory 6156/3860 ms, readonly 6163/3771 ms,
buffer 6139/3753 ms) because the cost is the COMPUTE→COMPUTE *drain* (44–48
us/dispatch), not the access masks. Default (full `VkMemoryBarrier`,
`kindi_vk_decode.cpp:389-393`) is therefore fine; `readonly` is not faster,
just narrower. The only real fix is fewer dispatches (sibling elision,
`dispatch_nb()`), which is Shard A/B territory. **No change.**

### KINDI_BATCH_MAX tuning — documented in code, keep 2048

`PERFORMANCE.md:520-521`: `KINDI_BATCH_MAX=3500` measured *slower* per pass
(1885 vs 1767 ms) than 2048. Measured ops/submit are 446 (2B n=40) to 936
(27B spec: 13109 ops / 14 submits), so fence+submit cost is already amortized
to ~1 fence per ~30 ms of device work. A one-line rationale comment was added
at the `BATCH_MAX` definition (`kindi_vk_decode.cpp:477-482`); the default is
unchanged. (`KINDI_BATCH_MAX` env override already existed.)

## 3. Code changes made

- `kindi_vk_decode.cpp` (BATCH_MAX site): **comment-only** — 3-line tuning note
  recording why 2048 stays. Zero behavior change; default path bit-identical.
- No shader changes (none touched). No new env flags: adding
  `KINDI_DESCBUF`/`KINDI_TIMELINE` flags for paths that are measured <0.1%
  would be dead code behind flags — rejected per YAGNI.
- Net functional diff: **none** ("no code change, analysis only" apart from the
  comment).

## 4. Validation

- Build: `.\build.ps1` must pass (comment-only edit; rerun before merge).
- `--selftest` requires a `--model` GGUF; no full model GGUF found on this
  machine (only vocab/toy fixtures), so selftest could not run here — but the
  edit cannot affect it (comment text only, no codegen change). Default path is
  bit-identical by construction.
