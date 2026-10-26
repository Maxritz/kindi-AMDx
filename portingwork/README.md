# kindi-vk - Kindi inference engine on Vulkan, AMD GCN (RX 580)

A port of [Kindi](https://github.com/ahtx/kindi)'s inference engine from CUDA/CuPy to
Vulkan compute, targeting a single **Radeon RX 580 (Polaris / gfx803, 8 GB)** on Windows.
No ROCm, no containers.

Reads the original **GGUF files directly** - no conversion step, no weight pack.

## Status

| model | status |
|---|---|
| Ternary-Bonsai-2 **2B** PTQ1_0 | token-exact vs llama.cpp |
| Ternary-Bonsai-2 **27B** PQ2_0 | correct output, token-exact |
| Ternary-Bonsai-2 **27B** PTQ1_0 + MTP head | runs; MTP head loaded but not drafting |
| Llama-3.2-1B Q8_0 | not implemented |

Verified against Prism's `llama-completion` (the reference fork):

```
2B prompt "The sky is blue because" -> " of the scattering of light"
27B prompt "The sky is blue because" -> " of Rayleigh scattering. This is a phenomenon where light is"
```

`kindi_cpu.exe` is a CPU mirror used as a cross-check on the 2B.

## Performance (RX 580, 2B, decode)

**36 tok/s**, up from ~3.3 at the start of the optimisation work (~11x). Every step
verified token-identical. See [PERFORMANCE.md](PERFORMANCE.md) for the full measurements.

| stage | tok/s |
|---|---|
| original (per-dispatch fence, simple GEMV) | ~3.3 |
| + dispatch batching | 5.5 |
| + `g4lut` (256-entry LUT, 4 trits/byte) | 26.0 |
| + `g4lut2` (4 rows/workgroup, subgroup reduce) | 34.5 |
| + `gemv3` (3 tensors per launch) | 36.3 |

Prefill batching (`KINDI_CHUNK`), 144-token prompt:

| model | serial | CHUNK=4 | speedup |
|---|---|---|---|
| 2B | 6,122 ms | 4,538 ms | 1.35x |
| 27B | 92,744 ms | 56,056 ms | 1.65x |

## Prebuilt binaries

`kindi-vk-polaris-win64.zip` (in this folder) has the engine ready to run, with no Vulkan
SDK or compiler needed. It contains `kindi_vk_decode.exe`, `kindi_cpu.exe`, the 59 compiled
shaders in `shaders/`, and a README. Unzip it and run from that folder, because the engine
loads shaders from `shaders/` relative to the working directory.

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf `
 --prompt "The sky is blue because" --temp 0 -n 20
```

Built with MSVC 14.44 and the Vulkan SDK 1.4.363.0.

## Build

```powershell
.\build.ps1
```

Needs the Vulkan SDK (glslangValidator + vulkan-1.lib), MSVC, and `nlohmann/json.hpp`.
Edit the paths at the top of `build.ps1`. Shaders compile to `shaders/*.spv`; the engine
loads them at runtime, so run it from this directory.

## Run

```powershell
.\bin\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf `
 --prompt "The sky is blue because" --temp 0 -n 20
```

| flag | meaning |
|---|---|
| `--model <path.gguf>` | GGUF file (required) |
| `--prompt`, `-n`, `--temp`, `--top-k` | generation |
| `--ctx N` | context length (VRAM budget) |
| `--native5` | keep PTQ1_0 in its 28-byte/128-weight form (no host expansion; needed for the 27B) |
| `--selftest` | validate the batched GEMV against the single-token path (both weight formats) |
| `--dump` | per-layer probe values to stderr |
| `--dumpvec` | full-vector dumps for offline fold verification |
| `--trace` | per-component and per-shader performance table |
| `--trace-json <file>` | write a machine-readable baseline |
| `--force-gen` | force the generic GQA attention path |

### Environment switches (all default off)

| var | effect |
|---|---|
| `KINDI_CHUNK=N` | chunked prefill, N tokens per pass (1..4) |
| `KINDI_DEVICE=N\|name` | pick a device (default: prefers RX 580 by name) |
| `KINDI_QUEUE_PRIORITY=low\|medium\|high\|realtime` | queue scheduling class |
| `KINDI_BATCH_MAX=N` | dispatches per batch before flushing (default 2048) |
| `KINDI_NO_BATCH=1` | regression A/B: one submit+fence per dispatch |
| `KINDI_NO_BARRIER=1` | **diagnostic only, produces wrong results** - measures barrier cost |
| `KINDI_BARRIER=memory\|readonly\|buffer` | barrier scoping variant |

## Architecture

`kindi_vk_decode.cpp` parses the GGUF, uploads weights, and runs the decode loop.
All matrix products are ternary GEMVs; activations are Hadamard-folded with the sign
vectors stored in the GGUF (`prism.hadamard.*`).

Weight formats decoded natively on the device:

| format | shader | used by |
|---|---|---|
| PQ2_0 (2-bit, 34-byte blocks) | `g2pq`, `g2pqM` | 27B |
| PTQ1_0 native (5-trit, 28-byte blocks) | `g5ptq`, `deq_emb5` | 27B PTQ1_0 |
| 4-trit pack (4 trits/byte) | `g4lut2`, `g4lut3`, `g4lutM` | 2B, 9B |
| Q8_0 | `g4q8`, `q8deq` | MTP head |

`M` suffix = batched over T tokens (`g4lutM`, `g2pqM`); `_T` suffix = T-token variant of
an elementwise or recurrent kernel (`rmsfold_T`, `smfold_T`, `gdn_conv_T`,
`gdn_delta_T`, `gdn_gates_T`, `l2w_T`).

## Instrumentation

`--trace` prints a per-component and per-shader table with device / idle / host time,
bound bytes, waves and waves/CU, plus traps for synchronisation, allocation, transfer,
accuracy (token hash) and a measured instrumentation floor. Always compiled in; enabled
by a CLI flag, never an env var.

```
 component ops %dev dev us idle us host us bindMB
 projections 1355 97.0% 837615.36 175037.94 1012653.30 4454.86
 recurrent 486 1.1% 9388.00 63420.80 72808.80 244.32
 ...
 synchronisation: submits=40 fence_waits=40 (446.3 ops per submit)
 queue wait : 1282.77 ms in 40 flushes (32.07 ms each)
 accuracy : 5 tokens, hash=...
 floor : 0.090 us host per recorded op
```

## Docs

| file | contents |
|---|---|
| [PERFORMANCE.md](PERFORMANCE.md) | measurements, bottleneck analysis, A/B switches |
| [CHUNK-STATUS.md](CHUNK-STATUS.md) | chunked prefill, the bugs it uncovered |
| [STATUS-27B.md](STATUS-27B.md) | 27B correctness history |
| [MTP-FINDINGS.md](MTP-FINDINGS.md) | MTP head layout and what it still needs |

## Known limits

- Attention is per token even in the chunked prefill path (a T-row flash kernel is the
 next step and is what limits prefill to 1.35-1.65x).
- MTP speculative decode is not implemented; the head tensors are loaded but unused.
 On a single-token-per-pass engine it would be slower than plain decode.
- Queue priority has no measurable effect on this card (the GPU is idle ~35% of wall
 time, so it is not contending); it is requested anyway for desktop contention.
- Llama-3.2-1B (Q8_0, standard transformer) is not implemented.

## License

Kindi is AGPL-3.0 (see [LICENSE](../LICENSE) in the upstream project). This port follows
the same licence.
