# Benchmarks, RX 580 8 GB (Polaris, gfx803)

Measured on one Radeon RX 580 (36 CU, 4 SIMD, wave 64, 8 GB), Windows, AMD proprietary
driver 26.5.2, Vulkan 1.3.260. Greedy decoding (`temperature 0`). Every model here reads
its GGUF directly; there is no conversion step and no weight pack.

Method: each model is served with `--serve` and driven over the OpenAI-compatible API,
because the server reports prefill and decode separately and the CLI does not. Nothing is
measured under `--trace`: the trace adds a timestamp pair per dispatch and prefill has
tens of thousands of dispatches, which would distort the number being measured.

- **Prefill**: an 85-token prompt, cold (a prompt the prefix cache has not seen).
- **Decode**: 32 tokens on the same prompt, so the prefix cache is warm and prefill is 0.
- Model load is excluded.

## Results

| Model | Prefill tok/s | Decode ms/token | Decode tok/s |
|---|---|---|---|
| **2B PTQ1_0** | **66.5** | 27.1 | **36.95** |
| **27B PQ2_0** | 3.7 | 616.5 | 1.62 |
| **27B PTQ1_0 + MTP** | **8.8** | 192.7 | **5.19** |
| **27B PTQ1_0 + MTP (Q4_0)** | **8.8** | 192.0 | **5.21** |

Commands:

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf --serve --port 8080

.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf --serve --port 8080

.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf `
    --native5 --serve --port 8080

.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-MTP-Q4_0.gguf `
    --native5 --serve --port 8080
```

`--native5` is required for the 27B PTQ1_0 files: it keeps the weights in their native
28-byte/128-weight form instead of expanding them to one byte per weight.

## Speculative decode

Only the models with an MTP head can draft: the two 27B PTQ1_0 files. On a short 20-token
CLI run the speedup is 5.12 -> 6.31 tok/s (1.23x). Through the server on a longer prompt
with lower draft acceptance it measured 4.72 and 4.66 tok/s, which is *below* the serial
5.19, because acceptance on that prompt was poor. Speculative decode is a win when drafts
are accepted and a small loss when they are not; break-even is around 27% acceptance.

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf `
    --native5 --ctx 2048 --prompt "The sky is blue because" --temp 0 --spec 20
```

## Correctness

Every number above was produced by an engine whose output is checked against a reference
at each change. The gate is token ids, not text:

| Model | Prompt | Expected first tokens |
|---|---|---|
| 2B PTQ1_0 | `The sky is blue because` | `314 279 69377 314 3000` |
| 27B PTQ1_0 + MTP | `The sky is blue because` | `314 12870 60845 69377 13 1061 369 264 ...` |

`--selftest` additionally checks, at bit level (must print `0.000e+00`):

- the batched GEMV against the single-token GEMV, for every quant format
- the fused 3-tensor kernel against three separate launches
- Q8_0, Q4_0 and PQ2_0 against a host dequantisation of the raw GGUF bytes
- PTQ1_0 T=1 against T=2

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf --selftest
```

## What limits each model

**2B, 66.5 prefill / 36.95 decode.** The 4-trit path. Decode is dominated by `g4lut3`
(44% of device time, the fused 3-tensor kernel) and `g4lutM` (21%). Neither has rows per
wave yet; `g4lut3r` exists but is disabled because it produced wrong values.

**27B PTQ1_0, 8.8 prefill / 5.19 decode.** The 5-trit path, and the most optimised.
`g5ptq_byte3nr2` is 61% of device time and `g5ptq_bytenr2` 32%. The kernel is
instruction-bound, not bandwidth-bound: making every weight read cache-resident changed
the time by 2%.

**27B PQ2_0, 3.7 prefill / 1.62 decode.** The slowest, and the least optimised. PQ2_0
uses `g2pq`/`g2pqM`, which never received the byte-granular decode, the float LUT or the
rows-per-wave treatment. It is 3.2x slower than the PTQ1_0 file for the same model.

## Hardware ceilings

Every weight participates in one multiply-accumulate per token, so the arithmetic floor
for prefill is `weights / (lanes * clock)`. This card has 36 CU x 64 lanes = 2304 lanes at
1.34 GHz = 3.09 T lane-ops/s.

| Model | Weights | FMA floor | Absolute ceiling | Measured prefill | % of ceiling |
|---|---|---|---|---|---|
| 2B PTQ1_0 | 2.0 G | 0.6 ms/token | 1544 tok/s | 66.5 | 4.3% |
| 27B PTQ1_0 | 27.0 G | 8.7 ms/token | 114 tok/s | 8.8 | 7.7% |

The gap is instruction count: our kernels do roughly 6-10 operations per weight where a
perfect one would do 1. That is where the remaining headroom is, and it is why the
numbers are far below the ceiling rather than near it.

For scale: a number like 2000 tokens/s on a 2B would need about 30x this card's FP32
throughput, which is an RTX 4090 class GPU, not a Polaris part.
