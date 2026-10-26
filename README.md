<p align="center"><img src="assets/kindi-banner.svg" alt="Kindi 0.2.0: modern 2B to 27B language models on low-cost NVIDIA cards, from a 1 GB Quadro K420 to an RTX 3080. Two Quadro K1200 cards work as one 8 GB card." width="100%"></p>

# Kindi 0.2.0

[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Commercial licenses available](https://img.shields.io/badge/commercial%20licenses-available-green.svg)](COMMERCIAL-LICENSE.md)
[![Release v0.2.0](https://img.shields.io/badge/release-v0.2.0-lightgrey.svg)](https://github.com/ahtx/kindi/releases/tag/v0.2.0)

Kindi is a custom inference engine for Prism ML's **Ternary-Bonsai-2** language models (2B, 4B, 9B and 27B
parameters, ternary weights). It started on the **NVIDIA Quadro K420** (Kepler, 1 GB), where it runs the 2B at
**20-23 tokens per second**. Version 0.2.0 adds the Quadro K1200, two cards in tensor parallel, the GeForce GTX 1080,
the GeForce RTX 3080, a CPU engine and a one-command installer. It has an **OpenAI-compatible server** for chat
clients, scripts and agents that write code. Its default output is token-for-token identical to the output of
llama.cpp for the same model file. On the K420, with the same file and the same context, it generates about 13x
faster than llama.cpp. The comparisons for each card are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). The one-page summary
is [WHY.md](WHY.md).

<p align="center"><img src="assets/kindi-demo.gif" alt="The pi coding agent driving Kindi on a Quadro K420: it runs nvidia-smi, writes and runs a Python script, and checks disk space" width="100%"></p>

<p align="center"><sub>Recorded live: the <a href="https://pi.dev">pi</a> coding agent (with a Kindi colour theme)
connected to <code>kindi serve</code> on one Quadro K420, default settings. Token streaming is real time. We
shortened pauses longer than 2.5 s. The <a href="https://github.com/ahtx/kindi/releases/download/v0.1.0/kindi-demo-untrimmed.mp4">full
video</a> (123 s, no pauses shortened) and its <a href="https://github.com/ahtx/kindi/releases/download/v0.1.0/kindi-demo.cast">asciicast</a> are in the 0.1.0 release.</sub></p>

<p align="center"><img src="assets/kindi-speed.svg" alt="Write speed in tokens per second, Kindi 0.2.0 and llama.cpp on the same card with the same model file and context: Quadro K420 1 GB, 2B, 20.8 and 1.65; Quadro K1200, 2B, 81.0 and 20.8; 4B, 39.2 and 8.7; 9B, 23.4 and 5.9; two Quadro K1200 cards, 27B, 12.6 and 1.69; GeForce GTX 1080, 27B, 27.9 and 19.4 at 512 tokens, and 15.4 at 64,000 tokens, where llama.cpp runs out of memory; GeForce RTX 3080 at 190 W, 27B, 63.2 and 52.2 at 1K and 44.0 and 23.5 at 90K." width="100%"></p>

By [Amir Husain](https://www.amirhusain.com). Watch the [launch video](https://youtu.be/ddAnghj8HpE) on YouTube.

## AMD Polaris / GCN port (RX 580 / gfx803) - Vulkan, first working port

This is the first working AMD port of Kindi. It is a Vulkan compute engine for the Radeon
RX 580 (Polaris 10, gfx803, 8 GB) running on Windows. There is no ROCm and no container
anywhere in the stack. The card has no WMMA and no usable ROCm support, so every matrix
product here is a hand-written GLSL compute shader built for GCN's wave64 model.

It reads the original GGUF files directly. There is no conversion step and no weight pack.
PQ2_0, native PTQ1_0 (5-trit) and the 4-trit pack are all decoded on the card itself.

**Requirements:** Windows x64, an AMD driver with Vulkan support, and the GGUF model files.
The prebuilt engine needs no Vulkan SDK and no compiler.

**Get the models** from Prism ML on Hugging Face. For example the 2B:

```powershell
mkdir C:\x
curl.exe -L -o C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf `
  https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/Ternary-Bonsai-2-2B-PTQ1_0.gguf
```

The 27B files used below are `Ternary-Bonsai-2-27B-PQ2_0.gguf` and
`Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf` from the same publisher. Adjust the
paths in the commands if you keep them somewhere else.

**Run it.** Unzip `portingwork/kindi-vk-polaris-win64.zip` anywhere, then run from inside
the extracted folder. The engine loads its shaders from `shaders/` relative to the working
directory, so do not move the exe out on its own.

```powershell
Expand-Archive portingwork\kindi-vk-polaris-win64.zip -DestinationPath kindi-vk
cd kindi-vk
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf `
    --prompt "The sky is blue because" --temp 0 -n 20
```

To build from source instead of using the zip, run `portingwork/build.ps1`.

**Results on the RX 580 (8 GB):**

2B PTQ1_0 (Ternary-Bonsai-2-2B-PTQ1_0.gguf) - token-exact, 36.8 tok/s

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf `
    --prompt "The sky is blue because" --temp 0 -n 20
```

27B PQ2_0 (Ternary-Bonsai-2-27B-PQ2_0.gguf) - token-exact, 1.6 tok/s

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf `
    --ctx 4096 --prompt "The sky is blue because" --temp 0 -n 8
```

27B PTQ1_0 with MTP head (Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf) - token-exact,
5.19 tok/s, and the only model here that drafts: add `--spec 20` for 6.29 tok/s

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf `
    --ctx 4096 --native5 --prompt "The sky is blue because" --temp 0 -n 8
```

Output checked against Prism's `llama-completion`:

```
2B   "The sky is blue because"  ->  " of the scattering of light"
27B  "The sky is blue because"  ->  " of Rayleigh scattering. This is a phenomenon where light is"
```

**Prefill and token generation, RX 580 8 GB.** Model load is excluded (about 2.5 s for
the 2B and 8.6 s for the 27B). Prefill is measured on a 144-token prompt, so 143 tokens
are actually prefilled.

| Model | Prefill | Prefill (`KINDI_CHUNK=4`) | Token gen | Token gen (`--spec`) |
|---|---|---|---|---|
| 2B PTQ1_0 | 39.3 tok/s | 69.0 tok/s | 32-34 tok/s | n/a (no MTP head) |
| 27B PQ2_0 | 1.71 tok/s | 3.03 tok/s | 1.5 tok/s | n/a (no MTP head) |
| 27B PTQ1_0 + MTP | 3.75 tok/s | 3.75 tok/s | **5.19 tok/s** | **6.29 tok/s** |

The 27B PTQ1_0 generation figure is **5.19 tok/s**, and **6.29 with `--spec`**, against
0.848 when this port started. That is 7.4x, and every step is token-identical.

The work was on `g5ptq`, the PTQ1_0 GEMV that is 94% of a 27B token. It went from
2643 us per launch to 343:

- **byte-granular decode.** PTQ1_0 packs five base-3 digits per byte (3^5 = 243 < 256),
  so one byte load now yields five weights instead of one.
- **a float LUT.** The trits are pre-converted to floats in shared memory, so the inner
  loop is one LDS load and one FMA per weight instead of a shift, an and, a convert and
  a subtract.
- **rows per wave.** Two output rows per wave share every x load and give the scheduler
  two independent FMA chains. This was the largest single win, 1.49x.
- **one-word scale.** A block starts at a multiple of 28 and 28 is a multiple of 4, so
  the fp16 scale is always the high half of one word, not two separate byte loads.
- **fusion.** Up to three tensors that read the same input run in one dispatch, so the
  barrier between them disappears.

The kernel is instruction-bound, not bandwidth-bound: making every weight read
cache-resident changed the time by 2%. 5.19 tok/s needs ~19 instructions per weight on
this card's 3.09 T lane-ops/s; 8 tok/s would need ~14.

`--spec` uses a separate batched kernel (`g5ptqM2r2`), which gets the same rows-per-wave
treatment so it stays ahead of the serial path: one weight decode feeds both tokens, and
one x load feeds both rows.

Chunked prefill buys 1.76x on the 2B and 1.77x on the 27B. Token generation is not
affected, because decode is always a single token per pass. It is off by default; set the
environment variable before running:

```powershell
$env:KINDI_CHUNK = 4
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf --ctx 4096 `
    --prompt "The sky is blue because" --temp 0 -n 8
```

Sanity check that the build is working:

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf --selftest
```

### OpenAI-compatible API

`--serve` turns the engine into a local server that speaks the OpenAI chat and completion
APIs, so any OpenAI client works without knowing about Vulkan or GGUF:

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf --serve --port 8080
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = c.chat.completions.create(
    model="Ternary-Bonsai-2-2B-PTQ1_0",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=16, temperature=0,
    extra_body={"enable_thinking": False},
)
print(r.choices[0].message.content)
```

`POST /v1/chat/completions` and `POST /v1/completions` both stream. Reasoning is reported
separately in `reasoning_content`, split at `</think>`.

Add `--spec 1` to enable speculative decode in the server too. On the 27B PTQ1_0 + MTP
head that is 0.88 -> 1.20 tok/s (1.37x) with byte-identical output; it applies to greedy
requests only and leaves prefill untouched. The 2B has no MTP head, so the server reports
`--spec ignored` and serves normally.

**Prompt reuse** is on by default. The KV cache needs no copying here (it is
position-indexed and append-only), so a checkpoint of the recurrent state is the whole
cost, ~20 MiB on the 2B and ~150 MiB on the 27B. On the 27B the same request went from
31.4 s of prefill to 0 s, with byte-identical output.

`--ctx` is clamped to what the card can actually hold. `--ctx 16384` on the 27B now warns
and runs at 13824 instead of dying with a bare `Vulkan error -2`.

Full endpoint list, request fields and limits:
[portingwork/API.md](portingwork/API.md).

**What this port needed that the CUDA engine did not:**

- GCN has no WMMA and no `cp.async`, so the tensor core paths are replaced by subgroup-64
  GEMVs. The 4-trit weight layout exists upstream only to feed a 256-entry LUT kernel, and
  here it is decoded with a shared memory LUT at 4 trits per byte.
- Every launch used to carry its own fence. Batching the dispatches into one command buffer
  removed 21.6 ms out of 27.5 ms per token on the 2B.
- `VK_KHR_global_priority` was requested and the driver does accept REALTIME, but it made
  no measurable difference. The GPU sits idle for about 35% of the wall time, so there is
  nothing to contend with.
- AMD's `VkQueueFamilyGlobalPriorityPropertiesKHR` has to be queried through the queue
  family chain. In the device properties chain it quietly returns zero and you stay on
  MEDIUM without any error.

**Instrumentation** is compiled in always and switched on with a flag, never an env var.
`--trace` prints a per-component and per-shader table with device, idle and host time,
bytes moved, waves and waves per CU, along with traps for synchronisation, allocation,
transfer, accuracy and a measured instrumentation floor.

**Full details:** [portingwork/README.md](portingwork/README.md) covers the engine, the
build and every flag. [portingwork/PERFORMANCE.md](portingwork/PERFORMANCE.md) has the
measurements and the bottleneck analysis. [portingwork/CHUNK-STATUS.md](portingwork/CHUNK-STATUS.md)
records the chunked prefill work and the bugs it exposed.
[portingwork/MTP-FINDINGS.md](portingwork/MTP-FINDINGS.md) describes the MTP head layout and
the speculative decode loop, and [portingwork/API.md](portingwork/API.md) the server.

**MTP speculative decode works.** `--spec N` drafts one token with the MTP head, verifies
it in a single T=2 trunk pass and accepts the matching prefix. Output is token-identical to
plain decode. Measured on the 27B PTQ1_0 + MTP head, 20 tokens, greedy:

| prompt | plain | `--spec` | acceptance | speedup |
|---|---|---|---|---|
| `The sky is blue because` | 0.850 tok/s | **1.25 tok/s** | 100% | **1.47x** |
| `def fibonacci(n):` | 0.829 tok/s | **1.20 tok/s** | 100% | **1.45x** |
| `Once upon a time there was` | 0.871 tok/s | **1.00 tok/s** | 53.8% | **1.15x** |

Break-even is about 1.27 tokens per pass, so the draft needs roughly 27% acceptance to pay
for itself. The verify pass runs at 1.74x the serial trunk, close to the 2x ceiling that
one weight read per two tokens allows. See
[portingwork/MTP-FINDINGS.md](portingwork/MTP-FINDINGS.md) for the four bugs it took
(including a `block_q8_0` layout error that had made the Q8 selftest pass on garbage).

```powershell
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf `
    --ctx 2048 --native5 --prompt "The sky is blue because" --temp 0 --spec 20
```

**Not done yet:** k=2 drafts (a second draft would amortise the per-pass overhead over 3
tokens), a T-row attention kernel, a Q4 KV cache (would take the 27B from ~14k to ~50k
context on 8 GB), and Llama-3.2-1B.

**Quick Note:** 
I actually installed my rx580x just to port this to vulkan. Now will switch back to my 6700xt
Thanks to the amazing brother Amir for his original engine. It was fun to port (I find the frustration that it caused really worth it)
Now lets see if I can get this to speed up on Rx 6700xt and Rx 9070xt.
Anyone who wants to port to anything else, check the portingwork folder and have fun. (Deepseek is you friend)

---


## Quick start: one command

On Linux x86-64 with an NVIDIA card (Kepler to Ampere, one or two cards) or with no GPU:

```bash
curl -fsSL https://amirhusain.com/kindi/install.sh | bash
```

The script finds your cards and selects the model and the settings for them. It downloads the model and checks
its sha256, converts it, checks that the output is token-for-token identical to llama.cpp, and starts the server
on `127.0.0.1:8093`. It explains each step, and it asks before it installs Docker or the NVIDIA Container Toolkit.
To see each step before anything changes, add `-s -- --dry-run` after `bash`. Run it again to upgrade or repair;
`--uninstall` removes it. To check the script's sha256 before you run it, and for the model that each card gets,
see [docs/INSTALL.md](docs/INSTALL.md).

- [Quick start: one command](#quick-start-one-command)
- [Requirements](#requirements)
- [Install](#install) - with Docker, or natively
- [Get the model and convert it](#get-the-model-and-convert-it)
- [Run the server](#run-the-server)
- [Use it](#use-it) - curl, Python, streaming, tools, agents
- [Exact and fast output heads](#exact-and-fast-output-heads)
- [The prompt cache](#the-prompt-cache)
- [Commands](#commands)
- [Configuration](#configuration)
- [Performance](#performance)
- [Agent benchmark](#agent-benchmark)
- [Limits](#limits)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability 3.0 (Kepler) or later, one or two cards. Tested: Quadro K420 (1 GB and 2 GB), Quadro K1200 (one card, or two in tensor parallel), GeForce GTX 1080, GeForce RTX 3080. With no GPU, the CPU engine runs on x86-64 with AVX2. |
| Driver | Kepler: the NVIDIA **470** series (the last driver branch with Kepler support). Later cards: a current driver. The image has CUDA 10.2; the host needs only the driver. |
| OS | Linux x86-64 (developed on Ubuntu 24.04 with Docker; the image itself is Ubuntu 18.04) |
| RAM / disk | 2B: 4 GB RAM, ~1.5 GB disk. 27B: ~14 GB of free RAM for its first start (two K1200 cards: ~5 GB) and ~19 GB of disk. The Docker image takes ~4 GB more. |
| Software | Docker with the NVIDIA Container Toolkit (or the installer's device passthrough) - **or** CUDA 10.2 + Python 3.6 + CuPy 9.6 (see below) |

[docs/CARDS.md](docs/CARDS.md) gives the model and the settings for each card. The card must be free: Kindi fits the
context to the free card memory. On the K420 1 GB, Kindi uses about 870 MB of its 980 MB with the default
24,576-token context. A desktop session on the same card takes a few MB and is fine.

## Install

### With Docker (recommended)

CUDA 10.2 and CuPy 9.6 are the last releases that support Kepler, and they need an old userland; the image packages
exactly that.

```sh
git clone https://github.com/ahtx/kindi.git && cd kindi
docker build -t kindi:0.2.0 -f docker/Dockerfile .   # downloads the CUDA 10.2 toolkit (~2.5 GB), ~15 minutes
```

You can also use the release archive. Download `kindi-0.2.0.tar.gz` from the
[v0.2.0 release](https://github.com/ahtx/kindi/releases/tag/v0.2.0), then run
`tar xzf kindi-0.2.0.tar.gz && cd kindi-0.2.0` and build the image with the same `docker build` command.

Check that Docker sees the GPU (create `~/.kindi` first, so it belongs to you and not to root):

```sh
mkdir -p ~/.kindi
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.2.0 info
```

More in [docs/DOCKER.md](docs/DOCKER.md).

### Natively

You need CUDA 10.2 (toolkit with NVRTC and cuBLAS), Python 3.6 or later that CuPy 9.6 supports, and:

```sh
pip install "numpy==1.19.5" "fastrlock==0.8.3" "cupy-cuda102==9.6.0" "fastapi==0.79.1" "uvicorn==0.16.0" "pydantic==1.9.2"
```

Then run Kindi from the repository root with `python3 -m kindi ...` (put the repository on `PYTHONPATH` to run it
from elsewhere).

## Get the model and convert it

These steps use the 2B. For the 4B, 9B and 27B, use their PTQ1_0 files from the same publisher. The installer
selects and downloads the right one; [docs/INSTALL.md](docs/INSTALL.md) lists the files and their sha256. Download
Prism ML's PTQ1_0 file (430 MB, Apache 2.0) from Hugging Face:

```sh
mkdir -p ~/.kindi && cd ~/.kindi
wget https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf/resolve/main/Ternary-Bonsai-2-2B-PTQ1_0.gguf
```

Convert it once into Kindi's weight pack (about a minute; exact - the same weights, re-laid out):

```sh
# Docker
docker run --rm -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.2.0 convert /data/Ternary-Bonsai-2-2B-PTQ1_0.gguf
# native
python3 -m kindi convert ~/.kindi/Ternary-Bonsai-2-2B-PTQ1_0.gguf          # writes ~/.kindi/pack
```

Check the engine against llama.cpp's reference output:

```sh
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.2.0 verify
```

```
prompt 1 'The sky is blue because'          PASS  40/40 tokens match
prompt 2 'The capital of France is'         PASS  40/40 tokens match
prompt 3 'Water boils at a lower temper...' PASS  40/40 tokens match
prompt 4 'def add_two_numbers('             PASS  40/40 tokens match
ALL PASS: token-exact with llama.cpp - 22.1 tokens/s over the run
```

The speed in the last line includes the prompt tokens, so it is not the generation speed. It also depends on the
fast-head file of the next step. Without this file, `verify` uses a plain layout of the exact head and shows about 20
tokens/s. With the file, `verify` shows about 22 tokens/s. In both cases, the output is token-exact. `kindi bench`
measures the generation speed (see [Performance](#performance)).

Optional - build the cluster file for the fast output head (about 2 minutes):

```sh
docker run --rm --gpus all -u $(id -u):$(id -g) -v ~/.kindi:/data kindi:0.2.0 fast-head
```

## Run the server

```sh
docker run -d --name kindi --restart unless-stopped --gpus all -u $(id -u):$(id -g) \
    -p 127.0.0.1:8093:8093 -v ~/.kindi:/data kindi:0.2.0
# native:  python3 -m kindi serve
```

The first start compiles Kindi's CUDA kernels for your card (about 1.5 minutes on the K420). Kindi caches them under
`state_dir`, and later starts take about 10 seconds. Then:

```sh
curl -s localhost:8093/health
```

The server listens on `127.0.0.1:8093` (natively) or on the port you publish (Docker). To serve your network, publish
`-p 8093:8093` (Docker) or set `host = 0.0.0.0`, and set an `api_key`.

## Use it

The endpoint is `http://localhost:8093/v1`, the model id `bonsai2-2b` (configurable).

```sh
curl -s localhost:8093/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "bonsai2-2b",
  "messages": [{"role": "user", "content": "Write a bash one-liner that counts the .txt files here."}]
}'
```

Python (`pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8093/v1", api_key="none")   # your api_key if you set one
r = client.chat.completions.create(model="bonsai2-2b",
                                   messages=[{"role": "user", "content": "What is the capital of Italy?"}])
print(r.choices[0].message.content)
```

Kindi supports these request fields: `messages` (system / user / assistant / tool, text content), `tools` and
`tool_choice` (`"none"` disables tools), and `stream` and `stream_options.include_usage`. It also supports
`temperature` (0 = greedy), `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `stop` and `seed`. It also
supports `chat_template_kwargs.enable_thinking` (off by default). When thinking is on, the thinking comes back as
`reasoning_content` (also streamed as `delta.reasoning_content`). `thinking_budget_tokens` caps the thinking. At the
budget, the server closes the thinking and the model then gives its answer.
Responses carry OpenAI `usage` (with `prompt_tokens_details.cached_tokens`) and a `timing` block.

**Tool calls** come back as OpenAI `tool_calls` with `finish_reason: "tool_calls"`; send the results back as
`role: "tool"` messages.

**Agents and chat UIs.** Point any OpenAI-compatible client at `http://localhost:8093/v1` with model `bonsai2-2b`.
Two settings matter on the smaller cards - see [docs/AGENTS.md](docs/AGENTS.md):
- Give the client a **long request timeout**. An agent's first prompt of 5,000+ tokens takes a few minutes to read the
  first time. The cache makes later turns fast. Do not use automatic retries, because they queue behind the current
  request.
- Tell the client the context window that `/health` shows (on the K420 1 GB: **24,576** tokens).

`kindi chat` is a terminal chat that needs no server.

## Exact and fast output heads

The output layer scores all 248,320 vocabulary entries for every generated token - about 9 ms of a 46 ms step.

- **exact** (default): the full head. Greedy output is token-for-token identical to llama.cpp.
- **fast**: a shortlist head. `kindi fast-head` clusters the vocabulary once. For each token, Kindi scores the 1,024
  cluster centres. Then it computes exact scores only for the 256 clusters with the best scores (~13 % of the
  vocabulary). Generation is about 15 % faster. On 310 measured steps, it chose the same token as the full head
  99.4 % of the time.

To change the head while the server runs, with no restart, use `kindi head fast` / `kindi head exact` (Docker:
`docker exec kindi python3 -m kindi head fast`). `/health` shows the current mode.

## The prompt cache

Bonsai mixes attention with recurrent Gated-DeltaNet layers, whose state cannot be rewound. Kindi caches anyway,
exactly:

- **checkpoints** of the recurrent state every 256 tokens. A request that shares a start with the previous request
  resumes at the last checkpoint before the first difference.
- **parked conversations**: Kindi keeps up to 3 other conversations in host RAM. It resumes each conversation when its
  next turn comes. (Agents alternate between their main prompt and side prompts.)
- **saved openings** on disk. When two requests share an opening of 512+ tokens and then differ, Kindi saves the
  state of that opening under `state_dir`. Typically, this opening is a system prompt with tool definitions. Later
  sessions load it in a fraction of a second, also after restarts.

Resumed output is identical to the output when Kindi reads the prompt from scratch.

## Commands

| | |
|---|---|
| `kindi convert MODEL.gguf [--out DIR] [--force]` | build the weight pack (default: the configured `pack`) |
| `kindi fast-head [--clusters N]` | build the fast-head cluster file (stop the server first: needs ~300 MB of GPU memory) |
| `kindi verify` | check greedy output against llama.cpp's reference |
| `kindi bench` | speed at several context positions, prompt read speed, GPU memory |
| `kindi serve [--host H] [--port P] [--api-key K] [--head exact\|fast]` | the OpenAI-compatible server |
| `kindi chat [--system TEXT]` | terminal chat |
| `kindi head exact\|fast` | set the output head while the server runs |
| `kindi info` | configuration, pack and GPU |

Global options: `--config FILE`, `--pack DIR`, `--state-dir DIR`, `--ctx N`.

## Configuration

Defaults, then a config file (`--config`, `$KINDI_CONFIG`, `./kindi.conf` or `~/.kindi/kindi.conf`), then
environment variables `KINDI_<KEY>`, then command-line flags. Copy [kindi.conf.example](kindi.conf.example) and edit
it. The file [docs/CONFIGURATION.md](docs/CONFIGURATION.md) describes the keys: paths, server address, API key, context length,
head mode, cache sizes and generation defaults.

## Performance

Quadro K420, Ternary-Bonsai-2-2B, `kindi bench` (dev build de83d2f, blocking sync off, the installer's setting):

| | exact head | fast head |
|---|---|---|
| generation, short context | 20.9 tokens/s | 24.1 tokens/s |
| generation at 8,000 tokens of context | 18.5 | 20.9 |
| generation at 16,000 | 16.7 | 18.6 |
| generation at 24,000 | 15.2 | 16.8 |

Prompt reading on the same card (Kindi 0.2.0, exact head): 33.3 tokens/s at 512 tokens, 24.2 at 24,000.

The other cards, at a short context, exact head:

| Card | Model | Generation | Prompt reading (512 tokens) |
|---|---|---|---|
| Quadro K1200 | 2B / 4B / 9B | 81.0 / 39.2 / 23.4 tokens/s | 194.5 / 77.4 / 41.7 tokens/s |
| 2x Quadro K1200 | 27B | 12.6 | 19.5 |
| GeForce GTX 1080 | 27B | 27.9 | 254.0 |
| GeForce RTX 3080 (190 W) | 27B | 63.2 (99.9 with the 2B draft model, on agent prompts) | 1,093 |

llama.cpp on the same card, with the same model file and the same context: on the K420, it generates 1.65
tokens/s and reads prompts at 1.2 tokens/s (512 tokens). The tables for each card, with their method, are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).
How Kindi works: [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).
The settings for each tested card (K420, K1200, GTX 1080, RTX 3080, CPU): [docs/CARDS.md](docs/CARDS.md).

## Agent benchmark

[kbench](https://github.com/ahtx/kindi-bench) has 65 agent tasks with hidden tests and a runner. It also has
the results of our runs with Kindi on the K420. kbench works with any OpenAI-compatible server.

## Limits

- **One request at a time.** Concurrent requests queue (the card holds one engine).
- **On the K420, Kindi reads prompts at about 25-33 tokens/s**, so a long new prompt takes time. A new prompt of
  5,000 tokens takes about 3 minutes there. The cache makes repeated prompts fast. Kindi reads text that is not in
  the cache (for example, a large file or long command output) at this speed.
- **The Ternary-Bonsai-2 models only** (the PTQ1_0 GGUFs: 2B, 4B, 9B, 27B). The 27B needs 8 GB of card memory (GTX
  1080, RTX 3080) or two 4 GB cards (2x K1200).
- **At most two GPUs per model.** Two cards run one model in tensor parallel. More cards run more servers.
- **Small models make mistakes.** The 2B is good at short code, shell and factual questions, tool calls and
  summaries. Check what it writes.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). The usual causes are:

- Another program uses the card (out of memory at start).
- On Kepler, the driver is not the 470 branch.
- A client timed out while it waited for a long first prompt.

## About

[Amir Husain](https://www.amirhusain.com). The [Deep Tech with Amir](https://www.youtube.com/@DeepTechWithAmir)
series on VectorFall covers this work. The [Kindi launch video](https://youtu.be/ddAnghj8HpE) shows Kindi on the K420. More broadly, it also shows how to get the most from inexpensive GPUs.

## License

**AGPL-3.0; commercial licenses available.**

Kindi is free software under the [GNU Affero General Public License v3.0](LICENSE). If you distribute Kindi or a
changed version, the AGPL says that you must give the source code to the recipients. This is also true when users
use your version over a network. For use in a closed product or a hosted service, a paid commercial license is
available. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).

The model is Prism ML's Ternary-Bonsai-2-2B under Apache 2.0. You download it from its publisher. The third-party
notices are in [NOTICE](NOTICE).

## Acknowledgements

Kindi exists because of the model it runs. Thank you to [Prism ML](https://huggingface.co/prism-ml) for
[Ternary-Bonsai-2-2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) and for its release under
Apache 2.0. Thank you to the Qwen team at Alibaba Cloud for Qwen3.5-2B, the base model of Ternary-Bonsai-2-2B. Thank
you also to the [llama.cpp](https://github.com/ggml-org/llama.cpp) authors. Their implementation is the reference
that we check Kindi's output against.
