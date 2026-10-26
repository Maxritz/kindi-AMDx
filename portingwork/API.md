# OpenAI-compatible API

`kindi_vk_decode.exe --serve` runs the engine as a local HTTP server that speaks the
OpenAI chat and completion APIs, so any OpenAI client library can talk to the RX 580
without knowing anything about Vulkan, GGUF or ternary weights.

```
.\kindi_vk_decode.exe --model C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf --serve
```

```
api: listening on http://127.0.0.1:8099  (model Ternary-Bonsai-2-2B-PTQ1_0, ctx 4096, chunked prefill)
api: endpoints /v1/models /v1/chat/completions /v1/completions /health
```

## Flags

| flag | default | meaning |
|---|---|---|
| `--serve` | off | run the API instead of the one-shot CLI |
| `--host` | `127.0.0.1` | bind address; `0.0.0.0` exposes it on the LAN |
| `--port` | `8080` | port |
| `--ctx` | auto | context length; clamped to what the card can hold (see below) |
| `--native5` | off | required for the 27B PTQ1_0 (keeps the 5-trit blocks packed) |
| `--no-prefix-cache` | off | disable prompt reuse |
| `--spec N` | off | speculative decode with the MTP head (any N; the server runs k=1) |

Environment: `KINDI_CHUNK` overrides the prefill width (the server defaults it to 4,
`KINDI_CHUNK=0` forces the serial reference path).

## Endpoints

`GET /health` -> `{"status":"ok"}`

`GET /v1/models` -> the one loaded model.

`POST /v1/chat/completions` and `POST /v1/completions`, both with `"stream": true|false`.

Fields honoured: `messages` (chat), `prompt` (completions), `max_tokens` /
`max_completion_tokens`, `temperature`, `top_k`, `stream`, `enable_thinking` (also
accepted inside `chat_template_kwargs`). `top_p` is accepted and ignored.

### curl

```powershell
curl.exe -s http://127.0.0.1:8080/v1/chat/completions `
  -H "Content-Type: application/json" `
  --data-binary "@req.json"
```

with `req.json`:

```json
{"messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],
 "max_tokens":16,"temperature":0,"enable_thinking":false}
```

```json
{"id":"chatcmpl-b123f03f","object":"chat.completion","created":1792989004,
 "model":"Ternary-Bonsai-2-2B-PTQ1_0",
 "choices":[{"index":0,"message":{"role":"assistant","content":"4"},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":22,"completion_tokens":1,"total_tokens":23}}
```

Note: under PowerShell, `-d '{"a":1}'` loses the quotes before `curl.exe` sees them.
Write the body to a file and use `--data-binary "@file"`, or use the Python client.

### Python (official OpenAI SDK)

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

### Streaming

`"stream": true` returns `text/event-stream` with chunked transfer encoding:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Pieces are emitted on complete UTF-8 boundaries, so a multi-byte character split across
two tokens never reaches the client as mojibake.

### Reasoning

This model family thinks before answering. With thinking on (the default, matching the
model's chat template) the reasoning is reported in `reasoning_content` and the answer in
`content`, split at `</think>`, for both streaming and non-streaming. Pass
`"enable_thinking": false` to render an empty think block and get the answer directly.

## Prompt reuse (prefix cache)

KV needs no copying here: it is position-indexed and append-only, so a shared prefix is
already sitting in the right place. The only state that must be saved is the GDN
recurrent state, and one checkpoint of it costs:

| model | checkpoint |
|---|---|
| 2B | ~20 MiB |
| 27B | ~150 MiB |

When a request shares a prefix with the cached prompt, the server restores that state and
prefills only the new suffix. Measured on the 27B:

```
api: 73 prompt + 8 gen in 41482 ms (prefill 31410, decode 10072)     cold
api: prefix reuse 72/73 tokens
api: 73 prompt + 8 gen in  9024 ms (prefill     0, decode  9024)     warm
```

Output is unchanged: the same request cold and warm returns byte-identical text
(verified). The checkpoint advances as a conversation grows, so each turn reuses the
previous turn's prompt.

If the checkpoint cannot be allocated (a 27B near the top of the context), the server
warns and serves without it rather than failing.

## Context clamping

`--ctx` is honoured only up to what the card can actually hold. The fit is computed after
the weights are uploaded, because that is the first moment their real footprint is known:

```
ctx 8192 fits (8.32 GB heap, 6.74 GB weights, 0.92 GB free, 13824 max)
ctx reduced 16384 -> 13824 to fit VRAM (8.32 GB heap, 6.74 GB weights, 0.92 GB free)
```

Previously `--ctx 16384` on the 27B died with a bare `Vulkan error -2` from the allocator.

## Limits

- **One request at a time.** The engine is single-slot: a 5.9 GB model on an 8 GB card
  leaves no room for a second context. Requests are serialised; there is no queue depth
  beyond the TCP listen backlog.
- **HTTP/1.1, no TLS.** Plain HTTP on localhost, or on the LAN with `--host 0.0.0.0` if
  you accept that it is unauthenticated.
- **No `top_p`, no `stop`, no `logit_bias`, no `n > 1`, no function calling.** Accepted-
  and-ignored or rejected; the model's own tool syntax is still reachable through a plain
  completion prompt.
- **Chunked prefill changes summation order.** Token-exact output is verified for the
  prompts in the test set, but a different prefill width can change floating-point
  results and therefore a greedy token. `KINDI_CHUNK=0` forces the serial reference path
  if you need that guarantee.

## Speculative decode in the server

`--spec N` turns speculative decode on for the server too; the value is ignored and it
always runs k=1. It needs the model to carry an MTP head (the 27B
`-mtp-procreations` file does); on a model without one the server says so and serves
normally:

```
api: --spec ignored, this model has no MTP block
```

With it on:

```
api: speculative decode ON (k=1; greedy requests only, sampling runs serially)
```

Measured on the 27B PTQ1_0 + MTP, one request, 24 tokens:

| | prefill | decode | total | tok/s |
|---|---|---|---|---|
| serial | 6998 ms | 27311 ms | 34309 ms | 0.88 |
| `--spec` | 6887 ms | 20000 ms | 26887 ms | **1.20** |

Decode is **1.37x** and the text is byte-identical. A 60-token generation measured 1.12
tok/s against 0.88 serial.

Two constraints worth knowing:

- **Greedy only.** The draft is a single argmax; accepting it under sampling would change
  the output distribution. A request with `temperature > 0` silently runs the serial path,
  which is correct but does not get the speedup.
- **Prefill is unaffected.** The MTP layer keeps its own KV and needs a row at every
  position, so its catch-up runs alongside the chunked trunk prefill. It reads the chunk's
  per-token hidden states rather than forcing a serial pass; doing it per token cost 9.4 s
  of extra prefill on a 17-token prompt, which is why it reads the chunk instead.

Speculative decode and the prefix cache work together: after a cache restore, at least two
tokens are always fed, because the draft needs `h_{N-2}` and the checkpoint stores only the
recurrent state, not the residual stream. Verified byte-identical cold and warm.

`KINDI_SPEC_FORCE_REJECT=1` forces every draft to be rejected, which exercises the state
restore and commit replay on every pass. That is the regression gate for the commit path:
under it the server must still return exactly what the serial path returns, and it does.
