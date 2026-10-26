"""Export GGUF to a flat binary pack for the C++ Vulkan engine."""
import json, os, sys, struct, numpy as np
sys.path.insert(0, r"C:\Users\rr\Desktop\portings\kindi")
from kindi.gguf import GGUF
from kindi.convert import trits_from_raw, phase_bytes
import concurrent.futures

GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\rr\Desktop\portings\kindi-vk\pack"
os.makedirs(OUT, exist_ok=True)

g = GGUF(GGUF_PATH)
assert g.kv.get("general.architecture") == "qwen35"
dims = {
    "layers": g.kv["qwen35.block_count"], "width": g.kv["qwen35.embedding_length"],
    "ffn": g.kv["qwen35.feed_forward_length"], "heads": g.kv["qwen35.attention.head_count"],
    "kv_heads": g.kv["qwen35.attention.head_count_kv"], "groups": g.kv["qwen35.ssm.group_count"],
    "head_dim": g.kv["qwen35.attention.key_length"], "rope_dim": g.kv["qwen35.rope.dimension_count"],
    "state_size": g.kv["qwen35.ssm.state_size"], "gdn_inner": g.kv["qwen35.ssm.inner_size"],
    "rope_base": g.kv["qwen35.rope.freq_base"],
    "full_attention_interval": g.kv["qwen35.full_attention_interval"],
    "hb": g.kv.get("prism.hadamard.block_size", 512),
}
hd = {k.split("prism.hadamard.")[-1]: v for k, v in g.kv.items() if k.startswith("prism.hadamard.")}
sv = np.array(hd["sign_values"], dtype=np.float32)
signs = {}; off = 0
for w in hd["sign_widths"]:
    signs[int(w)] = sv[off:off + int(w)].astype(np.float32); off += int(w)
# write signs
for w, v in signs.items():
    v.astype("<f4").tofile(os.path.join(OUT, f"signs_{w}.f32"))

base = g.data_start
PTQ1_0 = 143
tensors = {}
code_buf = bytearray()
scale_buf = bytearray()
f32_buf = bytearray()

def load_one(item):
    name, ne, ty, toff = item
    ne0, rows = ne[0], 1
    for x in ne[1:]:
        rows *= x
    with open(GGUF_PATH, "rb") as f:
        f.seek(base + toff)
        if ty == PTQ1_0:
            bpr = ne0 // 128
            raw = np.frombuffer(f.read(28 * rows * bpr), dtype=np.uint8).reshape(rows * bpr, 28)
            pb = phase_bytes(trits_from_raw(raw)).reshape(rows, bpr * 32)  # [rows, nbytes]
            dm = np.frombuffer(raw[:, 26:28].tobytes(), dtype=np.uint16).reshape(rows, bpr)
            return (name, "ptq", rows, ne0, pb.tobytes(), dm.tobytes())
        elif ty == 30:  # BF16
            n = rows * ne0
            v = np.frombuffer(f.read(2 * n), dtype="<u2").astype(np.uint32) << 16
            return (name, "f32", rows, ne0, (v.astype("<u4").tobytes()), None)
        elif ty == 0:
            n = rows * ne0
            return (name, "f32", rows, ne0, f.read(4 * n), None)
        else:
            raise SystemExit(f"unsupported type {ty} for {name}")

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for name, kind, rows, ne0, a, b in ex.map(load_one, g.tensors):
        if kind == "ptq":
            co = len(code_buf); so = len(scale_buf)
            code_buf += a; scale_buf += b
            tensors[name] = {"type": "ptq", "rows": rows, "cols": ne0, "code_off": co, "scale_off": so, "code_bytes": len(a), "scale_bytes": len(b)}
        else:
            fo = len(f32_buf); f32_buf += a
            tensors[name] = {"type": "f32", "rows": rows, "cols": ne0, "f32_off": fo, "f32_bytes": len(a)}

with open(os.path.join(OUT, "codes.bin"), "wb") as f: f.write(code_buf)
with open(os.path.join(OUT, "scales.bin"), "wb") as f: f.write(scale_buf)
with open(os.path.join(OUT, "f32.bin"), "wb") as f: f.write(f32_buf)

manifest = {"dims": dims, "head_name": "output.weight" if any(t[0]=="output.weight" for t in g.tensors) else "token_embd.weight",
            "tensors": tensors, "vocab": len(g.kv["tokenizer.ggml.tokens"])}
with open(os.path.join(OUT, "manifest.json"), "w") as f: json.dump(manifest, f, indent=2)
# save vocab
with open(os.path.join(OUT, "vocab.json"), "w") as f: json.dump(g.kv["tokenizer.ggml.tokens"], f)
print("pack written to", OUT)
print("codes", len(code_buf), "scales", len(scale_buf), "f32", len(f32_buf))
print("dims", dims)
