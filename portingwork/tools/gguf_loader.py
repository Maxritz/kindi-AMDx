"""Direct GGUF PTQ1_0 loader: ternary codes stay integer, scales stay fp16 bits. No dequant."""
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\rr\Desktop\portings\kindi")
from kindi.gguf import GGUF
from kindi.convert import trits_from_raw, phase_bytes

PTQ1_0 = 143

class TernaryGGUF:
    def __init__(self, path):
        self.g = GGUF(path)
        g = self.g
        assert g.kv.get("general.architecture") == "qwen35", "only qwen35 2B in v1"
        self.dims = {
            "layers": g.kv["qwen35.block_count"], "width": g.kv["qwen35.embedding_length"],
            "ffn": g.kv["qwen35.feed_forward_length"], "heads": g.kv["qwen35.attention.head_count"],
            "kv_heads": g.kv["qwen35.attention.head_count_kv"], "groups": g.kv["qwen35.ssm.group_count"],
            "head_dim": g.kv["qwen35.attention.key_length"], "rope_dim": g.kv["qwen35.rope.dimension_count"],
            "state_size": g.kv["qwen35.ssm.state_size"], "gdn_inner": g.kv["qwen35.ssm.inner_size"],
            "rope_base": g.kv["qwen35.rope.freq_base"],
            "full_attention_interval": g.kv["qwen35.full_attention_interval"],
        }
        hd = {k.split("prism.hadamard.")[-1]: v for k, v in g.kv.items() if k.startswith("prism.hadamard.")}
        sv = np.array(hd["sign_values"], dtype=np.float32)
        self.signs = {}
        off = 0
        for w in hd["sign_widths"]:
            self.signs[int(w)] = sv[off:off + int(w)].astype(np.float32)
            off += int(w)
        self.hb = int(hd.get("block_size", 512))
        self.head_name = "output.weight" if any(t[0] == "output.weight" for t in g.tensors) else "token_embd.weight"
        self._load_tensors(path)

    def _load_tensors(self, path):
        import concurrent.futures
        import os as _os
        g = self.g
        items = [(name, ne, ty, toff) for name, ne, ty, toff in g.tensors]
        base = g.data_start
        self.codes = {}
        self.scales = {}
        self.f32 = {}

        def _work(item):
            name, ne, ty, toff = item
            ne0, rows = ne[0], 1
            for x in ne[1:]:
                rows *= x
            with open(path, "rb") as f:
                f.seek(base + toff)
                if ty == PTQ1_0:
                    bpr = ne0 // 128
                    nbytes = 28 * rows * bpr
                    buf = f.read(nbytes)
                    raw = np.frombuffer(buf, dtype=np.uint8).reshape(rows * bpr, 28)
                    pb = phase_bytes(trits_from_raw(raw)).reshape(rows, bpr * 32)
                    dm = np.frombuffer(raw[:, 26:28].tobytes(), dtype=np.uint16).reshape(rows, bpr)
                    return (name, "ptq", np.ascontiguousarray(pb), np.ascontiguousarray(dm))
                elif ty == 30:  # BF16
                    n = rows * ne0
                    buf = f.read(2 * n)
                    v = np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16
                    return (name, "f32", np.frombuffer(v.astype("<u4").tobytes(), dtype="<f4").copy(), None)
                elif ty == 0:  # F32
                    n = rows * ne0
                    buf = f.read(4 * n)
                    return (name, "f32", np.frombuffer(buf, dtype="<f4").copy(), None)
                else:
                    raise SystemExit(f"kindi-vk v1: unsupported tensor type {ty} for {name}")

        workers = min(8, _os.cpu_count() or 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for name, kind, a, b in ex.map(_work, items):
                if kind == "ptq":
                    self.codes[name], self.scales[name] = a, b
                else:
                    self.f32[name] = a

    def summary(self):
        tot_c = sum(v.nbytes for v in self.codes.values())
        tot_s = sum(v.nbytes for v in self.scales.values())
        tot_f = sum(v.nbytes for v in self.f32.values())
        return f"ternary codes {tot_c/2**20:.0f} MiB + scales {tot_s/2**20:.0f} MiB + f32 {tot_f/2**20:.1f} MiB = {(tot_c+tot_s+tot_f)/2**20:.0f} MiB VRAM"

if __name__ == "__main__":
    import sys as _s
    t = TernaryGGUF(_s.argv[1] if len(_s.argv) > 1 else r"C:\x\Ternary-Bonsai-2-2B-PTQ1_0.gguf")
    print(t.dims, "HB=", t.hb, "head=", t.head_name)
    print(t.summary())
    print("signs:", {k: len(v) for k, v in t.signs.items()})
