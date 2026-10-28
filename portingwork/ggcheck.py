import struct, math

path = r"C:\models\Ternary-Bonsai-2-2B-PTQ1_0.gguf"
data_start = 10999008
name = b"blk.0.ssm_out.weight"

f = open(path, "rb")
assert f.read(4) == b"GGUF"
ver, ntensor, nkv = struct.unpack("<IQQ", f.read(20))

def rd_str():
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)

def rd_val(t):
    if t == 0: return struct.unpack("<B", f.read(1))[0]
    if t == 1: return struct.unpack("<b", f.read(1))[0]
    if t == 2: return struct.unpack("<H", f.read(2))[0]
    if t == 3: return struct.unpack("<h", f.read(2))[0]
    if t == 4: return struct.unpack("<I", f.read(4))[0]
    if t == 5: return struct.unpack("<i", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return struct.unpack("<B", f.read(1))[0]
    if t == 8: return rd_str().decode()
    if t == 9:
        t2 = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [rd_val(t2) for _ in range(n)]
    if t == 10:
        t2 = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [rd_val(t2) for _ in range(n)]
    raise ValueError(t)

for _ in range(nkv):
    rd_str(); rd_val(struct.unpack("<I", f.read(4))[0])

toff = None
for _ in range(ntensor):
    nm = rd_str()
    nd = struct.unpack("<I", f.read(4))[0]
    dims = struct.unpack("<" + "Q"*nd, f.read(8*nd))
    t = struct.unpack("<I", f.read(4))[0]
    o = struct.unpack("<Q", f.read(8))[0]
    if nm == name:
        toff, tt, dims = o, t, dims
        break
assert toff is not None, "tensor not found"
print("tensor type", tt, "dims", dims, "offset", toff)

base = data_start + toff
f.seek(base + 0)   # row 0, block 0
blk = f.read(28)
w = struct.unpack("<7I", blk)
print("words:", [hex(x) for x in w])

def digit(b, n):  # the proven mod-multiply decode
    return (((b * [1,3,9,27,81][n]) & 255) * 3) >> 8

def h2f(u16):
    return struct.unpack("<e", struct.pack("<H", u16))[0]

x = [math.sin(0.01*i)*0.5 for i in range(128)]

# bytenr semantics: per byte off, p = sum_n digit_n(b_off) * x[pos(off,n)]
def pos(off, n):
    if off < 16:  return off + 16*n
    if off < 24:  return 80 + (off-16) + 8*n
    return 120 + 2*n + (off-24)

bytenr_parts = []
bytes_ = []
for i in range(6):
    for q in range(4):
        bytes_.append((w[i] >> (8*q)) & 255)
# bytes 24,25 from w6
b24 = w[6] & 255; b25 = (w[6] >> 8) & 255

p = 0.0
for off in range(26):
    b = bytes_[off] if off < 24 else (b24 if off == 24 else b25)
    nn = 5 if off < 24 else 4
    pb = 0.0
    for n in range(nn):
        pb += (digit(b, n) - 1) * x[pos(off, n)]
    bytenr_parts.append(pb)
    p += pb
scale = h2f(w[6] >> 16)
print("bytenr  s =", scale * p)

# g5tab semantics: T[j*243+c] built with d_i = digit_i(c) (mod-multiply), then s = sum_j T[j][byte_j]
T = {}
for j in range(24):
    for c in range(256):
        s = 0.0
        for i in range(5):
            if j < 16:  col = j + 16*i
            else:       col = 80 + (j-16) + 8*i
            s += (digit(c, i) - 1) * x[col]
        T[(j, c)] = s
for m in range(2):
    for c in range(256):
        s = 0.0
        for i in range(4):
            s += (digit(c, i) - 1) * x[120 + m + 2*i]
        T[(24+m, c)] = s

s = 0.0
gt_parts = []
for off in range(26):
    b = bytes_[off] if off < 24 else (b24 if off == 24 else b25)
    gt_parts.append(T[(off, b)])
    s += T[(off, b)]
print("g5tab   s =", scale * s)

# per-byte compare
for off in range(26):
    b = bytes_[off] if off < 24 else (b24 if off == 24 else b25)
    d = abs(bytenr_parts[off] - T[(off, b)])
    if d > 1e-6:
        print(f"  MISMATCH byte {off} (val {b}): bytenr {bytenr_parts[off]:.6f} vs T {T[(off,b)]:.6f}  digits bytenr={[digit(b,n) for n in range(5)]}")
print("done")
