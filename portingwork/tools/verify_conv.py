"""Which Hadamard fold convention reproduces the reference ssm_out?

Inputs are all verified: swiglu (== ref node_61) and blk.0.ssm_out.weight (== GPU GEMV).
So solve for the transform T with  ssm_out_ref = W @ T(swiglu).
"""
import struct, numpy as np

GGUF = r"C:\x\Ternary-Bonsai-2-27B-PQ2_0.gguf"
REF = np.array([-0.0204, 0.0285, -0.0247], dtype=np.float64)

def rd_str(f):
    n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode('utf-8','replace')
def rd_val(f,t):
    if t==0: return struct.unpack('<B',f.read(1))[0]
    if t==1: return struct.unpack('<b',f.read(1))[0]
    if t==2: return struct.unpack('<H',f.read(2))[0]
    if t==3: return struct.unpack('<h',f.read(2))[0]
    if t==4: return struct.unpack('<I',f.read(4))[0]
    if t==5: return struct.unpack('<i',f.read(4))[0]
    if t==6: return struct.unpack('<f',f.read(4))[0]
    if t==7: return struct.unpack('<?',f.read(1))[0]
    if t==8: return rd_str(f)
    if t==9:
        et=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        return [rd_val(f,et) for _ in range(n)]
    if t==10: return struct.unpack('<Q',f.read(8))[0]
    if t==11: return struct.unpack('<q',f.read(8))[0]
    if t==12: return struct.unpack('<d',f.read(8))[0]

f=open(GGUF,'rb'); f.read(4); struct.unpack('<I',f.read(4))
nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
kv={}
for _ in range(nkv):
    k=rd_str(f); t=struct.unpack('<I',f.read(4))[0]; kv[k]=rd_val(f,t)
tensors={}
for _ in range(nt):
    nm=rd_str(f); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    tensors[nm]=(dims,ty,off)
kv_end=f.tell(); f.close()
data_start=11120992
HB=int(kv['prism.hadamard.block_size'])

sv=np.asarray(kv['prism.hadamard.sign_values'],dtype=np.float32)
off=0; signs={}
for wdt in kv['prism.hadamard.sign_widths']:
    signs[int(wdt)]=sv[off:off+int(wdt)]; off+=int(wdt)

dims,ty,toff=tensors['blk.0.ssm_out.weight']
C=dims[0]; R=dims[1]; bpr=C//128
with open(GGUF,'rb') as g:
    g.seek(data_start+toff); raw=np.frombuffer(g.read(bpr*34*R),dtype=np.uint8).reshape(R,bpr,34)
d=raw[:,:,0:2].copy().view('<f2').astype(np.float32).reshape(R,bpr)
qs=raw[:,:,2:34]
j=np.arange(128)
q=(qs[:,:,j>>2]>>(2*(j&3)))&3
W=((q.astype(np.float32)-1.0)*d[:,:,None]).reshape(R,C)

z   =np.fromfile('dv_z.bin',dtype=np.float32).astype(np.float64)
norm=np.fromfile('dv_gdn_norm.bin',dtype=np.float32).astype(np.float64)
swiglu=(z/(1.0+np.exp(-z)))*norm
sg=signs[C].astype(np.float64)

def fwht_blocks(x, hb):
    y=x.copy(); C=x.size
    for c0 in range(0,C,hb):
        blk=y[c0:c0+hb]; h=1
        while h<hb:
            blk=blk.reshape(-1,2,h)
            a=blk[:,0,:].copy(); b=blk[:,1,:].copy()
            blk[:,0,:]=a+b; blk[:,1,:]=a-b
            blk=blk.reshape(-1); h*=2
        y[c0:c0+hb]=blk
    return y

cands = {
 'FWHT(sg*x)/sqrt(HB)   [current]': fwht_blocks(sg*swiglu,HB)/np.sqrt(HB),
 'sg*FWHT(x)/sqrt(HB)             ': sg*(fwht_blocks(swiglu,HB)/np.sqrt(HB)),
 'FWHT(x)*sg/sqrt(HB)  (== above) ': (fwht_blocks(swiglu,HB)/np.sqrt(HB))*sg,
 'FWHT(x)/sqrt(HB) no signs       ': fwht_blocks(swiglu,HB)/np.sqrt(HB),
 'sg*x  (no FWHT)                 ': sg*swiglu,
 'x     (no fold)                 ': swiglu,
 'FWHT(sg*x) unnormalised         ': fwht_blocks(sg*swiglu,HB),
 'sg*FWHT(x) unnormalised         ': sg*fwht_blocks(swiglu,HB),
 'x*HB                              ': swiglu*HB,
 'FWHT_HB(sg*x) where sg=xor      ': fwht_blocks(sg*swiglu,HB)/np.sqrt(HB),
}

print('ref ssm_out = %s' % np.round(REF,4))
print()
best=None
for name,T in cands.items():
    out = W @ T
    d0 = np.abs(out[0:3]-REF).max()
    rel = d0/max(np.abs(REF).max(),1e-9)
    print('%-34s %s  maxerr=%.5f' % (name, np.round(out[0:3],4), d0))
    if best is None or d0<best[1]: best=(name,d0)
print()
print('BEST: %s  (maxerr %.5f)' % best)
