"""Extract per-node reference values from llama-eval-callback output (token 0 row)."""
import re, sys

PATH = 'ref27.txt'
WANT = ['model.input_embed','norm-0','attn_norm-0','node_17','node_19','node_21',
        'conv_output_silu-0','qk_conv_l2-0','gate-0','beta_sigmoid-0','attn_output-0',
        'node_61','linear_attn_out-0','attn_residual-0','ffn_gate-0','l_out-0',
        'result_output','node_0']

lines = open(PATH, encoding='utf-8', errors='replace').read().splitlines()

def find_node(name):
    """Return index of the node's definition line."""
    for i, l in enumerate(lines):
        if 'common_debug_cb_eval' not in l: continue
        # match "  <name> = (f32)" or "  <name> (something) ="
        m = re.match(r'common_debug_cb_eval:\s+(.+?)\s*=\s*\((\w+)\)', l)
        if m and m.group(1).strip() == name:
            return i
    return None

def first_row_values(i):
    """From line i, read the following value block; return first 3 and last 3 floats."""
    vals = []
    depth_rows = []
    j = i + 1
    started = False
    cur = []
    while j < len(lines) and j < i + 4000:
        l = lines[j]
        if 'common_debug_cb_eval' in l:
            break
        if l.strip().startswith('['):
            started = True
        nums = re.findall(r'-?\d+\.\d+(?:[eE][-+]?\d+)?', l)
        if nums:
            cur.extend(float(x) for x in nums)
        if started and l.strip().startswith('],') :
            if cur:
                depth_rows.append(cur); cur = []
            if len(depth_rows) >= 1:
                break
        j += 1
    if cur and not depth_rows:
        depth_rows.append(cur)
    if not depth_rows:
        return None
    r = depth_rows[0]
    return r

for name in WANT:
    i = find_node(name)
    if i is None:
        print('%-22s  NOT FOUND' % name); continue
    r = first_row_values(i)
    if not r:
        print('%-22s  (no values)' % name); continue
    head = r[:3]
    tail = r[-3:] if len(r) > 3 else []
    print('%-22s  first3=%s%s' % (name, ['%.5f'%v for v in head],
          ('  last3=%s' % ['%.5f'%v for v in tail]) if tail else ''))
