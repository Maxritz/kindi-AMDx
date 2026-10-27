import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 't1.json'))
tot = 0
for s in sorted(d['shaders'], key=lambda s: -s['dev_us']):
    tot += s['dev_us']
    print("%-24s ops=%5d  dev_us=%10.1f  us/op=%7.1f" % (s['name'], s['ops'], s['dev_us'], s['dev_us']/max(1,s['ops'])))
print("TOTAL dev_us = %.1f" % tot)
