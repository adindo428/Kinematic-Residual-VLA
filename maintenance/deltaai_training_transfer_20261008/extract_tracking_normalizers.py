"""Copy only the exact training normalization tensors from Palmetto."""
import json, struct, sys
from pathlib import Path
source=Path('/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_baseline_16623336/model.best_validation_action_mse.safetensors')
with source.open('rb') as f:
    length=struct.unpack('<Q',f.read(8))[0]
    header=json.loads(f.read(length)); base=8+length
    out={}; payload=bytearray()
    for key,value in header.items():
        if key.split('.')[0] not in ('normalize_inputs','normalize_targets','unnormalize_outputs'): continue
        start,end=value['data_offsets']; f.seek(base+start)
        raw=f.read(end-start)
        assert len(raw)==end-start
        entry=dict(value); entry['data_offsets']=[len(payload),len(payload)+len(raw)]
        out[key]=entry; payload.extend(raw)
    assert all(any(k.startswith(prefix+'.') for k in out) for prefix in ('normalize_inputs','normalize_targets','unnormalize_outputs'))
    encoded=json.dumps(out,separators=(',',':')).encode(); encoded+=b' '*((-len(encoded))%8)
    sys.stdout.buffer.write(struct.pack('<Q',len(encoded))+encoded+payload)
