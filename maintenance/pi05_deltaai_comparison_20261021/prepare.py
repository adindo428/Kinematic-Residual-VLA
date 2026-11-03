"""DeltaAI-only path adaptations to a diagnostic copy, never production code."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE/'source'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    original=json.loads((ROOT/'data-provenance.json').read_text())
    original['dataset_root']='/work/nvme/bivv/aparameshwaran/dynamicvla/huggingface/lerobot/hzxie/DOM'
    (ROOT/'data-provenance.json').write_text(json.dumps(original,indent=2)+'\n')
    refs=json.loads((ROOT/'checkpoints.json').read_text())
    refs['google/paligemma-3b-pt-224']['path']=str(HERE/'tokenizer')
    (ROOT/'checkpoints.json').write_text(json.dumps(refs,indent=2)+'\n')
    p=ROOT/'policy.py';s=p.read_text()
    if 'Diagnostic restores the full trusted production snapshot' not in s:
        begin=s.index('        if sha(checkpoint /')
        end=s.index("        if self.method == 'smolvla':",begin)
        s=s[:begin]+s[end:]
        begin=s.index("        weights = load_file(checkpoint / 'model.safetensors')")
        end=s.index('        self.tokenizer =',begin)
        s=s[:begin]+'        # Diagnostic restores the full trusted production snapshot immediately below construction.\n'+s[end:]
    p.write_text(s)
    # Pin the same modified Gemma implementation in this isolated env only.
    import transformers
    gemma=Path(transformers.__file__).parent/'models/gemma/modeling_gemma.py'
    assert str(gemma).startswith(str(HERE/'env')), 'Refuse editing a shared environment'
    gemma.write_bytes((HERE/'pinned/modeling_gemma.py').read_bytes())
    assert sha(HERE/'pinned/lerobot/policies/pi05/modeling_pi05.py')=='5757fbf69fa54f9d4f05b8c113ffbb647ee7d31ae341b7e956dbcb88d2468f3c'
    assert sha(gemma)=='236196720ba5446704cc95a221aa43dd217e39afd72d92496f6307b1cd13a18a'
    paths=[*ROOT.glob('*.py'),*HERE.glob('*.py'),*HERE.glob('*.sh'),*(HERE/'pinned').rglob('*.py'),*ROOT.glob('*.json'),*ROOT.glob('recipes/*.json'),gemma]
    manifest={'files':{str(p.relative_to(HERE)):sha(p) for p in paths},'diagnostic_only':True,'checkpoint_sha256':'09c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c','path_adaptation':True,'production_changed':False}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
