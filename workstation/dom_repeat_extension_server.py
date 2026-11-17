"""Original DOM wrapper with pass-aware prompt keys for repeats11--20."""
from pathlib import Path
import krv_full_dom_server as full

def prompt(self,cfg):
    from utils.instruction_generator import InstructionGenerator
    resident=full.resident;scene=resident.CURRENT['scene']
    index=resident.COUNTS[Path(scene).stem]+resident.CURRENT.get('repeat_offset',0)+1
    saved=resident.PROMPTS.setdefault(scene,{})
    if str(index) not in saved:saved[str(index)]=InstructionGenerator.generate_instruction(cfg['instruction'])
    resident.LAST_PROMPT=saved[str(index)]
    return resident.LAST_PROMPT

if __name__=='__main__':
    full.profile.server.prompt=prompt
    full.resident.patch_source=full.patch_source
    full.profile.server.main()
