"""Bounded completion relay for this launch's existing checkpoint transfers."""
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT='/work/nvme/bivv/aparameshwaran/dynamicvla/main8-20k-10epochs-20261021'
PALROOT='/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021'
TARGET_PIDS=[27103,27582,27872]
PAL=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=15','aparame@slogin.palmetto.clemson.edu']
DELTA=['ssh','-F','/dev/null','-S','/tmp/delta-krvla.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','aparameshwaran@login.delta.ncsa.illinois.edu']


def persist(value):
    p=HERE/'relay-status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def call(prefix,command,input=None):
    return subprocess.run(prefix+[command],input=input,text=True,capture_output=True,check=True,timeout=60).stdout


def main():
    started=time.time()
    while time.time()-started<4*3600:
        try:
            ps=subprocess.run(['ps','-p',','.join(map(str,TARGET_PIDS)),'-o','pid=,args='],text=True,capture_output=True)
            active=[line.strip() for line in ps.stdout.splitlines() if 'tar -xf -' in line and ROOT in line]
            if active:
                persist({'complete':False,'waiting_for_transfer_pids':[x.split()[0] for x in active], 'seconds':time.time()-started})
                time.sleep(30);continue
            script='''import json
from pathlib import Path
r=Path("/work/nvme/bivv/aparameshwaran/dynamicvla/main8-20k-10epochs-20261021")
sizes={"parents/pi05_baseline/latest.pt":40456772509,"parents/pi05_kr/latest.pt":40456772509,"parents/vlash_baseline/latest.pt":20946855289,"parents/vlash_kr/latest.pt":20946855289,"assets/models--lerobot--pi05_base/snapshots/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0/model.safetensors":14467165872}
assert all((r/name).is_file() and (r/name).stat().st_size==n for name,n in sizes.items()),"Transfer exited before all file sizes completed"
assert (r/"assets/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/tokenizer.json").is_file()
print(json.dumps({"all_target_sizes_complete":True}))
'''
            call(DELTA,'python3 -',script)
            hashes=json.loads(call(PAL,'cat '+PALROOT+'/source-checkpoint-hashes.json'))
            assert set(hashes)=={'pi05_baseline','pi05_kr','vlash_baseline','vlash_kr'}
            (HERE/'source-checkpoint-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
            writer='import json,sys; from pathlib import Path; r=Path('+repr(ROOT)+'); p=r/"source-checkpoint-hashes.json"; t=p.with_suffix(".tmp"); t.write_text(json.dumps(json.load(sys.stdin),indent=2)+"\\n"); t.replace(p)'
            call(DELTA,'python3 -c '+shlex.quote(writer),json.dumps(hashes))
            persist({'complete':True,'all_transfer_processes_finished':True,'all_target_sizes_complete':True,'source_digest_commit_transferred':True,'seconds':time.time()-started})
            print('All checkpoint transfers committed; scheduler release gates can proceed.',flush=True)
            return
        except Exception as exc:
            persist({'complete':False,'error':str(exc),'seconds':time.time()-started})
            time.sleep(30)
    raise RuntimeError('Launch transfer relay exceeded4h; held jobs remain held')


if __name__=='__main__':main()
