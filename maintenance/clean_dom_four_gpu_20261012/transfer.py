"""SCP the immutable, individually hash-checked bundle; submit checksum extraction only."""
import json,subprocess,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONFIG='/tmp/krvla-clean-transfer-ssh.conf'
REMOTE='/scratch/aparame/Dynamic_VLA/clean-dom-four-gpu-20261012'
DEST='/work/nvme/bivv/aparameshwaran/dynamicvla/dom-audited-20261012'
state={'pack_job':'17129784','delta_audit_cancelled':'3369190','stage':'waiting_for_pack','pid':os.getpid()}
def save(**kw):
 state.update(kw,updated_unix=time.time());p=ROOT/'transfer-state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'transfer-state.json')
def remote(host,cmd):return subprocess.check_output(['ssh','-F',CONFIG,host,cmd],text=True,timeout=60).strip()
def scp(source,dest):subprocess.run(['scp','-3','-F',CONFIG,source,dest],check=True)
try:
 save()
 while True:
  value=remote('pal-audit',"test ! -f "+REMOTE+"/dom-clean.status.json || cat "+REMOTE+"/dom-clean.status.json")
  if value:
   progress=json.loads(value);save(pack_progress=progress)
   if progress['state']=='complete':break
  status=remote('pal-audit','sacct -X -n -j 17129784 -P --format=State').strip('|\n ')
  if any(x in status for x in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY']):raise RuntimeError('Packaging job '+status)
  time.sleep(30)
 save(stage='scp_in_progress')
 scp('pal-audit:'+REMOTE+'/dom-clean.tar','delta-audit:'+DEST+'/dom-clean.tar')
 scp('pal-audit:'+REMOTE+'/dom-clean.sha256','delta-audit:'+DEST+'/dom-clean.sha256')
 save(stage='scp_complete')
 jid=remote('delta-audit','sbatch --parsable '+DEST+'/verify_unpack.sbatch').split(';')[0]
 assert jid.isdigit(),jid
 save(stage='waiting_for_destination_verification',verification_job=jid)
except Exception as exc:
 save(stage='error',error=str(exc));raise
