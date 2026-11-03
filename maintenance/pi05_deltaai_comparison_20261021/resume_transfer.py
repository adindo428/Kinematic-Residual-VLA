"""Resume only a verified exact prefix of the approved immutable checkpoint."""
import subprocess,json,datetime,shlex
from stage import HERE,P,D,DEST,stream
SOURCE='/scratch/aparame/Dynamic_VLA/pi05-cpu-pipeline-20261021/snapshot/checkpoint-1792595272349606763.pt'
EXPECTED='09c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c'
TOTAL=40456772509
def output(session,command):return subprocess.check_output(session+[command],text=True).strip()
def main():
    for session in (P,D):subprocess.run(session[:-1]+['-O','check',session[-1]],check=True)
    partial=DEST+'/checkpoint.pt.partial';final=DEST+'/checkpoint.pt'
    assert output(P,f'stat -c %s {SOURCE}')==str(TOTAL),'Source snapshot size changed'
    assert output(D,f'test ! -e {final} && echo absent')=='absent','Reconcile completed checkpoint before retry'
    before=output(D,f'stat -c "%s:%Y:%i" {partial}');size=int(before.split(':')[0])
    assert 0<size<TOTAL,'Unexpected partial size; reconcile before transfer'
    target_hash=output(D,f'sha256sum {partial}').split()[0]
    source_hash=output(P,f'set -o pipefail; head -c {size} {SOURCE} | sha256sum').split()[0]
    assert target_hash==source_hash,'Partial file is not the source prefix'
    assert output(D,f'stat -c "%s:%Y:%i" {partial}')==before,'Partial file changed during reconciliation'
    receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified_prefix_bytes':size,'prefix_sha256':source_hash,'source_checkpoint':SOURCE,'target_checkpoint':final,'checkpoint_sha256':EXPECTED,'production_modified':False,'mac_checkpoint_files_created':False}
    (HERE/'resume-prefix-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    check='test "$(stat -c \'%s:%Y:%i\' '+partial+')" = '+shlex.quote(before)
    target=f'{check} && cat >> {partial} && test "$(stat -c %s {partial})" = {TOTAL} && test "$(sha256sum {partial} | cut -d\" \" -f1)" = {EXPECTED} && mv {partial} {final}'
    stream(f'tail -c +{size+1} {SOURCE}',target)
    receipt['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (HERE/'transfer.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
