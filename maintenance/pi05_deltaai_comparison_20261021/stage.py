"""Stage a read-only benchmark copy; checkpoint bytes never enter a Mac file."""
import subprocess, json, pathlib, datetime, hashlib
HERE=pathlib.Path(__file__).resolve().parent
P=['ssh','-o','ControlMaster=no','-o','ControlPath=/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','aparame@slogin.palmetto.clemson.edu']
D=['ssh','-o','ControlMaster=no','-o','ControlPath=/private/tmp/deltaai-kinematic.sock','-o','BatchMode=yes','deltaai']
DEST='/work/nvme/bivv/aparameshwaran/dynamicvla/pi05-comparison-20261021'
ROOT='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020'
SITE='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/env/lib/python3.10/site-packages'
def stream(command, target):
    source=subprocess.Popen(P+[command],stdout=subprocess.PIPE)
    sink=subprocess.run(D+[target],stdin=source.stdout)
    source.stdout.close()
    assert source.wait()==0 and sink.returncode==0, 'Transfer failed; reconcile partial assets before retry'
def main():
    # Refuse to fall back to a new login when an authenticated master expires.
    for session in (P,D):
        subprocess.run(session[:-1]+['-O','check',session[-1]],check=True)
    # Source packages contain code/resources only. Dereference tokenizer symlinks.
    stream(f'tar -czf - -C {SITE} lerobot -C {SITE}/transformers/models/gemma modeling_gemma.py',f'tar -xzf - -C {DEST}/pinned')
    stream('tar -chzf - -C /scratch/aparame/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c .',f'mkdir -p {DEST}/tokenizer && tar -xzf - -C {DEST}/tokenizer')
    names=['statistics.json','data-provenance.json','split.json','parquet_hashes.json','video_hashes.json','implementation-lock.json','checkpoints.json','source-manifest.json','recipes/pi05_kr.json']
    stream(f'tar -czf - -C {ROOT} '+ ' '.join(names),f'tar -xzf - -C {DEST}/source')
    snapshot='/scratch/aparame/Dynamic_VLA/pi05-cpu-pipeline-20261021/snapshot/checkpoint-1792595272349606763.pt'
    # An encrypted pipe connects the two authenticated sessions. No local weight,
    # checkpoint, or dataset file is created; final SHA is checked at destination.
    stream(f'cat {snapshot}',f'cat > {DEST}/checkpoint.pt.partial && sha256sum {DEST}/checkpoint.pt.partial && test "$(sha256sum {DEST}/checkpoint.pt.partial | cut -d\" \" -f1)" = 09c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c && mv {DEST}/checkpoint.pt.partial {DEST}/checkpoint.pt')
    receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_checkpoint':snapshot,'target_checkpoint':DEST+'/checkpoint.pt','checkpoint_sha256':'09c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c','progress':{'updates':60004,'samples':1920128,'epoch':0,'cursor':1920128},'production_modified':False,'mac_checkpoint_files_created':False}
    (HERE/'transfer.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
