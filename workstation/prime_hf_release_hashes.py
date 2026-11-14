"""Reuse verified local hashes in the Hub SDK's resumable upload cache.

This records no upload/commit claims. The SDK still negotiates and uploads each
file, and the publisher independently verifies remote content hashes afterward.
"""
import json
import time
from huggingface_hub._local_folder import read_upload_metadata, get_local_upload_paths
from prepare_hf_checkpoint_release import ROOT, STAGE


def main():
    seen=set();primed=0
    while True:
        try:
            state=json.loads((ROOT/'upload_progress.json').read_text())
            if state['stage'] in {'uploading_all_retained_checkpoints','verifying_remote_files','complete','failed'}:break
            cached=json.loads((ROOT/'hash_cache.private.json').read_text())
            for key,entry in cached.items():
                if key in seen:continue
                path=STAGE/key
                if not path.is_file():continue
                stat=path.stat()
                if (stat.st_size,stat.st_mtime_ns)!=(entry['bytes'],entry['mtime_ns']):continue
                metadata=read_upload_metadata(STAGE,key)
                if metadata.sha256 is None:
                    metadata.sha256=entry['sha256'];metadata.save(get_local_upload_paths(STAGE,key));primed+=1
                else:assert metadata.sha256==entry['sha256']
                seen.add(key)
            (ROOT/'hash_reuse.json').write_text(json.dumps({'verified_hashes_reused':primed,'files_seen':len(seen),'remote_upload_or_commit_status_modified':False})+'\n')
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(5)
    print(json.dumps({'verified_hashes_reused':primed}),flush=True)


if __name__=='__main__':main()
