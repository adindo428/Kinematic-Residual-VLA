"""Summarize completed and partial audit shards without touching dataset files."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args();root=Path(a.results)
rows=[json.loads(p.read_text()) for p in sorted(root.glob('status-*.json'))]
report={'shards_seen':len(rows),'shards_complete':sum(r['state']=='complete' for r in rows),'completed_episodes':sum(r['completed_episodes'] for r in rows),'failed_episodes':sum(r['failed_episodes'] for r in rows),'frames_decoded':sum(r['frames_decoded'] for r in rows),'metadata_errors':sorted({s for r in rows for s in r['metadata_errors']}),'warnings':sorted({s for r in rows for s in r['warnings']})}
if rows:
 report['expected_shards']=rows[0]['shards'];report['total_retained_episodes']=rows[0]['total_retained_episodes'];report['all_episodes_checked']=report['shards_complete']==report['expected_shards'] and report['completed_episodes']==report['total_retained_episodes']
 report['elapsed_hours']=(time.time()-min(r['started_unix'] for r in rows))/3600
 report['clean_for_checked_criteria']=report['all_episodes_checked'] and not report['failed_episodes'] and not report['metadata_errors']
print(json.dumps(report,indent=2))
