"""Stage a controlled native contact-friction sweep without modifying prior runs."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path('/scratch/aparame/Dynamic_VLA/runs/official_dom_diagnostic_20261009')
STAGE=Path('/scratch/aparame/Dynamic_VLA/runs/official_dom_friction_20261009')
ROOT=Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_friction/20261009')
LOCAL=Path('/home/aparame/Research/Dynamic_VLA/maintenance/official_dom_friction_20261009')
if STAGE.exists() or ROOT.exists():
    raise FileExistsError('Preserve existing stages/results')
STAGE.mkdir(parents=True); ROOT.mkdir(parents=True)
(STAGE/'scripts').mkdir()
(STAGE/'checkpoint').symlink_to(BASE/'checkpoint',target_is_directory=True)
for name in ('report.py','prepare.py','probe_policy.py'):
    shutil.copy2(BASE/'scripts'/name,STAGE/'scripts'/name)
shutil.copy2(LOCAL/'friction.py',STAGE/'scripts/friction.py')
text=(BASE/'scripts/observer.py').read_text()
needle="        robot = env.unwrapped.scene['robot']"
assert text.count(needle)==1
text=text.replace(needle,needle+'''
        from friction import apply as apply_contact_friction
        coefficient = os.environ.get('DOM_FINGER_CONTACT_FRICTION', 'unchanged')
        coefficient = None if coefficient == 'unchanged' else float(coefficient)
        friction_report = apply_contact_friction(env, coefficient)
''')
needle="            with destination.open('a') as f: f.write(json.dumps(row)+'\\n')"
assert text.count(needle)==1
text=text.replace(needle,"            row['finger_contact_friction'] = friction_report\n"+needle)
# Post-trial native readback ensures the reset/control loop did not overwrite the setting.
needle="        n_steps = len(result['ee_path'])"
assert text.count(needle)==1
text=text.replace(needle,'''        after_report = apply_contact_friction(env, None)
        if after_report['finger_materials_after'] != friction_report['finger_materials_after']:
            raise RuntimeError('Finger contact friction changed during the trial')
'''+needle)
ast.parse(text); (STAGE/'scripts/observer.py').write_text(text)
text=(BASE/'scripts/run.py').read_text()
needle="        port=22000+(int(os.environ['SLURM_JOB_ID'])%17000)+2*number"
assert text.count(needle)==1
text=text.replace(needle,"        env['DOM_FINGER_CONTACT_FRICTION']=str(protocol['variants'][variant]['finger_contact_coefficient'])\n"+needle)
needle="        for k,v in values.items(): server_cmd+=['--env',k+'='+v]"
assert text.count(needle)==1
text=text.replace(needle,"        values['DOM_FINGER_CONTACT_FRICTION']=env['DOM_FINGER_CONTACT_FRICTION']\n"+needle)
ast.parse(text);(STAGE/'scripts/run.py').write_text(text)
protocol=json.loads((Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_diagnostic/20261009')/'protocol.json').read_text())
variants={}
for label, coefficient in [('control','unchanged'),('contact_mu1',1.0),('contact_mu2',2.0)]:
    source=STAGE/'sources'/label
    shutil.copytree(BASE/'sources/gripper_aligned',source,ignore=shutil.ignore_patterns('__pycache__'))
    variants[label]={'source':str(source),'gripper_fix':True,'align_actions':True,
                     'finger_contact_coefficient':coefficient,'source_sha256':{
                     str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source.rglob('*.py')}}
protocol.update(benchmark='DOM finger contact friction diagnostic, not official Table 1 reproduction',
                variants=variants,checkpoint=str(STAGE/'checkpoint'),requested_gpu='L40S:1',gpu_label='shared L40S',
                expected_trials_per_variant=6,total_requested_trials=18,
                manipulated_factor='Only static and dynamic contact friction on left/right finger collision shapes',
                controls_unchanged=['object contact material','all nonfinger materials','restitution','combine modes',
                                    'joint friction','finger speed/effort','model weights','task configs','scoring','latency sleep'],
                limitations=['Three cases with two repeats per arm are diagnostic only.',
                             'Coefficients 1 and 2 are exploratory physics interventions, not published benchmark scores.'])
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
shutil.copy2(Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_diagnostic/20261009/test-envs.txt'),ROOT/'test-envs.txt')
print(json.dumps({'stage':str(STAGE),'root':str(ROOT),'variants':list(variants)},indent=2))
