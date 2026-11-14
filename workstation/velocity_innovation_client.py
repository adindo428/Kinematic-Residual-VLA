"""Isolated resident client for the training-free comparison."""
from pathlib import Path

source = Path(__file__).with_name('resident_table1_client.py').read_text()
source = source.replace('from resident_table1_worker import ResidentPool,minimum_retention',
    'from velocity_innovation_worker import ResidentPool,minimum_retention')
old = "    return source.replace('time.sleep(10)','_resident_fence()')"
new = '''    source = source.replace('time.sleep(10)', '_resident_fence()')
    old = 'act_socket.send_pyobj({"action": action}, flags=zmq.NOBLOCK)'
    assert source.count(old) == 1
    return source.replace(old, 'act_socket.send_pyobj({"action": action, "chunk_origin_index": vla_model.chunk_origin_index}, flags=zmq.NOBLOCK)')'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '            front.reset();module._resident_fence=front.idle'
assert source.count(old) == 1
source = source.replace(old, old+"\n            front._episode_seed = int(job['seed'])\n            front.chunk_origin_index = None")
old = "            module.save_test_results(results,str(folder/'actions'),job['model'])"
new = '''            if hasattr(module, 'save_test_results'):
                module.save_test_results(results,str(folder/'actions'),job['model'])
            else:
                write(folder/'actions/closed_loop_results.json', {
                    'variant':job['model'], 'correction_mode':job['correction_mode'],
                    'environments':{name:dict(trials=len(trials),
                        successes=sum(bool(t['success']) for t in trials))
                        for name,trials in results.items()},
                    'initial_speed_mps':job['initial_speed_mps']})'''
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(Path(__file__).with_name('resident_table1_client.py')), 'exec'))
