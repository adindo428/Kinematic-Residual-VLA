"""Explicitly authorized independent replicas; no canonical tracker is reused."""
import socket

NAMES = ('pi05_baseline', 'pi05_kr', 'vlash_baseline', 'vlash_kr')


def layout(kind):
    if kind == 'a100':
        return dict(nodes=2, local_world=2, world=4, cpus=32, memory_gib=224,
                    gpu='a100', constraint='gpu_a100_80gb', minimum_gib=75)
    if kind == 'h200':
        return dict(nodes=1, local_world=2, world=2, cpus=32, memory_gib=256,
                    gpu='h200', constraint='gpu_h200', minimum_gib=120)
    raise ValueError('Unknown placement')


def memory_by_node(observations):
    nodes = {}
    for item in observations:
        node = nodes.setdefault(item['hostname'], {'rss': 0, 'limit': item['limit']})
        if node['limit'] != item['limit']:
            raise ValueError('Ranks disagree about node memory allocation')
        node['rss'] += item['rss']
    return nodes, all(n['rss'] <= .85*n['limit'] for n in nodes.values())


def memory_observation(rss, limit):
    return dict(hostname=socket.gethostname(), rss=rss, limit=limit)
