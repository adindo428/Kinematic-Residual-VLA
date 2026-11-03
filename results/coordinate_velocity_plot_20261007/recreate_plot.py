"""Recreate only the requested comparison plot from retained evaluation metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ('baseline', 'velocity', 'coordinate', 'coordinate_velocity')
LABELS = {'baseline': 'DOM-VLA', 'velocity': 'DOM-VLA+Velocity',
          'coordinate': 'KR-VLA-Coordinate', 'coordinate_velocity': 'KR-VLA-Coordinate+Velocity'}
COLORS = {'baseline': '#2878B5', 'velocity': '#218C64',
          'coordinate': '#8555B5', 'coordinate_velocity': '#C94F66'}


def object_speeds(root, manifest):
    report = {}
    seconds = manifest['config']['collection']['seconds']
    hz = manifest['config']['collection']['physics_hz']
    for speed in manifest['speed_multipliers']:
        scenario = 'iid' if speed == 1 else f'ood_{speed:g}x'
        values, by_episode = [], {}
        for item in manifest['episodes']:
            ep = item['episode']
            path = root/'baseline'/scenario/f'episode_{ep:06d}'/'steps.jsonl'
            episode_values = []
            for line in path.open():
                row = json.loads(line)
                magnitude = float(np.linalg.norm(row['object_velocity']))
                assert np.isclose(magnitude, row['object_speed_mps'], rtol=1e-9, atol=1e-12)
                episode_values.append(magnitude)
            assert len(episode_values) == round(seconds*hz), 'Average speed needs a full object trajectory'
            values.extend(episode_values)
            by_episode[str(ep)] = float(np.mean(episode_values))
        report[f'{speed:g}x'] = {
            'mean_object_speed_mps': float(np.mean(values)), 'samples': len(values),
            'episodes': len(by_episode), 'duration_seconds_per_episode': seconds,
            'physics_hz': hz, 'per_episode_mean_object_speed_mps': by_episode,
            'calculation': 'Time average of norm(current robot-base XYZ object velocity); each trajectory counted once, using full baseline telemetry.'}
    return report


def recreate(root):
    summary = json.loads((root/'summary.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert summary['complete'] and summary['pairing_valid']
    speeds = manifest['speed_multipliers']
    speed_report = object_speeds(root, manifest)
    x = np.arange(len(speeds)); width = .8/len(MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plotted = {}
    for i, model in enumerate(MODELS):
        groups = [summary['groups'][f'{model}_{s:g}x'] for s in speeds]
        xx = x+(i-(len(MODELS)-1)/2)*width
        errors = [1000*g['completed_mean_hover_rmse_m'] for g in groups]
        coverage = [100*g['completed_mean_position_within_2cm_fraction'] for g in groups]
        bars = axes[0].bar(xx, errors, width, color=COLORS[model], label=LABELS[model])
        axes[0].bar_label(bars, labels=[f'{v:.1f}' for v in errors], padding=4, fontsize=9)
        bars = axes[1].bar(xx, coverage, width, color=COLORS[model])
        axes[1].bar_label(bars, labels=[f'{v:.1f}' for v in coverage], padding=4, fontsize=9)
        plotted[model] = {'hover_rmse_mm': errors, 'time_within_2cm_percent': coverage}
    ticks = [('IID' if speed == 1 else 'OOD')+
             f'\n{speed:g}x speed\nMean: {speed_report[f"{speed:g}x"]["mean_object_speed_mps"]:.3f} m/s'
             for speed in speeds]
    for ax in axes:
        ax.set_xticks(x, ticks, fontsize=10)
        ax.set_xlabel('Scenario and mean object speed', fontsize=10, labelpad=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_title('Position tracking error (lower is better)', fontsize=12)
    axes[0].set_ylabel('Mean episode 3D hover RMSE (mm)', fontsize=10)
    axes[0].set_ylim(0, 130)
    axes[1].set_title('Time within 2 cm of hover target (higher is better)', fontsize=12)
    axes[1].set_ylabel('Mean rollout time (%)', fontsize=10)
    axes[1].set_ylim(0, 40); axes[1].set_yticks(range(0, 41, 10))
    fig.suptitle('Live closed-loop tracking | accuracy and time within tolerance',
                 fontsize=17, fontweight='bold', y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5,.915), ncol=2, frameon=False, fontsize=11)
    fig.text(.5,.058,'Metrics use completed rollouts; one original Coordinate trial terminated early at 2x speed.',
             ha='center', fontsize=9)
    fig.text(.5,.022,'Simulation pauses for inference. Mean object speed is the time average of velocity magnitude over full 10-second trajectories.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.74, bottom=.255, left=.075, right=.98, wspace=.27)
    destination = root/'tracking_comparison_no_field.png'
    fig.savefig(destination, dpi=180); plt.close(fig)
    metadata = {'source_summary': str(root/'summary.json'), 'models': LABELS,
                'speed_multipliers': speeds, 'object_speed': speed_report, 'plotted_values': plotted,
                'metric_population': 'Means over completed rollouts, preserving the original summary denominators.',
                'other_evaluations_rerun': False}
    (root/'tracking_comparison_no_field_metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'plot': str(destination), 'mean_object_speed_mps':
                      {k: v['mean_object_speed_mps'] for k,v in speed_report.items()}}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    recreate(p.parse_args().root)
