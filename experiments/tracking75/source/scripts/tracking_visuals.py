"""Consistent, explicit method/scenario labels for tracking plots and videos."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tracking_variants import VARIANT_LABELS

MODELS = ('baseline', 'velocity', 'residual', 'coordinate', 'coordinate_velocity')
COLORS = {'baseline':'#2878B5', 'velocity':'#218C64', 'residual':'#D97718', 'coordinate':'#8555B5', 'coordinate_velocity':'#C94F66'}
DESCRIPTIONS = {'baseline':'Standard action flow; no analytic velocity drift',
 'residual':'Raw coordinates; drift at each flow step',
 'coordinate':'Residual coordinates; drift restored once',
 'velocity':'Velocity as neural input; no analytic drift',
 'coordinate_velocity':'Neural velocity + residual coordinates'}
CAMERAS = {'opst_cam':'Overhead camera', 'wrist_cam':'Wrist camera', 'side_cam':'Side camera'}
WIDTH, CAMERA_HEIGHT, HEADER, FOOTER = 480, 360, 96, 56
HEIGHT = HEADER + CAMERA_HEIGHT + FOOTER

@lru_cache(None)
def font(size, bold=False):
    import matplotlib
    filename='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf'/filename),size)


def scenario_text(speed):
    return ('IID' if speed == 1 else 'OOD') + f' | {speed:g}x object speed'


class PanelRenderer:
    def __init__(self, row, camera='opst_cam', position=None):
        self.row=row;self.camera=camera
        self.base=Image.new('RGB',(WIDTH,HEIGHT),'#101620')
        d=ImageDraw.Draw(self.base);model=row['model'];color=COLORS[model]
        d.rectangle((0,0,WIDTH,5),fill=color)
        title=VARIANT_LABELS[model]
        if position:title=f'{position} | {title}'
        title_size=21
        while d.textbbox((0,0),title,font=font(title_size,True))[2] > WIDTH-24:
            title_size-=1
        d.text((12,12),title,font=font(title_size,True),fill=color)
        d.text((12,43),DESCRIPTIONS[model],font=font(14),fill='white')
        d.text((12,69),f"{scenario_text(row.get('speed_multiplier',1))} | Episode {row.get('episode',0):03d}",font=font(15),fill='#D7DFE9')

    def render(self, rgb, index, fps=25):
        panel=self.base.copy();d=ImageDraw.Draw(panel)
        if rgb is not None:
            # Support regenerated comparisons from already-captioned camera videos.
            if rgb.shape[:2] == (HEIGHT,WIDTH):rgb=rgb[HEADER:HEADER+CAMERA_HEIGHT]
            if rgb.shape[:2] != (CAMERA_HEIGHT,WIDTH):raise ValueError(f'Unexpected camera size {rgb.shape}')
            panel.paste(Image.fromarray(rgb),(0,HEADER))
        else:
            d.text((20,HEADER+112),'ROLLOUT STOPPED EARLY',font=font(23,True),fill='#FF8989')
            d.text((20,HEADER+155),'Joint-limit guard',font=font(20),fill='white')
            d.text((20,HEADER+191),f"Recorded {self.row.get('recorded_seconds',0):.3f} of {self.row.get('requested_seconds',10):g} seconds",font=font(16),fill='#D7DFE9')
            d.text((20,HEADER+220),'No further robot frames were recorded.',font=font(16),fill='#D7DFE9')
        y=HEADER+CAMERA_HEIGHT
        d.text((12,y+6),f"{CAMERAS[self.camera]} | t={index/fps:.2f} / {self.row.get('requested_seconds',10):g} s",font=font(16),fill='white')
        if self.row['metrics']['completed']:
            status='Outcome: full horizon recorded; see tracking metrics'
            color='#D7DFE9'
        else:
            status=f"Early stop at {self.row.get('recorded_seconds',0):.3f} s | joint-limit guard"
            color='#FF8989'
        d.text((12,y+31),status,font=font(14),fill=color)
        return np.asarray(panel)


def _style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)


def plot_offline(report, rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=list(report['models']);eps=report['episodes'];x=np.arange(len(eps));width=.8/len(models)
    fig,ax=plt.subplots(figsize=(12.5,5.5))
    for i,m in enumerate(models):
        by_ep={int(r['episode']):r for r in rows if r['model']==m}
        values=[float(by_ep[e]['xyz_coordinate_rmse_mm']) for e in eps]
        ax.bar(x+(i-(len(models)-1)/2)*width,values,width,color=COLORS[m],label=VARIANT_LABELS[m])
    ax.set_xticks(x,[f'{e:03d}' for e in eps]);ax.set_xlabel('Held-out episode (IID recorded observations)')
    ax.set_ylabel('XYZ coordinate RMSE (mm) — lower is better');_style(ax)
    fig.suptitle('Offline imitation | position accuracy by episode',fontsize=16,fontweight='bold',y=.98)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=len(models),frameon=False)
    fig.text(.5,.035,'Same 2,310 valid anchors and 20 predicted actions per anchor. Recorded-data imitation; no policy-controlled rollout.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.78,bottom=.18,left=.09,right=.98);fig.savefig(destination,dpi=180);plt.close(fig)


def plot_live(summary, speeds, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=list(summary['models']);x=np.arange(len(speeds));width=.8/len(models)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for i,m in enumerate(models):
        groups=[summary['groups'][f'{m}_{s:g}x'] for s in speeds]
        xx=x+(i-(len(models)-1)/2)*width
        values=[1000*g['completed_mean_hover_rmse_m'] if g['completed_mean_hover_rmse_m'] is not None else np.nan for g in groups]
        bars=axes[0].bar(xx,values,width,color=COLORS[m],label=VARIANT_LABELS[m])
        axes[0].bar_label(bars,labels=[f'{v:.1f}\nn={g["completed"]}' if np.isfinite(v) else 'no completed\nrollouts' for v,g in zip(values,groups)],padding=4,fontsize=9)
        counts=[g['completed'] for g in groups]
        bars=axes[1].bar(xx,counts,width,color=COLORS[m])
        axes[1].bar_label(bars,labels=[f'{g["completed"]}/{g["expected"]}' for g in groups],padding=4,fontsize=10)
    for ax in axes:
        ax.set_xticks(x,[('IID' if s==1 else 'OOD')+f'\n{s:g}x speed' for s in speeds]);ax.set_xlabel('Object trajectory speed scenario');_style(ax)
    axes[0].set_title('Position tracking error | completed rollouts only',fontsize=12)
    axes[0].set_ylabel('Mean episode 3D hover RMSE (mm) — lower is better');axes[0].margins(y=.25)
    axes[1].set_title('Full 10-second rollouts | all attempted trials',fontsize=12)
    axes[1].set_ylabel('Completed / attempted episodes — higher is better');axes[1].set_ylim(0,12);axes[1].set_yticks(range(0,11,2))
    fig.suptitle('Live closed-loop tracking | accuracy and completion',fontsize=17,fontweight='bold',y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.915),ncol=min(3,len(models)) if len(models)>4 else len(models),frameon=False,fontsize=11)
    fig.text(.5,.06,'n = completed episodes contributing to the error mean. Early terminations are excluded from error means and retained in completion counts.',ha='center',fontsize=9)
    fig.text(.5,.025,'Completion is not task success. Simulation pauses for inference. Faster scenarios also increase acceleration.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.74 if len(models)>4 else .77,bottom=.21,left=.075,right=.98,wspace=.26);fig.savefig(destination,dpi=180);plt.close(fig)
