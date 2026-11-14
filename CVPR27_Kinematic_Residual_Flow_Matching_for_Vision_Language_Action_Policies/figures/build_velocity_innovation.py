#!/usr/bin/env python3
"""Single-column KRA schematic; synthetic trajectories, no measured results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
GRAY='#A9AFB7'; BLUE='#3676BC'; ORANGE='#D17A24'; GREEN='#198468'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.4,
 'mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none'})
t=np.linspace(0,1,7)
x=np.column_stack((.30+2.65*t,.22+.42*t))
# A shared elapsed-time translation keeps the straight residual and blue
# trajectories parallel; this is a snapshot illustration, not v0*t_h.
constant=x+np.array([0.,.38])
planned=constant+np.column_stack((.24*t*t,.46*t*t))
w=np.maximum((t-.4)/.6,0)
q=np.column_stack((-.32*w*w,.37*w*w))
executed=planned+q
assert np.allclose(np.diff(x,axis=0),np.diff(constant,axis=0))
assert np.allclose(executed-planned,q)
fig=plt.figure(figsize=(3.35,1.85),facecolor='white')
ax=fig.add_axes([.03,.275,.94,.49]);ax.set_xlim(0,3.65);ax.set_ylim(.08,2.02);ax.axis('off')
intervals=fig.add_axes([.03,.01,.94,.235])
intervals.set_xlim(0,3.65);intervals.set_ylim(-.75,1.0);intervals.axis('off')
def chain(points,color,ls='-',lw=1.15,empty=False,z=2):
 ax.plot(points[:,0],points[:,1],linestyle=ls,color=color,lw=lw,zorder=z)
 ax.scatter(points[:,0],points[:,1],s=10,facecolors='white' if empty else color,
  edgecolors=color,linewidths=.65,zorder=z+1)
def arrow(a,b,color):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',mutation_scale=7,
  lw=.85,color=color,shrinkA=2.5,shrinkB=2.5),zorder=3)
chain(x,GRAY,z=1)
chain(constant,BLUE,ls=(0,(3,2)),lw=.9,empty=True,z=2)
chain(planned,ORANGE,z=3)
chain(executed,GREEN,z=4)
arrow(x[1],constant[1],BLUE)
h_index=4  # The long bracket and addition arrow refer to the same action h.
arrow(x[h_index],planned[h_index],ORANGE)
for j in (5,6):arrow(planned[j],executed[j],GREEN)
ax.text(constant[-1,0]+.02,constant[-1,1]+.04,r'$\mathbf{v}^o(t_0)$',color=BLUE,
 fontsize=7.4,ha='left',va='bottom')
ax.text(1.60,.62,r'$\Delta_{\mathbf{c},h}(\mathbf{v}^o)$',
 color=ORANGE,fontsize=7.4,ha='center',va='center',rotation=5)
m=(planned[6]+executed[6])/2
ax.text(m[0]+.11,m[1],r'$\mathbf{q}(t)$',color=GREEN,fontsize=7.4,ha='left',va='center')
# Prefix accumulation beneath the action samples: individual time intervals
# contribute velocity-scaled movement to the displacement restored at h.
interval_labels=[r'$\delta\tau_1$',r'$\delta\tau_2$',r'$\cdots$',r'$\delta\tau_{h-1}$']
for j,label in enumerate(interval_labels):
 left,right=x[j,0],x[j+1,0]
 intervals.plot([left,left,right,right],[.95,.70,.70,.95],
  color=GRAY,lw=.65,solid_capstyle='butt')
 intervals.text((left+right)/2,.63,label,ha='center',va='top',
  fontsize=7.4,color='#68717C')
left,right=x[0,0],x[h_index,0]
intervals.plot([left,left,right,right],[.03,-.12,-.12,.03],
 color=ORANGE,lw=.8,solid_capstyle='butt')
intervals.text((left+right)/2,-.20,r'$d_{\mathbf{c}}(\mathbf{v}^o)$',
 ha='center',va='top',fontsize=7.4,color=ORANGE)
for j in (0,h_index):
 start=fig.transFigure.inverted().transform(intervals.transData.transform((x[j,0],.95)))
 end=fig.transFigure.inverted().transform(ax.transData.transform(x[j]))
 fig.add_artist(Line2D([start[0],end[0]],[start[1],end[1]],
  transform=fig.transFigure,color=GRAY,lw=.45,ls=':',zorder=0))
handles=[Line2D([],[],color=GRAY,marker='o',lw=1.1,markersize=3,
 label=r'Residual Chunk ($\tilde{\mathbf{x}}$)'),
 Line2D([],[],color=ORANGE,marker='o',lw=1.1,markersize=3,
 label='Residual + time-dependent object displacement'),
 Line2D([],[],color=GREEN,marker='o',lw=1.1,markersize=3,
 label='After velocity innovation')]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.035,.985),
 frameon=False,fontsize=7.4,handlelength=1.9,handletextpad=.65,labelspacing=.35,
 borderpad=0,borderaxespad=0)
for ext in ('pdf','svg','png'):
 fig.savefig(ROOT/f'velocity_innovation.{ext}',dpi=300,
  metadata={'Title':'Residual action reconstruction and velocity innovation'} if ext!='png' else None)
plt.close(fig)
print(ROOT/'velocity_innovation.png')
