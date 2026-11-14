#!/usr/bin/env python3
"""Illustrative KRA displacement addition; synthetic geometry, no measured results.

Visual reference: VLASH, Fig. 5 (sampled trajectories and displacement arrows).
https://arxiv.org/pdf/2512.01031
Run with the repository's existing NumPy/Matplotlib environment.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
GRAY = '#ACB2B9'
BLUE = '#2765AD'
GREEN = '#16816B'
INK = '#222A34'
MUTED = '#68717C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
 'mathtext.fontset':'dejavusans','pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white'})

# One shared residual chunk makes the three addition rules comparable.
t = np.linspace(0, 1, 7)
r = np.column_stack((.35+2.6*t, .55+.18*np.sin(1.3*np.pi*t)))
d_const = np.column_stack((np.zeros_like(t), 1.35*t))
# Integral of a changing velocity: v_plan(t)=(.9 t, .8+1.3 t).
d_plan = np.column_stack((.45*t*t, .8*t+.65*t*t))
plan = r+d_plan
# Synthetic, unbounded innovation: observations diverge after t=.4.
w = np.maximum((t-.4)/.6, 0)
e = np.column_stack((-.55*w*w, .62*w*w))
executed = plan+e
assert np.allclose(e[t<=.4], 0)
assert np.allclose((r+d_plan)+e, r+(d_plan+e))

fig, axes = plt.subplots(1,3,figsize=(12.8,4.6))
fig.subplots_adjust(left=.025,right=.985,top=.75,bottom=.25,wspace=.12)
fig.text(.5,.965,'Adding object displacement to a residual action chunk',
 ha='center',va='top',fontsize=15,fontweight='semibold',color=INK)
handles=[Line2D([],[],color=GRAY,marker='o',lw=1.5,label='Residual chunk'),
 Line2D([],[],color=BLUE,marker='o',lw=1.8,label='Residual + planned displacement'),
 Line2D([],[],color=GREEN,marker='o',lw=1.8,label='After velocity innovation')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.91),
 ncol=3,frameon=False,fontsize=10,columnspacing=2.0,handlelength=2.2)

def chain(ax, points, color, lw=1.8, ls='-', filled=True, z=3):
 ax.plot(points[:,0],points[:,1],ls,color=color,lw=lw,zorder=z)
 ax.scatter(points[:,0],points[:,1],s=33,facecolor=color if filled else 'white',
  edgecolor=color,linewidth=1.2,zorder=z+1)

def offset(ax, start, end, color, lw=1.35):
 ax.annotate('',xy=end,xytext=start,
  arrowprops=dict(arrowstyle='-|>',color=color,lw=lw,
                 shrinkA=5,shrinkB=5,mutation_scale=10),zorder=2)

titles=['(a) Constant velocity','(b) Time-dependent velocity','(c) Velocity innovation']
for k,ax in enumerate(axes):
 ax.set_xlim(0,3.65);ax.set_ylim(.1,2.8);ax.set_aspect('equal')
 ax.axis('off')
 ax.set_title(titles[k],fontsize=12,fontweight='semibold',color=INK,pad=12)
 chain(ax,r,GRAY,lw=1.4,z=1)
 if k<2:
  d = d_const if k==0 else d_plan
  a = r+d
  chain(ax,a,BLUE)
  for j in [2,4,6]:offset(ax,r[j],a[j],BLUE)
  j=4
  ax.text(r[j,0]-.1,(r[j,1]+a[j,1])/2,
   r'$d_h$',color=BLUE,ha='right',va='center',fontsize=11)
  ax.text(r[0,0]-.04,r[0,1]-.15,r'$t_0$',ha='center',va='top',fontsize=9,color=MUTED)
  ax.text(r[-1,0],r[-1,1]-.15,r'$t_H$',ha='center',va='top',fontsize=9,color=MUTED)
 else:
  chain(ax,plan,BLUE,lw=1.5,ls='--',filled=False,z=2)
  chain(ax,executed,GREEN,lw=2.0,z=4)
  for j in [4,5,6]:offset(ax,plan[j],executed[j],GREEN,lw=1.6)
  # Planned lift from the residual and the extra innovation are separate arrows.
  offset(ax,r[5],plan[5],BLUE,lw=1.15)
  ax.text(r[5,0]+.08,(r[5,1]+plan[5,1])/2,r'$d_h$',
          color=BLUE,ha='left',va='center',fontsize=11)
  mid=(plan[6]+executed[6])/2
  ax.text(mid[0]+.14,mid[1]+.02,r'$e_h$',color=GREEN,fontsize=11)
  ax.annotate('New motion\nobservations',xy=plan[3],xytext=(.55,1.78),
   color=MUTED,fontsize=9,ha='center',va='center',
   arrowprops=dict(arrowstyle='-',lw=.8,color=MUTED,connectionstyle='arc3,rad=-.18'))
 # Subtle coordinate cue keeps the illustration a spatial path, not a time plot.
 if k==0:
  ax.annotate('',xy=(.45,.12),xytext=(.05,.12),arrowprops=dict(arrowstyle='->',lw=.75,color=MUTED))
  ax.annotate('',xy=(.05,.52),xytext=(.05,.12),arrowprops=dict(arrowstyle='->',lw=.75,color=MUTED))
  ax.text(.47,.12,'x',fontsize=8,color=MUTED,va='center')
  ax.text(.05,.55,'y',fontsize=8,color=MUTED,ha='center')

centers=[ax.get_position().x0+ax.get_position().width/2 for ax in axes]
for x,formula,subtitle in zip(centers,
 [r'$a_h=r_h+d_h,\qquad d_h=(t_h-t_0)\,v_0^o$',
  r'$a_h=r_h+d_h,\qquad d_h=\int_{t_0}^{t_h} v_{\rm plan}^o(u)\,du$',
  r'$a_h^{\rm exec}=r_h+d_h+e_h$'],
 ['One velocity; displacement grows with time.',
  'Integrate the changing velocity over the chunk.',
  'Add only the newly observed displacement error.']):
 fig.text(x,.208,formula,ha='center',va='center',fontsize=10.5,color=INK)
 fig.text(x,.136,subtitle,ha='center',va='center',fontsize=9.2,color=MUTED)
fig.text(centers[2],.072,
 r'$e_h=\int_{t_0}^{t_h}[v_{\rm obs}^o(u)-v_{\rm plan}^o(u)]\,du$',
 ha='center',va='center',fontsize=10.5,color=GREEN)
fig.text(.025,.022,'Illustrative XY trajectories; centering, frame transforms, and execution limits omitted.',
 ha='left',va='bottom',fontsize=8.2,color=MUTED)
for ext in ['png','pdf','svg']:
 fig.savefig(ROOT/f'kra_displacement_addition.{ext}',dpi=220,
             metadata={'Title':'KRA displacement addition: constant velocity, time-dependent velocity, and innovation'}
             if ext!='png' else {'Software':'Matplotlib'})
plt.close(fig)
print(ROOT/'kra_displacement_addition.png')
