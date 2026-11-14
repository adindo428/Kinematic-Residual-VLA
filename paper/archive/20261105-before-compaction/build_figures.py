#!/usr/bin/env python3
"""Build the KRFM figure suite from the archived, unmodified DOM RGB frames.

Run: MPLCONFIGDIR=/tmp/krfm-mpl python paper/figures/build_figures.py
Only NumPy, Pillow, and Matplotlib are required; no simulator or GPU is used.
PDF is the publication master; SVG preserves editable labels and vector overlays.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
INK = "#202B3B"
MUTED = "#596779"
LINE = "#CCD4DE"
BLUE = "#326BB0"
BLUE_BG = "#EDF3FC"
TEAL = "#087F83"
TEAL_BG = "#EAF6F3"
ORANGE = "#C77924"
ORANGE_BG = "#FFF3E4"
PURPLE = "#7853A2"
PAPER = "#F5F7FA"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "text.color": INK, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "savefig.facecolor": "white"})
FIGURES = []


def canvas(w, h):
    f = plt.figure(figsize=(w, h), facecolor="white")
    ax = f.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    return f, ax


def text(ax, x, y, s, size=8, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", zorder=8, linespacing=1.35, **kw)


def box(ax, x, y, w, h, fill=PAPER, edge=LINE, radius=.06, lw=.7, ls="solid"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                      facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls,
                      zorder=2)
    ax.add_patch(p)
    return p


def arrow(ax, a, b, color=MUTED, lw=1, rad=0, ls="solid", z=5):
    p = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                        linestyle=ls, zorder=z, shrinkA=1.5, shrinkB=1.5)
    ax.add_patch(p)


def line(ax, pts, color=LINE, lw=.8, ls="solid"):
    ax.plot(*zip(*pts), color=color, lw=lw, ls=ls, zorder=4)


def frame(ax, ep, idx, x, y, w, h, camera="opst_cam", crop=(0, 75, 480, 335)):
    p = ROOT / f"assets/ep{ep:06d}_{camera}_f{idx:03d}.png"
    im = Image.open(p).crop(crop)
    # Match the axes extent to the crop aspect, never stretch a screenshot.
    ratio = im.width / im.height
    if w / h > ratio:
        actual_w = h * ratio
        x += (w - actual_w) / 2
        w = actual_w
    else:
        actual_h = w / ratio
        y += (h - actual_h) / 2
        h = actual_h
    ax.imshow(im, extent=(x, x+w, y+h, y), interpolation="none", zorder=1)
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=LINE,linewidth=.5,zorder=3))
    return (x, y, w, h, crop)


def point(region, px, py):
    x,y,w,h,c = region
    return (x+(px-c[0])/(c[2]-c[0])*w, y+(py-c[1])/(c[3]-c[1])*h)


def ring(ax, region, px, py, radius=22, color=TEAL, ls="solid"):
    x,y,w,h,c=region
    ax.add_patch(Ellipse(point(region,px,py), 2*radius*w/(c[2]-c[0]),
                        2*radius*h/(c[3]-c[1]),fill=False,edgecolor="white",
                        lw=2.6,zorder=5))
    ax.add_patch(Ellipse(point(region,px,py), 2*radius*w/(c[2]-c[0]),
                        2*radius*h/(c[3]-c[1]),fill=False,edgecolor=color,
                        lw=1.3,linestyle=ls,zorder=6))


def save(f, name):
    f.savefig(ROOT/f"{name}.pdf", dpi=300)
    f.savefig(ROOT/f"{name}.svg", dpi=300)
    f.savefig(ROOT/f"previews/{name}.png", dpi=300)
    FIGURES.append((name,f))


def overview():
    f,ax=canvas(7,4.72)
    text(ax,.06,.16,"Dynamic manipulation through kinematic residuals",11,weight="bold")
    text(ax,.06,.40,"(a) Task observations",8.5,weight="bold")
    text(ax,2.60,.40,"(b) Shared framework",8.5,weight="bold")
    text(ax,4.93,.40,"(c) Intended outcomes",8.5,weight="bold")
    rows=[(5881,3,99,"Pick a moving apple","Target lifted",(182,191),(238,165)),
          (103899,3,105,"Place the can into the bowl","Object in destination",(138,223),(314,238)),
          (5224,3,197,"Pack both potatoes into the box","Multi-object completion",None,(325,215))]
    for k,(ep,start,end,title,result,p0,p1) in enumerate(rows):
        y=.67+k*1.23
        text(ax,.07,y,title,8,weight="bold")
        crop=(0,110,480,360) if ep==3265 else (0,75,480,335)
        a=frame(ax,ep,start,.06,y+.13,2.05,1.02,crop=crop)
        b=frame(ax,ep,end,4.89,y+.13,2.05,1.02,crop=crop)
        if p0:ring(ax,a,*p0,radius=19,color=ORANGE)
        if p1:ring(ax,b,*p1,radius=27 if ep==5224 else 23,color=TEAL)
        text(ax,4.94,y,result,8,color=TEAL,weight="bold")
        arrow(ax,(2.14,y+.65),(2.43,y+.65),color=BLUE)
        arrow(ax,(4.57,y+.65),(4.86,y+.65),color=TEAL)
    box(ax,2.45,.67,2.10,3.60,fill="#FAFBFD")
    text(ax,3.50,.90,"KINEMATIC RESIDUAL",8,weight="bold",ha="center")
    text(ax,3.50,1.08,"FLOW MATCHING",8,weight="bold",ha="center")
    box(ax,2.60,1.31,1.80,.55,BLUE_BG,BLUE)
    text(ax,3.50,1.49,"Vision + language + state",7.5,ha="center")
    text(ax,3.50,1.68,"Compatible VLA backbone",7,ha="center",color=BLUE)
    arrow(ax,(3.50,1.87),(3.50,2.06),BLUE)
    box(ax,2.60,2.08,1.80,.56,BLUE_BG,BLUE)
    text(ax,3.50,2.26,"Learn residual manipulation",7.3,ha="center",weight="bold")
    text(ax,3.50,2.46,r"Residual action $\widehat{\widetilde{\mathbf{x}}}_1$",8,ha="center",color=BLUE)
    box(ax,2.60,2.86,1.80,.55,ORANGE_BG,ORANGE)
    text(ax,3.50,3.04,"Project target motion",7.5,ha="center",weight="bold")
    text(ax,3.50,3.23,r"Velocity $\rightarrow\ \Delta_{\mathbf{c}}(\mathbf{v}^{o})$",8,ha="center",color=ORANGE)
    # Both branches feed composition; neither is a serial input to the other.
    line(ax,[(2.60,2.37),(2.52,2.37),(2.52,3.75),(2.65,3.75)],BLUE)
    arrow(ax,(3.50,3.42),(3.50,3.60),ORANGE)
    box(ax,2.65,3.63,1.70,.41,TEAL_BG,TEAL)
    text(ax,3.50,3.83,r"$\widehat{\mathbf{x}}_1=\widehat{\widetilde{\mathbf{x}}}_1+\Delta_{\mathbf{c}}$",10,ha="center",color=TEAL)
    text(ax,.07,4.47,"DOM demonstrations illustrate tasks and goals; they are not KRFM policy results.",7,color=MUTED)
    text(ax,.07,4.62,"All scene panels use recorded simulation frames. Circles identify the target or destination.",7,color=MUTED)
    save(f,"task_framework_overview")


def method():
    f,ax=canvas(7,4.30)
    text(ax,.06,.16,"One decomposition, shared across compatible action policies",11,weight="bold")
    # Compact context strip.
    a=frame(ax,103899,3,.08,.39,.84,.63,crop=(0,0,480,360))
    frame(ax,103899,3,.98,.39,.84,.63,camera="wrist_cam",crop=(0,0,480,360))
    text(ax,.50,1.11,"Scene RGB",7,ha="center",color=MUTED)
    text(ax,1.40,1.11,"Wrist RGB",7,ha="center",color=MUTED)
    box(ax,1.98,.41,2.06,.60,BLUE_BG,BLUE)
    text(ax,3.01,.60,"VLA context encoder",8.5,weight="bold",ha="center")
    text(ax,3.01,.82,r"$\mathbf{c}=(\mathcal{I},\ell,\mathbf{q})$",10,ha="center",color=BLUE)
    arrow(ax,(1.85,.70),(1.96,.70),BLUE)
    text(ax,4.25,.56,'Instruction: “Place the can in the bowl.”',7.2)
    text(ax,4.25,.80,"Robot state and camera observations",7.2,color=MUTED)
    arrow(ax,(4.22,.70),(4.06,.70),BLUE)
    line(ax,[(3.01,1.02),(3.01,1.19),(6.02,1.19)],BLUE)
    arrow(ax,(6.02,1.19),(6.02,1.70),BLUE)
    text(ax,4.60,1.18,r"Shared context $\mathbf{c}$",7,color=BLUE,
         bbox={"facecolor":"white","edgecolor":"none","pad":1.5})
    # Training: raw target -> subtract displacement -> residual target -> learn.
    box(ax,.06,1.31,6.88,1.10,"#FAFBFD")
    text(ax,.20,1.49,"(a) Training: remove predictable displacement",8.5,weight="bold")
    items=[(.20,1.01,"Expert actions",r"$\mathbf{x}_1$",PAPER,MUTED),
           (1.63,1.43,"Residualize",r"$\widetilde{\mathbf{x}}_1=\mathbf{x}_1-\Delta_{\mathbf{c}}$",ORANGE_BG,ORANGE),
           (3.47,1.43,"Residual flow path",r"$\widetilde{\mathbf{x}}_s=(1-s)\mathbf{x}_0+s\widetilde{\mathbf{x}}_1$",BLUE_BG,BLUE),
           (5.31,1.43,"Train action expert",r"$\mathbf{u}_{\theta}(\widetilde{\mathbf{x}}_s,s,\mathbf{c})$",BLUE_BG,BLUE)]
    for x,w,title,eq,bg,c in items:
        box(ax,x,1.72,w,.49,bg,c)
        text(ax,x+w/2,1.85,title,7,ha="center",weight="bold")
        text(ax,x+w/2,2.06,eq,7.6 if x==3.47 else 9,ha="center",color=c)
    for x1,x2 in [(1.22,1.61),(3.08,3.45),(4.92,5.29)]:arrow(ax,(x1,1.97),(x2,1.97))
    # Inference: noise -> sample residual -> add displacement -> executable chunk.
    box(ax,.06,2.54,6.88,1.12,"#FAFBFD")
    text(ax,.20,2.73,"(b) Inference: restore motion in action coordinates",8.5,weight="bold")
    for x,w,title,eq,bg,c in [(.20,1.01,"Base sample",r"$\mathbf{x}_0\sim p_0$",PAPER,MUTED),
        (1.63,1.43,"Integrate residual flow",r"$\widehat{\widetilde{\mathbf{x}}}_1$",BLUE_BG,BLUE),
        (3.47,1.43,"Add motion",r"$\widehat{\mathbf{x}}_1=\widehat{\widetilde{\mathbf{x}}}_1+\Delta_{\mathbf{c}}$",TEAL_BG,TEAL),
        (5.31,1.43,"Execute action chunk",r"$\mathbf{a}_{1:H}$",TEAL_BG,TEAL)]:
        box(ax,x,2.96,w,.49,bg,c)
        text(ax,x+w/2,3.09,title,7,ha="center",weight="bold")
        text(ax,x+w/2,3.30,eq,9,ha="center",color=c)
    for x1,x2 in [(1.22,1.61),(3.08,3.45),(4.92,5.29)]:arrow(ax,(x1,3.21),(x2,3.21))
    # The projection bar is the shared analytic interface.
    box(ax,.06,3.80,6.88,.43,ORANGE_BG,ORANGE)
    text(ax,.20,4.01,r"Analytic branch:  $\mathbf{v}^{o}\ \rightarrow\ \Phi_{\mathbf{c}}\ \rightarrow\ d_{\mathbf{c}}\ \rightarrow\ \Delta_{\mathbf{c}}=d_{\mathbf{c}}-\bar d_{\mathbf{c}}$",8.3,color=ORANGE)
    text(ax,6.79,4.01,"Current study: simulator-state velocity",7,ha="right",color=MUTED)
    # Explicit conditioning inputs, unobstructed above each band.
    text(ax,6.73,2.29,r"Flow target: $\widetilde{\mathbf{x}}_1-\mathbf{x}_0$",7,ha="right",color=BLUE)
    text(ax,1.64,3.54,r"Conditioned on $\mathbf{c}$",7,color=BLUE)
    text(ax,6.73,3.54,"Reobserve and repeat after execution",7,ha="right",color=TEAL)
    save(f,"krfm_method")


def sequence():
    f,ax=canvas(7,3.62)
    text(ax,.06,.16,"From moving target to completed placement",11,weight="bold")
    text(ax,.06,.40,'“Place the red can into the bowl.”',8.5)
    text(ax,6.94,.40,"DOM demonstration · episode 103899",7,ha="right",color=MUTED)
    ids=[3,43,83,105]
    stages=["Observe target motion","Approach and grasp","Transport to bowl","Release in destination"]
    for k,(idx,title) in enumerate(zip(ids,stages)):
        x=.06+k*1.75
        text(ax,x,.68,f"{k+1}  {title}",7.5,weight="bold")
        a=frame(ax,103899,idx,x,.82,1.63,1.00,crop=(0,65,480,360))
        frame(ax,103899,idx,x,2.02,1.63,1.00,camera="wrist_cam",crop=(0,65,480,360))
        text(ax,x+.815,1.90,f"t = {idx/25:.2f} s",7,ha="center",color=MUTED)
        if k==0:
            ring(ax,a,138,223,20,ORANGE)
            arrow(ax,point(a,158,217),point(a,222,218),ORANGE,lw=1.5)
        if k==3:ring(ax,a,314,238,24,TEAL)
        if k<3:arrow(ax,(x+1.64,1.32),(x+1.74,1.32),TEAL)
    text(ax,.07,3.15,"Top: opposite view   ·   Bottom: synchronized wrist view   ·   Timestamps from episode metadata",7,color=MUTED)
    line(ax,[(.08,3.30),(6.92,3.30)])
    text(ax,.07,3.46,"KRFM interpretation: compensate target motion + learn grasp, transport, and release behavior.",7.5)
    save(f,"annotated_manipulation")


def residual_geometry():
    f,ax=canvas(7,2.72)
    text(ax,.06,.16,"Why residualize the action target?",11,weight="bold")
    text(ax,.08,.45,"(a) Raw action distributions",8.5,weight="bold")
    text(ax,3.84,.45,"(b) Aligned residual distributions",8.5,weight="bold")
    cols=[BLUE,ORANGE,PURPLE]
    # Deterministic analytic densities; intentionally no empirical scale or points.
    x=np.linspace(-2.5,2.5,500)
    def density(v):return .63*np.exp(-.5*((v+.4)/.25)**2)+.43*np.exp(-.5*((v-.42)/.32)**2)
    for side in [0,1]:
        left=.18 if side==0 else 3.96
        arrow(ax,(left,1.81),(left+2.71,1.81),MUTED,.7)
        for i,(shift,c) in enumerate(zip([-.95,0,.95],cols)):
            xx=left+(x+2.5)/5*2.57
            yy=1.78-density(x-(shift if side==0 else 0))*.96
            ax.plot(xx,yy,color=c,lw=1.4,ls=["solid","--",":"][i],zorder=4+i)
        text(ax,left+1.28,1.99,"One action coordinate (schematic)",7,ha="center",color=MUTED)
    text(ax,1.42,.78,r"$p(\mathbf{x}_1\mid\mathbf{c},\mathbf{v}^{o})$",9,ha="center")
    text(ax,5.24,.78,r"$p(\widetilde{\mathbf{x}}_1\mid\mathbf{c},\mathbf{v}^{o})$",9,ha="center")
    arrow(ax,(2.94,1.30),(3.80,1.30),TEAL,1.3)
    text(ax,3.38,1.09,r"$-\Delta_{\mathbf{c}}$",10,ha="center",color=TEAL)
    text(ax,3.38,1.55,"Align",7,ha="center",color=TEAL)
    for i,c in enumerate(cols):
        x0=.18+i*1.03
        line(ax,[(x0,2.22),(x0+.22,2.22)],c,1.5,["solid","--",":"][i])
        text(ax,x0+.29,2.22,rf"Velocity $v_{i+1}$",7)
    text(ax,3.96,2.22,"Grasp modes remain; velocity shifts are removed.",7,color=TEAL)
    text(ax,.08,2.53,"Conceptual illustration under the local additive-shift assumption; not a measured distribution.",7,color=MUTED)
    save(f,"residual_geometry")


def main():
    (ROOT/"previews").mkdir(exist_ok=True)
    manifest=json.loads((ROOT/"frame_manifest.json").read_text())
    for row in manifest["frames"]:
        assert hashlib.sha256((ROOT/row["file"]).read_bytes()).hexdigest()==row["frame_sha256"],row["file"]
    overview();method();sequence();residual_geometry()
    with PdfPages(ROOT/"figure_review.pdf") as pdf:
        for _,f in FIGURES:pdf.savefig(f,dpi=300)
    for name,f in FIGURES:
        plt.close(f)
        print(name)


if __name__=="__main__":main()
