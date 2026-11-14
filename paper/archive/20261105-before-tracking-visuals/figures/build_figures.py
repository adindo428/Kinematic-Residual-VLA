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
    """Two-inch overview: paired task frames, shared decomposition, and geometry."""
    f,ax=canvas(7,2.08)
    text(ax,.03,.12,"(a) Tasks and intended outcomes",8.5,weight="bold")
    text(ax,3.40,.12,"(b) Kinematic residual framework",8.5,weight="bold")
    rows=[(5881,3,99,"Pick\napple",(182,191),(238,165)),
          (103899,3,105,"Place\ncan",(138,223),(314,238)),
          (5224,3,197,"Pack\nobjects",None,(325,215))]
    text(ax,1.21,.32,"Observation",7,ha="center",color=MUTED)
    text(ax,2.58,.32,"Goal",7,ha="center",color=TEAL)
    for k,(ep,start,end,label,p0,p1) in enumerate(rows):
        y=.43+k*.54
        text(ax,.03,y+.22,label,7.5,weight="bold")
        a=frame(ax,ep,start,.60,y,1.23,.47,crop=(0,125,480,305))
        b=frame(ax,ep,end,1.98,y,1.23,.47,crop=(0,125,480,305))
        if p0:ring(ax,a,*p0,radius=19,color=ORANGE)
        if p1:ring(ax,b,*p1,radius=27 if ep==5224 else 23,color=TEAL)
        arrow(ax,(1.84,y+.23),(1.97,y+.23),TEAL,.9)
    line(ax,[(3.29,.04),(3.29,2.04)])
    box(ax,3.42,.32,1.58,.48,BLUE_BG,BLUE)
    text(ax,4.21,.47,"VLA residual action",7.5,weight="bold",ha="center")
    text(ax,4.21,.67,r"$\widehat{\widetilde{\mathbf{x}}}_1$",9,ha="center",color=BLUE)
    text(ax,5.16,.57,"+",12,ha="center",color=TEAL)
    box(ax,5.34,.32,1.61,.48,ORANGE_BG,ORANGE)
    text(ax,6.145,.47,"Kinematic displacement",7,weight="bold",ha="center")
    text(ax,6.145,.67,r"$\Delta_{\mathbf{c}}(\mathbf{v}^{o})$",9,ha="center",color=ORANGE)
    text(ax,3.45,.96,r"Train: $\widetilde{\mathbf{x}}_1=\mathbf{x}_1-\Delta_{\mathbf{c}}$",8)
    text(ax,5.34,.96,r"Execute: $\widehat{\mathbf{x}}_1=\widehat{\widetilde{\mathbf{x}}}_1+\Delta_{\mathbf{c}}$",8,color=TEAL)
    text(ax,4.12,1.20,"Raw targets",7.5,ha="center",weight="bold")
    text(ax,6.24,1.20,"Residual targets",7.5,ha="center",weight="bold")
    x=np.linspace(-2.5,2.5,400)
    def density(v):return .63*np.exp(-.5*((v+.4)/.25)**2)+.43*np.exp(-.5*((v-.42)/.32)**2)
    for side,left in enumerate([3.46,5.56]):
        arrow(ax,(left,1.82),(left+1.30,1.82),MUTED,.6)
        for i,(shift,c) in enumerate(zip([-.95,0,.95],[BLUE,ORANGE,PURPLE])):
            yy=1.79-density(x-(shift if side==0 else 0))*.68
            ax.plot(left+(x+2.5)/5*1.23,yy,color=c,lw=1.1,ls=["solid","--",":"][i],zorder=4+i)
    arrow(ax,(4.83,1.57),(5.45,1.57),TEAL,1.2)
    text(ax,5.14,1.38,r"$-\Delta_{\mathbf{c}}$",9,ha="center",color=TEAL)
    text(ax,5.17,1.99,"Different velocities; shared residual modes",7,ha="center",color=MUTED)
    save(f,"task_framework_overview")


def sequence():
    f,ax=canvas(7,2.18)
    ids=[3,43,83,105]
    stages=["Observe motion","Approach and grasp","Transport","Release"]
    for k,(idx,title) in enumerate(zip(ids,stages)):
        x=.04+k*1.75
        text(ax,x,.12,f"{k+1}  {title}",8,weight="bold")
        a=frame(ax,103899,idx,x,.24,1.64,.94,crop=(0,65,480,340))
        frame(ax,103899,idx,x,1.22,1.64,.94,camera="wrist_cam",crop=(0,50,480,325))
        text(ax,x+.04,.34,f"{idx/25:.2f} s",7,color=INK,
             bbox={"facecolor":"white","edgecolor":"none","alpha":.88,"pad":1.4})
        if k==0:
            ring(ax,a,138,223,20,ORANGE)
            arrow(ax,point(a,158,217),point(a,222,218),ORANGE,lw=1.3)
        if k==3:ring(ax,a,314,238,24,TEAL)
        if k<3:arrow(ax,(x+1.65,.67),(x+1.74,.67),TEAL)
    save(f,"annotated_manipulation")


def main():
    (ROOT/"previews").mkdir(exist_ok=True)
    manifest=json.loads((ROOT/"frame_manifest.json").read_text())
    for row in manifest["frames"]:
        assert hashlib.sha256((ROOT/row["file"]).read_bytes()).hexdigest()==row["frame_sha256"],row["file"]
    overview();sequence()
    with PdfPages(ROOT/"figure_review.pdf") as pdf:
        for _,f in FIGURES:pdf.savefig(f,dpi=300)
    for name,f in FIGURES:
        plt.close(f)
        print(name)


if __name__=="__main__":main()
