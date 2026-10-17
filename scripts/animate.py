import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from hydro_reader import read_meta, read_grid, read_particle, canvas_shape, stitch, primitives, get_mu, coord_origin

"""
Animation of the AMR gas output (grid.bin) with the mesh blocks and the particle

 panels: log(rho/rho_0), z-velocity u, T, p on a uniform canvas at the finest level of the run
 axes:   r horizontal, z vertical, measured from rmin / zmin (the absolute values ~1e12-1e13 cm
         can't be resolved on a few-cm wide axis)
 usage:  python animate.py [outdir]
"""

SHOW_BLOCKS = True  # mesh block outlines on the density panel
FPS = 20

outdir = sys.argv[1] if len(sys.argv) > 1 else input("Output directory: ").strip()

meta = read_meta(outdir)
zmin, zmax = meta['zmin'], meta['zmax']
rmin, rmax = meta['rmin'], meta['rmax']
gamma = meta['gamma']
mu = get_mu(meta)
z_org, r_org = coord_origin(meta)  # subtract from file coordinates to get z - zmin, r - rmin

print("Loading grid.bin...")
frames = read_grid(os.path.join(outdir, "grid.bin"))
if not frames:
    raise RuntimeError("grid.bin is empty or could not be read")
nz_c, nr_c, max_level = canvas_shape(frames, meta)
print(f"Loaded {len(frames)} frames, finest level {max_level}, canvas {nz_c} x {nr_c}")

times = [t for t, _ in frames]


#fields of one frame on the canvas (stitched on demand: all frames at once would not fit in memory at high levels)
def fields(n):
    rho, u, v, p, T = primitives(stitch(frames[n][1], (nz_c, nr_c), max_level), gamma, mu)
    return [np.log10(rho/rho_base), u, T, p]


rho_base = primitives(stitch(frames[0][1], (nz_c, nr_c), max_level), gamma, mu)[0]

#colour limits over all frames
print("Computing colour limits...")
lims = None
for n in range(len(frames)):
    f = fields(n)
    cur = [(np.nanmin(a), np.nanmax(a)) for a in f]
    lims = cur if lims is None else [(min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(lims, cur)]
#a constant field (e.g. log(rho/rho_0) = 0 at t=0) needs a non-empty colour range
lims = [(lo, hi) if hi > lo else (lo - 1e-12 - 1e-6*abs(lo), hi + 1e-12 + 1e-6*abs(hi)) for lo, hi in lims]

part = read_particle(outdir)
has_particle = part is not None and len(part) > 0
if has_particle:
    print(f"Loaded particle data with {len(part)} rows")
else:
    print("No particle.bin found, skipping particle overlay")

#cell edges relative to the domain corner
z_edges = np.linspace(0.0, zmax - zmin, nz_c + 1)
r_edges = np.linspace(0.0, rmax - rmin, nr_c + 1)

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
fig.subplots_adjust(hspace=0.4, wspace=0.45)

titles = [r"Relative density  $\log(\rho/\rho_0)$", r"$z$-velocity  $u$ [cm/s]", r"$T$ [K]", r"Pressure  $p$"]
cmaps = ["inferno", "RdBu_r", "RdBu_r", "viridis"]

meshes, p_dots = [], []
for ax, field, title, lim, cmap in zip(axes.flat, fields(0), titles, lims, cmaps):
    mesh = ax.pcolormesh(r_edges, z_edges, field, cmap=cmap, vmin=lim[0], vmax=lim[1], shading='flat')
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$r - r_{min}$ [cm]")
    ax.set_ylabel(r"$z - z_{min}$ [cm]")
    ax.tick_params(labelsize=7)
    meshes.append(mesh)
    if has_particle:
        dot, = ax.plot([], [], 'w+', ms=8, mew=1.5)
        p_dots.append(dot)

suptitle = fig.suptitle("", fontsize=12, fontweight='bold')
fig.text(0.5, 0.01, f"$z_{{min}}$ = {zmin:.6e} cm,   $r_{{min}}$ = {rmin:.10e} cm", ha='center', fontsize=8)


def draw_blocks(n):
    ax = axes.flat[0]
    for p in list(ax.patches):
        p.remove()
    for b in frames[n][1]:
        ax.add_patch(patches.Rectangle((b['x2min'] - r_org, b['x1min'] - z_org),
                                       b['x2max'] - b['x2min'], b['x1max'] - b['x1min'],
                                       linewidth=0.4 + 0.2*b['level'], edgecolor='white',
                                       facecolor='none', alpha=0.6))


def update(n):
    for mesh, field in zip(meshes, fields(n)):
        mesh.set_array(field.ravel())
    if SHOW_BLOCKS:
        draw_blocks(n)
    suptitle.set_text(f"t = {times[n]:.4g} s   ({len(frames[n][1])} blocks)")
    if has_particle:
        row = part[np.argmin(np.abs(part['t'] - times[n]))]
        for dot in p_dots:
            if row['active']:
                dot.set_data([row['r'] - r_org], [row['z'] - z_org])
            else:
                dot.set_data([], [])
    return meshes + p_dots + [suptitle]


update(0)
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
plt.show()

#save
if input("Save as mp4? (y/n): ").strip().lower() == 'y':
    out = os.path.join(outdir, "animation.mp4")
    try:
        ani.save(out, writer='ffmpeg', fps=FPS, dpi=150)
    except Exception as e:  #no ffmpeg -> gif
        print(f"ffmpeg failed ({e}), saving a gif instead")
        out = os.path.join(outdir, "animation.gif")
        ani.save(out, writer='pillow', fps=FPS)
    print(f"Saved to {out}")
