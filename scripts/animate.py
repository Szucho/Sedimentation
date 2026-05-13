import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def read_meta(outdir):
    meta = {}
    with open(os.path.join(outdir, "meta.txt")) as f:
        for line in f:
            k, v = line.split()
            meta[k] = float(v)
    return meta


def read_grid(path):
    frames = []
    with open(path, 'rb') as f:
        nx = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        ny = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        while True:
            tbuf = f.read(8)
            if not tbuf: break
            t   = np.frombuffer(tbuf, dtype=np.float64)[0]
            raw = f.read(int(nx * ny * 4 * 8))
            if len(raw) != nx * ny * 4 * 8: break
            data = np.frombuffer(raw, dtype=np.float64).reshape(nx, ny, 4)
            frames.append((t, data))
    return frames

#primitive values from state vector
def compute_primitives(data, gamma=1.4):
    rho = data[:,:,0]
    u   = data[:,:,1] / rho
    v   = data[:,:,2] / rho
    E   = data[:,:,3]
    p   = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    return rho, u, v, p

#color lims
def lims(arr_list):
    return min(a.min() for a in arr_list), max(a.max() for a in arr_list)

#load frames
outdir = input("Output directory: ").strip()

meta = read_meta(outdir)
xmin, xmax = meta['xmin'], meta['xmax']
ymin, ymax = meta['ymin'], meta['ymax']
nx, ny     = int(meta['Nx']), int(meta['Ny'])
gamma      = meta['gamma']


print("Loading grid.bin...")
frames = read_grid(os.path.join(outdir, "grid.bin"))
if not frames:
    raise RuntimeError("grid.bin is empty or could not be read")
print(f"Loaded {len(frames)} frames")


times = [f[0] for f in frames]
prims = [compute_primitives(f[1], gamma) for f in frames]


particle_path = os.path.join(outdir, "particle.csv")
has_particle  = os.path.exists(particle_path)
if has_particle:
    part = pd.read_csv(particle_path)
    part_by_n = part.set_index('n')
else:
    print("No particle.csv found, skipping particle overlay")


#color lim of primitive vals
rho_lim = lims([p[0] for p in prims])
u_lim   = lims([p[1] for p in prims])
v_lim   = lims([p[2] for p in prims])
p_lim   = lims([p[3] for p in prims])

#figure
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.subplots_adjust(hspace=0.4, wspace=0.35)

titles = [r"Density  $\rho$", r"$x$-velocity  $u$", r"$y$-velocity  $v$", r"Pressure  $p$"]
clims  = [rho_lim, u_lim, v_lim, p_lim]
cmaps  = ["inferno", "RdBu_r", "RdBu_r", "viridis"]

x_ticks = np.linspace(0, nx, 5)
x_labels = [f"{xmin + v*(xmax-xmin)/nx:.2g}" for v in x_ticks]
y_ticks  = np.linspace(0, ny, 3)
y_labels = [f"{ymin + v*(ymax-ymin)/ny:.2g}" for v in y_ticks]

rho0, u0, v0, p0 = prims[0]
meshes = []
p_dots = []

for ax, field, title, clim, cmap in zip(axes.flat, [rho0,u0,v0,p0], titles, clims, cmaps):
    mesh = ax.pcolormesh(field.T, cmap=cmap, vmin=clim[0], vmax=clim[1], shading='auto')
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    meshes.append(mesh)
    if has_particle:
        dot, = ax.plot([], [], 'w+', ms=8, mew=1.5)
        p_dots.append(dot)


suptitle = fig.suptitle(f"t = {times[0]:.4f}", fontsize=13, fontweight='bold')

#update
def update(n):
    rho, u, v, p = prims[n]
    for mesh, field in zip(meshes, [rho, u, v, p]):
        mesh.set_array(field.T.ravel())
    suptitle.set_text(f"t = {times[n]:.4f}")
    if has_particle and n in part_by_n.index: #put particle on animation too
        row = part_by_n.loc[n]
        if row['active']:
            px = (row['x'] - xmin) / (xmax - xmin) * nx
            py = (row['y'] - ymin) / (ymax - ymin) * ny
            for dot in p_dots:
                dot.set_data([px], [py])
        else:
            for dot in p_dots:
                dot.set_data([], [])
    return meshes + p_dots + [suptitle]


ani = animation.FuncAnimation(fig, update, frames=len(prims), interval=50, blit=False)
plt.show()

#save
if input("Save as mp4? (y/n): ").strip().lower() == 'y':
    out = os.path.join(outdir, "animation.mp4")
    ani.save(out, writer='ffmpeg', fps=200, dpi=150)
    print(f"Saved to {out}")

# #check at t=0.2
# rho_final = prims[-1][0]  #shape: (nx, ny)
# u_final = prims[-1][1]
# p_final = prims[-1][3]
# j_mid = rho_final.shape[1] // 2
# rho_slice = rho_final[:, j_mid]
# u_slice = u_final[:, j_mid]
# p_slice = p_final[:, j_mid]
# x = np.linspace(0, 1, len(rho_slice))
#
# plt.figure(figsize=(8, 4))
# plt.plot(x, rho_slice, 'b-', lw=1.5, label=r'$\rho(x,y=0.5)$')
# plt.plot(x, u_slice, 'r--', lw=1.5, label=r'$u(x,y=0.5)$')
# plt.plot(x, p_slice, 'g-.', lw=1.5, label=r'$p(x,y=0.5)$')
# plt.xlabel("x")
# plt.ylabel(f"Primitive vals HLLC t={times[-1]:.4f}")
# plt.title("shock tube snippet primitive vals at t~0.2")
# plt.legend()
# plt.grid(ls=":",alpha=0.5)
# plt.tight_layout()
# plt.savefig("./Sod_shock_tube/sod_shock_tube_1d.png", bbox_inches="tight", dpi=300)
# plt.show()
