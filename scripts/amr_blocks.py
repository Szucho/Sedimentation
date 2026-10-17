"""
Block layout check: ln(rho/rho_bg) and T/T_bg - 1 with the AMR block outlines, at 3 times.
usage: python scripts/amr_blocks.py <outdir> [rho_range] [T_range]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import hydro_reader as h

outdir = sys.argv[1]
crho = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
cT = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

meta = h.read_meta(outdir)
frames = h.read_grid(outdir + '/grid.bin')
part = h.read_particle(outdir)
gamma, mu = meta['gamma'], h.get_mu(meta)
kB, mp = 1.380649e-16, 1.672622e-24

def temperature(Q):
    rho = Q[..., 0]
    p = (gamma - 1.0)*(Q[..., 3] - 0.5*(Q[..., 1]**2 + Q[..., 2]**2)/rho)
    return p*mu*mp/(rho*kB)

rho_bg = np.median(frames[0][1][0]['data'][:, :, 0])
T_bg = np.median(temperature(frames[0][1][0]['data']))
zlen, rlen = meta['zmax'] - meta['zmin'], meta['rmax'] - meta['rmin']

picks = [len(frames)//3, 2*len(frames)//3, len(frames) - 1]
fig, axs = plt.subplots(2, 3, figsize=(13, 10), sharex=True, sharey=True)
for col, k in enumerate(picks):
    t, blocks = frames[k]
    for row, (f, c, label) in enumerate([(lambda Q: np.log(Q[:, :, 0]/rho_bg), crho, 'ln(rho/rho_bg)'),
                                         (lambda Q: temperature(Q)/T_bg - 1.0, cT, 'T/T_bg - 1')]):
        ax = axs[row, col]
        for b in blocks:
            ext = [b['x2min'], b['x2max'], b['x1min'], b['x1max']]
            im = ax.imshow(f(b['data']), origin='lower', extent=ext, vmin=-c, vmax=c, cmap='RdBu_r',
                           aspect='auto', interpolation='nearest')
            ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False, lw=0.4, ec='k'))
        if part is not None:
            i = np.argmin(np.abs(part['t'] - t))
            ax.plot(part['r'][i], part['z'][i], 'g+', ms=10)
        ax.set_xlim(0, rlen); ax.set_ylim(0, zlen)
        ax.set_title(f"{label}, t={t:.2e} s, {len(blocks)} blocks", fontsize=9)
        if col == 2: fig.colorbar(im, ax=ax)
for ax in axs[1]: ax.set_xlabel('r - rmin [cm]')
for ax in axs[:, 0]: ax.set_ylabel('z - zmin [cm]')
plt.tight_layout()
plt.savefig(outdir + '/amr_blocks.png', dpi=90)
print("saved", outdir + '/amr_blocks.png')
