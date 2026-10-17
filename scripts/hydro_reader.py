"""
Readers for the AMR hydro output (shared by animate.py and particle.py)

 meta.txt     : "key value" lines written by writeMetadata (hydro_io.cpp)
 grid.bin     : sequence of frames, each frame is
                  double t, uint64 nblocks,
                  nblocks x (int32 level, int64 lx1, int64 lx2, double x1min, x1max, x2min, x2max, uint64 nx1, nx2),
                  nblocks x (nx1*nx2*4 doubles, cell (i,j) -> [rho, rho*u, rho*v, E], j fastest)
 particle.bin : records of (int32 n, double t, z, r, vz, vr, T, int32 active), no padding

 x1 is z (vertical), x2 is r.
 Runs with "local_coords 1" in meta.txt store block bounds and particle z, r relative to zmin / rmin (absolute
 values ~1e12-1e13 cm lose the cm-scale detail); older runs store absolute values. See coord_origin. Every block has the same cell count; a block on level L is 2^L times smaller.
"""
import os
import struct
import numpy as np

k_B = 1.380649e-16  # erg/K
m_p = 1.672622e-24  # g

BLOCK_FMT = "<i q q d d d d Q Q"
BLOCK_SIZE = struct.calcsize(BLOCK_FMT)
NVAR = 4

PARTICLE_DTYPE = np.dtype([
    ('n', 'i4'),
    ('t', 'f8'),
    ('z', 'f8'),
    ('r', 'f8'),
    ('vz', 'f8'),
    ('vr', 'f8'),
    ('T', 'f8'),
    ('active', 'i4')
])


def read_meta(outdir):
    meta = {}
    with open(os.path.join(outdir, "meta.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0]] = float(parts[1])
    return meta


def coord_origin(meta):
    """(z0, r0) that the coordinates in grid.bin / particle.bin are measured from:
    0 for local-coordinate runs, so x - zmin is never formed from two ~1e12 numbers"""
    if meta.get('local_coords', 0) == 1:
        return 0.0, 0.0
    return meta['zmin'], meta['rmin']


def read_grid(path):
    """list of frames (t, blocks), blocks = list of dicts with the geometry and 'data' of shape (nx1, nx2, 4)"""
    frames = []
    with open(path, 'rb') as f:
        while True:
            head = f.read(16)
            if len(head) < 16:
                break
            t, nblocks = struct.unpack("<d Q", head)

            blocks = []
            for _ in range(nblocks):
                raw = f.read(BLOCK_SIZE)
                if len(raw) < BLOCK_SIZE:
                    return frames
                level, lx1, lx2, x1min, x1max, x2min, x2max, nx1, nx2 = struct.unpack(BLOCK_FMT, raw)
                blocks.append(dict(level=level, lx1=lx1, lx2=lx2,
                                   x1min=x1min, x1max=x1max, x2min=x2min, x2max=x2max,
                                   nx1=int(nx1), nx2=int(nx2)))

            for b in blocks:
                n = b['nx1']*b['nx2']*NVAR
                raw = f.read(n*8)
                if len(raw) < n*8:  # unfinished last frame (simulation still running / crashed)
                    return frames
                b['data'] = np.frombuffer(raw, dtype=np.float64).reshape(b['nx1'], b['nx2'], NVAR)
            frames.append((t, blocks))
    return frames


def read_particle(outdir):
    path = os.path.join(outdir, "particle.bin")
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=PARTICLE_DTYPE)


def canvas_shape(frames, meta):
    """uniform canvas on the finest level reached in ANY frame, so every frame has the same shape"""
    nx1_b, nx2_b = frames[0][1][0]['nx1'], frames[0][1][0]['nx2']
    max_level = max(b['level'] for _, blocks in frames for b in blocks)
    nblk1 = int(round(meta['Nz']))//nx1_b  # root blocks per direction
    nblk2 = int(round(meta['Nr']))//nx2_b
    return nblk1*nx1_b << max_level, nblk2*nx2_b << max_level, max_level


def stitch(blocks, shape, max_level):
    """paste every block onto the uniform finest-level canvas (coarse cells are repeated).
    Placement uses the logical location (exact), not the float bounds."""
    canvas = np.empty(shape[:2] + (NVAR,))
    for b in blocks:
        f = 1 << (max_level - b['level'])  # canvas cells per block cell
        i0 = b['lx1']*b['nx1']*f
        j0 = b['lx2']*b['nx2']*f
        canvas[i0:i0 + b['nx1']*f, j0:j0 + b['nx2']*f] = np.repeat(np.repeat(b['data'], f, axis=0), f, axis=1)
    return canvas


def primitives(Q, gamma, mu):
    """rho, u (z velocity), v (r velocity), p, T from conserved variables (last axis)"""
    rho = Q[..., 0]
    u = Q[..., 1]/rho
    v = Q[..., 2]/rho
    p = (gamma - 1.0)*(Q[..., 3] - 0.5*rho*(u**2 + v**2))
    T = p*m_p*mu/(rho*k_B)
    return rho, u, v, p, T


def get_mu(meta):
    if 'gas_mu' in meta:
        return meta['gas_mu']
    print("meta.txt has no gas_mu (older run) -> using the solver default mu = 2.3")
    return 2.3
