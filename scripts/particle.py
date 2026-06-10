import numpy as np
import os
import matplotlib.pyplot as plt

def read_particle_bin(filepath):
    # Define the structure to match your C++ writeParticleBin:
    # i4 = 4-byte integer (n)
    # f8 = 8-byte float/double (t, x, y, vx, vy)
    dt_struct = np.dtype([
        ('n', 'i4'),
        ('t', 'f8'),
        ('z', 'f8'),
        ('r', 'f8'),
        ('vz', 'f8'),
        ('vr', 'f8'),
        ('T', 'f8'),
        ('active', 'i4')
    ])
    
    # Read the data into a structured array
    data = np.fromfile(filepath, dtype=dt_struct)
    return data

# Load the meta data
def read_meta(outdir):
    meta = {}
    with open(os.path.join(outdir, "meta.txt")) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Split the line into the key and the value
            # e.g., "Nx 80" -> ["Nx", "80"]
            parts = line.split()
            
            if len(parts) == 2:
                key, val = parts
                meta[key] = float(val)
                
    return meta

# Execution
outdir = input("Output directory: ").strip()
meta = read_meta(outdir)
data = read_particle_bin(os.path.join(outdir, "particle.bin"))

# Conversions
# Simulated time in years
print(data.shape)
t_yr = data['t'] / (60 * 60 * 24 * 365)
z = data["z"]/meta["H_iso"]
vz = data['vz']/np.sqrt(meta["cs_iso"])
T = data["T"]

# Plotting
print(f"Latest simulation time: {t_yr[-1]:.2f} years")

fig, ax1 = plt.subplots(figsize=(10, 7))

# Axis 1: Position (Left)
ax1.plot(t_yr, z, 'k-', label=r'$z: s_0 = 1e-2$cm')
ax1.set_xlabel('t [yr]', fontsize=12)
ax1.set_ylabel(r'$z/H_1$', fontsize=12)
ax1.set_xscale('log')

# Axis 2: Velocity (Right)
ax2 = ax1.twinx()
ax2.plot(t_yr, vz, 'k--', label=r'$v: s_0 = 1e-2$cm')
ax2.set_ylabel(r'$v/c_s$', fontsize=12)

# Grid and Legend
ax1.grid(True, which="both", ls=":", alpha=0.5)

# Mimic the combined legend style
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right', frameon=False)

# Info text in bottom left
info_str = f"$H_1$={meta['H_iso']:.3e} cm\n$c_s$={np.sqrt(meta['cs_iso']):.3e} cm/s"
ax1.text(0.05, 0.05, info_str, transform=ax1.transAxes, verticalalignment='bottom')

plt.tight_layout()
plt.savefig("particle.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
plt.show()


fig, ax1 = plt.subplots(figsize=(5,5))
ax1.plot(t_yr, T/T[0], "k-", label=r"$T: s_0 = 1e-2$cm")
ax1.set_xlabel(r"t [yr]")
ax1.set_ylabel(r"T/T_0")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid(ls=":", alpha=0.5)
ax1.legend()
plt.tight_layout()
plt.savefig("particle_temperature.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
plt.show()
