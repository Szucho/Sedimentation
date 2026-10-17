import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from hydro_reader import read_meta, read_particle, coord_origin

"""
Particle trajectory, velocity and temperature from particle.bin
 usage: python particle.py [outdir]
"""

# Execution
outdir = sys.argv[1] if len(sys.argv) > 1 else input("Output directory: ").strip()
meta = read_meta(outdir)
data = read_particle(outdir)
if data is None or len(data) == 0:
    raise RuntimeError(f"no particle data in {outdir}/particle.bin")

# Scale height and sound speed of the gaussian_z profile
# cs_iso in meta.txt is p0/rho0 = c_s^2 (name kept from the old writer)
if 'cs_iso' in meta and 'H_iso' in meta:
    cs = np.sqrt(meta['cs_iso'])
    H = meta['H_iso']
elif all(k in meta for k in ('gas_p0', 'gas_rho0', 'Omega')):
    cs = np.sqrt(meta['gas_p0']/meta['gas_rho0'])
    H = np.sqrt(2.0)*cs/meta['Omega']
else:
    # meta.txt written before the gas parameters were added to it -> plot in physical units
    print("meta.txt has no gas parameters (run made with an older build): plotting z in cm and v in cm/s.\n"
          "Rerun the simulation to get z/H and v/c_s.")
    cs = H = None
scaled = H is not None

# legend label with the particle size (if the run wrote it)
s0 = f"{meta['particle_radius']:g}" if 'particle_radius' in meta else "0.01"

# Conversions
# Simulated time in years
print(data.shape)
t_yr = data['t'] / (60 * 60 * 24 * 365)
z_org, _ = coord_origin(meta)
z_abs = data["z"] + (meta['zmin'] - z_org)  # absolute z (local runs store z - zmin)
z = z_abs/H if scaled else z_abs
vz = data['vz']/cs if scaled else data['vz']
T = data["T"]

# Plotting
print(f"Latest simulation time: {t_yr[-1]:.3e} years")
if not data['active'][-1]:
    print("particle left the domain (inactive)")

fig, ax1 = plt.subplots(figsize=(10, 7))

# Axis 1: Position (Left)
ax1.plot(t_yr, z, 'k-', label=rf'$z: s_0 = {s0}$cm')
ax1.set_xlabel('t [yr]', fontsize=12)
ax1.set_ylabel(r'$z/H_1$' if scaled else r'$z$ [cm]', fontsize=12)
ax1.set_xscale('log')

# Axis 2: Velocity (Right)
ax2 = ax1.twinx()
ax2.plot(t_yr, vz, 'k--', label=rf'$v: s_0 = {s0}$cm')
ax2.set_ylabel(r'$v/c_s$' if scaled else r'$v$ [cm/s]', fontsize=12)

# Grid and Legend
ax1.grid(True, which="both", ls=":", alpha=0.5)

# Mimic the combined legend style
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right', frameon=False)

# Info text in bottom left
if scaled:
    info_str = f"$H_1$={H:.3e} cm\n$c_s$={cs:.3e} cm/s"
    ax1.text(0.05, 0.05, info_str, transform=ax1.transAxes, verticalalignment='bottom')

plt.tight_layout()
plt.savefig("particle.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
plt.show()


fig, ax1 = plt.subplots(figsize=(5,5))
ax1.plot(t_yr, T/T[0], "k-", label=rf"$T: s_0 = {s0}$cm")
ax1.set_xlabel(r"t [yr]")
ax1.set_ylabel(r"T/T$_0$")
# ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.grid(ls=":", alpha=0.5)
ax1.legend()
plt.tight_layout()
plt.savefig("particle_temperature.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
plt.show()
