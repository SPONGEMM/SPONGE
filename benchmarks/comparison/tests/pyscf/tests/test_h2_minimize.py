"""H2 bond length optimization: SPONGE minimization vs PySCF reference."""
import subprocess, os, math
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_script_dir, '..', '..', '..', '..', '..'))
SPONGE_BIN = os.path.join(_repo_root, '.pixi', 'envs', 'dev-cpu', 'bin', 'SPONGE')
if not os.path.exists(SPONGE_BIN):
    SPONGE_BIN = os.path.join(_repo_root, 'build-dev-cpu', 'SPONGE')
STATICS = os.path.join(_script_dir, '..', 'statics')


def pyscf_h2_equilibrium(basis):
    """Scan H2 PES to find equilibrium bond length."""
    from pyscf import gto, scf
    best_r, best_e = None, 1e10
    for r in np.arange(0.5, 1.2, 0.001):
        mol = gto.M(atom=f'H 0 0 {-r/2}; H 0 0 {r/2}',
                     basis=basis, unit='Angstrom', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        if mf.e_tot < best_e:
            best_e = mf.e_tot
            best_r = r
    return best_r, best_e


def run_sponge_minimize(case_dir):
    """Run SPONGE minimization and return final bond length and energy."""
    result = subprocess.run(
        [SPONGE_BIN, '-mdin', 'mdin.txt'],
        cwd=case_dir, capture_output=True, text=True, timeout=600)

    # Parse restart_coordinate.txt for final geometry
    restart = os.path.join(case_dir, 'restart_coordinate.txt')
    with open(restart) as f:
        lines = f.readlines()
    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            coords.append([float(x) for x in parts])
        elif len(parts) == 6:
            break  # box line

    # Parse final energy from mdout (data lines have numeric 'step' field)
    mdout = os.path.join(case_dir, 'mdout.txt')
    last_qc = None
    with open(mdout) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    int(parts[0])  # step number
                    last_qc = float(parts[3])  # QC column
                except (ValueError, IndexError):
                    pass

    # Compute bond length (PBC-aware)
    if len(coords) >= 2:
        box_z = 40.0  # from input
        dz = coords[1][2] - coords[0][2]
        if dz > box_z / 2: dz -= box_z
        if dz < -box_z / 2: dz += box_z
        bond_len = math.sqrt(
            (coords[1][0] - coords[0][0])**2 +
            (coords[1][1] - coords[0][1])**2 +
            dz**2)
    else:
        bond_len = None

    energy_ha = last_qc / 627.509474 if last_qc else None
    return bond_len, energy_ha


print("=" * 60)
print("H2 Minimization Test: SPONGE vs PySCF")
print("=" * 60)

# PySCF reference
pyscf_r, pyscf_e = pyscf_h2_equilibrium('sto-3g')
print(f"\nPySCF HF/STO-3G equilibrium:")
print(f"  Bond length: {pyscf_r:.4f} A")
print(f"  Energy:      {pyscf_e:.8f} Ha")

# SPONGE minimization
case_dir = os.path.join(STATICS, 'h2_min_sto3g', 'sponge')
sponge_r, sponge_e = run_sponge_minimize(case_dir)

print(f"\nSPONGE minimization (5000 Adam steps from 1.0 A):")
print(f"  Bond length: {sponge_r:.4f} A")
print(f"  Energy:      {sponge_e:.8f} Ha")

r_err = abs(sponge_r - pyscf_r)
e_err = abs(sponge_e - pyscf_e)

print(f"\nComparison:")
print(f"  Bond length error: {r_err:.4f} A ({r_err/pyscf_r*100:.2f}%)")
print(f"  Energy error:      {e_err:.8f} Ha ({e_err*627.509:.4f} kcal/mol)")

# Pass criteria: bond length within 0.02 A, energy within 0.001 Ha
r_pass = r_err < 0.02
e_pass = e_err < 0.001
status = "PASS" if (r_pass and e_pass) else "FAIL"
print(f"\n  Bond length: {'PASS' if r_pass else 'FAIL'} (tol = 0.02 A)")
print(f"  Energy:      {'PASS' if e_pass else 'FAIL'} (tol = 0.001 Ha)")
print(f"  Overall:     {status}")
