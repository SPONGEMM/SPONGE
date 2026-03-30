"""Full RHF gradient sanity check: SPONGE energy finite difference vs PySCF analytical.

NOTE: SPONGE uses float precision for energies, so finite difference accuracy
is limited to ~0.001 Ha/Bohr for small molecules. This test validates that
forces have the correct sign and magnitude, not exact numerical agreement.
For precise gradient validation, see test_h2_minimize.py.
"""
import subprocess, os, tempfile, shutil
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_script_dir, '..', '..', '..', '..', '..'))
SPONGE_BIN = os.path.join(_repo_root, '.pixi', 'envs', 'dev-cpu', 'bin', 'SPONGE')
if not os.path.exists(SPONGE_BIN):
    SPONGE_BIN = os.path.join(_repo_root, 'build-dev-cpu', 'SPONGE')
STATICS = os.path.join(_script_dir, '..', 'statics')


def sponge_energy(case_dir, coords, box=40.0):
    """Run SPONGE single point, return QC energy in kcal/mol."""
    tmpdir = tempfile.mkdtemp()
    try:
        for f in ['qc_type.txt', 'mass.txt', 'charge.txt', 'mdin.txt']:
            shutil.copy(os.path.join(case_dir, f), tmpdir)
        with open(os.path.join(tmpdir, 'coordinate.txt'), 'w') as f:
            f.write(f"{len(coords)}\n")
            for c in coords:
                f.write(f"{c[0]:.10f} {c[1]:.10f} {c[2]:.10f}\n")
            f.write(f"{box} {box} {box} 90.0 90.0 90.0\n")
        with open(os.path.join(tmpdir, 'mdin.txt'), 'r') as f:
            mdin = f.read()
        if 'qc_need_gradient' not in mdin:
            mdin += "\nqc_need_gradient = 0\n"
        with open(os.path.join(tmpdir, 'mdin.txt'), 'w') as f:
            f.write(mdin)
        result = subprocess.run(
            [SPONGE_BIN, '-mdin', 'mdin.txt'],
            cwd=tmpdir, capture_output=True, text=True, timeout=300)
        for line in result.stdout.split('\n'):
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'QC' and i + 2 < len(parts):
                    val = parts[i + 2].rstrip(',')
                    if val != '=':
                        try:
                            return float(val)
                        except ValueError:
                            pass
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def sponge_fd_gradient(case_dir, coords, h=0.002):
    """Compute gradient via central finite differences (h in Angstrom)."""
    natm = len(coords)
    grad = np.zeros((natm, 3))
    for ia in range(natm):
        for d in range(3):
            cp = [list(c) for c in coords]
            cm = [list(c) for c in coords]
            cp[ia][d] += h
            cm[ia][d] -= h
            ep = sponge_energy(case_dir, cp)
            em = sponge_energy(case_dir, cm)
            if ep is not None and em is not None:
                grad[ia, d] = (ep - em) / (2 * h)
    # kcal/mol/Å → Ha/Bohr
    grad /= (627.509474 * 1.8897259886)
    return grad


def pyscf_gradient(atoms_str, basis):
    from pyscf import gto, scf
    mol = gto.M(atom=atoms_str, basis=basis, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mf.nuc_grad_method().kernel(), mf.e_tot


test_cases = [
    ("H2/STO-3G", "h2_sto3g",
     [[0, 0, -0.37], [0, 0, 0.37]],
     "H 0 0 -0.37; H 0 0 0.37", "sto-3g"),
    ("H2/6-31G", "h2",
     [[0, 0, -0.37], [0, 0, 0.37]],
     "H 0 0 -0.37; H 0 0 0.37", "6-31g"),
]

print("=" * 70)
print("RHF Gradient: SPONGE (finite diff, h=0.002A) vs PySCF (analytical)")
print("=" * 70)

results = []
for name, case_dir, coords, atoms_str, basis in test_cases:
    full_dir = os.path.join(STATICS, case_dir, 'sponge')
    if not os.path.isdir(full_dir):
        print(f"SKIP {name}")
        continue

    print(f"\n--- {name} ---")
    pyscf_grad, _ = pyscf_gradient(atoms_str, basis)
    sponge_grad = sponge_fd_gradient(full_dir, coords)

    max_abs = np.max(np.abs(sponge_grad - pyscf_grad))
    status = 'PASS' if max_abs < 0.005 else 'FAIL'
    print(f"  SPONGE FD:  {sponge_grad[0]}")
    print(f"  PySCF anal: {pyscf_grad[0]}")
    print(f"  Max abs err = {max_abs:.6f}  [{status}]")
    results.append((name, max_abs, status))

print(f"\n{'='*70}")
for name, err, status in results:
    print(f"  {name:<20} err={err:.6f}  {status}")
