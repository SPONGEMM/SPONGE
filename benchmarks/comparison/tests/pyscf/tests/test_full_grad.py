"""Full RHF gradient (nuclear + 1e + 2e) comparison: SPONGE vs PySCF."""
import subprocess, os, sys, json, tempfile, shutil
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.normpath(os.path.join(_script_dir, '..', '..', '..', '..', '..'))
SPONGE_BIN = os.path.join(_repo_root, '.pixi', 'envs', 'dev-cpu', 'bin', 'SPONGE')
if not os.path.exists(SPONGE_BIN):
    SPONGE_BIN = os.path.join(_repo_root, 'build-dev-cpu', 'SPONGE')
STATICS = os.path.join(_script_dir, '..', 'statics')

def run_sponge_gradient(case_dir):
    """Run SPONGE and extract gradient via env var debug output."""
    env = os.environ.copy()
    env['SPONGE_DEBUG_GRAD'] = '1'
    result = subprocess.run(
        [SPONGE_BIN, '-mdin', 'mdin.txt'],
        cwd=case_dir, capture_output=True, text=True, env=env, timeout=300)
    grad = {}
    for line in result.stderr.split('\n'):
        if line.startswith('AFTER_2E') or line.startswith('AFTER_XC'):
            label = line.split()[0]
        if line.strip().startswith('atom '):
            parts = line.strip().split()
            iatm = int(parts[1])
            vals = parts[3].strip('()').split(',')
            grad[iatm] = [float(v) for v in vals]
    # Use AFTER_XC if available (DFT), else AFTER_2E (HF)
    # Actually just get the last set of atom lines
    atoms = {}
    for line in result.stderr.split('\n'):
        if 'AFTER_2E' in line:
            atoms = {}  # reset
        if line.strip().startswith('atom '):
            parts = line.strip().split()
            iatm = int(parts[1])
            vals = parts[3].strip('()').split(',')
            atoms[iatm] = [float(v) for v in vals]
    return atoms

def run_sponge_gradient_simple(case_dir):
    """Run SPONGE with SPONGE_DEBUG_GRAD, parse AFTER_2E block."""
    env = os.environ.copy()
    env['SPONGE_DEBUG_GRAD'] = '1'
    result = subprocess.run(
        [SPONGE_BIN, '-mdin', 'mdin.txt'],
        cwd=case_dir, capture_output=True, text=True, env=env, timeout=300)

    # Parse all debug blocks, keep the last complete one before force writeback
    blocks = {}
    current_label = None
    current_atoms = {}
    for line in result.stderr.split('\n'):
        for label in ['AFTER_NUCLEAR', 'AFTER_1E', 'AFTER_2E', 'AFTER_XC']:
            if label in line:
                if current_label and current_atoms:
                    blocks[current_label] = dict(current_atoms)
                current_label = label
                current_atoms = {}
                break
        if line.strip().startswith('atom '):
            parts = line.strip().split()
            iatm = int(parts[1])
            # Parse "(x, y, z)" format
            rest = line.split(':', 1)[1].strip().strip('()')
            vals = [float(v.strip()) for v in rest.split(',')]
            current_atoms[iatm] = vals
    if current_label and current_atoms:
        blocks[current_label] = dict(current_atoms)

    # Use last available block (AFTER_XC > AFTER_2E > AFTER_1E)
    for key in ['AFTER_XC', 'AFTER_2E', 'AFTER_1E']:
        if key in blocks:
            return blocks[key]
    return {}


def pyscf_gradient(atoms_str, basis, unit='Angstrom'):
    from pyscf import gto, scf
    mol = gto.M(atom=atoms_str, basis=basis, unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    g = mf.nuc_grad_method()
    return g.kernel(), mf.e_tot

# Test cases: (name, case_dir_name, atoms_str, basis)
test_cases = [
    ("H2/STO-3G", "h2_sto3g",
     "H 0 0 -0.37; H 0 0 0.37", "sto-3g"),
    ("H2/6-31G", "h2",
     "H 0 0 -0.37; H 0 0 0.37", "6-31g"),
    ("H2O/STO-3G", "h2o_sto3g",
     "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692", "sto-3g"),
    ("H2O/6-31G", "h2o_631g",
     "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692", "6-31g"),
    ("CH4/STO-3G", "ch4_sto3g",
     """C 0 0 0; H 0.6276 0.6276 0.6276; H 0.6276 -0.6276 -0.6276;
        H -0.6276 0.6276 -0.6276; H -0.6276 -0.6276 0.6276""", "sto-3g"),
    ("CH4/6-31G", "ch4_631g",
     """C 0 0 0; H 0.6276 0.6276 0.6276; H 0.6276 -0.6276 -0.6276;
        H -0.6276 0.6276 -0.6276; H -0.6276 -0.6276 0.6276""", "6-31g"),
    ("benzene/STO-3G", "benzene_sto3g",
     """C  1.2124  0.7000  0.0; C  1.2124 -0.7000  0.0; C  0.0 -1.4000  0.0;
        C -1.2124 -0.7000  0.0; C -1.2124  0.7000  0.0; C  0.0  1.4000  0.0;
        H  2.1562  1.2450  0.0; H  2.1562 -1.2450  0.0; H  0.0 -2.4900  0.0;
        H -2.1562 -1.2450  0.0; H -2.1562  1.2450  0.0; H  0.0  2.4900  0.0""",
     "sto-3g"),
]

print("=" * 76)
print("Full RHF Gradient: SPONGE vs PySCF")
print("=" * 76)

results = []
for name, case_dir, atoms_str, basis in test_cases:
    full_dir = os.path.join(STATICS, case_dir, 'sponge')
    if not os.path.isdir(full_dir):
        print(f"SKIP {name}: directory not found")
        continue

    print(f"\n--- {name} ---")

    # PySCF reference
    pyscf_grad, pyscf_e = pyscf_gradient(atoms_str, basis)
    natm = len(pyscf_grad)

    # SPONGE
    sponge_grad = run_sponge_gradient_simple(full_dir)

    if not sponge_grad:
        print(f"  SPONGE: no gradient output (check SPONGE_DEBUG_GRAD)")
        results.append((name, float('inf'), 'ERROR'))
        continue

    # Compare
    max_abs = 0
    max_rel = 0
    for ia in range(natm):
        for d in range(3):
            p = pyscf_grad[ia, d]
            s = sponge_grad.get(ia, [0,0,0])[d]
            err = abs(s - p)
            max_abs = max(max_abs, err)
            if abs(p) > 0.001:
                max_rel = max(max_rel, err / abs(p))

    status = 'PASS' if max_abs < 0.01 else 'FAIL'
    print(f"  PySCF E = {pyscf_e:.10f} Ha")
    print(f"  Max abs err = {max_abs:.6f}  Max rel err = {max_rel:.4%}  [{status}]")

    if max_abs > 0.001 or status == 'FAIL':
        print(f"  Per-atom comparison:")
        for ia in range(natm):
            p = pyscf_grad[ia]
            s = sponge_grad.get(ia, [0,0,0])
            err = max(abs(s[d] - p[d]) for d in range(3))
            flag = " <<<" if err > 0.001 else ""
            print(f"    atom {ia}: SPONGE=({s[0]:10.6f},{s[1]:10.6f},{s[2]:10.6f})"
                  f"  PySCF=({p[0]:10.6f},{p[1]:10.6f},{p[2]:10.6f})  err={err:.6f}{flag}")

    results.append((name, max_abs, status))

print(f"\n{'='*76}")
print(f"{'Case':<20} {'Max Abs Err':>12} {'Status':>8}")
print(f"{'-'*76}")
for name, err, status in results:
    print(f"{name:<20} {err:12.6f} {status:>8}")
