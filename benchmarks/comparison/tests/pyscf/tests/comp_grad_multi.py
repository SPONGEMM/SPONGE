"""Test 1e gradient (nuclear + T + V - Pulay) for multiple molecules/bases against PySCF."""
import numpy as np
from pyscf import gto, scf

def get_1e_gradient_pyscf(mol, mf):
    """Decompose 1e gradient into components."""
    dm = mf.make_rdm1()
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    dme = np.einsum('i,pi,qi->pq', mo_energy * mo_occ, mo_coeff, mo_coeff)

    g = mf.nuc_grad_method()
    grad_nuc = g.grad_nuc()

    h1_T = -mol.intor('int1e_ipkin', comp=3)
    h1_V = -mol.intor('int1e_ipnuc', comp=3)
    s1 = -mol.intor('int1e_ipovlp', comp=3)

    aoslices = mol.aoslice_by_atom()
    nao = mol.nao

    grad_T = np.zeros((mol.natm, 3))
    grad_V_ao = np.zeros((mol.natm, 3))
    grad_S = np.zeros((mol.natm, 3))

    for ia in range(mol.natm):
        _, _, p0, p1 = aoslices[ia]
        for h1, g_arr in [(h1_T, grad_T), (h1_V, grad_V_ao), (s1, grad_S)]:
            ao = np.zeros((3, nao, nao))
            ao[:, p0:p1] += h1[:, p0:p1]
            ao[:, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
            mat = dm if g_arr is not grad_S else dme
            g_arr[ia] = np.einsum('xij,ij->x', ao, mat)

    grad_V_nuc = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(ia))
        vrinv = -mol.intor('int1e_iprinv', comp=3) * mol.atom_charge(ia)
        vrinv_sym = vrinv + vrinv.transpose(0, 2, 1)
        grad_V_nuc[ia] = np.einsum('xij,ij->x', vrinv_sym, dm)

    grad_1e_no_eri = grad_nuc + grad_T + grad_V_ao + grad_V_nuc - grad_S
    return {
        'nuc': grad_nuc, 'T': grad_T, 'V_ao': grad_V_ao,
        'V_nuc': grad_V_nuc, 'pulay': -grad_S, '1e_total': grad_1e_no_eri,
    }


test_cases = [
    # (name, atom_str, basis, unit)
    ("H2/STO-3G", "H 0 0 -0.37; H 0 0 0.37", "sto-3g", "Angstrom"),
    ("H2/6-31G", "H 0 0 -0.37; H 0 0 0.37", "6-31g", "Angstrom"),
    ("H2/6-31G*", "H 0 0 -0.37; H 0 0 0.37", "6-31g*", "Angstrom"),
    ("H2/def2-SVP", "H 0 0 -0.37; H 0 0 0.37", "def2-svp", "Angstrom"),
    ("H2O/STO-3G",
     "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
     "sto-3g", "Angstrom"),
    ("H2O/6-31G",
     "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
     "6-31g", "Angstrom"),
    ("CH4/STO-3G",
     """C  0.0000  0.0000  0.0000;
        H  0.6276  0.6276  0.6276;
        H  0.6276 -0.6276 -0.6276;
        H -0.6276  0.6276 -0.6276;
        H -0.6276 -0.6276  0.6276""",
     "sto-3g", "Angstrom"),
    ("CH4/6-31G",
     """C  0.0000  0.0000  0.0000;
        H  0.6276  0.6276  0.6276;
        H  0.6276 -0.6276 -0.6276;
        H -0.6276  0.6276 -0.6276;
        H -0.6276 -0.6276  0.6276""",
     "6-31g", "Angstrom"),
]

print("=" * 80)
print("1e Gradient (nuclear + T + V_ao + V_nuc + Pulay) vs PySCF")
print("=" * 80)

for name, atoms, basis, unit in test_cases:
    mol = gto.M(atom=atoms, basis=basis, unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    result = get_1e_gradient_pyscf(mol, mf)
    grad_1e = result['1e_total']

    print(f"\n--- {name} (nao={mol.nao}, nbas={mol.nbas}) ---")
    print(f"  Energy: {mf.e_tot:.10f} Ha")
    print(f"  1e gradient (nuclear+T+V-Pulay, no ERI):")
    for i in range(mol.natm):
        sym = mol.atom_symbol(i)
        gx, gy, gz = grad_1e[i]
        print(f"    {sym} {i}: {gx:12.8f} {gy:12.8f} {gz:12.8f}")

    # Also print components for first atom
    i = 0
    sym = mol.atom_symbol(0)
    print(f"  Components for {sym} 0:")
    for key in ['nuc', 'T', 'V_ao', 'V_nuc', 'pulay']:
        gx, gy, gz = result[key][0]
        print(f"    {key:6s}: {gx:12.8f} {gy:12.8f} {gz:12.8f}")

    # Total gradient (with ERI)
    g = mf.nuc_grad_method()
    grad_total = g.kernel()
    print(f"  Full gradient (with ERI):")
    for i in range(mol.natm):
        sym = mol.atom_symbol(i)
        gx, gy, gz = grad_total[i]
        print(f"    {sym} {i}: {gx:12.8f} {gy:12.8f} {gz:12.8f}")
