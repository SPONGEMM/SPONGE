"""Decompose RHF gradient into individual components for comparison with SPONGE."""
import numpy as np
from pyscf import gto, scf, grad

def decompose_gradient_h2():
    """H2 STO-3G gradient decomposition."""
    mol = gto.M(
        atom='H 0 0 -0.37; H 0 0 0.37',  # Angstrom, matches SPONGE test
        basis='sto-3g',
        unit='Angstrom'
    )
    mf = scf.RHF(mol)
    mf.kernel()

    print(f"RHF energy: {mf.e_tot:.10f} Ha")
    print(f"Orbital energies: {mf.mo_energy}")

    dm = mf.make_rdm1()
    print(f"\nDensity matrix:\n{dm}")

    # Energy-weighted density
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    dme = np.einsum('i,pi,qi->pq', mo_energy * mo_occ, mo_coeff, mo_coeff)
    print(f"\nEnergy-weighted density (W):\n{dme}")

    # Full gradient
    g = mf.nuc_grad_method()
    grad_total = g.kernel()
    print(f"\nTotal gradient (Ha/Bohr):\n{grad_total}")

    # Nuclear repulsion gradient
    grad_nuc = g.grad_nuc()
    print(f"\nNuclear gradient:\n{grad_nuc}")

    # Electronic gradient decomposition
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()

    # Get integral derivatives
    h1_T = -mol.intor('int1e_ipkin', comp=3)  # d<i|T|j>/dR_i, with ip = -nabla_R
    h1_V = -mol.intor('int1e_ipnuc', comp=3)  # d<i|V|j>/dR_i
    s1 = -mol.intor('int1e_ipovlp', comp=3)   # d<i|S|j>/dR_i

    print(f"\nip_overlap (raw, before symmetrization):")
    print(f"  s1[z, 0, 1] = {s1[2, 0, 1]:.10f}")
    print(f"  s1[z, 1, 0] = {s1[2, 1, 0]:.10f}")

    grad_T = np.zeros((mol.natm, 3))
    grad_V_ao = np.zeros((mol.natm, 3))
    grad_S = np.zeros((mol.natm, 3))

    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = aoslices[ia]

        # Symmetrize: bra + ket contributions
        # h1[x, p0:p1, :] is d<mu|O|nu>/dR_A for mu on atom A
        # Adding transpose gives the ket contribution

        T_ao = np.zeros((3, nao, nao))
        T_ao[:, p0:p1] += h1_T[:, p0:p1]
        T_ao[:, :, p0:p1] += h1_T[:, p0:p1].transpose(0, 2, 1)

        V_ao_sym = np.zeros((3, nao, nao))
        V_ao_sym[:, p0:p1] += h1_V[:, p0:p1]
        V_ao_sym[:, :, p0:p1] += h1_V[:, p0:p1].transpose(0, 2, 1)

        S_ao = np.zeros((3, nao, nao))
        S_ao[:, p0:p1] += s1[:, p0:p1]
        S_ao[:, :, p0:p1] += s1[:, p0:p1].transpose(0, 2, 1)

        grad_T[ia] = np.einsum('xij,ij->x', T_ao, dm)
        grad_V_ao[ia] = np.einsum('xij,ij->x', V_ao_sym, dm)
        grad_S[ia] = np.einsum('xij,ij->x', S_ao, dme)

    # Nuclear center derivative of V (rinv contribution)
    grad_V_nuc = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(ia))
        vrinv = -mol.intor('int1e_iprinv', comp=3)
        vrinv *= mol.atom_charge(ia)
        # symmetrize
        vrinv_sym = vrinv + vrinv.transpose(0, 2, 1)
        grad_V_nuc[ia] = np.einsum('xij,ij->x', vrinv_sym, dm)

    print(f"\n=== Gradient Decomposition (Ha/Bohr) ===")
    print(f"Nuclear repulsion gradient:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_nuc[i,0]:12.8f} {grad_nuc[i,1]:12.8f} {grad_nuc[i,2]:12.8f}")

    print(f"\nKinetic (T) gradient [Tr(P*dT/dR)]:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_T[i,0]:12.8f} {grad_T[i,1]:12.8f} {grad_T[i,2]:12.8f}")

    print(f"\nV AO-center gradient [Tr(P*dV_ao/dR)]:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_V_ao[i,0]:12.8f} {grad_V_ao[i,1]:12.8f} {grad_V_ao[i,2]:12.8f}")

    print(f"\nV nuclear-center gradient [Tr(P*dV_nuc/dR)]:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_V_nuc[i,0]:12.8f} {grad_V_nuc[i,1]:12.8f} {grad_V_nuc[i,2]:12.8f}")

    print(f"\nPulay (S) gradient [-Tr(W*dS/dR)]:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {-grad_S[i,0]:12.8f} {-grad_S[i,1]:12.8f} {-grad_S[i,2]:12.8f}")

    print(f"\nT+V_ao+V_nuc gradient:")
    tv = grad_T + grad_V_ao + grad_V_nuc
    for i in range(mol.natm):
        print(f"  Atom {i}: {tv[i,0]:12.8f} {tv[i,1]:12.8f} {tv[i,2]:12.8f}")

    print(f"\nElectronic gradient (T+V_ao+V_nuc-S):")
    elec = grad_T + grad_V_ao + grad_V_nuc - grad_S
    for i in range(mol.natm):
        print(f"  Atom {i}: {elec[i,0]:12.8f} {elec[i,1]:12.8f} {elec[i,2]:12.8f}")

    print(f"\nTotal (nuclear + electronic):")
    total = grad_nuc + elec
    for i in range(mol.natm):
        print(f"  Atom {i}: {total[i,0]:12.8f} {total[i,1]:12.8f} {total[i,2]:12.8f}")

    # Bra-only (what SPONGE currently computes)
    print(f"\n=== Bra-only Components (half of correct) ===")
    grad_T_bra = np.zeros((mol.natm, 3))
    grad_V_bra = np.zeros((mol.natm, 3))
    grad_S_bra = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = aoslices[ia]
        grad_T_bra[ia] = np.einsum('xij,ij->x', h1_T[:, p0:p1], dm[p0:p1])
        grad_V_bra[ia] = np.einsum('xij,ij->x', h1_V[:, p0:p1], dm[p0:p1])
        grad_S_bra[ia] = np.einsum('xij,ij->x', s1[:, p0:p1], dme[p0:p1])

    print(f"T bra-only:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_T_bra[i,0]:12.8f} {grad_T_bra[i,1]:12.8f} {grad_T_bra[i,2]:12.8f}")

    print(f"V AO bra-only:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_V_bra[i,0]:12.8f} {grad_V_bra[i,1]:12.8f} {grad_V_bra[i,2]:12.8f}")

    print(f"S bra-only:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_S_bra[i,0]:12.8f} {grad_S_bra[i,1]:12.8f} {grad_S_bra[i,2]:12.8f}")

    # V nuclear center (no bra/ket distinction)
    print(f"\nV nuclear-center (same as above, no bra/ket issue):")
    # Actually the rinv integral also has bra/ket symmetrization in PySCF
    grad_V_nuc_bra = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(ia))
        vrinv = -mol.intor('int1e_iprinv', comp=3)
        vrinv *= mol.atom_charge(ia)
        # bra-only: just contract with full dm
        grad_V_nuc_bra[ia] = np.einsum('xij,ij->x', vrinv, dm)
    print(f"V nuclear bra-only:")
    for i in range(mol.natm):
        print(f"  Atom {i}: {grad_V_nuc_bra[i,0]:12.8f} {grad_V_nuc_bra[i,1]:12.8f} {grad_V_nuc_bra[i,2]:12.8f}")

    # Check: is iprinv the nuclear center derivative?
    # int1e_iprinv gives nabla_C <mu|1/|r-C||nu>
    # This is NOT an AO bra/ket derivative, it's the operator derivative
    # So vrinv[x, mu, nu] = d<mu|V_C|nu>/dC_x for ALL (mu, nu)
    # The full matrix sum Tr[dm * vrinv] gives the correct nuclear center gradient
    # BUT PySCF also symmetrizes: vrinv + vrinv.T = 2*vrinv (since V_C is symmetric)
    # So we need to check if vrinv is already symmetric
    print(f"\nIs iprinv symmetric? max|V-V^T| = {np.max(np.abs(vrinv - vrinv.transpose(0,2,1))):.2e}")
    print(f"V_nuc full / V_nuc bra-only ratio: {grad_V_nuc[0,2]/grad_V_nuc_bra[0,2]:.6f}")


if __name__ == '__main__':
    decompose_gradient_h2()
