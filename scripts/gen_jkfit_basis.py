#!/usr/bin/env python3
"""Generate def2-universal-JKFIT C++ basis set header from PySCF _env data."""

from pyscf import gto
from pyscf.data.elements import ELEMENTS


def format_float(s):
    """Convert float to C++ float literal."""
    val = float(s)
    return f"{val}f"


def supported_elements(basis_name):
    result = []
    for z in range(1, len(ELEMENTS)):
        sym = ELEMENTS[z]
        if not sym:
            continue
        try:
            gto.basis.load(basis_name, sym)
        except Exception:
            continue
        result.append(sym)
    return result


def build_shells(basis_name, symbol):
    spin = gto.charge(symbol) % 2
    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=basis_name,
        unit="Angstrom",
        verbose=0,
        spin=spin,
    )

    shells = []
    for ib in range(mol.nbas):
        l = int(mol._bas[ib, gto.ANG_OF])
        nprim = int(mol._bas[ib, gto.NPRIM_OF])
        nctr = int(mol._bas[ib, gto.NCTR_OF])
        if nctr != 1:
            raise RuntimeError(
                f"{basis_name} {symbol} shell {ib} has nctr={nctr}, unsupported"
            )
        ptr_exp = int(mol._bas[ib, gto.PTR_EXP])
        ptr_coeff = int(mol._bas[ib, gto.PTR_COEFF])
        exps = [float(v) for v in mol._env[ptr_exp : ptr_exp + nprim]]
        coeffs = [float(v) for v in mol._env[ptr_coeff : ptr_coeff + nprim]]
        shells.append((l, exps, coeffs))
    return shells


def gen_element(symbol, shells_data):
    """Generate C++ code for one element."""
    lines = [f'        data["{symbol}"] = {{']
    for l, exps, coeffs in shells_data:
        exp_str = ", ".join(format_float(e) for e in exps)
        coeff_str = ", ".join(format_float(c) for c in coeffs)
        lines.append(f"            {{{l}, {{{exp_str}}}, {{{coeff_str}}}}},")
    lines.append("        };")
    return "\n".join(lines)


def main():
    # Generate C++ header
    out = []
    out.append("#pragma once")
    out.append('#include "../prototype.h"')
    out.append("")
    out.append("// def2-universal-JKFIT auxiliary basis set")
    out.append("// Source: Basis Set Exchange (Turbomole 7.3)")
    out.append(
        "// Reference: F. Weigend, Phys. Chem. Chem. Phys. 8, 1057 (2006)"
    )
    out.append("")
    out.append("struct QC_BASIS_DEF2_UNIVERSAL_JKFIT : QC_BASIS_SET")
    out.append("{")
    out.append("    QC_BASIS_DEF2_UNIVERSAL_JKFIT()")
    out.append("    {")
    out.append('        name = "def2-universal-jkfit";')
    out.append("        spherical = true;")
    out.append("    }")
    out.append("    void Initialize() override")
    out.append("    {")
    out.append("        if (!data.empty()) return;")
    out.append("")

    for sym in supported_elements("def2-universal-jkfit"):
        out.append(gen_element(sym, build_shells("def2-universal-jkfit", sym)))

    out.append("    }")
    out.append("};")
    out.append("")
    out.append(
        "static QC_BASIS_DEF2_UNIVERSAL_JKFIT _qc_basis_def2_universal_jkfit_instance;"
    )
    out.append(
        "static QC_BASIS_SET* QC_BASIS_DEF2_UNIVERSAL_JKFIT_PTR = &_qc_basis_def2_universal_jkfit_instance;"
    )
    out.append("")

    outpath = "SPONGE/quantum_chemistry/basis/jkfit/def2_universal_jkfit.h"
    with open(outpath, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"Written to {outpath}")


if __name__ == "__main__":
    main()
