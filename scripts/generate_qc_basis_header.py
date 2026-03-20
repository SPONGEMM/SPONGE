#!/usr/bin/env python3

import argparse
from pathlib import Path

from pyscf import gto
from pyscf.data.elements import ELEMENTS


BASIS_SPECS = {
    "def2-svp": {
        "var_name": "BASIS_DEF2_SVP",
        "func_name": "Initialize_Basis_Def2_SVP",
        "output": Path("SPONGE/quantum_chemistry/basis/basis_def2_svp.hpp"),
    },
    "def2-qzvp": {
        "var_name": "BASIS_DEF2_QZVP",
        "func_name": "Initialize_Basis_Def2Qzvp",
        "output": Path("SPONGE/quantum_chemistry/basis/basis_def2_qzvp.hpp"),
    },
}


def format_float(value: float) -> str:
    text = format(float(value), ".15g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def supported_elements(basis_name: str):
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


def build_shells(basis_name: str, symbol: str):
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


def render_shell(shell, indent: str) -> list[str]:
    l, exps, coeffs = shell
    exp_items = ", ".join(format_float(v) for v in exps)
    coeff_items = ", ".join(format_float(v) for v in coeffs)
    return [
        f"{indent}{{{l},",
        f"{indent} {{{exp_items}}},",
        f"{indent} {{{coeff_items}}}}}",
    ]


def render_header(basis_name: str, spec: dict) -> str:
    lines = [
        "#pragma once",
        '#include "basis_common.hpp"',
        "",
        f"inline std::map<std::string, std::vector<ShellData>> {spec['var_name']};",
        "",
        f"inline void {spec['func_name']}()",
        "{",
        f"    if (!{spec['var_name']}.empty()) return;",
        "",
    ]

    elements = supported_elements(basis_name)
    for symbol in elements:
        shells = build_shells(basis_name, symbol)
        lines.append(f'    {spec["var_name"]}["{symbol}"] = {{')
        for idx, shell in enumerate(shells):
            shell_lines = render_shell(shell, "        ")
            if idx + 1 < len(shells):
                shell_lines[-1] += ","
            lines.extend(shell_lines)
        lines.append("    };")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPONGE QC basis header from PySCF _env coefficients."
    )
    parser.add_argument(
        "basis_name",
        choices=sorted(BASIS_SPECS),
        help="Basis name to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output header path. Defaults to the standard SPONGE location.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the existing header matches the generated content.",
    )
    args = parser.parse_args()

    spec = BASIS_SPECS[args.basis_name]
    output = args.output or spec["output"]
    header = render_header(args.basis_name, spec)
    if args.check:
        if not output.exists():
            raise SystemExit(f"[FAIL] Missing {output}")
        existing = output.read_text(encoding="utf-8")
        if existing != header:
            raise SystemExit(f"[FAIL] {output} is out of date")
        print(f"[PASS] {output} matches PySCF _env data")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header, encoding="utf-8")
    print(f"[OK] Wrote {output}")


if __name__ == "__main__":
    main()
