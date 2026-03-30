#!/usr/bin/env python3

import argparse
from pathlib import Path

from pyscf import gto
from pyscf.data.elements import ELEMENTS


# ---------------------------------------------------------------------------
# 基组规格表
# ---------------------------------------------------------------------------
# 每个条目包含:
#   pyscf_name  - PySCF 使用的基组名
#   struct_name - C++ 结构体名
#   ptr_name    - C++ 全局指针名
#   display_name - 用户可见的基组名 (name 字段)
#   spherical   - 是否使用球谐基
#   output      - 输出 .cpp 文件路径
# ---------------------------------------------------------------------------

BASIS_SPECS = {
    # ======================== Pople 基组 ========================
    "sto-3g": {
        "pyscf_name": "sto-3g",
        "struct_name": "QC_BASIS_STO_3G",
        "ptr_name": "QC_BASIS_STO_3G_PTR",
        "display_name": "sto-3g",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/sto-3g.cpp"),
    },
    "3-21g": {
        "pyscf_name": "3-21g",
        "struct_name": "QC_BASIS_3_21G",
        "ptr_name": "QC_BASIS_3_21G_PTR",
        "display_name": "3-21g",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/3-21g.cpp"),
    },
    "6-31g": {
        "pyscf_name": "6-31g",
        "struct_name": "QC_BASIS_631G",
        "ptr_name": "QC_BASIS_631G_PTR",
        "display_name": "6-31g",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-31g.cpp"),
    },
    "6-31g*": {
        "pyscf_name": "6-31g*",
        "struct_name": "QC_BASIS_631G_STAR",
        "ptr_name": "QC_BASIS_631G_STAR_PTR",
        "display_name": "6-31g*",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-31g_star.cpp"),
    },
    "6-31g**": {
        "pyscf_name": "6-31g**",
        "struct_name": "QC_BASIS_631G_STARSTAR",
        "ptr_name": "QC_BASIS_631G_STARSTAR_PTR",
        "display_name": "6-31g**",
        "spherical": False,
        "output": Path(
            "SPONGE/quantum_chemistry/basis/pople/6-31g_starstar.cpp"
        ),
    },
    "6-311g": {
        "pyscf_name": "6-311g",
        "struct_name": "QC_BASIS_6311G",
        "ptr_name": "QC_BASIS_6311G_PTR",
        "display_name": "6-311g",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-311g.cpp"),
    },
    "6-311g*": {
        "pyscf_name": "6-311g*",
        "struct_name": "QC_BASIS_6311G_STAR",
        "ptr_name": "QC_BASIS_6311G_STAR_PTR",
        "display_name": "6-311g*",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-311g_star.cpp"),
    },
    "6-311g**": {
        "pyscf_name": "6-311g**",
        "struct_name": "QC_BASIS_6311G_STARSTAR",
        "ptr_name": "QC_BASIS_6311G_STARSTAR_PTR",
        "display_name": "6-311g**",
        "spherical": False,
        "output": Path(
            "SPONGE/quantum_chemistry/basis/pople/6-311g_starstar.cpp"
        ),
    },
    # Pople 弥散基组
    "6-31+g": {
        "pyscf_name": "6-31+g",
        "struct_name": "QC_BASIS_631PG",
        "ptr_name": "QC_BASIS_631PG_PTR",
        "display_name": "6-31+g",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-31+g.cpp"),
    },
    "6-31++g": {
        "pyscf_name": "6-31++g",
        "struct_name": "QC_BASIS_631PPG",
        "ptr_name": "QC_BASIS_631PPG_PTR",
        "display_name": "6-31++g",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-31++g.cpp"),
    },
    "6-31+g*": {
        "pyscf_name": "6-31+g*",
        "struct_name": "QC_BASIS_631PG_STAR",
        "ptr_name": "QC_BASIS_631PG_STAR_PTR",
        "display_name": "6-31+g*",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-31+g_star.cpp"),
    },
    "6-31+g**": {
        "pyscf_name": "6-31+g**",
        "struct_name": "QC_BASIS_631PG_STARSTAR",
        "ptr_name": "QC_BASIS_631PG_STARSTAR_PTR",
        "display_name": "6-31+g**",
        "spherical": False,
        "output": Path(
            "SPONGE/quantum_chemistry/basis/pople/6-31+g_starstar.cpp"
        ),
    },
    "6-31++g**": {
        "pyscf_name": "6-31++g**",
        "struct_name": "QC_BASIS_631PPG_STARSTAR",
        "ptr_name": "QC_BASIS_631PPG_STARSTAR_PTR",
        "display_name": "6-31++g**",
        "spherical": False,
        "output": Path(
            "SPONGE/quantum_chemistry/basis/pople/6-31++g_starstar.cpp"
        ),
    },
    "6-311+g*": {
        "pyscf_name": "6-311+g*",
        "struct_name": "QC_BASIS_6311PG_STAR",
        "ptr_name": "QC_BASIS_6311PG_STAR_PTR",
        "display_name": "6-311+g*",
        "spherical": False,
        "output": Path("SPONGE/quantum_chemistry/basis/pople/6-311+g_star.cpp"),
    },
    "6-311++g**": {
        "pyscf_name": "6-311++g**",
        "struct_name": "QC_BASIS_6311PPG_STARSTAR",
        "ptr_name": "QC_BASIS_6311PPG_STARSTAR_PTR",
        "display_name": "6-311++g**",
        "spherical": False,
        "output": Path(
            "SPONGE/quantum_chemistry/basis/pople/6-311++g_starstar.cpp"
        ),
    },
    # ======================== Def2 基组 ========================
    "def2-svp": {
        "pyscf_name": "def2-svp",
        "struct_name": "QC_BASIS_DEF2_SVP",
        "ptr_name": "QC_BASIS_DEF2_SVP_PTR",
        "display_name": "def2-svp",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/def2-svp.cpp"),
    },
    "def2-tzvp": {
        "pyscf_name": "def2-tzvp",
        "struct_name": "QC_BASIS_DEF2_TZVP",
        "ptr_name": "QC_BASIS_DEF2_TZVP_PTR",
        "display_name": "def2-tzvp",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/def2-tzvp.cpp"),
    },
    "def2-tzvpp": {
        "pyscf_name": "def2-tzvpp",
        "struct_name": "QC_BASIS_DEF2_TZVPP",
        "ptr_name": "QC_BASIS_DEF2_TZVPP_PTR",
        "display_name": "def2-tzvpp",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/def2-tzvpp.cpp"),
    },
    "def2-qzvp": {
        "pyscf_name": "def2-qzvp",
        "struct_name": "QC_BASIS_DEF2_QZVP",
        "ptr_name": "QC_BASIS_DEF2_QZVP_PTR",
        "display_name": "def2-qzvp",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/def2-qzvp.cpp"),
    },
    # Def2 弥散基组 (diffuse)
    "def2-svpd": {
        "pyscf_name": "def2-svpd",
        "struct_name": "QC_BASIS_DEF2_SVPD",
        "ptr_name": "QC_BASIS_DEF2_SVPD_PTR",
        "display_name": "def2-svpd",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/def2-svpd.cpp"),
    },
    "def2-tzvpd": {
        "pyscf_name": "def2-tzvpd",
        "struct_name": "QC_BASIS_DEF2_TZVPD",
        "ptr_name": "QC_BASIS_DEF2_TZVPD_PTR",
        "display_name": "def2-tzvpd",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/def2-tzvpd.cpp"),
    },
    # Def2 弥散基组 (minimally augmented)
    "ma-def2-svp": {
        "pyscf_name": "ma-def2-svp",
        "struct_name": "QC_BASIS_MA_DEF2_SVP",
        "ptr_name": "QC_BASIS_MA_DEF2_SVP_PTR",
        "display_name": "ma-def2-svp",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/ma-def2-svp.cpp"),
    },
    "ma-def2-tzvp": {
        "pyscf_name": "ma-def2-tzvp",
        "struct_name": "QC_BASIS_MA_DEF2_TZVP",
        "ptr_name": "QC_BASIS_MA_DEF2_TZVP_PTR",
        "display_name": "ma-def2-tzvp",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/def2/ma-def2-tzvp.cpp"),
    },
    # ======================== CC 基组 ========================
    "cc-pvdz": {
        "pyscf_name": "cc-pvdz",
        "struct_name": "QC_BASIS_CC_PVDZ",
        "ptr_name": "QC_BASIS_CC_PVDZ_PTR",
        "display_name": "cc-pvdz",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/cc/cc-pvdz.cpp"),
    },
    "cc-pvtz": {
        "pyscf_name": "cc-pvtz",
        "struct_name": "QC_BASIS_CC_PVTZ",
        "ptr_name": "QC_BASIS_CC_PVTZ_PTR",
        "display_name": "cc-pvtz",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/cc/cc-pvtz.cpp"),
    },
    # CC augmented 弥散基组
    "aug-cc-pvdz": {
        "pyscf_name": "aug-cc-pvdz",
        "struct_name": "QC_BASIS_AUG_CC_PVDZ",
        "ptr_name": "QC_BASIS_AUG_CC_PVDZ_PTR",
        "display_name": "aug-cc-pvdz",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/cc/aug-cc-pvdz.cpp"),
    },
    "aug-cc-pvtz": {
        "pyscf_name": "aug-cc-pvtz",
        "struct_name": "QC_BASIS_AUG_CC_PVTZ",
        "ptr_name": "QC_BASIS_AUG_CC_PVTZ_PTR",
        "display_name": "aug-cc-pvtz",
        "spherical": True,
        "output": Path("SPONGE/quantum_chemistry/basis/cc/aug-cc-pvtz.cpp"),
    },
}


def format_float(value: float) -> str:
    text = format(float(value), ".15g")
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def supported_elements(basis_name: str, max_z: int = 36):
    result = []
    for z in range(1, min(max_z + 1, len(ELEMENTS))):
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
        ptr_exp = int(mol._bas[ib, gto.PTR_EXP])
        ptr_coeff = int(mol._bas[ib, gto.PTR_COEFF])
        exps = [float(v) for v in mol._env[ptr_exp : ptr_exp + nprim]]
        # 展开通用收缩: 每列系数作为独立壳层
        for ic in range(nctr):
            coeffs = [
                float(v)
                for v in mol._env[
                    ptr_coeff + ic * nprim : ptr_coeff + (ic + 1) * nprim
                ]
            ]
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


def render_cpp(basis_key: str, spec: dict, max_z: int = 36) -> str:
    """生成与手写 .cpp 文件格式一致的基组源文件."""
    basis_name = spec["pyscf_name"]
    struct_name = spec["struct_name"]
    ptr_name = spec["ptr_name"]
    display_name = spec["display_name"]
    spherical = "true" if spec["spherical"] else "false"

    lines = [
        '#include "../prototype.h"',
        "",
        f"struct {struct_name} : QC_BASIS_SET",
        "{",
        f"    {struct_name}()",
        "    {",
        f'        name = "{display_name}";',
        f"        spherical = {spherical};",
        "    }",
        "    void Initialize() override",
        "    {",
        "        if (!data.empty()) return;",
        "",
    ]

    elements = supported_elements(basis_name, max_z)
    for symbol in elements:
        shells = build_shells(basis_name, symbol)
        lines.append(f'        data["{symbol}"] = {{')
        for idx, shell in enumerate(shells):
            shell_lines = render_shell(shell, "            ")
            if idx + 1 < len(shells):
                shell_lines[-1] += ","
            lines.extend(shell_lines)
        lines.append("        };")

    lines.append("    }")
    lines.append("};")
    lines.append("")

    # 静态实例和全局指针
    instance_name = struct_name.lower() + "_instance"
    lines.append(f"static {struct_name} {instance_name};")
    lines.append(f"QC_BASIS_SET* {ptr_name} = &{instance_name};")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPONGE QC basis .cpp from PySCF data."
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
        help="Output .cpp path. Defaults to the standard SPONGE location.",
    )
    parser.add_argument(
        "--max-z",
        type=int,
        default=36,
        help="Maximum atomic number to include (default: 36 = Kr).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the existing file matches the generated content.",
    )
    args = parser.parse_args()

    spec = BASIS_SPECS[args.basis_name]
    output = args.output or spec["output"]
    content = render_cpp(args.basis_name, spec, max_z=args.max_z)

    if args.check:
        if not output.exists():
            raise SystemExit(f"[FAIL] Missing {output}")
        existing = output.read_text(encoding="utf-8")
        if existing != content:
            raise SystemExit(f"[FAIL] {output} is out of date")
        print(f"[PASS] {output} matches PySCF data")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"[OK] Wrote {output}")


if __name__ == "__main__":
    main()
