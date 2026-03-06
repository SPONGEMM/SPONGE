from pathlib import Path

from benchmarks.utils import Extractor


def write_sits_mdin(
    case_dir,
    *,
    step_limit=200,
    dt=0.001,
    cutoff=12.0,
    thermostat="middle_langevin",
    thermostat_tau=0.01,
    thermostat_seed=2026,
    target_temperature=300.0,
    write_information_interval=20,
    write_mdout_interval=1,
    default_in_file_prefix="sys_flexible",
    sits_mode="iteration",
    sits_atom_numbers=23,
    sits_k_numbers=4,
    sits_t_low=280.0,
    sits_t_high=420.0,
    sits_record_interval=1,
    sits_update_interval=20,
    sits_nk_fix=False,
    sits_nk_in_file=None,
    sits_pe_a=None,
    sits_pe_b=None,
    constrain_mode=None,
    coordinate_in_file=None,
    velocity_in_file=None,
    write_restart_file_interval=None,
):
    mdin = (
        'md_name = "benchmark alanine_dipeptide_tip3p_water SITS"\n'
        'mode = "nvt"\n'
        f"step_limit = {step_limit}\n"
        f"dt = {dt}\n"
        f"cutoff = {cutoff}\n"
        f'thermostat = "{thermostat}"\n'
        f"thermostat_tau = {thermostat_tau}\n"
        f"thermostat_seed = {thermostat_seed}\n"
        f"target_temperature = {target_temperature}\n"
        f'default_in_file_prefix = "{default_in_file_prefix}"\n'
    )
    if coordinate_in_file is not None:
        mdin += f'coordinate_in_file = "{coordinate_in_file}"\n'
    if velocity_in_file is not None:
        mdin += f'velocity_in_file = "{velocity_in_file}"\n'
    mdin += (
        "print_zeroth_frame = 1\n"
        f"write_mdout_interval = {write_mdout_interval}\n"
        f"write_information_interval = {write_information_interval}\n"
    )
    if write_restart_file_interval is not None:
        mdin += f"write_restart_file_interval = {write_restart_file_interval}\n"
    mdin += (
        f'SITS_mode = "{sits_mode}"\nSITS_atom_numbers = {sits_atom_numbers}\n'
    )
    if sits_mode in ("iteration", "production"):
        mdin += (
            f"SITS_k_numbers = {sits_k_numbers}\n"
            f"SITS_T_low = {sits_t_low}\n"
            f"SITS_T_high = {sits_t_high}\n"
            f"SITS_record_interval = {sits_record_interval}\n"
            f"SITS_update_interval = {sits_update_interval}\n"
            f"SITS_nk_fix = {1 if sits_nk_fix else 0}\n"
        )
        if sits_nk_in_file is not None:
            mdin += f'SITS_nk_in_file = "{sits_nk_in_file}"\n'
    if sits_mode == "empirical":
        mdin += f"SITS_T_low = {sits_t_low}\nSITS_T_high = {sits_t_high}\n"
    if sits_pe_a is not None:
        mdin += f"SITS_pe_a = {sits_pe_a}\n"
    if sits_pe_b is not None:
        mdin += f"SITS_pe_b = {sits_pe_b}\n"
    if constrain_mode is not None:
        mdin += f'constrain_mode = "{constrain_mode}"\n'
    Path(case_dir, "mdin.spg.toml").write_text(mdin)


def parse_column_series(mdout_path, column_name):
    rows = Extractor.parse_mdout_rows(mdout_path, [column_name], int_columns=())
    return [row[column_name] for row in rows]


def parse_numeric_values(path):
    values = []
    for token in Path(path).read_text().split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"No numeric values parsed from {path}")
    return values
