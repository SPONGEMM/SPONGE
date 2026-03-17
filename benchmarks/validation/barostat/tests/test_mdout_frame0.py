import math

import pytest

from benchmarks.utils import Outputer
from benchmarks.validation.barostat.tests.utils import (
    AMU_PER_A3_TO_G_PER_CM3,
    read_total_mass_amu,
    run_sponge_barostat,
    write_barostat_mdin,
)
from benchmarks.validation.utils import parse_mdout_column

PRESSURE_BAROSTAT_CASES = [
    pytest.param("andersen_barostat", id="andersen_barostat"),
    pytest.param("bussi_barostat", id="bussi_barostat"),
    pytest.param("berendsen_barostat", id="berendsen_barostat"),
]


@pytest.mark.parametrize("barostat", PRESSURE_BAROSTAT_CASES)
def test_barostat_mdout_frame0_thermo_is_initialized(
    statics_path, outputs_path, barostat, mpi_np
):
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name="tip3p_water",
        mpi_np=mpi_np,
        run_name=f"{barostat}_frame0_thermo",
    )
    write_barostat_mdin(
        case_dir,
        step_limit=1,
        barostat=barostat,
        barostat_tau=0.1,
        barostat_update_interval=10,
        write_information_interval=1,
        write_mdout_interval=1,
    )

    run_sponge_barostat(case_dir, timeout=600, mpi_np=mpi_np)

    density_series = parse_mdout_column(case_dir / "mdout.txt", "density")
    pressure_series = parse_mdout_column(case_dir / "mdout.txt", "pressure")

    total_mass_amu = read_total_mass_amu(case_dir / "tip3p_mass.txt")
    box_fields = (
        (case_dir / "tip3p_coordinate.txt").read_text().splitlines()[-1].split()
    )
    box_lx, box_ly, box_lz = map(float, box_fields[:3])
    expected_density = (
        total_mass_amu * AMU_PER_A3_TO_G_PER_CM3 / (box_lx * box_ly * box_lz)
    )

    frame0_density = density_series[0]
    frame0_pressure = pressure_series[0]

    assert math.isfinite(frame0_density)
    assert math.isfinite(frame0_pressure)
    assert abs(frame0_density - expected_density) <= 5.0e-4
    assert abs(frame0_pressure) > 1.0
