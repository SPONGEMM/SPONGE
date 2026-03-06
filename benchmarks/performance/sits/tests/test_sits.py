import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.fft import fft2, fftfreq, ifft2

from benchmarks.performance.sits.tests.utils import (
    parse_column_series,
    parse_numeric_values,
    write_sits_mdin,
)
from benchmarks.utils import Outputer, Runner


def write_phi_psi_cv(case_dir):
    cv_text = (
        "print\n"
        "{\n"
        "    CV = phi psi\n"
        "}\n"
        "phi\n"
        "{\n"
        "    CV_type = dihedral\n"
        "    atom = 4 6 8 14\n"
        "}\n"
        "psi\n"
        "{\n"
        "    CV_type = dihedral\n"
        "    atom = 6 8 14 16\n"
        "}\n"
    )
    Path(case_dir, "cv.txt").write_text(cv_text)


def set_cv_in_file(case_dir, cv_file="cv.txt"):
    mdin_path = Path(case_dir, "mdin.spg.toml")
    lines = mdin_path.read_text().splitlines()
    lines = [
        line for line in lines if not line.strip().startswith("cv_in_file")
    ]
    lines.append(f'cv_in_file = "{cv_file}"')
    mdin_path.write_text("\n".join(lines) + "\n")


def reweighted_1d_pmf_minimum(angles, weights, kT, bins=72, min_count=1):
    period = 2.0 * math.pi
    wrapped = ((angles + math.pi) % period) - math.pi
    edges = np.linspace(-math.pi, math.pi, bins + 1)

    hist_w, _ = np.histogram(wrapped, bins=edges, weights=weights)
    hist_n, _ = np.histogram(wrapped, bins=edges)
    valid = (hist_n >= min_count) & np.isfinite(hist_w) & (hist_w > 0.0)
    if not np.any(valid):
        raise AssertionError(
            "No valid PMF bins; increase sampling or reduce bins."
        )

    free_energy = np.full(bins, np.nan, dtype=float)
    free_energy[valid] = -kT * np.log(hist_w[valid])
    free_energy[valid] -= np.nanmin(free_energy[valid])

    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = int(np.nanargmin(free_energy))
    return centers, free_energy, centers[idx], float(free_energy[idx])


def wrap_angles(angles):
    period = 2.0 * math.pi
    return ((angles + math.pi) % period) - math.pi


def fft_kde_logp_2d(
    phi,
    psi,
    *,
    weights=None,
    gridsize=240,
    bandwidth=0.15,
    extent=(-math.pi, math.pi, -math.pi, math.pi),
    eps=1e-300,
):
    xmin, xmax, ymin, ymax = extent
    lx = xmax - xmin
    ly = ymax - ymin

    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)

    hist, xedges, yedges = np.histogram2d(
        phi,
        psi,
        bins=gridsize,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=weights,
        density=False,
    )

    dx = lx / gridsize
    dy = ly / gridsize

    kx = fftfreq(gridsize, d=dx)
    ky = fftfreq(gridsize, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    sigma = float(bandwidth)
    gauss_hat = np.exp(
        -2.0 * (math.pi**2) * (sigma**2) * (kx_grid**2 + ky_grid**2)
    )

    hist_smooth = np.real(ifft2(fft2(hist) * gauss_hat))
    hist_smooth = np.clip(hist_smooth, 0.0, None)

    alpha = 1e-12
    hist_smooth = hist_smooth + alpha * hist_smooth.sum() / hist_smooth.size

    mass = hist_smooth.sum()
    prob = hist_smooth / (mass * dx * dy)
    log_prob = np.log(prob + eps)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    phi_grid, psi_grid = np.meshgrid(xcenters, ycenters, indexing="xy")

    return phi_grid, psi_grid, log_prob.T, prob.T


def write_phi_psi_2d_pmf_ala2_style(
    case_dir, phi, psi, weights, phi_ref, psi_ref, kT
):
    ecap = 6.0
    p_mult = 30.0
    bandwidth = 0.15
    levels_f = np.linspace(0.0, ecap, 7)

    phi_grid, psi_grid, logp_ref, prob_ref = fft_kde_logp_2d(
        phi_ref,
        psi_ref,
        gridsize=240,
        bandwidth=bandwidth,
    )
    _, _, logp_cur, prob_cur = fft_kde_logp_2d(
        phi,
        psi,
        weights=weights,
        gridsize=240,
        bandwidth=bandwidth,
    )

    f_ref = -kT * logp_ref
    f_cur = -kT * logp_cur
    f_ref_plot = f_ref - np.nanmin(f_ref)
    f_cur_plot = f_cur - np.nanmin(f_cur)

    pmin_ref = np.min(prob_ref[prob_ref > 0.0])
    pmin_cur = np.min(prob_cur[prob_cur > 0.0])
    p_cut = p_mult * max(pmin_ref, pmin_cur)

    mask_ref = prob_ref > p_cut
    mask_cur = prob_cur > p_cut

    f_ref_show = np.where(mask_ref, np.clip(f_ref_plot, 0.0, ecap), np.nan)
    f_cur_show = np.where(mask_cur, np.clip(f_cur_plot, 0.0, ecap), np.nan)

    f_ref_masked = np.ma.masked_invalid(f_ref_show)
    f_cur_masked = np.ma.masked_invalid(f_cur_show)

    extent = [-math.pi, math.pi, -math.pi, math.pi]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

    im0 = axes[0].imshow(
        f_ref_masked,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.0,
        vmax=ecap,
        interpolation="nearest",
    )
    c0 = axes[0].contour(
        f_ref_masked,
        levels=levels_f,
        colors="k",
        linewidths=0.9,
        origin="lower",
        extent=extent,
    )
    axes[0].clabel(c0, inline=True, fontsize=7, fmt="%.1f")
    axes[0].set_title("Reference F (reliable only)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        f_cur_masked,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0.0,
        vmax=ecap,
        interpolation="nearest",
    )
    c1 = axes[1].contour(
        f_cur_masked,
        levels=levels_f,
        colors="k",
        linewidths=0.9,
        origin="lower",
        extent=extent,
    )
    axes[1].clabel(c1, inline=True, fontsize=7, fmt="%.1f")
    axes[1].set_title("SITS F (weighted, reliable only)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel(r"$\phi$ (rad)")
        ax.set_ylabel(r"$\psi$ (rad)")
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)

    fig.text(
        0.5,
        -0.01,
        "White regions indicate low-density uncertain areas filtered by p_cut.",
        ha="center",
        va="top",
        fontsize=10,
    )

    plot_path = Path(case_dir) / "phi_psi_pmf_2d.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": plot_path,
        "p_cut": float(p_cut),
        "ref_kept": float(np.mean(mask_ref)),
        "cur_kept": float(np.mean(mask_cur)),
        "phi_grid": phi_grid,
        "psi_grid": psi_grid,
    }


def test_sits_iteration_to_production_reweight_phi_psi(
    statics_path,
    outputs_path,
    sits_iter_steps,
    sits_prod_steps,
    mpi_np,
):
    case_name = "ala2_sits"
    iteration_step_limit = sits_iter_steps
    prod_step_limit = sits_prod_steps
    iter_write_information_interval = 200
    prod_write_information_interval = 500
    iter_mdout_interval = 200
    prod_mdout_interval = 500
    case_dir = Outputer.prepare_output_case(
        statics_path=statics_path,
        outputs_path=outputs_path,
        case_name=case_name,
        mpi_np=mpi_np,
        run_name="sits_iter_prod_reweight",
    )
    write_phi_psi_cv(case_dir)

    # 1) Iteration stage: generate Nk
    write_sits_mdin(
        case_dir,
        step_limit=iteration_step_limit,
        dt=0.002,
        cutoff=8.0,
        thermostat_tau=1.0,
        write_information_interval=iter_write_information_interval,
        write_mdout_interval=iter_mdout_interval,
        write_restart_file_interval=iter_write_information_interval,
        default_in_file_prefix="ALA",
        sits_mode="iteration",
        sits_atom_numbers=22,
        sits_k_numbers=100,
        sits_t_low=273.0,
        sits_t_high=650.0,
        sits_record_interval=1,
        sits_update_interval=20,
        sits_nk_fix=False,
        sits_pe_a=1.0,
        sits_pe_b=34.23,
        constrain_mode="SHAKE",
    )
    set_cv_in_file(case_dir)
    iter_log = Runner.run_sponge(case_dir, timeout=21600, mpi_np=mpi_np)
    assert "SITS mode = iteration" in iter_log

    nk_path = Path(case_dir) / "SITS_nk_rest.txt"
    assert nk_path.exists()
    nk_values = np.array(parse_numeric_values(nk_path), dtype=float)
    assert len(nk_values) == 100
    finite_positive = nk_values[np.isfinite(nk_values) & (nk_values > 0.0)]
    assert len(finite_positive) >= 1
    nk_fill = float(np.median(finite_positive))
    nk_safe = np.array(nk_values, copy=True)
    nk_safe[~np.isfinite(nk_safe)] = nk_fill
    nk_safe[nk_safe <= 0.0] = nk_fill
    nk_in_path = Path(case_dir) / "SITS_nk_in_for_prod.txt"
    nk_in_path.write_text(" ".join(f"{v:.8e}" for v in nk_safe) + "\n")
    iter_restart_coordinate = Path(case_dir) / "restart_coordinate.txt"
    iter_restart_velocity = Path(case_dir) / "restart_velocity.txt"
    assert iter_restart_coordinate.exists()
    assert iter_restart_velocity.exists()

    # 2) Production stage: use Nk from iteration
    write_sits_mdin(
        case_dir,
        step_limit=prod_step_limit,
        dt=0.002,
        cutoff=8.0,
        thermostat_tau=1.0,
        write_information_interval=prod_write_information_interval,
        write_mdout_interval=prod_mdout_interval,
        write_restart_file_interval=prod_write_information_interval,
        default_in_file_prefix="ALA",
        coordinate_in_file=iter_restart_coordinate.name,
        velocity_in_file=iter_restart_velocity.name,
        sits_mode="production",
        sits_atom_numbers=22,
        sits_k_numbers=100,
        sits_t_low=273.0,
        sits_t_high=650.0,
        sits_record_interval=1,
        sits_update_interval=20,
        sits_nk_fix=True,
        sits_nk_in_file=nk_in_path.name,
        sits_pe_a=1.0,
        sits_pe_b=34.23,
        constrain_mode="SHAKE",
    )
    set_cv_in_file(case_dir)
    prod_log = Runner.run_sponge(case_dir, timeout=172800, mpi_np=mpi_np)
    assert "SITS mode = production" in prod_log

    phi = np.array(
        parse_column_series(case_dir / "mdout.txt", "phi"), dtype=float
    )
    psi = np.array(
        parse_column_series(case_dir / "mdout.txt", "psi"), dtype=float
    )
    bias = np.array(
        parse_column_series(case_dir / "mdout.txt", "SITS_bias"), dtype=float
    )

    expected_samples = prod_step_limit // prod_mdout_interval + 1
    assert len(phi) == expected_samples
    assert len(psi) == expected_samples
    assert len(bias) == expected_samples
    finite_mask = np.isfinite(phi) & np.isfinite(psi) & np.isfinite(bias)
    assert np.sum(finite_mask) >= int(0.8 * expected_samples)
    phi = phi[finite_mask]
    psi = psi[finite_mask]
    bias = bias[finite_mask]

    # 3) Reweight using SITS bias: w = exp(bias / kT), stabilized by max-shift.
    kT = 8.31446261815324 * 300.0 / 4184.0
    centered = (bias - np.max(bias)) / kT
    weights = np.exp(np.clip(centered, -700.0, 0.0))

    _, _, phi_min_angle, phi_min_energy = reweighted_1d_pmf_minimum(
        phi, weights, kT, bins=36, min_count=1
    )
    _, _, psi_min_angle, psi_min_energy = reweighted_1d_pmf_minimum(
        psi, weights, kT, bins=36, min_count=1
    )

    ref_path = Path(case_dir) / "bf_ref" / "phi_psi_ref.npy"
    assert ref_path.exists(), f"Reference dihedral file not found: {ref_path}"
    phi_psi_ref = np.load(ref_path)
    assert phi_psi_ref.ndim == 2 and phi_psi_ref.shape[1] == 2

    phi_wrapped = wrap_angles(phi)
    psi_wrapped = wrap_angles(psi)
    phi_ref = wrap_angles(np.asarray(phi_psi_ref[50000:, 0], dtype=float))
    psi_ref = wrap_angles(np.asarray(phi_psi_ref[50000:, 1], dtype=float))

    plot_result = write_phi_psi_2d_pmf_ala2_style(
        case_dir,
        phi_wrapped,
        psi_wrapped,
        weights,
        phi_ref,
        psi_ref,
        kT,
    )
    print(f"2D PMF plot saved to: {plot_result['path']}")

    phi_span = float(np.max(phi) - np.min(phi))
    psi_span = float(np.max(psi) - np.min(psi))
    bias_span = float(np.max(bias) - np.min(bias))

    headers = [
        "Case",
        "IterSteps",
        "ProdSteps",
        "PhiMin(rad)",
        "PhiMin(deg)",
        "PsiMin(rad)",
        "PsiMin(deg)",
        "BiasSpan",
        "RefKept",
        "SITSKept",
        "Status",
    ]
    rows = [
        [
            case_name,
            str(iteration_step_limit),
            str(prod_step_limit),
            f"{phi_min_angle:.4f}",
            f"{math.degrees(phi_min_angle):.2f}",
            f"{psi_min_angle:.4f}",
            f"{math.degrees(psi_min_angle):.2f}",
            f"{bias_span:.4f}",
            f"{plot_result['ref_kept']:.4f}",
            f"{plot_result['cur_kept']:.4f}",
            "PASS",
        ]
    ]
    Outputer.print_table(
        headers,
        rows,
        title="SITS Iteration->Production Reweight: Phi/Psi PMF minima",
    )

    assert -math.pi <= phi_min_angle <= math.pi
    assert -math.pi <= psi_min_angle <= math.pi
    assert math.isfinite(phi_min_energy)
    assert math.isfinite(psi_min_energy)
    assert phi_span >= 1.0
    assert psi_span >= 1.0
    assert bias_span >= 1.0
    assert plot_result["ref_kept"] > 0.85
    assert plot_result["cur_kept"] > 0.10
