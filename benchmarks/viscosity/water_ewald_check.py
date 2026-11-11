#!/usr/bin/env python3
"""Independent explicit Ewald diagnostic for the frozen water fixture.

This double-precision calculation uses no SPONGE/LAMMPS force implementation.
It separates the omitted real-space contribution from mesh/roundoff errors.
Finite image and reciprocal sums require convergence checks, not an exact label.
"""
import argparse
import itertools
import math
from pathlib import Path
import time

import numpy as np

import water_model as water


def erfc(values):
    return np.fromiter((math.erfc(float(x)) for x in values.flat), float,
                       count=values.size).reshape(values.shape)


def native_beta(cutoff, tolerance):
    """Mirror only the documented float32 bisection, not the force code."""
    low, high = np.float32(0), np.float32(1)
    while math.erfc(float(high*np.float32(cutoff)))/cutoff > tolerance:
        high *= np.float32(2)
    for _ in range(55):
        beta = np.float32((low+high)/np.float32(2))
        value = np.float32(math.erfc(float(beta*np.float32(cutoff)))/cutoff)
        if value >= np.float32(tolerance):
            low = beta
        else:
            high = beta
    return float(beta)


def explicit_ewald(positions, charges, pairs, box, cutoff, beta, kmax=20, images=1):
    """Energy and physical virial for neutral charges in an orthorhombic box.

    All real-space lattice pairs in [-images,images]^3 are summed with 1/2
    weight; self at zero displacement is removed. Excluded molecular pairs
    subtract q_i*q_j/r once. Tin-foil boundary: no dipole surface term.
    Short and omitted real-space contributions are returned separately.
    """
    x, q, box = np.asarray(positions, float), np.asarray(charges, float), np.broadcast_to(box, (3,)).astype(float)
    if (x.shape != (len(q), 3) or not np.all(np.isfinite(x)) or
            not np.all(np.isfinite(q)) or abs(q.sum()) > 1e-8 or
            not np.all(np.isfinite(box)) or np.any(box <= 0) or
            not np.isfinite(beta) or beta <= 0 or
            not 0 < cutoff < min(box)/2 or kmax < 1 or images < 1):
        raise ValueError('finite neutral periodic system and positive sum limits required')
    x = x % box
    energy = dict(real_short=0., real_omitted=0., exclusion=0., reciprocal=0., self=0.)
    virial = {name: np.zeros((3, 3)) for name in energy}
    pair_q = q[:, None]*q[None, :]
    for cell in itertools.product(range(-images, images+1), repeat=3):
        delta = x[:, None, :]-x[None, :, :]+np.array(cell)*box
        radius = np.linalg.norm(delta, axis=2)
        mask = radius > 0
        r, d, qq = radius[mask], delta[mask], pair_q[mask]
        g = erfc(beta*r)
        e = .5*qq*g/r
        force_factor = .5*qq*(g+2*beta*r/np.sqrt(np.pi)*np.exp(-(beta*r)**2))/r**3
        for name, select in [('real_short', r < cutoff), ('real_omitted', r >= cutoff)]:
            energy[name] += e[select].sum()
            virial[name] += np.einsum('i,ij,ik->jk', force_factor[select], d[select], d[select])
    for a, b, _ in pairs:
        a, b = int(a), int(b)
        d = x[a]-x[b]
        d -= box*np.floor(d/box+.5)
        r = np.linalg.norm(d)
        energy['exclusion'] -= q[a]*q[b]/r
        virial['exclusion'] -= q[a]*q[b]*np.outer(d, d)/r**3
    integers = np.array(list(itertools.product(range(-kmax, kmax+1), repeat=3)))
    wave = integers[np.any(integers, axis=1)]*2*np.pi/box
    volume = np.prod(box)
    for start in range(0, len(wave), 512):
        k = wave[start:start+512]
        k2 = np.einsum('ij,ij->i', k, k)
        structure = abs(np.exp(1j*(x@k.T)).T@q)**2
        e = 2*np.pi/volume*np.exp(-k2/(4*beta**2))/k2*structure
        energy['reciprocal'] += e.sum()
        virial['reciprocal'] += np.eye(3)*e.sum()-np.einsum(
            'i,ij,ik->jk', 2*e*(1/k2+1/(4*beta**2)), k, k)
    energy['self'] = -beta/np.sqrt(np.pi)*(q@q)
    return dict(energy={key: float(value) for key, value in energy.items()}, virial=virial)


def diagnose(source, output, sponge, lammps):
    source, output, sponge, lammps = [Path(p).resolve() for p in (source, output, sponge, lammps)]
    model, state = water.load_model(source)
    output.mkdir(parents=True, exist_ok=False)
    water.save(output/'plan.json', dict(model_sha256=water.digest(source/'model.json'),
               source_sha256=water.digest(__file__), water_source_sha256=water.digest(water.__file__),
               sponge_sha256=water.digest(sponge), lammps_sha256=water.digest(lammps),
               controls=[[128, 1e-7], [256, 1e-7], [128, 1e-10], [256, 1e-10]],
               kmax=20, real_images=1, reference_tables=[0, 12],
               purpose='post-failure error decomposition; no viscosity acceptance'))
    started = time.perf_counter()
    references = {}
    for bits in (0, 12):
        directory = output/f'lammps-table{bits}'
        directory.mkdir()
        water.lammps_input(directory, source, model, 'coulomb')
        p = directory/'in.lammps'
        p.write_text(p.read_text().replace('pair_modify table 0', f'pair_modify table {bits}'))
        wall = water.execute([lammps, '-in', 'in.lammps'], directory)
        values = np.loadtxt(directory/'static.dat')
        references[bits] = values[4:]*101325
        water.save(directory/'execution.json', dict(wall_seconds=wall))
    results = []
    for tolerance in (1e-7, 1e-10):
        beta = native_beta(model['cutoff_angstrom'], tolerance)
        exact = explicit_ewald(state['positions'], state['native_charges'], state['pairs'],
                               model['box_angstrom'], model['cutoff_angstrom'], beta)
        # Sum includes full bare-Coulomb exclusion. Removing omitted real terms
        # gives the same cutoff convention as erfc-direct + -erf exclusions.
        full = sum(exact['virial'].values())
        omitted = exact['virial']['real_omitted']
        to_pressure = lambda v: np.array([v[a, b] for a, b in water.AXES])*water.PA/model['box_angstrom']**3
        full_p, omitted_p = to_pressure(full), to_pressure(omitted)
        record = dict(direct_tolerance=tolerance, beta=beta, energy_parts=exact['energy'],
                      full_energy=sum(exact['energy'].values()), full_pressure_pa=full_p.tolist(),
                      omitted_real_pressure_pa=omitted_p.tolist(),
                      full_minus_lammps_pa={str(b): (full_p-p).tolist() for b, p in references.items()},
                      sponge=[])
        for grid in (128, 256):
            directory = output/f'sponge-g{grid}-tol{tolerance}'
            directory.mkdir()
            water.sponge_input(directory, source, model, 'coulomb', grid, direct_tolerance=tolerance)
            wall = water.execute([sponge, '-mdin', 'mdin.spg.toml'], directory)
            _, pressure, energy, _ = water.read_sponge(directory, model['atoms'])
            record['sponge'].append(dict(grid=grid, wall_seconds=wall, energy=energy['eff_pot'],
                     pressure_pa=pressure.tolist(),
                     minus_explicit_same_cutoff_pa=(pressure-(full_p-omitted_p)).tolist()))
        results.append(record)
        water.save(output/f'tolerance-{tolerance}.json', record)
    summary = dict(cases=results, wall_seconds=time.perf_counter()-started,
                   beta_invariance_pressure_max_pa=float(np.max(abs(np.array(results[0]['full_pressure_pa'])-
                                                                    results[1]['full_pressure_pa']))),
                   formal_viscosity_acceptance=False)
    water.save(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'output', 'sponge', 'lammps'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = diagnose(args.model, args.output, args.sponge, args.lammps)
    print('beta-invariance pressure max (Pa):', result['beta_invariance_pressure_max_pa'])
