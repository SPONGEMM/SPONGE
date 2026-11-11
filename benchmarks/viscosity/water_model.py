#!/usr/bin/env python3
"""Frozen SPC/E inputs and same-state molecular stress gates; not liquid viscosity.

Parameters: https://docs.lammps.org/Howto_spc.html and
Berendsen, Grigera, Straatsma (1987), DOI 10.1021/j100308a038.
No thermostat, equilibration or experimental fitting is hidden in preparation.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import subprocess
import time

import h5py
import numpy as np

from lj_model import digest, execute

ROOT = Path(__file__).resolve().parents[2]
PA = 4184 / 6.02214076e23 / 1e-30
SP_TIME = float(np.float32(20.455))
KB = float(np.float32(.00198716))
LM_COULOMB = 332.06371  # frozen 29Aug2024 real-unit qqrd2e
AXES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
NAMES = ('pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz')
MODEL_FILES = {'state.npz', 'mass.txt', 'charge.txt', 'zero_charge.txt', 'coordinate.txt',
               'velocity.txt', 'zero_velocity.txt', 'lj.txt', 'zero_lj.txt', 'constraint.txt',
               'exclude.txt', 'residue.txt', 'lammps.data'}


def internal_timestep(dt_ps):
    """MD_core Read_dt for an explicit input: atof(dt)*float constant, then float.

    Precasting dt to float introduces a second rounding that the engine does
    not perform. All benchmark inputs provide dt explicitly.
    """
    return float(np.float32(float(dt_ps)*SP_TIME))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def array_file(path, values, box=None):
    values = np.asarray(values)
    rows = values.reshape(len(values), -1)
    text = str(len(values))+'\n'+'\n'.join(' '.join(f'{x:.17g}' for x in row) for row in rows)+'\n'
    if box is not None:
        text += f'{box:.17g} {box:.17g} {box:.17g}\n90 90 90\n'
    Path(path).write_text(text)


def constrained_reference(positions, half, force, mass, pairs, box, dt, virial):
    """Independent NumPy dense 3x3-per-water solves, no engine observer call."""
    positions, half, force, mass = [np.asarray(x, dtype=float) for x in (positions, half, force, mass)]
    pairs = np.asarray(pairs, dtype=float)
    virial = np.asarray(virial, dtype=float)
    n = len(mass)
    if (positions.shape != (n, 3) or half.shape != (n, 3) or force.shape != (n, 3) or
            mass.shape != (n,) or n < 3 or n % 3 or pairs.shape != (n, 3) or virial.shape != (3, 3)):
        raise ValueError('water arrays must have three consecutive sites per molecule')
    if (not all(np.all(np.isfinite(a)) for a in (positions, half, force, mass, pairs, virial)) or
            np.any(mass <= 0) or not np.isfinite(box) or box <= 0 or
            not np.isfinite(dt) or dt < 0 or np.any(pairs[:, 2] <= 0) or
            np.any(pairs[:, :2] != np.floor(pairs[:, :2]))):
        raise ValueError('finite data, positive masses/box/lengths and integer indices required')
    velocity = half + .5*dt*force/mass[:, None]
    constraint = np.zeros((3, 3))
    max_length = 0.
    for start in range(0, n, 3):
        block = pairs[start:start+3]
        jac = np.zeros((3, 9))
        distances = []
        for row, (a, b, length) in enumerate(block):
            a, b = int(a), int(b)
            if not start <= a < start+3 or not start <= b < start+3:
                raise ValueError('reference requires complete per-water distance triples')
            d = positions[a]-positions[b]
            d -= box*np.floor(d/box+.5)
            r = np.linalg.norm(d)
            if r <= 0:
                raise ValueError('coincident constrained sites')
            max_length = max(max_length, abs(r/length-1))
            distances.append(d)
            jac[row, (a-start)*3:(a-start+1)*3] = d/r
            jac[row, (b-start)*3:(b-start+1)*3] = -d/r
        inverse = 1/np.repeat(mass[start:start+3], 3)
        metric = (jac*inverse)@jac.T
        u = velocity[start:start+3].ravel()
        u = u + inverse*(jac.T@np.linalg.solve(metric, -jac@u))
        velocity[start:start+3] = u.reshape(3, 3)
        curvature = []
        for (a, b, _), d in zip(block, distances):
            v = velocity[int(a)]-velocity[int(b)]
            r = np.linalg.norm(d)
            curvature.append((v@v-(d@v/r)**2)/r)
        rhs = -(jac@(force[start:start+3].ravel()*inverse)+curvature)
        multipliers = np.linalg.solve(metric, rhs)
        constraint += sum(lam*np.outer(d, d)/np.linalg.norm(d) for lam, d in zip(multipliers, distances))
    com = np.average(velocity, weights=mass, axis=0)
    thermal = velocity-com
    kinetic = (thermal.T*mass)@thermal
    dof = 3*n-len(pairs)-3
    return dict(velocity=velocity, constraint=constraint, kinetic=kinetic,
                pressure=(kinetic+constraint+virial)/box**3*PA,
                temperature=np.trace(kinetic)/float(np.float32(dof)*np.float32(KB)),
                dof=dof, max_relative_length_error=max_length)


def prepare(directory, cells=6, density=998., temperature=298.15, cutoff=8., seed=20261028):
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 2:
        raise ValueError('cells must be an integer >= 2')
    if not all(not isinstance(x, bool) and np.isfinite(x) and x > 0 for x in (density, temperature, cutoff)):
        raise ValueError('positive finite density/temperature/cutoff required')
    molecules = cells**3
    masses = np.tile(np.array([15.9994, 1.008, 1.008], dtype=np.float32).astype(float), molecules)
    box = float(np.float32((masses.sum()*1.66053906660e3/density)**(1/3)))
    cutoff = float(np.float32(cutoff))
    if 2*cutoff >= box:
        raise ValueError('cutoff must be smaller than half the actual box')
    angle = np.deg2rad(109.47)
    hh = float(np.float32(2*np.sin(angle/2)))
    # Effective triangle follows the float32 distance table exactly before
    # final position storage rounding, rather than mixing two angle roundings.
    half_width = hh/2
    body = np.array([[0, 0, 0], [half_width, np.sqrt(1-half_width**2), 0],
                     [-half_width, np.sqrt(1-half_width**2), 0]])
    body -= np.average(body, weights=masses[:3], axis=0)
    rng = np.random.default_rng(seed)
    positions = []
    for cell in np.ndindex(cells, cells, cells):
        # Normalized Gaussian quaternion gives isotropic orientation.
        w, x, y, z = (lambda q: q/np.linalg.norm(q))(rng.normal(size=4))
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        center = (np.array(cell)+.5)*box/cells
        positions.extend((body@rotation.T+center) % box)
    positions = np.array(positions, dtype=np.float32).astype(float)
    pairs = np.array([(i+a, i+b, length) for i in range(0, len(masses), 3)
                      for a, b, length in ((0, 1, 1.), (0, 2, 1.), (1, 2, hh))])
    raw = rng.normal(size=positions.shape)*np.sqrt(KB*temperature/masses[:, None])
    projected = constrained_reference(positions, raw, np.zeros_like(raw), masses, pairs, box, 0, np.zeros((3, 3)))
    velocity = projected['velocity']-np.average(projected['velocity'], weights=masses, axis=0)
    velocity *= np.sqrt(temperature/projected['temperature'])
    velocity = velocity.astype(np.float32).astype(float)
    charges = np.tile(np.array([-.8476, .4238, .4238])*float(np.float32(18.2223)), molecules).astype(np.float32).astype(float)
    c12_input, c6_input = [float(np.float32(v)) for v in (4*.1553*3.166**12, 4*.1553*3.166**6)]
    # Native loader stores 12*C12 and 6*C6 as float32. Match those effective
    # coefficients in the reference, and record this storage quantization.
    c12 = float(np.float32(np.float32(c12_input)*np.float32(12)))/12
    c6 = float(np.float32(np.float32(c6_input)*np.float32(6)))/6
    epsilon, sigma = c6*c6/(4*c12), (c12/c6)**(1/6)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory/'state.npz', positions=positions, half_velocity=velocity,
             masses=masses, native_charges=charges, pairs=pairs)
    array_file(directory/'mass.txt', masses)
    array_file(directory/'charge.txt', charges)
    array_file(directory/'zero_charge.txt', charges*0)
    array_file(directory/'coordinate.txt', positions, box)
    array_file(directory/'velocity.txt', velocity)
    array_file(directory/'zero_velocity.txt', velocity*0)
    types = np.tile([0, 1, 1], molecules)
    for filename, a, b in [('lj.txt', c12_input, c6_input), ('zero_lj.txt', 0, 0)]:
        (directory/filename).write_text(f'{len(masses)} 2\n{a:.17g}\n0\n0\n{b:.17g}\n0\n0\n'+'\n'.join(map(str, types))+'\n')
    (directory/'constraint.txt').write_text(str(len(pairs))+'\n'+'\n'.join(f'{int(a)} {int(b)} {r:.17g}' for a, b, r in pairs)+'\n')
    (directory/'exclude.txt').write_text(f'{len(masses)} {len(pairs)}\n'+''.join(f'2 {i+1} {i+2}\n1 {i+2}\n0\n' for i in range(0, len(masses), 3)))
    (directory/'residue.txt').write_text(f'{len(masses)} {molecules}\n'+'3\n'*molecules)
    # Bonds/angles have zero potential in LAMMPS: only intramolecular exclusions
    # and optional SHAKE topology are defined by these records.
    lines = ['SPC/E frozen Hamiltonian; effective native precision', '',
             f'{len(masses)} atoms', f'{2*molecules} bonds', f'{molecules} angles',
             '2 atom types', '1 bond types', '1 angle types', '',
             f'0 {box:.17g} xlo xhi', f'0 {box:.17g} ylo yhi', f'0 {box:.17g} zlo zhi', '',
             'Masses', '', f'1 {masses[0]:.17g}', f'2 {masses[1]:.17g}', '', 'Atoms # full', '']
    lines += [f'{i+1} {i//3+1} {types[i]+1} {charges[i]/np.sqrt(LM_COULOMB):.17g} '+
              ' '.join(f'{v:.17g}' for v in row) for i, row in enumerate(positions)]
    lines += ['', 'Bonds', '']+[f'{2*m+j} 1 {3*m+1} {3*m+j+1}' for m in range(molecules) for j in (1, 2)]
    lines += ['', 'Angles', '']+[f'{m+1} 1 {3*m+2} {3*m+1} {3*m+3}' for m in range(molecules)]
    (directory/'lammps.data').write_text('\n'.join(lines)+'\n')
    model = dict(contract='sponge.spce.transport-model.v1', model='SPC/E with Ewald electrostatics',
                 molecules=molecules, atoms=len(masses), seed=seed, cells=cells,
                 box_angstrom=box, cutoff_angstrom=cutoff, requested_density_kg_m3=density,
                 actual_density_kg_m3=masses.sum()*1.66053906660e3/box**3,
                 requested_initial_temperature_k=temperature, thermal_dof=projected['dof'],
                 target_parameters=dict(mass_da=[15.9994, 1.008], charge_e=[-.8476, .4238],
                                        epsilon_kcal_mol=.1553, sigma_angstrom=3.166, oh_angstrom=1., hoh_degree=109.47),
                 effective_c12=c12, effective_c6=c6, lammps_epsilon=epsilon, lammps_sigma=sigma,
                 native_charge_scale=float(np.float32(18.2223)), lammps_qqrd2e=LM_COULOMB,
                 mapped_lammps_charge_e=(charges[:3]/np.sqrt(LM_COULOMB)).tolist(), hh_angstrom=hh,
                 model_sources=['https://docs.lammps.org/Howto_spc.html', 'https://doi.org/10.1021/j100308a038'],
                 source_access_date='2026-10-28',
                 force_convention='unshifted truncated OO LJ; full periodic tin-foil electrostatics; all intramolecular nonbonded pairs excluded',
                 tail='SPONGE C6-only N_O^2 energy/diagonal stress; reference tail disabled and corrected analytically',
                 polarization_self_energy='constant molecular polarization correction omitted in both engines; not an absolute vaporization-energy benchmark',
                 initialization='oriented cubic molecular lattice and projected Gaussian velocities; not an equilibrated liquid',
                 input_sha256={p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()})
    save(directory/'model.json', model)
    return model


def load_model(directory):
    directory = Path(directory)
    model = json.loads((directory/'model.json').read_text())
    if model['contract'] != 'sponge.spce.transport-model.v1':
        raise ValueError('unsupported water model contract')
    if set(model['input_sha256']) != MODEL_FILES:
        raise ValueError('incomplete model input hash manifest')
    for name, expected in model['input_sha256'].items():
        if Path(name).name != name or digest(directory/name) != expected:
            raise ValueError(f'model input hash mismatch: {name}')
    return model, np.load(directory/'state.npz')


def sponge_input(directory, source, model, component='full', grid=128, algorithm=None, dt_ps=.00025,
                 direct_tolerance=1e-10):
    config = dict(md_name='water_same_state', mode='nve', pbc=True,
                  dt=dt_ps if algorithm else 0, step_limit=4 if algorithm else 1,
                  cutoff=model['cutoff_angstrom'], skin=1., print_zeroth_frame=True,
                  write_mdout_interval=1, write_information_interval=1,
                  write_trajectory_interval=1, write_restart_file_interval=0,
                  crd='crd.dat', vel='vel.dat', frc='force.dat', output_h5_observable_path='observables.h5')
    config.update({'PM.MPI_size': 1, 'PM.fftx': grid, 'PM.ffty': grid, 'PM.fftz': grid,
                   'PM.Direct_Tolerance': direct_tolerance, 'PM.print_detail': True})
    for key, file in [('mass', 'mass.txt'), ('charge', 'zero_charge.txt' if component == 'lj' else 'charge.txt'),
                      ('coordinate', 'coordinate.txt'), ('velocity', 'velocity.txt' if algorithm else 'zero_velocity.txt'),
                      ('LJ', 'zero_lj.txt' if component == 'coulomb' else 'lj.txt'),
                      ('exclude', 'exclude.txt'), ('residue', 'residue.txt')]:
        config[f'{key}_in_file'] = str(source/file)
    if algorithm:
        config.update({'constrain_mode': algorithm, 'settle_disable': algorithm == 'SHAKE',
                       'constrain.in_file': str(source/'constraint.txt'),
                       'SHAKE.iteration_numbers': 100, 'SHAKE.tolerance': 1e-6})
    write_engine_input(directory/'mdin.spg.toml', '\n'.join(f'{k}={json.dumps(v)}' for k, v in config.items())+'\n[viscosity]\nmethod="green_kubo"\n')


def lammps_input(directory, source, model, component):
    cutoff = model['cutoff_angstrom']
    epsilon = model['lammps_epsilon'] if component != 'coulomb' else 0
    style = f'lj/cut {cutoff:.17g}' if component == 'lj' else f'lj/cut/coul/long {cutoff:.17g}'
    electrostatics = 'set group all charge 0' if component == 'lj' else 'kspace_style ewald 1e-10'
    fields = ' '.join(f'$(c_p[{i}]:%.17g)' for i in range(1, 7))
    (directory/'in.lammps').write_text(f'''units real
atom_style full
boundary p p p
read_data {source/'lammps.data'}
bond_style zero
bond_coeff 1 1.0
angle_style zero
angle_coeff 1 {np.rad2deg(2*np.arcsin(model['hh_angstrom']/2)):.17g}
pair_style {style}
pair_coeff 1 1 {epsilon:.17g} {model['lammps_sigma']:.17g}
pair_coeff 1 2 0 1
pair_coeff 2 2 0 1
pair_modify shift no tail no
{'pair_modify table 0' if component != 'lj' else ''}
special_bonds lj/coul 0 0 0
{electrostatics}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
compute p all pressure NULL pair kspace
thermo_style custom step pe evdwl ecoul elong c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6]
thermo_modify norm no format float %.17g
run 0
write_dump all custom forces.dump id fx fy fz modify sort id format float %.17g
print "$(pe:%.17g) $(evdwl:%.17g) $(ecoul:%.17g) $(elong:%.17g) {fields}" file static.dat
''')


def read_sponge(directory, atoms):
    force = np.fromfile(directory/'force.dat', np.float32).reshape(-1, atoms, 3)[0].astype(float)
    with h5py.File(directory/'viscosity.h5') as file:
        values = {name: group['value'][0] for name, group in file['observables/all'].items() if isinstance(group, h5py.Group)}
    # Ordinary observables retain each Step_Print field's decimal formatting.
    # eff_pot has higher precision; PM/LJ diagnostic fields are only %.2f.
    with h5py.File(directory/'observables.h5') as file:
        energy = {name: float(np.ravel(group['value'][()])[0]) for name, group in file['observables/all'].items() if isinstance(group, h5py.Group)}
    return force, np.array([values[name] for name in NAMES]), energy, values


def static_check(source, output, sponge, lammps, grids=(128, 256), direct_tolerance=1e-10):
    source, output, sponge, lammps = [Path(p).resolve() for p in (source, output, sponge, lammps)]
    model, state = load_model(source)
    if not grids or any(isinstance(g, bool) or not isinstance(g, int) or g < 16 for g in grids):
        raise ValueError('integer PME grids >=16 required')
    if isinstance(direct_tolerance, bool) or not np.isfinite(direct_tolerance) or not 0 < direct_tolerance < 1:
        raise ValueError('direct tolerance must be finite and between 0 and 1')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(contract='sponge.spce.same-state-gate.v1', model_sha256=digest(source/'model.json'),
                grids=list(grids), direct_tolerance=direct_tolerance,
                lammps_coulomb_table_bits=0, algorithms=['SETTLE', 'SHAKE'], dt_ps=.00025,
                gates=dict(force_relative_l2=3e-4, force_max_absolute=2e-3,
                           energy_relative=2e-5, energy_absolute=.02,
                           pressure_relative_max_norm=5e-4, pressure_absolute_pa=100),
                semantics='same initial coordinates; no LAMMPS SHAKE/fix virial; independent NumPy instantaneous constraint reconstruction',
                sponge_sha256=digest(sponge), lammps_sha256=digest(lammps), source_sha256=digest(__file__),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                formal_viscosity_acceptance=False)
    plan['precision_selection'] = ('exploratory follow-up to failed 64/128, direct_tolerance=1e-7; '
                                   'gate thresholds unchanged; not a preregistered scientific acceptance')
    plan['energy_precision'] = 'eff_pot formatted %.7e; PM and LJ diagnostics formatted %.2f'
    save(output/'plan.json', plan)
    results, references = [], {}
    started = time.perf_counter()
    tail_energy = -2*np.pi/3*model['molecules']**2*model['effective_c6']/model['cutoff_angstrom']**3/model['box_angstrom']**3
    tail_pressure = np.array([1, 1, 1, 0, 0, 0])*2*tail_energy/model['box_angstrom']**3*PA
    try:
        for component in ('lj', 'coulomb', 'full'):
            lm = output/f'lammps-{component}'
            lm.mkdir(); lammps_input(lm, source, model, component)
            wall = execute([lammps, '-in', 'in.lammps'], lm)
            values = np.loadtxt(lm/'static.dat')
            force = np.loadtxt(lm/'forces.dump', skiprows=9)
            if not np.array_equal(force[:, 0], np.arange(1, model['atoms']+1)):
                raise ValueError('LAMMPS atom ordering mismatch')
            references[component] = (force[:, 1:], values[4:]*101325, values[0])
            save(lm/'execution.json', dict(wall_seconds=wall, component=component,
                 version_line=next(line for line in (lm/'run.log').read_text().splitlines() if line.startswith('LAMMPS ('))))
            for grid in grids:
                sp = output/f'sponge-{component}-{grid}'
                sp.mkdir(); sponge_input(sp, source, model, component, grid, direct_tolerance=direct_tolerance)
                wall = execute([sponge, '-mdin', 'mdin.spg.toml'], sp)
                actual_f, actual_p, energy, _ = read_sponge(sp, model['atoms'])
                expected_p = references[component][1]+(tail_pressure if component != 'coulomb' else 0)
                expected_e = references[component][2]+(tail_energy if component != 'coulomb' else 0)
                force_error = actual_f-references[component][0]
                record = dict(component=component, grid=grid, wall_seconds=wall,
                              force_relative_l2=float(np.linalg.norm(force_error)/max(np.linalg.norm(references[component][0]), 1e-20)),
                              force_max_absolute=float(np.max(abs(force_error))),
                              pressure_max_error_pa=float(np.max(abs(actual_p-expected_p))),
                              pressure_scale_pa=float(np.max(abs(expected_p))),
                              energy_error=float(energy['eff_pot']-expected_e),
                              sponge_energy=energy, reference_energy=references[component][2],
                              sponge_pressure_pa=actual_p.tolist(), expected_pressure_pa=expected_p.tolist())
                record['passed'] = bool(record['force_relative_l2'] <= 3e-4 and record['force_max_absolute'] <= 2e-3 and
                                    record['pressure_max_error_pa'] <= 100+5e-4*record['pressure_scale_pa'] and
                                    abs(record['energy_error']) <= .02+2e-5*abs(expected_e))
                results.append(record); save(sp/'comparison.json', record)
        reference_force, reference_pressure, _ = references['full']
        virial = np.zeros((3, 3))
        for value, (a, b) in zip(reference_pressure, AXES):
            virial[a, b] = virial[b, a] = value*model['box_angstrom']**3/PA
        h = internal_timestep(plan['dt_ps'])
        expected = constrained_reference(state['positions'], state['half_velocity'], reference_force,
                                         state['masses'], state['pairs'], model['box_angstrom'], h, virial)
        for algorithm in plan['algorithms']:
            sp = output/f'sponge-constrained-{algorithm}'
            sp.mkdir(); sponge_input(sp, source, model, 'full', max(grids), algorithm, plan['dt_ps'], direct_tolerance)
            wall = execute([sponge, '-mdin', 'mdin.spg.toml'], sp)
            _, actual_p, _, values = read_sponge(sp, model['atoms'])
            expected_p = np.array([expected['pressure'][a, b] for a, b in AXES])+tail_pressure
            error = float(np.max(abs(actual_p-expected_p)))
            record = dict(algorithm=algorithm, wall_seconds=wall, pressure_max_error_pa=error,
                          pressure_scale_pa=float(np.max(abs(expected_p))),
                          temperature_error_k=float(values['temperature']-expected['temperature']),
                          sponge_pressure_pa=actual_p.tolist(), reference_pressure_pa=expected_p.tolist(),
                          thermal_dof=expected['dof'], max_initial_length_residual=expected['max_relative_length_error'])
            record['passed'] = bool(error <= 100+5e-4*record['pressure_scale_pa'] and abs(record['temperature_error_k']) <= .01)
            results.append(record); save(sp/'comparison.json', record)
        summary = dict(status='passed_same_state_gates' if all(r['passed'] for r in results) else 'failed_same_state_gates',
                       cases=results, wall_seconds=time.perf_counter()-started,
                       tail_energy_kcal_mol=tail_energy, formal_viscosity_acceptance=False)
        save(output/'summary.json', summary)
        return summary
    except Exception as error:
        save(output/'failure.json', dict(error=str(error), completed_cases=results, wall_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', required=True)
    p.add_argument('--cells', type=int, default=6); p.add_argument('--density', type=float, default=998)
    p.add_argument('--temperature', type=float, default=298.15)
    p.add_argument('--cutoff', type=float, default=8); p.add_argument('--seed', type=int, default=20261028)
    p = sub.add_parser('static'); p.add_argument('--model', required=True); p.add_argument('--output', required=True)
    p.add_argument('--sponge', required=True); p.add_argument('--lammps', required=True)
    p.add_argument('--grids', type=int, nargs='+', default=[128, 256])
    p.add_argument('--direct-tolerance', type=float, default=1e-10)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output, args.cells, args.density, args.temperature, args.cutoff, args.seed)
    else:
        result = static_check(args.model, args.output, args.sponge, args.lammps, tuple(args.grids), args.direct_tolerance)
    print(json.dumps(result, indent=2))
    if result.get('status', '').startswith('failed'):
        raise SystemExit(1)
