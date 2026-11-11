#!/usr/bin/env python3
"""SPC/E liquid preparation and explicit structural handoff to NVE pilots.

No automatic equilibration, viscosity, experimental agreement or dynamic
thermostat restart claim. Raw trajectories and unsuccessful diagnostics remain.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import subprocess
import shutil
import time

import h5py
import numpy as np

import water_model as water
from water_ewald_check import native_beta


def write_input(directory, source, model, steps, dt, stride, mode, seed=18273,
                grid=128, coordinate=None, velocity=None):
    config = dict(md_name='spce_liquid_pilot', mode=mode, pbc=True, dt=dt,
                  step_limit=steps, cutoff=model['cutoff_angstrom'], skin=1.,
                  print_zeroth_frame=True, write_mdout_interval=stride,
                  write_information_interval=stride, write_trajectory_interval=stride,
                  write_restart_file_interval=0, crd='crd.dat', vel='vel.dat',
                  output_h5_observable_path='observables.h5', constrain_mode='SETTLE')
    config.update({'PM.MPI_size': 1, 'PM.fftx': grid, 'PM.ffty': grid, 'PM.fftz': grid,
                   'PM.Direct_Tolerance': 1e-10, 'constrain.in_file': str(source/'constraint.txt')})
    for name, filename in [('mass', 'mass.txt'), ('charge', 'charge.txt'), ('LJ', 'lj.txt'),
                           ('exclude', 'exclude.txt'), ('residue', 'residue.txt'),
                           ('coordinate', 'coordinate.txt'), ('velocity', 'velocity.txt')]:
        config[name+'_in_file'] = str(source/filename)
    if coordinate is not None:
        water.array_file(directory/'coordinate.txt', coordinate, model['box_angstrom'])
        config['coordinate_in_file'] = str(directory/'coordinate.txt')
    if velocity is not None:
        water.array_file(directory/'velocity.txt', velocity)
        config['velocity_in_file'] = str(directory/'velocity.txt')
    if mode == 'nvt':
        config.update(thermostat='middle_langevin', thermostat_tau=.1, thermostat_seed=seed,
                      target_temperature=model['requested_initial_temperature_k'])
    else:
        # Force output includes q0 and the final force-time sample. Native q/v
        # omit q0; the explicit input supplies it for independent kinetic audits.
        config['frc'] = 'force.dat'
    text = '\n'.join(f'{key}={json.dumps(value)}' for key, value in config.items())+'\n'
    if mode == 'nve':
        text += '[viscosity]\nmethod="green_kubo"\nsample_interval=1\n'
    write_engine_input(directory/'mdin.spg.toml', text)


def read_observables(path, expected):
    with h5py.File(path) as file:
        journal = file['parameters/sponge/output']
        if journal['status'][()] != b'finalized' or int(journal['last_complete_step'][-1]) != expected[-1]:
            raise ValueError('incomplete observable output')
        group = file['observables/all']
        if not np.array_equal(group['step'][()], expected) or int(journal['frame_count'][-1]) != len(expected):
            raise ValueError('observable cadence/journal mismatch')
        values = {name: np.ravel(item['value'][()]) for name, item in group.items() if isinstance(item, h5py.Group)}
    if any(not np.all(np.isfinite(v)) or len(v) != len(expected) for v in values.values()):
        raise ValueError('nonfinite/missing observable values')
    return values


def native_frames(path, atoms, expected):
    result = np.fromfile(path, np.float32).reshape(-1, atoms, 3).astype(float)
    if len(result) != expected or not np.all(np.isfinite(result)):
        raise ValueError('native trajectory count/nonfinite values')
    return result


def geometry(position, state, model):
    pairs, box = state['pairs'], model['box_angstrom']
    a, b = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    delta = position[:, a]-position[:, b]
    delta -= box*np.floor(delta/box+.5)
    maximum = float(np.max(abs(np.linalg.norm(delta, axis=2)/pairs[:, 2]-1)))
    if maximum > 1e-4:
        raise ValueError(f'constraint length residual {maximum} exceeds observer limit')
    return maximum


def motion_and_rdf(position, state, model, spacing_ps, output):
    box = model['box_angstrom']
    water_xyz = position.reshape(len(position), -1, 3, 3)
    local = water_xyz-water_xyz[:, :, :1]
    local -= box*np.floor(local/box+.5)
    mass = state['masses'][:3]
    centers = (water_xyz[:, :, 0]+np.einsum('fmij,i->fmj', local, mass)/mass.sum()) % box
    delta = np.diff(centers, axis=0)
    delta -= box*np.floor(delta/box+.5)
    jump = float(abs(delta).max())
    if jump >= box/4:
        raise ValueError('resolved COM jump too large for sampled unwrapping')
    continuous = np.concatenate((centers[:1], centers[:1]+np.cumsum(delta, axis=0)))
    start = len(position)//2
    displacement = continuous[start:]-continuous[start]
    displacement -= displacement.mean(axis=1, keepdims=True)
    msd = np.mean(np.sum(displacement**2, axis=2), axis=1)
    np.savetxt(output/'msd.csv', np.column_stack((np.arange(len(msd))*spacing_ps, msd)),
               delimiter=',', header='elapsed_ps,single_origin_com_msd_angstrom2', comments='')
    n = model['molecules']; i, j = np.triu_indices(n, 1)
    edges = np.linspace(0, box/2, 101); counts = np.zeros(100)
    for frame in position[start:]:
        d = frame[::3][i]-frame[::3][j]; d -= box*np.floor(d/box+.5)
        counts += np.histogram(np.linalg.norm(d, axis=1), edges)[0]
    shell = 4*np.pi/3*np.diff(edges**3)
    rdf = counts/((len(position)-start)*n*(n-1)/2*shell/box**3)
    np.savetxt(output/'rdf-oo.csv', np.column_stack(((edges[1:]+edges[:-1])/2, rdf)),
               delimiter=',', header='radius_angstrom,g_oo_finite_N_normalization', comments='')
    return dict(max_resolved_com_jump_angstrom=jump, latter_half_single_origin_msd_angstrom2=float(msd[-1]),
                interpretation='mobility/structure diagnostics only; no diffusion fit or equilibrium assertion')


def centered_seed(position, state, model, seed):
    raw = np.random.default_rng(seed).normal(size=position.shape)*np.sqrt(
        water.KB*model['requested_initial_temperature_k']/state['masses'][:, None])
    result = water.constrained_reference(position, raw, np.zeros_like(raw), state['masses'],
                                         state['pairs'], model['box_angstrom'], 0, np.zeros((3, 3)))
    velocity = result['velocity']-np.average(result['velocity'], weights=state['masses'], axis=0)
    return velocity, result['temperature']


def half_step(velocity, force, state, position, model, dt):
    h = water.internal_timestep(dt)
    half = (velocity-.5*h*force/state['masses'][:, None]).astype(np.float32).astype(float)
    result = water.constrained_reference(position, half, force, state['masses'], state['pairs'],
                                         model['box_angstrom'], h, np.zeros((3, 3)))
    return half, result, h


def force_probe(directory, source, model, sponge, position, grid):
    directory.mkdir()
    water.array_file(directory/'coordinate.txt', position, model['box_angstrom'])
    water.sponge_input(directory, source, model, grid=grid)
    config = directory/'mdin.spg.toml'
    config.write_text(config.read_text().replace(str(source/'coordinate.txt'), str(directory/'coordinate.txt')))
    wall = water.execute([sponge, '-mdin', 'mdin.spg.toml'], directory)
    force, _, _, _ = water.read_sponge(directory, model['atoms'])
    if not np.all(np.isfinite(force)):
        raise ValueError('nonfinite handoff force')
    return force, wall


def snapshot_model(source, seed_file, output, dt=.00025):
    """Export a derived frozen fixture for the existing cross-engine static gates."""
    source, seed_file, output = [Path(p).resolve() for p in (source, seed_file, output)]
    model, state = water.load_model(source)
    report = json.loads((seed_file.parent/'summary.json').read_text())
    if (water.digest(seed_file) != report['seed_sha256'] or
            water.digest(source/'model.json') != report['model_sha256'] or
            water.digest(seed_file.parent/'plan.json') != report['plan_sha256']):
        raise ValueError('structural snapshot identity mismatch')
    with np.load(seed_file) as seed:
        q, u, f = [seed[name] for name in ('positions', 'tangent_velocity', 'force')]
    half, _, _ = half_step(u, f, state, q, model, dt)
    output.mkdir(parents=True, exist_ok=False)
    for name in water.MODEL_FILES:
        shutil.copy2(source/name, output/name)
    water.array_file(output/'coordinate.txt', q, model['box_angstrom'])
    water.array_file(output/'velocity.txt', half)
    np.savez(output/'state.npz', positions=q, half_velocity=half, masses=state['masses'],
             native_charges=state['native_charges'], pairs=state['pairs'])
    data = (source/'lammps.data').read_text()
    prefix, rest = data.split('Atoms # full\n\n')
    atoms, suffix = rest.split('\nBonds\n')
    lines = [line for line in atoms.splitlines() if line.strip()]
    if len(lines) != model['atoms']:
        raise ValueError('unexpected molecular data format')
    updated = []
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 7 or int(fields[0]) != index+1:
            raise ValueError('unexpected molecular atom ordering')
        updated.append(' '.join(fields[:4])+' '+' '.join(f'{value:.17g}' for value in q[index]))
    (output/'lammps.data').write_text(prefix+'Atoms # full\n\n'+'\n'.join(updated)+'\n\nBonds\n'+suffix)
    model['initialization'] = 'derived preparation snapshot and explicit incoming half velocity; no equilibrium acceptance'
    model['snapshot'] = dict(parent_model_sha256=water.digest(source/'model.json'), seed_sha256=water.digest(seed_file),
                             source_plan_sha256=report['plan_sha256'], velocity_conversion_dt_ps=dt)
    model['input_sha256'] = {name: water.digest(output/name) for name in sorted(water.MODEL_FILES)}
    water.save(output/'model.json', model)
    return model


def plan_run(source, output, sponge, steps, dt, stride, grid, mode):
    source, output, sponge = [Path(p).resolve() for p in (source, output, sponge)]
    model, state = water.load_model(source)
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps < 100 or
            not isinstance(stride, int) or stride < 1 or steps % stride or steps//stride < 10 or
            steps//stride > 10000 or not np.isfinite(dt) or not 0 < dt <= .002 or
            not isinstance(grid, int) or grid < 32):
        raise ValueError('invalid bounded steps/dt/cadence/grid')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(contract='sponge.spce.liquid-pilot.v1', mode=mode, steps=steps, dt_ps=dt,
                duration_ps=steps*dt, audit_interval=stride, grid=grid, direct_tolerance=1e-10,
                model_sha256=water.digest(source/'model.json'), sponge_sha256=water.digest(sponge),
                source_sha256=water.digest(__file__), water_source_sha256=water.digest(water.__file__),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=water.ROOT, text=True).strip(),
                formal_viscosity_acceptance=False)
    return source, output, sponge, model, state, plan


def equilibrate(source, output, sponge, steps=20000, dt=.0005, stride=50, grid=128,
                thermostat_seed=18273, velocity_seed=71027):
    source, output, sponge, model, state, plan = plan_run(source, output, sponge, steps, dt, stride, grid, 'nvt')
    plan.update(thermostat_seed=thermostat_seed, velocity_seed=velocity_seed, thermostat_tau_ps=.1,
                handoff='final trajectory q at requested completed step; independent projected Maxwell draw; no RNG/mode continuity')
    write_input(output, source, model, steps, dt, stride, 'nvt', thermostat_seed, grid)
    plan['mdin_sha256'] = water.digest(output/'mdin.spg.toml')
    water.save(output/'plan.json', plan)
    started = time.perf_counter()
    try:
        elapsed = water.execute([sponge, '-mdin', 'mdin.spg.toml'], output, timeout=1800)
        water.save(output/'execution.json', dict(status='engine_completed', wall_seconds=elapsed))
        count = steps//stride
        q = native_frames(output/'crd.dat', model['atoms'], count)
        v = native_frames(output/'vel.dat', model['atoms'], count)
        obs = read_observables(output/'observables.h5', np.arange(0, steps+1, stride))
        length = geometry(q, state, model)
        mobility = motion_and_rdf(q, state, model, stride*dt, output)
        position = q[-1]
        # Probe the exact exported position without another position update.
        force, force_wall = force_probe(output/'handoff-force', source, model, sponge, position, grid)
        velocity, temperature = centered_seed(position, state, model, velocity_seed)
        np.savez(output/'structural_seed.npz', positions=position, tangent_velocity=velocity,
                 force=force, native_nvt_velocity=v[-1], box=model['box_angstrom'])
        blocks = []
        for indices in np.array_split(np.arange(len(obs['temperature'])//2, len(obs['temperature'])), 5):
            blocks.append(dict(mean_native_temperature_k=float(np.mean(obs['temperature'][indices])),
                               mean_potential_kcal_mol=float(np.mean(obs['eff_pot'][indices]))))
        report = dict(status='completed_preparation_pilot_not_equilibrium_acceptance', model_sha256=plan['model_sha256'],
                      plan_sha256=water.digest(output/'plan.json'), seed_sha256=water.digest(output/'structural_seed.npz'),
                      duration_ps=steps*dt, engine_wall_seconds=elapsed, force_probe_wall_seconds=force_wall,
                      total_wall_seconds=time.perf_counter()-started, maximum_length_residual=length,
                      latter_half_blocks=blocks, mobility=mobility, new_nve_velocity_seed=velocity_seed,
                      new_nve_initial_temperature_k=float(temperature), thermal_dof=3*model['atoms']-len(state['pairs'])-3,
                      warnings=['Native NVT temperature uses its stored velocity convention and includes COM DOF.',
                                'NVT velocity is retained but not reinterpreted as integer NVE velocity.',
                                'Independent tangent Maxwell draw removes COM; no temperature rescaling.',
                                'One preparation trajectory does not prove equilibrium or liquid viscosity.'],
                      formal_viscosity_acceptance=False)
        report['raw_sha256'] = {p.name: water.digest(p) for p in (output/'crd.dat', output/'vel.dat', output/'observables.h5')}
        water.save(output/'summary.json', report)
        return report
    except Exception as error:
        water.save(output/'failure.json', dict(error=str(error), wall_seconds=time.perf_counter()-started))
        raise


def energy_shift(position, state, model):
    """Continuous cutoff-shift energy offset; forces and trajectories unchanged."""
    i, j = np.triu_indices(model['atoms'], 1)
    keep = i//3 != j//3; i, j = i[keep], j[keep]
    d = position[i]-position[j]; d -= model['box_angstrom']*np.floor(d/model['box_angstrom']+.5)
    r = np.linalg.norm(d, axis=1); active = r < model['cutoff_angstrom']
    oo = (i % 3 == 0) & (j % 3 == 0)
    rc = model['cutoff_angstrom']
    lj = (model['effective_c12']/rc**12-model['effective_c6']/rc**6)*np.count_nonzero(active & oo)
    import math
    beta = native_beta(rc, 1e-10)
    coulomb = math.erfc(beta*rc)/rc*np.sum((state['native_charges'][i]*state['native_charges'][j])[active])
    return float(lj+coulomb)


def nve(source, seed_file, output, sponge, steps=2000, dt=.0005, stride=20, grid=128):
    seed_file = Path(seed_file).resolve()
    report = json.loads((seed_file.parent/'summary.json').read_text())
    if water.digest(seed_file) != report['seed_sha256'] or water.digest(Path(source)/'model.json') != report['model_sha256']:
        raise ValueError('structural seed/model hash mismatch')
    if water.digest(seed_file.parent/'plan.json') != report['plan_sha256']:
        raise ValueError('structural seed plan hash mismatch')
    source, output, sponge, model, state, plan = plan_run(source, output, sponge, steps, dt, stride, grid, 'nve')
    with np.load(seed_file) as seed:
        q, u = [seed[name] for name in ('positions', 'tangent_velocity')]
        if float(seed['box']) != model['box_angstrom']:
            raise ValueError('seed box mismatch')
    plan.update(seed_sha256=water.digest(seed_file), seed_source_plan_sha256=report['plan_sha256'],
                handoff='new NVE clock; freshly evaluated force at requested grid; half = float32(tangent_velocity - h*force/(2*mass)); instantaneous observer projects at q0')
    water.save(output/'plan.json', plan)
    started = time.perf_counter()
    try:
        force, probe_wall = force_probe(output/'handoff-force', source, model, sponge, q, grid)
        half, initial, h = half_step(u, force, state, q, model, dt)
        write_input(output, source, model, steps, dt, stride, 'nve', grid=grid, coordinate=q, velocity=half)
        water.save(output/'handoff.json', dict(dt_internal=h, initial_temperature_k=float(initial['temperature']),
                   maximum_handoff_velocity_rounding=float(abs(initial['velocity']-u).max()),
                   force_sha256=water.digest(output/'handoff-force/force.dat'),
                   mdin_sha256=water.digest(output/'mdin.spg.toml'), force_probe_wall_seconds=probe_wall))
        wall = water.execute([sponge, '-mdin', 'mdin.spg.toml'], output, timeout=1800)
        water.save(output/'execution.json', dict(status='engine_completed', wall_seconds=wall))
        expected = np.arange(0, steps+1, stride)
        obs = read_observables(output/'observables.h5', expected)
        positions = np.concatenate((q[None], native_frames(output/'crd.dat', model['atoms'], steps//stride)))
        length = geometry(positions, state, model)
        import sys
        sys.path.insert(0, str(water.ROOT/'scripts'))
        import analyze_green_kubo as gk
        times, stress, temperature, volume, provenance = gk.load_stress(output/'viscosity.h5')
        if not np.allclose(times, np.arange(steps+1)*dt, rtol=1e-12, atol=1e-12):
            raise ValueError('NVE force-time sample phase mismatch')
        if abs(temperature[0]-initial['temperature']) > 1e-3:
            raise ValueError('initial centered temperature handoff mismatch')
        # Temperature denominator is the exact native float32 DOF*kB product.
        dof = 3*model['atoms']-len(state['pairs'])-3
        kinetic = .5*float(np.float32(dof)*np.float32(water.KB))*temperature[::stride]
        offset = np.array([energy_shift(frame, state, model) for frame in positions])
        total = obs['eff_pot']-offset+kinetic
        delta = total-total[0]; thermal = .5*dof*water.KB*model['requested_initial_temperature_k']
        np.savetxt(output/'energy.csv', np.column_stack((expected, expected*dt, obs['eff_pot'], offset, kinetic, total)),
                   delimiter=',', header='step,time_ps,potential_kcal_mol,cutoff_shift_kcal_mol,kinetic_com_free_kcal_mol,total_shifted_kcal_mol', comments='')
        result = dict(status='completed_nve_pilot_not_transport_acceptance', duration_ps=steps*dt,
                      plan_sha256=water.digest(output/'plan.json'), engine_wall_seconds=wall,
                      force_probe_wall_seconds=probe_wall,
                      total_wall_seconds=time.perf_counter()-started, maximum_length_residual=length,
                      mean_temperature_k=float(np.mean(temperature)), initial_temperature_k=float(temperature[0]),
                      final_temperature_k=float(temperature[-1]),
                      energy_endpoint_delta_kcal_mol=float(delta[-1]), energy_range_kcal_mol=float(np.ptp(total)),
                      energy_slope_kcal_mol_ps=float(np.polyfit(expected*dt, delta, 1)[0]),
                      energy_max_excursion_over_target_thermal=float(np.max(abs(delta))/thermal),
                      scope='centered constrained NVE energy with continuous cutoff-shift bookkeeping; no convergence acceptance',
                      stress_contract=provenance, formal_viscosity_acceptance=False)
        result['raw_sha256'] = {name: water.digest(output/name) for name in
                               ('coordinate.txt', 'velocity.txt', 'crd.dat', 'vel.dat', 'force.dat',
                                'observables.h5', 'viscosity.h5')}
        water.save(output/'summary.json', result)
        return result
    except Exception as error:
        water.save(output/'failure.json', dict(error=str(error), wall_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['nvt', 'nve'])
    for key in ('model', 'output', 'sponge'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seed-file')
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--dt', type=float, default=.0005)
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--thermostat-seed', type=int, default=18273)
    parser.add_argument('--velocity-seed', type=int, default=71027)
    args = parser.parse_args()
    common = dict(source=args.model, output=args.output, sponge=args.sponge, steps=args.steps,
                  dt=args.dt, stride=args.stride, grid=args.grid)
    result = (equilibrate(**common, thermostat_seed=args.thermostat_seed, velocity_seed=args.velocity_seed)
              if args.mode == 'nvt' else nve(**common, seed_file=args.seed_file))
    print(json.dumps(result, indent=2))
