#!/usr/bin/env python3
"""Independent sampled kinetic/momentum audit of a completed SPC/E NVE pilot.

Uses exported q, incoming half-step velocity and unconstrained force. Does not
modify MD or infer transport convergence from an energy or momentum diagnostic.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import water_model as water
import water_pilot as pilot


def frame_audit(position, half, force, state, model, h):
    mass = state['masses']
    centered = half + .5*h*force/mass[:, None]
    reference = water.constrained_reference(position, half, force, mass, state['pairs'],
                                             model['box_angstrom'], h, np.zeros((3, 3)))
    tangent = reference['velocity']
    momentum = np.sum(mass[:, None]*tangent, axis=0)
    before = np.sum(mass[:, None]*centered, axis=0)
    com = momentum/np.sum(mass)
    full = .5*np.sum(mass[:, None]*tangent**2)
    thermal = .5*np.sum(mass[:, None]*(tangent-com)**2)
    com_energy = .5*(momentum@momentum)/np.sum(mass)
    if abs(full-thermal-com_energy) > 1e-11*max(1., abs(full)):
        raise ValueError('kinetic COM decomposition failed')
    return dict(momentum=momentum, net_force=np.sum(force, axis=0),
                projection_momentum_error=float(np.linalg.norm(momentum-before)),
                kinetic_full=float(full), kinetic_thermal=float(thermal), kinetic_com=float(com_energy),
                max_relative_length_error=reference['max_relative_length_error'])


def audit(model_path, run_path, output):
    model_path, run_path, output = [Path(p).resolve() for p in (model_path, run_path, output)]
    model, state = water.load_model(model_path)
    plan = json.loads((run_path/'plan.json').read_text())
    report = json.loads((run_path/'summary.json').read_text())
    if ((run_path/'failure.json').exists() or
            report.get('status') != 'completed_nve_pilot_not_transport_acceptance' or
            not (run_path/'execution.json').is_file()):
        raise ValueError('NVE pilot did not complete successfully')
    execution = json.loads((run_path/'execution.json').read_text())
    if execution.get('status') != 'engine_completed':
        raise ValueError('NVE engine did not complete successfully')
    if (plan['mode'] != 'nve' or plan['contract'] != 'sponge.spce.liquid-pilot.v1' or
            plan['model_sha256'] != water.digest(model_path/'model.json') or
            report['plan_sha256'] != water.digest(run_path/'plan.json')):
        raise ValueError('NVE model/plan identity mismatch')
    handoff = json.loads((run_path/'handoff.json').read_text())
    if handoff['mdin_sha256'] != water.digest(run_path/'mdin.spg.toml'):
        raise ValueError('NVE input identity mismatch')
    if handoff['force_sha256'] != water.digest(run_path/'handoff-force/force.dat'):
        raise ValueError('NVE handoff force identity mismatch')
    required = {'coordinate.txt', 'velocity.txt', 'crd.dat', 'vel.dat', 'force.dat',
                'observables.h5', 'viscosity.h5'}
    if set(report.get('raw_sha256', {})) != required:
        raise ValueError('NVE raw identity manifest missing or incomplete')
    for name in required:
        if water.digest(run_path/name) != report['raw_sha256'][name]:
            raise ValueError(f'NVE raw identity mismatch: {name}')
    h = water.internal_timestep(plan['dt_ps'])
    if h != handoff['dt_internal']:
        raise ValueError('NVE internal timestep mismatch')
    expected = np.arange(0, plan['steps']+1, plan['audit_interval'])
    n = model['atoms']; count = len(expected)
    q0 = np.loadtxt(run_path/'coordinate.txt', skiprows=1, max_rows=n)
    v0 = np.loadtxt(run_path/'velocity.txt', skiprows=1, max_rows=n)
    q = np.concatenate((q0[None], pilot.native_frames(run_path/'crd.dat', n, count-1)))
    v = np.concatenate((v0[None], pilot.native_frames(run_path/'vel.dat', n, count-1)))
    force = pilot.native_frames(run_path/'force.dat', n, count)
    # The static handoff probe writes its initial and one terminal force frame.
    handoff_force = pilot.native_frames(run_path/'handoff-force/force.dat', n, 2)
    if not np.array_equal(force[0], handoff_force[0]):
        raise ValueError('NVE initial force does not match handoff phase')
    obs = pilot.read_observables(run_path/'observables.h5', expected)
    sys.path.insert(0, str(water.ROOT/'scripts'))
    import analyze_green_kubo as gk
    times, _, temperature, _, metadata = gk.load_stress(run_path/'viscosity.h5')
    if (len(times) != plan['steps']+1 or
            not np.allclose(times, np.arange(plan['steps']+1)*plan['dt_ps'], rtol=1e-12, atol=1e-12)):
        raise ValueError('GK force-time cadence mismatch')
    dof = 3*n-len(state['pairs'])-3
    if ((metadata.get('observable_contract') != 'sponge.observables.physical_state.v1' and metadata.get('viscosity_contract') != gk.CONSTRAINED_CONTRACT) or
            int(metadata['atom_count']) != n or int(metadata['constraint_count']) != len(state['pairs']) or
            int(metadata['thermal_dof']) != dof or int(metadata['sample_interval']) != 1 or
            float(metadata['dt_ps']) != plan['dt_ps']):
        raise ValueError('GK model/timestep contract mismatch')
    measured = .5*float(np.float32(dof)*np.float32(water.KB))*temperature[::plan['audit_interval']]
    source_files = ['plan.json', 'summary.json', 'execution.json', 'handoff.json', 'handoff-force/force.dat',
                    'mdin.spg.toml', 'coordinate.txt',
                    'velocity.txt', 'crd.dat', 'vel.dat', 'force.dat', 'observables.h5', 'viscosity.h5']
    output.mkdir(parents=True, exist_ok=False)
    water.save(output/'plan.json', dict(contract='sponge.spce.nve-kinetic-audit.v1',
        model_sha256=plan['model_sha256'], run_path=str(run_path),
        input_sha256={name: water.digest(run_path/name) for name in source_files},
        source_sha256=water.digest(__file__), reference_sha256=water.digest(water.__file__),
        sample_phase='q_n, incoming half velocity, force(q_n); mass-metric tangent projection',
        momentum_unit='Da * native velocity (angstrom / internal time)',
        force_unit='kcal/(mol angstrom)', energy_unit='kcal/mol',
        formal_viscosity_acceptance=False))
    started = time.perf_counter()
    try:
        rows = [frame_audit(x, s, f, state, model, h) for x, s, f in zip(q, v, force)]
        thermal = np.array([r['kinetic_thermal'] for r in rows])
        full = np.array([r['kinetic_full'] for r in rows])
        com = np.array([r['kinetic_com'] for r in rows])
        momentum = np.array([r['momentum'] for r in rows])
        net_force = np.array([r['net_force'] for r in rows])
        error = thermal-measured
        agreement = bool(np.allclose(thermal, measured, rtol=1e-9, atol=1e-8))
        potential = obs['eff_pot']-np.array([pilot.energy_shift(x, state, model) for x in q])
        total = potential+full
        data = np.column_stack((expected, expected*plan['dt_ps'], momentum, net_force,
                                full, thermal, com, error, total))
        np.savetxt(output/'audit.csv', data, delimiter=',', comments='',
                   header='step,time_ps,px_native,py_native,pz_native,fx,fy,fz,kinetic_full,kinetic_com_free,kinetic_com,kinetic_reference_minus_engine,total_full_shifted')
        result = dict(status='passed_sampled_kinetic_agreement' if agreement else 'failed_sampled_kinetic_agreement',
            samples=count, maximum_kinetic_difference_kcal_mol=float(abs(error).max()),
            maximum_com_kinetic_kcal_mol=float(com.max()), com_kinetic_endpoint_change_kcal_mol=float(com[-1]-com[0]),
            maximum_momentum_change_native=float(np.linalg.norm(momentum-momentum[0], axis=1).max()),
            maximum_net_force_kcal_mol_angstrom=float(np.linalg.norm(net_force, axis=1).max()),
            maximum_projection_momentum_error_native=max(r['projection_momentum_error'] for r in rows),
            maximum_length_residual=max(r['max_relative_length_error'] for r in rows),
            total_full_endpoint_delta_kcal_mol=float(total[-1]-total[0]),
            total_full_max_excursion_kcal_mol=float(abs(total-total[0]).max()),
            wall_seconds=time.perf_counter()-started, stress_contract=metadata,
            scope='independent sampled kinetic/momentum only; potential is native plus analytic cutoff bookkeeping',
            formal_viscosity_acceptance=False)
        water.save(output/'summary.json', result)
        return result
    except Exception as error:
        water.save(output/'failure.json', dict(error=str(error), wall_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'run', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = audit(args.model, args.run, args.output)
    print(json.dumps(result, indent=2))
    if result['status'] != 'passed_sampled_kinetic_agreement':
        raise SystemExit(1)
