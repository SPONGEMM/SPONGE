#!/usr/bin/env python3
"""Structural handoff and fixed-bond hexane NVE/GK numerical pilot.

The preparation's physical velocity is reprojected at the quantized snapshot.
A freshly evaluated force defines each new incoming half step. No thermostat
state is continued, and no short-pilot viscosity acceptance is inferred.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import hexane_model as model
import hexane_pilot as prep
import water_model as water
import water_pilot as io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import analyze_green_kubo as gk


def reference(q, half, force, mass, box, h):
    """Independent local five-constraint solves, with continuous reaction stress."""
    q, half, force, mass = [np.asarray(v, float) for v in (q, half, force, mass)]
    n = len(mass)
    if n % 6 or n < 6 or any(v.shape != (n, 3) for v in (q, half, force)) or np.any(mass <= 0):
        raise ValueError('six-site chain arrays required')
    if any(not np.all(np.isfinite(v)) for v in (q, half, force, mass)) or not np.isfinite(box) or box <= 0:
        raise ValueError('finite chain arrays and positive box required')
    local = q.reshape(-1, 6, 3)
    d = local[:, :-1]-local[:, 1:]; d -= box*np.floor(d/box+.5)
    r = np.linalg.norm(d, axis=2)
    if np.any(r < 1e-10): raise ValueError('coincident constrained sites')
    jac = np.zeros((n//6, 5, 18))
    for k in range(5):
        jac[:, k, 3*k:3*k+3] = d[:, k]/r[:, k, None]
        jac[:, k, 3*k+3:3*k+6] = -d[:, k]/r[:, k, None]
    inverse = 1/np.repeat(mass.reshape(-1, 6), 3, axis=1)
    metric = (jac*inverse[:, None, :])@jac.swapaxes(1, 2)
    u = (half+.5*h*force/mass[:, None]).reshape(-1, 18, 1)
    u -= inverse[:, :, None]*(jac.swapaxes(1, 2)@np.linalg.solve(metric, jac@u))
    velocity = u.reshape(n, 3)
    dv = np.diff(-velocity.reshape(-1, 6, 3), axis=1)
    curvature = (np.sum(dv*dv, axis=2)-np.sum(d*dv, axis=2)**2/r**2)/r
    rhs = -(jac@(force.reshape(-1, 18)*inverse)[:, :, None]+curvature[:, :, None])
    multipliers = np.linalg.solve(metric, rhs)[:, :, 0]
    constraint = np.einsum('mi,mia,mib->ab', multipliers/r, d, d)
    c = velocity-np.average(velocity, weights=mass, axis=0)
    kinetic = (c.T*mass)@c
    dof = 3*n-5*(n//6)-3
    return dict(velocity=velocity, constraint=constraint, kinetic=kinetic,
                full_kinetic=.5*np.sum(mass[:, None]*velocity**2),
                temperature=np.trace(kinetic)/float(np.float32(dof)*np.float32(water.KB)),
                length_residual=float(np.max(abs(r/model.LENGTH-1))),
                tangent_residual=float(np.max(abs(np.sum(d*dv, axis=2)/r))))


def snapshot(source, preparation, output):
    source, preparation, output = [Path(p).resolve() for p in (source, preparation, output)]
    m, state = model.load(source)
    report = json.loads((preparation/'summary.json').read_text())
    plan = json.loads((preparation/'plan.json').read_text())
    if (report['status'] != 'completed_preparation_pilot_not_equilibrium_acceptance' or
            (preparation/'failure.json').exists() or
            report['plan_sha256'] != water.digest(preparation/'plan.json') or
            plan['model_sha256'] != water.digest(source/'model.json')):
        raise ValueError('preparation identity/status')
    for name, sha in report['raw_sha256'].items():
        if Path(name).name != name or water.digest(preparation/name) != sha:
            raise ValueError('preparation raw hash: '+name)
    with np.load(preparation/'final-state.npz') as saved:
        qd, u = saved['positions'], saved['tangent_velocity']
        if qd.shape != state['positions'].shape or u.shape != qd.shape:
            raise ValueError('preparation shape')
        box = m['box_angstrom']
        q = np.asarray(qd % box, np.float32).astype(float)
        q[q >= box] = 0
    # A new experiment: project physical velocity at the actual stored geometry.
    u = reference(q, u, np.zeros_like(u), state['masses'], box, 0)['velocity']
    u -= np.average(u, weights=state['masses'], axis=0)
    output.mkdir(parents=True, exist_ok=False)
    for name in model.MODEL_FILES: shutil.copy2(source/name, output/name)
    water.array_file(output/'coordinate.txt', q, box)
    water.array_file(output/'velocity.txt', np.zeros_like(q))  # static gate has no kinetic term
    np.savez(output/'state.npz', positions=q, physical_velocity=u,
             masses=state['masses'], pairs=state['pairs'], types=state['types'])
    text = (source/'lammps.data').read_text()
    prefix, rest = text.split('Atoms # full\n\n'); atoms, suffix = rest.split('\nBonds\n')
    rows = [row.split() for row in atoms.splitlines() if row.strip()]
    if len(rows) != len(q): raise ValueError('LAMMPS atom count')
    new = []
    for i, (fields, xyz) in enumerate(zip(rows, q)):
        if len(fields) != 7 or int(fields[0]) != i+1: raise ValueError('LAMMPS atom order')
        new.append(' '.join(fields[:4])+' '+' '.join(f'{v:.17g}' for v in xyz))
    (output/'lammps.data').write_text(prefix+'Atoms # full\n\n'+'\n'.join(new)+'\n\nBonds\n'+suffix)
    m['initialization'] = 'derived preparation snapshot; physical tangent velocity separate from zero static velocity; no equilibrium acceptance'
    m['snapshot'] = dict(parent_model_sha256=water.digest(source/'model.json'), preparation_plan_sha256=water.digest(preparation/'plan.json'), preparation_summary_sha256=water.digest(preparation/'summary.json'), final_state_sha256=water.digest(preparation/'final-state.npz'), preparation_duration_ps=plan['duration_ps'])
    # The ordered-lattice spacing/minimum-distance fields do not describe this snapshot.
    m.pop('minimum_included_pair_distance_angstrom', None); m.pop('cells', None)
    m['input_sha256'] = {name:water.digest(output/name) for name in sorted(model.MODEL_FILES)}
    water.save(output/'model.json', m)
    return m


def cutoff_offset(q, m, state):
    i, j = np.triu_indices(len(q), 1)
    keep = (i//6 != j//6)|(j-i > 3); i, j = i[keep], j[keep]
    d = q[i]-q[j]; d -= m['box_angstrom']*np.floor(d/m['box_angstrom']+.5)
    active = np.sum(d*d, axis=1) < m['cutoff_angstrom']**2
    coef = np.zeros((2, 2))
    for p in m['coefficients']:
        a, b = p['types']; c12, c6 = p['effective']; rc = m['cutoff_angstrom']
        coef[a, b] = coef[b, a] = c12/rc**12-c6/rc**6
    return float(coef[state['types'][i[active]], state['types'][j[active]]].sum())


def run(source, static, output, sponge, duration=2., dt=.0005):
    source, static, output, sponge = [Path(p).resolve() for p in (source, static, output, sponge)]
    m, st = model.load(source)
    if 'physical_velocity' not in st: raise ValueError('derived physical-velocity snapshot required')
    if not np.isfinite(dt) or not 0 < dt <= .0005 or not np.isfinite(duration) or not .1 <= duration <= 10:
        raise ValueError('bounded dt/duration required')
    steps, stride = round(duration/dt), round(.01/dt)
    if steps % stride or abs(steps*dt-duration)>1e-10 or abs(stride*dt-.01)>1e-10:
        raise ValueError('cadence alignment')
    gate = json.loads((static/'summary.json').read_text()); sp = json.loads((static/'plan.json').read_text())
    if (not gate['passed'] or sp['model_sha256'] != water.digest(source/'model.json') or
            sp['sponge_sha256'] != water.digest(sponge) or sp['coordinate_sha256'] != water.digest(source/'coordinate.txt') or
            gate['force_sha256'] != water.digest(static/'sponge/force.dat')):
        raise ValueError('static reference identity')
    f0 = io.native_frames(static/'sponge/force.dat', m['atoms'], 2)[0]
    q, mass, u = st['positions'], st['masses'], st['physical_velocity']
    h = water.internal_timestep(dt)
    half = np.asarray(u-.5*h*f0/mass[:, None], np.float32).astype(float)
    output.mkdir(parents=True, exist_ok=False)
    water.array_file(output/'velocity.txt', half)
    extra = {'constrain_mode':'SHAKE', 'settle_disable':True, 'constrain.in_file':str(source/'constraint.txt'), 'velocity_in_file':str(output/'velocity.txt')}
    model.write_input(output, source, m, steps, dt, stride, extra, 'method="green_kubo"\nsample_interval=1')
    plan = dict(contract='sponge.hexane.nve-pilot.v1', dt_ps=dt, h=h, steps=steps, stride=stride, duration_ps=duration, model_sha256=water.digest(source/'model.json'), static_plan_sha256=water.digest(static/'plan.json'), static_summary_sha256=water.digest(static/'summary.json'), sponge_sha256=water.digest(sponge), source_sha256=water.digest(__file__), mdin_sha256=water.digest(output/'mdin.spg.toml'), gates=dict(kinetic_rtol=1e-9, kinetic_atol=1e-8, length=1e-4, initial_pressure_pa='100 + 5e-4 maxabs(reference)', initial_temperature_k=.01), handoff='new NVE experiment, projected physical velocity; fresh q0 force and dt-specific incoming half', formal_viscosity_acceptance=False)
    water.save(output/'plan.json', plan); started = time.perf_counter()
    try:
        wall = water.execute([sponge, '-mdin', 'mdin.spg.toml'], output, timeout=1800)
        water.save(output/'execution.json', dict(engine_wall_seconds=wall))
        times, _, temperature, volume, meta = gk.load_stress(output/'viscosity.h5')
        np.testing.assert_allclose(times, np.arange(steps+1)*dt, rtol=1e-12, atol=1e-12)
        dof = 3*m['atoms']-len(st['pairs'])-3
        if (meta.get('observable_contract') != 'sponge.observables.physical_state.v1' and meta.get('viscosity_contract') != gk.CONSTRAINED_CONTRACT) or int(meta['thermal_dof']) != dof or int(meta['constraint_count']) != len(st['pairs']) or 'shake_pairs='+str(len(st['pairs'])) not in meta['constraint_solver']:
            raise ValueError('GK constraint contract')
        np.testing.assert_allclose(volume, m['box_angstrom']**3*.001, rtol=1e-7)
        expected = np.arange(0, steps+1, stride); count = len(expected)
        obs = io.read_observables(output/'observables.h5', expected)
        positions = np.concatenate((q[None], io.native_frames(output/'crd.dat', len(q), count-1)))
        halves = np.concatenate((half[None], io.native_frames(output/'vel.dat', len(q), count-1)))
        forces = io.native_frames(output/'force.dat', len(q), count)
        np.testing.assert_array_equal(forces[0], f0)
        rows = [reference(x, s, f, mass, m['box_angstrom'], h) for x,s,f in zip(positions, halves, forces)]
        kinetic = np.array([np.trace(r['kinetic'])/2 for r in rows])
        measured = .5*float(np.float32(dof)*np.float32(water.KB))*temperature[::stride]
        np.testing.assert_allclose(kinetic, measured, rtol=1e-9, atol=1e-8)
        length = max(r['length_residual'] for r in rows)
        if length > 1e-4: raise ValueError('native constraint length')
        # LAMMPS unrestrained mechanical pressure already includes the matched tail.
        lf = np.loadtxt(static/'lammps/forces.dump', skiprows=9)[:, 1:]
        ref = reference(q, half, lf, mass, m['box_angstrom'], h)
        pressure = np.array(gate['reference_pressure_pa'])+np.array([(ref['constraint']+ref['kinetic'])[a,b] for a,b in water.AXES])/m['box_angstrom']**3*water.PA
        with h5py.File(output/'viscosity.h5') as file:
            p0 = np.array([file['observables/all/'+name+'/value'][0] for name in water.NAMES])
        pressure_error = float(abs(p0-pressure).max())
        if pressure_error > 100+5e-4*abs(pressure).max() or abs(temperature[0]-ref['temperature']) > .01:
            raise ValueError('independent initial constrained pressure/temperature')
        full = np.array([r['full_kinetic'] for r in rows])
        offset = np.array([cutoff_offset(x, m, st) for x in positions])
        total = obs['eff_pot']-offset+full
        momentum = np.array([np.sum(mass[:,None]*r['velocity'],axis=0) for r in rows])
        np.savetxt(output/'audit.csv', np.column_stack((expected*dt, obs['eff_pot'], offset, full, kinetic, measured, total, momentum)), delimiter=',', comments='', header='time_ps,U,cutoff_offset,Kfull,Kcom_free,Kengine,total_shifted,px,py,pz')
        np.savez(output/'audited-state.npz', positions=positions, physical_velocity=np.array([r['velocity'] for r in rows]))
        result = dict(status='passed_nve_numerical_pilot_not_transport_acceptance', plan_sha256=water.digest(output/'plan.json'), duration_ps=duration, engine_wall_seconds=wall, total_wall_seconds=time.perf_counter()-started, thermal_dof=dof, mean_temperature_k=float(temperature.mean()), kinetic_max_error=float(abs(kinetic-measured).max()), length_residual=length, initial_pressure_error_pa=pressure_error, initial_reference_pressure_pa=pressure.tolist(), initial_sponge_pressure_pa=p0.tolist(), initial_temperature_error_k=float(temperature[0]-ref['temperature']), energy_max_excursion_kcal_mol=float(abs(total-total[0]).max()), energy_endpoint_delta_kcal_mol=float(total[-1]-total[0]), energy_slope_kcal_mol_ps=float(np.polyfit(expected*dt,total-total[0],1)[0]), momentum_max_change_native=float(np.linalg.norm(momentum-momentum[0],axis=1).max()), stress_contract=meta, limitations=['No energy convergence gate; compare declared dt cases.', 'Independent pressure gate covers q0 only.', 'No equilibrium or viscosity acceptance.'], raw_sha256={p.name:water.digest(p) for p in output.iterdir() if p.suffix in ('.h5','.dat','.npz')}, formal_viscosity_acceptance=False)
        water.save(output/'summary.json', result); return result
    except Exception as error:
        water.save(output/'failure.json', dict(error=str(error), plan_sha256=water.digest(output/'plan.json'), wall_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('snapshot')
    for key in ('model','preparation','output'):p.add_argument('--'+key,required=True)
    p=sub.add_parser('run')
    for key in ('model','static','output','sponge'):p.add_argument('--'+key,required=True)
    p.add_argument('--duration',type=float,default=2.);p.add_argument('--dt',type=float,default=.0005)
    a=parser.parse_args()
    result=snapshot(a.model,a.preparation,a.output) if a.command=='snapshot' else run(a.model,a.static,a.output,a.sponge,a.duration,a.dt)
    print(json.dumps(result,indent=2))
