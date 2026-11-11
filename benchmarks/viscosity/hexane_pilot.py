#!/usr/bin/env python3
"""Zero-drive fixed-bond hexane preparation; no automatic liquid acceptance.

Independent per-chain tangent projection audits force-time kinetic energy.
Structural diagnostics describe relaxation from the declared ordered lattice.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np

import hexane_model as model
import water_model as water
import water_pilot as io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import analyze_cosine as cosine


def projection(q, mass, box):
    """Five independent bond rows per six-site chain, in mass coordinates."""
    q = np.asarray(q, float).reshape(-1, 6, 3)
    mass = np.asarray(mass, float).reshape(-1, 6)
    if len(q) != len(mass) or np.any(mass <= 0):
        raise ValueError('six-site chain masses')
    d = q[:, :-1]-q[:, 1:]
    d -= box*np.floor(d/box+.5)
    lengths = np.linalg.norm(d, axis=2, keepdims=True)
    if not np.all(np.isfinite(d)) or np.any(lengths < 1e-10):
        raise ValueError('degenerate constraint geometry')
    d /= lengths
    a = np.zeros((len(q), 5, 18))
    for i in range(5):
        a[:, i, 3*i:3*i+3] = d[:, i]/np.sqrt(mass[:, i, None])
        a[:, i, 3*i+3:3*i+6] = -d[:, i]/np.sqrt(mass[:, i+1, None])
    gram = a@a.swapaxes(1, 2)

    def project(w):
        w = np.asarray(w).reshape(-1, 18, 1)
        return (w-a.swapaxes(1, 2)@np.linalg.solve(gram, a@w)).reshape(-1, 3)
    return project


def frame(q, half, force, mass, box, h):
    root = np.sqrt(mass[:, None])
    project = projection(q, mass, box)
    w = project((half+.5*h*force/mass[:, None])*root)
    v = w/root
    c = v-np.average(v, weights=mass, axis=0)
    flow = np.zeros_like(q)
    flow[:, 0] = np.cos(2*np.pi*q[:, 2]/box)
    b = project(flow*root)/root
    b -= np.average(b, weights=mass, axis=0)
    norm = np.sum(mass[:, None]*b*b)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError('cosine mode rank')
    thermal = c-b*np.sum(mass[:, None]*c*b)/norm
    return v, .5*np.sum(mass[:, None]*thermal**2)


def structure(positions, mass, box, spacing, output):
    xyz = positions.reshape(len(positions), -1, 6, 3)
    delta = np.diff(xyz, axis=2)
    delta -= box*np.floor(delta/box+.5)
    local = np.concatenate((np.zeros_like(xyz[:, :, :1]), np.cumsum(delta, axis=2)), axis=2)
    local_mass = mass.reshape(-1, 6)
    centers = (xyz[:, :, 0]+np.einsum('fmij,mi->fmj', local, local_mass)/local_mass.sum(axis=1)[None, :, None]) % box
    axis = local[:, :, -1]
    length = np.linalg.norm(axis, axis=2)
    if not np.all(np.isfinite(length)) or np.any(length < 1e-10):
        raise ValueError('degenerate chain end-to-end axis')
    axis = axis/length[:, :, None]
    tensor = (3*np.einsum('fmi,fmj->fij', axis, axis)/axis.shape[1]-np.eye(3))/2
    order = np.linalg.eigvalsh(tensor)[:, -1]
    # Conventional four-point torsion: all-trans gives |phi| = pi.
    normals = np.cross(delta[:, :, :-1], delta[:, :, 1:])
    norm = np.linalg.norm(normals, axis=3, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm < 1e-10):
        raise ValueError('degenerate dihedral plane')
    normals /= norm
    cosine_phi = np.sum(normals[:, :, :-1]*normals[:, :, 1:], axis=3)
    phi = np.arccos(np.clip(cosine_phi, -1, 1))
    gauche = np.mean(phi < 2*np.pi/3, axis=(1, 2))
    jumps = np.diff(centers, axis=0)
    jumps -= box*np.floor(jumps/box+.5)
    maximum = float(np.max(np.linalg.norm(jumps, axis=2)))
    if maximum >= box/4:
        raise ValueError('resolved COM displacement too large for sampled unwrap')
    continuous = np.concatenate((centers[:1], centers[:1]+np.cumsum(jumps, axis=0)))
    start = len(positions)//2
    displacement = continuous[start:]-continuous[start]
    displacement -= displacement.mean(axis=1, keepdims=True)
    msd = np.mean(np.sum(displacement**2, axis=2), axis=1)
    np.savetxt(output/'structure.csv', np.column_stack((np.arange(len(positions))*spacing, order, gauche, length.mean(axis=1))), delimiter=',', header='time_ps,nematic_S,gauche_fraction_phi_abs_lt_120,mean_end_to_end_angstrom', comments='')
    np.savetxt(output/'msd.csv', np.column_stack((np.arange(len(msd))*spacing, msd)), delimiter=',', header='elapsed_ps,latter_half_single_origin_com_msd_angstrom2', comments='')
    n = centers.shape[1]; i, j = np.triu_indices(n, 1)
    edges = np.linspace(0, box/2, 101); counts = np.zeros(100)
    density = np.zeros(20)
    for c in centers[start:]:
        d = c[i]-c[j]; d -= box*np.floor(d/box+.5)
        counts += np.histogram(np.linalg.norm(d, axis=1), edges)[0]
        density += np.histogram(c[:, 2], np.linspace(0, box, 21))[0]
    rdf = counts/((len(centers)-start)*n*(n-1)/2*4*np.pi/3*np.diff(edges**3)/box**3)
    np.savetxt(output/'rdf-com.csv', np.column_stack(((edges[1:]+edges[:-1])/2, rdf)), delimiter=',', header='r_angstrom,g_com_finite_N', comments='')
    np.savetxt(output/'density-z.csv', np.column_stack(((np.arange(20)+.5)*box/20, density/((len(centers)-start)*n/20))), delimiter=',', header='z_angstrom,relative_com_density', comments='')
    return dict(initial_nematic_S=float(order[0]), final_nematic_S=float(order[-1]), latter_mean_nematic_S=float(order[start:].mean()), initial_gauche_fraction=float(gauche[0]), latter_gauche_fraction=float(gauche[start:].mean()), max_resolved_com_jump_angstrom=maximum, latter_single_origin_msd_angstrom2=float(msd[-1]), interpretation='diagnostics only; 120-degree torsion bin; finite-N nematic bias; no diffusion fit or equilibrium assertion; sampled unwrap cannot exclude multiple crossings'), order, gauche


def identity(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run(source, output, sponge, static_force, duration=100., dt=.0005, velocity_seed=75101, thermostat_seed=75201):
    source, output, sponge, static_force = [Path(p).resolve() for p in (source, output, sponge, static_force)]
    m, state = model.load(source)
    if not np.isfinite(duration) or duration < 1. or not np.isfinite(dt) or not 0 < dt <= .0005:
        raise ValueError('positive bounded duration/dt')
    steps, stride, native = round(duration/dt), round(.01/dt), round(.1/dt)
    if steps % native or native % stride or abs(steps*dt-duration)>1e-10 or abs(native*dt-.1)>1e-10 or abs(stride*dt-.01)>1e-10:
        raise ValueError('duration and cadences must align to dt')
    if steps > 2000000:
        raise ValueError('bounded preparation maximum 2 million steps')
    q, mass, box = state['positions'], state['masses'], m['box_angstrom']
    forces = io.native_frames(static_force, m['atoms'], 2)
    # The static comparison uses q0. The extra dt=0 step can republish native
    # coordinates through fixed-point conversion, so use only its first force.
    force = forces[0]
    # Require that the force was computed at exactly this model and that its gate passed.
    static_root = static_force.parent.parent
    static_plan = json.loads((static_root/'plan.json').read_text())
    static_summary = json.loads((static_root/'summary.json').read_text())
    if (static_plan['model_sha256'] != water.digest(source/'model.json') or
            static_plan['sponge_sha256'] != water.digest(sponge) or
            static_plan['coordinate_sha256'] != water.digest(source/'coordinate.txt') or
            static_summary['force_sha256'] != water.digest(static_force) or
            not static_summary['passed']):
        raise ValueError('static gate identity')
    output.mkdir(parents=True, exist_ok=False)
    raw = np.random.default_rng(velocity_seed).normal(size=q.shape)*np.sqrt(water.KB*m['temperature_k']/mass[:, None])
    u = frame(q, raw, np.zeros_like(raw), mass, box, 0)[0]
    u -= np.average(u, weights=mass, axis=0)
    h = water.internal_timestep(dt)
    half = np.asarray(u-.5*h*force/mass[:, None], np.float32).astype(float)
    water.array_file(output/'coordinate.txt', q, box); water.array_file(output/'velocity.txt', half)
    extra = dict(mode='nvt', thermostat='middle_langevin', thermostat_tau=.1, thermostat_seed=thermostat_seed, target_temperature=m['temperature_k'], constrain_mode='SHAKE', settle_disable=True, write_restart_file_interval=steps, output_h5_restart_path='checkpoint.spgr.h5', coordinate_in_file=str(output/'coordinate.txt'), velocity_in_file=str(output/'velocity.txt'))
    extra['constrain.in_file'] = str(source/'constraint.txt')
    extra['output_h5_restart_topology_hash'] = identity(m['input_sha256'])
    extra['output_h5_restart_atom_order_hash'] = identity(mass.tolist())
    extra['output_h5_restart_protocol_hash'] = identity(dict(dt=dt, A=0, tau=.1))
    model.write_input(output, source, m, steps, dt, native, extra, f'method="cosine"\nacceleration=0\nsample_interval={stride}\nbins=20')
    plan = dict(contract='sponge.hexane.preparation-pilot.v1', steps=steps, dt=dt, h=h, duration_ps=duration, profile_stride=stride, native_stride=native, thermal_dof=m['thermal_dof'], velocity_seed=velocity_seed, thermostat_seed=thermostat_seed, model_sha256=water.digest(source/'model.json'), sponge_sha256=water.digest(sponge), source_sha256=water.digest(__file__), dependencies={Path(p).name:water.digest(p) for p in (model.__file__, water.__file__, io.__file__, cosine.__file__)}, force_sha256=water.digest(static_force), mdin_sha256=water.digest(output/'mdin.spg.toml'), numerical_gates=dict(kinetic_rtol=2e-6, kinetic_atol=1e-5, length=2e-12, tangent_nm_ps=1e-9, final_native_position_angstrom=3e-6), formal_viscosity_acceptance=False)
    plan.update(static_plan_sha256=water.digest(static_root/'plan.json'), static_summary_sha256=water.digest(static_root/'summary.json'))
    water.save(output/'plan.json', plan)
    started = time.perf_counter()
    try:
        wall = water.execute([sponge, '-mdin', 'mdin.spg.toml'], output, timeout=14400)
        water.save(output/'execution.json', dict(engine_wall_seconds=wall))
        return audit(output, m, state, plan, wall, started)
    except Exception as error:
        water.save(output/'failure.json', dict(error=str(error), phase='execution_or_analysis', plan_sha256=water.digest(output/'plan.json'), wall_seconds=time.perf_counter()-started))
        raise


def audit(output, m, state, plan, wall, started):
    output = Path(output); mass = state['masses']; n = m['atoms']; box = m['box_angstrom']
    data, _ = cosine.load_profile(output/'viscosity.h5')
    expected = np.arange(0, plan['steps']+1, plan['profile_stride'])
    np.testing.assert_array_equal(data['step'], expected)
    np.testing.assert_allclose(data['time'], expected*plan['dt'], rtol=1e-12, atol=1e-12)
    ratio = plan['native_stride']//plan['profile_stride']; sampled = {k:v[::ratio] for k,v in data.items()}
    native_steps = expected[::ratio]; count = len(native_steps)
    obs = io.read_observables(output/'observables.h5', native_steps)
    q0 = np.loadtxt(output/'coordinate.txt', skiprows=1, max_rows=n)
    s0 = np.loadtxt(output/'velocity.txt', skiprows=1, max_rows=n)
    q = np.concatenate((q0[None], io.native_frames(output/'crd.dat', n, count-1)))
    s = np.concatenate((s0[None], io.native_frames(output/'vel.dat', n, count-1)))
    f = io.native_frames(output/'force.dat', n, count)
    com = np.column_stack([sampled['com_v'+a] for a in 'xyz'])*10/water.SP_TIME
    full = .5*float(np.float32(m['thermal_dof']+1)*np.float32(water.KB))*sampled['kinetic_temperature']+.5*mass.sum()*np.sum(com**2, axis=1)
    thermal = .5*float(np.float32(m['thermal_dof'])*np.float32(water.KB))*sampled['thermal_temperature']
    kf, kt = [], []
    for qi, si, fi in zip(q, s, f):
        v, k = frame(qi, si, fi, mass, box, plan['h'])
        kf.append(.5*np.sum(mass[:, None]*v*v)); kt.append(k)
    np.testing.assert_allclose(kf, full, rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(kt, thermal, rtol=2e-6, atol=1e-5)
    length = float(data['constraint_length_residual'].max()); tangent = float(data['constraint_tangent_residual'].max())
    if length > 2e-12 or tangent > 1e-9 or np.any(data['coupling_work'] != 0):
        raise ValueError('constraint/zero-work gate')
    with h5py.File(output/'checkpoint.spgr.h5') as file:
        assert file['parameters/sponge/output/status'][()] == b'finalized'
        assert int(file['parameters/restart/coupling/steps'][()]) == plan['steps']
        assert int(file['parameters/restart/integrator_state/step'][()]) == plan['steps']-1
        assert abs(float(file['parameters/restart/coupling/time'][()])-plan['duration_ps']) < 1e-10
        group = file['parameters/restart/integrator_state/double']
        qd = group['constrained_flow_position'][()].reshape(n, 3)
        sd = group['constrained_flow_velocity'][()].reshape(n, 3)
        assert qd.dtype == sd.dtype == np.dtype('float64')
        config = cosine.text(file['parameters/restart/integrator_state/constrained_flow_configuration'][()])
        assert 'native_periodic_double_v1' in config
        assert float(re.search(r';dt=([^;]+)', config).group(1)) == plan['h']
    difference = q[-1]-qd; difference -= box*np.floor(difference/box+.5)
    np.testing.assert_allclose(difference, 0, rtol=0, atol=3e-6)
    vd, kd = frame(qd, sd, f[-1], mass, box, plan['h'])
    np.testing.assert_allclose(.5*np.sum(mass[:, None]*vd**2), full[-1], rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(kd, thermal[-1], rtol=2e-6, atol=1e-5)
    np.savez(output/'final-state.npz', positions=qd, incoming_half_velocity=sd, tangent_velocity=vd, force=f[-1])
    diagnostics, order, gauche = structure(q, mass, box, plan['native_stride']*plan['dt'], output)
    heat = sampled['thermostat_heat']-sampled['thermostat_heat'][0]
    residual = obs['eff_pot']+full-obs['eff_pot'][0]-full[0]-heat
    np.savetxt(output/'energy.csv', np.column_stack((native_steps*plan['dt'], obs['eff_pot'], full, thermal, heat, residual)), delimiter=',', header='time_ps,U,K,Kthermal,Q,unshifted_deltaE_minus_Q', comments='')
    blocks = []
    for ids in np.array_split(np.arange(count//2, count), 5):
        blocks.append(dict(start_ps=float(native_steps[ids[0]]*plan['dt']), end_ps=float(native_steps[ids[-1]]*plan['dt']), thermal_temperature_k=float(sampled['thermal_temperature'][ids].mean()), potential_kcal_mol=float(obs['eff_pot'][ids].mean()), pressure_diagonal_pa=[float(sampled[k][ids].mean()) for k in ('pxx', 'pyy', 'pzz')], nematic_S=float(order[ids].mean()), gauche_fraction=float(gauche[ids].mean())))
    result = dict(status='completed_preparation_pilot_not_equilibrium_acceptance', engine_wall_seconds=wall, total_wall_seconds=time.perf_counter()-started, duration_ps=plan['duration_ps'], length_residual=length, tangent_residual_nm_ps=tangent, full_kinetic_max_error=float(np.max(abs(np.array(kf)-full))), thermal_kinetic_max_error=float(np.max(abs(np.array(kt)-thermal))), final_native_position_error_angstrom=float(abs(difference).max()), unshifted_energy_max_residual=float(abs(residual).max()), structure=diagnostics, latter_blocks=blocks, limitations=['Unshifted energy includes cutoff-crossing discontinuities; no energy convergence gate.', 'No independent dynamic pressure audit in this bulk pilot.', 'One ordered-start trajectory does not establish equilibrium, viscosity, or experimental agreement.'], plan_sha256=water.digest(output/'plan.json'), raw_sha256={p.name:water.digest(p) for p in output.iterdir() if p.suffix in ('.dat', '.h5', '.npz')}, formal_viscosity_acceptance=False)
    water.save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'output', 'sponge', 'static-force'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--duration', type=float, default=100.)
    parser.add_argument('--dt', type=float, default=.0005)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.output, args.sponge, args.static_force, args.duration, args.dt), indent=2))
