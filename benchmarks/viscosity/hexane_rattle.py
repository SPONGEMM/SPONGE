#!/usr/bin/env python3
"""Mechanical RATTLE liquid NVE audit at the actual projected initial state.

Preparation probe shares the engine's core and only exposes the initial state;
NumPy force-time constraints and LAMMPS static force/virial remain independent.
No automatic liquid transport or experimental acceptance.
"""
import argparse
import io as stream
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import h5py
import numpy as np
import hexane_model as model
import hexane_nve as nve
import hexane_pilot as prep
import water_model as water
import water_pilot as io


def project(probe, q, u, mass, pairs, box, output):
    """Test-only C++ preparation; not an independent dynamics reference."""
    text = f'{len(q)} {len(pairs)}\n'+f'{box:.17g} {box:.17g} {box:.17g}\n'
    text += ' '.join(f'{v:.17g}' for v in mass)+'\n'
    for values in (q, u):
        text += '\n'.join(' '.join(f'{v:.17g}' for v in row) for row in values)+'\n'
    text += ''.join(f'{int(a)} {int(b)} {r:.17g}\n' for a,b,r in pairs)
    output = Path(output)
    output.with_suffix('.input').write_text(text)
    result = subprocess.run([str(probe), '--prepare'], input=text, capture_output=True, text=True, timeout=60)
    output.with_suffix('.stdout').write_text(result.stdout)
    output.with_suffix('.stderr').write_text(result.stderr)
    if result.returncode: raise ValueError('preparation probe failed: '+result.stderr)
    value = np.loadtxt(stream.StringIO(result.stdout))
    if value.shape != (len(q),6) or not np.all(np.isfinite(value)):
        raise ValueError('invalid preparation probe output')
    return value[:,:3], value[:,3:]


def prepare(source, output, probe):
    source, output, probe = [Path(p).resolve() for p in (source, output, probe)]
    m, state = model.load(source)
    if 'physical_velocity' not in state: raise ValueError('prepared physical velocity required')
    output.mkdir(parents=True, exist_ok=False)
    qraw = state['positions']
    q, u = project(probe, qraw, state['physical_velocity'], state['masses'], state['pairs'],
                   m['box_angstrom'], output/'prepare')
    u -= np.average(u, weights=state['masses'], axis=0)
    ref = nve.reference(q, u, np.zeros_like(q), state['masses'], m['box_angstrom'], 0)
    if ref['length_residual'] > 2e-12 or ref['tangent_residual'] > 1e-10:
        raise ValueError('projected initial geometry/tangent')
    qnative = np.asarray(q % m['box_angstrom'], np.float32).astype(float)
    qnative[qnative >= m['box_angstrom']] = 0
    for name in model.MODEL_FILES: shutil.copy2(source/name, output/name)
    water.array_file(output/'coordinate.txt', qnative, m['box_angstrom'])
    water.array_file(output/'velocity.txt', np.zeros_like(q))
    np.savez(output/'state.npz', positions=qnative, canonical_position=q,
             runtime_input_position=qraw, physical_velocity=u, masses=state['masses'],
             pairs=state['pairs'], types=state['types'])
    text = (source/'lammps.data').read_text()
    before, body = text.split('Atoms # full\n\n'); atoms, after = body.split('\nBonds\n')
    rows = [line.split() for line in atoms.splitlines() if line.strip()]
    if len(rows) != len(q) or [int(r[0]) for r in rows] != list(range(1,len(q)+1)):
        raise ValueError('LAMMPS atom order')
    atoms = '\n'.join(' '.join(row[:4])+' '+' '.join(f'{v:.17g}' for v in xyz)
                      for row,xyz in zip(rows,qnative))
    (output/'lammps.data').write_text(before+'Atoms # full\n\n'+atoms+'\n\nBonds\n'+after)
    m['rattle_snapshot'] = dict(parent_model_sha256=water.digest(source/'model.json'),
        probe_sha256=water.digest(probe), coordinate_projection_max_angstrom=float(abs(q-qraw).max()),
        contract='canonical double preparation; static model uses actual native publication; production uses saved original input',
        independent_projection_oracle=False)
    m['initialization'] = 'projected prepared snapshot; common physical target; no equilibrium acceptance'
    m['thermal_dof'] = 3*len(q)-len(state['pairs'])-3
    m['thermal_dof_convention'] = 'equilibrium COM-free independent distances; no cosine mode removed'
    m['input_sha256'] = {name:water.digest(output/name) for name in sorted(model.MODEL_FILES)}
    water.save(output/'model.json', m)
    return m


def sampling_schedule(duration, dt, native_ps=.01, sample_interval=1, checkpoint_ps=None):
    """Bounded production with aligned native, stress and restart cadences."""
    if (not np.isfinite(dt) or not 0 < dt <= .0005 or
            not np.isfinite(duration) or not .1 <= duration <= 200 or
            not np.isfinite(native_ps) or not 0 < native_ps <= 1):
        raise ValueError('bounded duration/dt/native interval required')
    if isinstance(sample_interval, bool) or not isinstance(sample_interval, (int, np.integer)) or sample_interval < 1:
        raise ValueError('positive integer stress sample interval required')
    checkpoint_ps = duration if checkpoint_ps is None else checkpoint_ps
    if not np.isfinite(checkpoint_ps) or not 0 < checkpoint_ps <= duration:
        raise ValueError('bounded checkpoint interval required')
    steps, stride, checkpoint = [round(x/dt) for x in (duration,native_ps,checkpoint_ps)]
    if min(steps,stride,checkpoint)<1 or any(abs(n*dt-x)>1e-10 for n,x in
            ((steps,duration),(stride,native_ps),(checkpoint,checkpoint_ps))):
        raise ValueError('duration and cadence alignment')
    if steps>2000000 or steps//stride>20000:
        raise ValueError('bounded step/native frame count required')
    if steps % stride or stride % sample_interval or steps % checkpoint:
        raise ValueError('native cadence must align with stress and duration; checkpoint must divide duration')
    return dict(steps=steps,stride=stride,sample_interval=int(sample_interval),checkpoint_interval=checkpoint)


def run(source, static, output, sponge, probe, duration=2., dt=.0005,
        native_ps=.01, sample_interval=1, checkpoint_ps=None):
    source, static, output, sponge, probe = [Path(p).resolve() for p in (source,static,output,sponge,probe)]
    m, state = model.load(source)
    if any(key not in state for key in ('canonical_position','runtime_input_position','physical_velocity')):
        raise ValueError('RATTLE prepared snapshot required')
    schedule = sampling_schedule(duration,dt,native_ps,sample_interval,checkpoint_ps)
    steps,stride = schedule['steps'],schedule['stride']
    sp = json.loads((static/'plan.json').read_text()); gate = json.loads((static/'summary.json').read_text())
    if (not gate['passed'] or not sp.get('rattle_initial_frame') or sp['model_sha256'] != water.digest(source/'model.json') or
            gate['plan_sha256'] != water.digest(static/'plan.json') or
            sp['sponge_sha256'] != water.digest(sponge) or
            gate['force_sha256'] != water.digest(static/'sponge/force.dat') or
            sp['coordinate_sha256'] != water.digest(source/'coordinate.txt') or
            m['rattle_snapshot']['probe_sha256'] != water.digest(probe)):
        raise ValueError('static/preparation identities')
    for name, sha in gate['raw_sha256'].items():
        path = (static/name).resolve()
        if not path.is_relative_to(static) or water.digest(path) != sha:
            raise ValueError('static raw identity: '+name)
    q0, mass, box, u = state['canonical_position'], state['masses'], m['box_angstrom'], state['physical_velocity']
    n = len(mass); dof = 3*n-len(state['pairs'])-3; h = water.internal_timestep(dt)
    f0 = io.native_frames(static/'sponge/force.dat',n,2)[0]
    incoming = np.asarray(u-.5*h*f0/mass[:,None],np.float32).astype(float)
    output.mkdir(parents=True,exist_ok=False)
    qcheck, s0 = project(probe,state['runtime_input_position'],incoming,mass,state['pairs'],box,output/'prepare')
    np.testing.assert_array_equal(qcheck,q0)
    actual = nve.reference(q0,s0,f0,mass,box,h)
    handoff_error = float(abs(actual['velocity']-u).max())
    if handoff_error > 5e-7: raise ValueError('physical handoff velocity error')
    water.array_file(output/'coordinate.txt',state['runtime_input_position'],box)
    water.array_file(output/'velocity.txt',incoming)
    np.savez(output/'initial-state.npz',canonical_position=q0,canonical_staggered_velocity=s0,
             target_physical_velocity=u,actual_physical_velocity=actual['velocity'],native_force=f0)
    extra = dict(constrain_mode='RATTLE',coordinate_in_file=str(output/'coordinate.txt'),
        velocity_in_file=str(output/'velocity.txt'),write_restart_file_interval=schedule['checkpoint_interval'],
        output_h5_restart_path='checkpoint.spgr.h5',
        output_h5_restart_topology_hash=prep.identity(m['input_sha256']),
        output_h5_restart_atom_order_hash=prep.identity(mass.tolist()),
        output_h5_restart_protocol_hash=prep.identity(dict(algorithm='RATTLE',dt=dt)))
    extra['constrain.in_file'] = str(source/'constraint.txt')
    model.write_input(output,source,m,steps,dt,stride,extra,f'method="green_kubo"\nsample_interval={sample_interval}')
    plan = dict(contract='sponge.hexane.rattle-nve-pilot.v2',dt_ps=dt,h=h,**schedule,
        duration_ps=duration,thermal_dof=dof,model_sha256=water.digest(source/'model.json'),
        static_plan_sha256=water.digest(static/'plan.json'),static_summary_sha256=water.digest(static/'summary.json'),
        sponge_sha256=water.digest(sponge),probe_sha256=water.digest(probe),source_sha256=water.digest(__file__),
        dependencies={Path(p).name:water.digest(p) for p in (model.__file__,nve.__file__,prep.__file__,water.__file__,io.__file__,nve.gk.__file__)},
        mdin_sha256=water.digest(output/'mdin.spg.toml'),initial_state_sha256=water.digest(output/'initial-state.npz'),
        physical_handoff_max_error_internal=handoff_error,
        gates=dict(native_kinetic_rtol=2e-6,native_kinetic_atol=1e-5,canonical_endpoint_kinetic_atol=1e-8,
                   length=2e-12,tangent_nm_ps=1e-9,native_endpoint_angstrom=3e-6,
                   physical_handoff_internal=5e-7,initial_pressure_pa='100 + 5e-4 maxabs(reference)',initial_temperature_k=.01),
        energy_gate=None,formal_viscosity_acceptance=False)
    water.save(output/'plan.json',plan); started=time.perf_counter()
    try:
        wall=water.execute([sponge,'-mdin','mdin.spg.toml'],output,timeout=14400)
        water.save(output/'execution.json',dict(engine_wall_seconds=wall))
        return audit(source,static,output,m,state,plan,wall,started)
    except Exception as error:
        water.save(output/'failure.json',dict(error=str(error),plan_sha256=water.digest(output/'plan.json'),wall_seconds=time.perf_counter()-started))
        raise


def audit(source,static,output,m,state,plan,wall,started):
    for path, key in ((source/'model.json','model_sha256'), (static/'plan.json','static_plan_sha256'),
                      (static/'summary.json','static_summary_sha256'),
                      (output/'initial-state.npz','initial_state_sha256'), (output/'mdin.spg.toml','mdin_sha256')):
        if water.digest(path) != plan[key]: raise ValueError('audit identity: '+str(path))
    n=m['atoms'];mass=state['masses'];box=m['box_angstrom'];h=plan['h'];dof=plan['thermal_dof']
    steps=plan['steps'];stride=plan['stride'];dt=plan['dt_ps']
    sample_interval=plan.get('sample_interval',1);native_ratio=stride//sample_interval
    times,_,temperature,volume,meta=nve.gk.load_stress(output/'viscosity.h5')
    np.testing.assert_allclose(times,np.arange(0,steps+1,sample_interval)*dt,rtol=1e-12,atol=1e-12)
    if meta['constraint_solver'] != f'RATTLE|rattle_pairs={len(state["pairs"])}|precision=canonical_double_native_force_v1|step=BA_full_drift' or int(meta['thermal_dof']) != dof:
        raise ValueError('RATTLE GK metadata')
    np.testing.assert_allclose(volume,box**3*.001,rtol=1e-7)
    sampled=np.arange(0,steps+1,stride);count=len(sampled)
    obs=io.read_observables(output/'observables.h5',sampled)
    initial=np.load(output/'initial-state.npz')
    q=np.concatenate((state['positions'][None],io.native_frames(output/'crd.dat',n,count-1)))
    s=np.concatenate((initial['canonical_staggered_velocity'].astype(np.float32).astype(float)[None],io.native_frames(output/'vel.dat',n,count-1)))
    f=io.native_frames(output/'force.dat',n,count)
    np.testing.assert_array_equal(f[0],initial['native_force'])
    with h5py.File(output/'viscosity.h5') as file:
        group=file['observables/all']
        com=np.column_stack([group['com_v'+a+'/value'][()][::native_ratio] for a in 'xyz'])*10/water.SP_TIME
        length=float(group['constraint_length_residual/value'][()].max())
        tangent=float(group['constraint_tangent_residual/value'][()].max())
        pressure0=np.array([group[name+'/value'][0] for name in water.NAMES])
    kinetic=.5*float(np.float32(dof)*np.float32(water.KB))*temperature[::native_ratio]
    full=kinetic+.5*mass.sum()*np.sum(com*com,axis=1)
    rows=[nve.reference(x,v,ff,mass,box,h) for x,v,ff in zip(q,s,f)]
    native_full=np.array([r['full_kinetic'] for r in rows])
    native_thermal=np.array([np.trace(r['kinetic'])/2 for r in rows])
    np.testing.assert_allclose(native_full,full,rtol=2e-6,atol=1e-5)
    np.testing.assert_allclose(native_thermal,kinetic,rtol=2e-6,atol=1e-5)
    if length>2e-12 or tangent>1e-9: raise ValueError('canonical constraint residual')
    with h5py.File(output/'checkpoint.spgr.h5') as file:
        if file['parameters/sponge/output/status'][()] != b'finalized': raise ValueError('checkpoint incomplete')
        group=file['parameters/restart/integrator_state'];double=group['double']
        if set(double) != {'constrained_nve_position','constrained_nve_velocity','constrained_nve_drift_energy'}:
            raise ValueError('mechanical restart state')
        qd=double['constrained_nve_position'][()].reshape(n,3);sd=double['constrained_nve_velocity'][()].reshape(n,3)
        drift=float(double['constrained_nve_drift_energy'][0])
        if qd.dtype != np.dtype('float64') or sd.dtype != np.dtype('float64'): raise ValueError('canonical dtype')
        if int(group['step'][()]) != steps-1 or abs(float(group['time'][()])-steps*dt)>1e-10: raise ValueError('restart phase')
        identity=nve.gk.text(group['constrained_nve_configuration'][()])
        if 'BA_full_drift' not in identity or float(re.search(r';dt=([^;]+)',identity).group(1)) != h:
            raise ValueError('mechanical integration identity')
    delta=q[-1]-qd;delta-=box*np.floor(delta/box+.5)
    np.testing.assert_allclose(delta,0,rtol=0,atol=3e-6)
    final=nve.reference(qd,sd,f[-1],mass,box,h)
    np.testing.assert_allclose(final['full_kinetic'],full[-1],rtol=0,atol=1e-8)
    np.testing.assert_allclose(np.trace(final['kinetic'])/2,kinetic[-1],rtol=0,atol=1e-8)
    lf=np.loadtxt(static/'lammps/forces.dump',skiprows=9)[:,1:]
    ref=nve.reference(initial['canonical_position'],initial['canonical_staggered_velocity'],lf,mass,box,h)
    gate=json.loads((static/'summary.json').read_text())
    reference_pressure=np.array(gate['reference_pressure_pa'])+np.array([(ref['kinetic']+ref['constraint'])[a,b] for a,b in water.AXES])/box**3*water.PA
    pressure_error=float(abs(pressure0-reference_pressure).max())
    if pressure_error>100+5e-4*abs(reference_pressure).max() or abs(temperature[0]-ref['temperature'])>.01:
        raise ValueError('independent initial constrained stress/temperature')
    offset=np.array([nve.cutoff_offset(x,m,state) for x in q])
    total=obs['eff_pot']-offset+full
    np.savetxt(output/'audit.csv',np.column_stack((sampled*dt,obs['eff_pot'],offset,full,kinetic,native_full,native_thermal,total,com)),
               delimiter=',',comments='',header='time_ps,U,cutoff_offset,Kfull,Kcom_free,Knative_full,Knative_thermal,total_shifted,com_x,com_y,com_z')
    np.savez(output/'final-state.npz',positions=qd,staggered_velocity=sd,physical_velocity=final['velocity'],force=f[-1])
    result=dict(status='passed_rattle_nve_numerical_pilot_not_transport_acceptance',plan_sha256=water.digest(output/'plan.json'),
        duration_ps=plan['duration_ps'],engine_wall_seconds=wall,total_wall_seconds=time.perf_counter()-started,
        mean_temperature_k=float(temperature.mean()),thermal_dof=dof,length_residual=length,tangent_residual_nm_ps=tangent,
        native_full_kinetic_max_error=float(abs(native_full-full).max()),native_thermal_kinetic_max_error=float(abs(native_thermal-kinetic).max()),
        canonical_endpoint_kinetic_error=float(final['full_kinetic']-full[-1]),native_endpoint_position_error_angstrom=float(abs(delta).max()),
        initial_pressure_error_pa=pressure_error,initial_reference_pressure_pa=reference_pressure.tolist(),initial_sponge_pressure_pa=pressure0.tolist(),
        initial_temperature_error_k=float(temperature[0]-ref['temperature']),physical_handoff_max_error_internal=plan['physical_handoff_max_error_internal'],
        energy_max_excursion_kcal_mol=float(abs(total-total[0]).max()),energy_endpoint_delta_kcal_mol=float(total[-1]-total[0]),
        energy_slope_kcal_mol_ps=float(np.polyfit(sampled*dt,total-total[0],1)[0]),drift_kinetic_change_kcal_mol=drift,
        momentum_max_change_native=float(mass.sum()*np.linalg.norm(com-com[0],axis=1).max()),
        limitations=['No energy convergence gate: compare all predeclared dt cases.',
                     'GK kinetic state is canonical; native independent audits have bounded float error.',
                     'Independent cross-engine potential pressure gate covers actual native q0 only.',
                     'Prepared snapshot and physical handoff differ from old SHAKE experiment; not identical initial states.',
                     'No equilibrium, transport or experimental acceptance.'],
        raw_sha256={p.name:water.digest(p) for p in output.iterdir() if p.suffix in ('.h5','.dat','.npz','.txt')},formal_viscosity_acceptance=False)
    water.save(output/'summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for key in ('source','output','probe'):p.add_argument('--'+key,required=True)
    p=sub.add_parser('run')
    for key in ('source','static','output','sponge','probe'):p.add_argument('--'+key,required=True)
    p.add_argument('--duration',type=float,default=2.);p.add_argument('--dt',type=float,default=.0005)
    p.add_argument('--native-ps',type=float,default=.01)
    p.add_argument('--sample-interval',type=int,default=1)
    p.add_argument('--checkpoint-ps',type=float)
    args=parser.parse_args()
    result=prepare(args.source,args.output,args.probe) if args.command=='prepare' else run(args.source,args.static,args.output,args.sponge,args.probe,args.duration,args.dt,args.native_ps,args.sample_interval,args.checkpoint_ps)
    print(json.dumps(result,indent=2))
