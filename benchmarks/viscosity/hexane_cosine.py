#!/usr/bin/env python3
"""Driven hexane from a canonical prepared snapshot, with numerical audits.

Uses the same q and physical velocity handoff as hexane_rattle, with the
drive included in the initial half kick. No automatic steady-state or
viscosity acceptance; profile analysis is a separate preregistered step.
"""
import argparse
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
import hexane_model as model
import hexane_nve as nve
import hexane_pilot as prep
import hexane_rattle as rattle
import water_model as water
import water_pilot as io


def external_force(q, mass, box, acceleration):
    """Native published q, force along x with gradient z; nm/ps^2 input."""
    result = np.zeros_like(q, dtype=np.float32)
    result[:, 0] = mass*(acceleration*10/water.SP_TIME**2)*np.cos(2*np.pi/box*q[:, 2])
    return result.astype(float)


def run(source, static, output, sponge, probe, duration=1., dt=.0005,
        acceleration=.2, tau=.1, seed=79101, native_ps=.01, sample_interval=1):
    source,static,output,sponge,probe = [Path(p).resolve() for p in (source,static,output,sponge,probe)]
    m,state = model.load(source)
    if any(key not in state for key in ('canonical_position','runtime_input_position','physical_velocity')):
        raise ValueError('canonical prepared snapshot required')
    schedule = rattle.sampling_schedule(duration,dt,native_ps,sample_interval)
    if (not np.isfinite(acceleration) or not np.isfinite(tau) or tau <= 0 or
            isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed < 2**31):
        raise ValueError('finite acceleration, positive tau and bounded integer seed required')
    sp=json.loads((static/'plan.json').read_text());gate=json.loads((static/'summary.json').read_text())
    if (not gate['passed'] or not sp.get('rattle_initial_frame') or
            sp['model_sha256'] != water.digest(source/'model.json') or
            sp['sponge_sha256'] != water.digest(sponge) or
            sp['coordinate_sha256'] != water.digest(source/'coordinate.txt') or
            gate['plan_sha256'] != water.digest(static/'plan.json') or
            gate['force_sha256'] != water.digest(static/'sponge/force.dat') or
            m['rattle_snapshot']['probe_sha256'] != water.digest(probe)):
        raise ValueError('static/preparation identities')
    for name,sha in gate['raw_sha256'].items():
        path=(static/name).resolve()
        if not path.is_relative_to(static) or water.digest(path) != sha:
            raise ValueError('static raw identity: '+name)
    q0,mass,box,u = state['canonical_position'],state['masses'],m['box_angstrom'],state['physical_velocity']
    n=len(mass);h=water.internal_timestep(dt)
    internal=io.native_frames(static/'sponge/force.dat',n,2)[0]
    external=external_force(state['positions'],mass,box,acceleration)
    total=(internal+external).astype(np.float32).astype(float)
    incoming=(u-.5*h*total/mass[:,None]).astype(np.float32).astype(float)
    output.mkdir(parents=True,exist_ok=False)
    qcheck,s0=rattle.project(probe,state['runtime_input_position'],incoming,mass,state['pairs'],box,output/'prepare')
    np.testing.assert_array_equal(qcheck,q0)
    actual=nve.reference(q0,s0,total,mass,box,h)
    handoff=float(abs(actual['velocity']-u).max())
    if handoff>5e-7:raise ValueError('physical handoff velocity error')
    water.array_file(output/'coordinate.txt',state['runtime_input_position'],box)
    water.array_file(output/'velocity.txt',incoming)
    np.savez(output/'initial-state.npz',canonical_position=q0,canonical_staggered_velocity=s0,
        target_physical_velocity=u,actual_physical_velocity=actual['velocity'],internal_force=internal,
        external_force=external,native_force=total)
    extra=dict(mode='nvt',thermostat='middle_langevin',thermostat_tau=tau,thermostat_seed=seed,
        target_temperature=m['temperature_k'],constrain_mode='SHAKE',settle_disable=True,
        coordinate_in_file=str(output/'coordinate.txt'),velocity_in_file=str(output/'velocity.txt'),
        write_restart_file_interval=schedule['checkpoint_interval'],output_h5_restart_path='checkpoint.spgr.h5',
        output_h5_restart_topology_hash=prep.identity(m['input_sha256']),
        output_h5_restart_atom_order_hash=prep.identity(mass.tolist()),
        output_h5_restart_protocol_hash=prep.identity(dict(algorithm='constrained_cosine_v2',dt=dt,A=acceleration,tau=tau)))
    extra['constrain.in_file']=str(source/'constraint.txt')
    model.write_input(output,source,m,schedule['steps'],dt,schedule['stride'],extra,
        f'method="cosine"\nacceleration={acceleration:.17g}\nsample_interval={sample_interval}\nbins=20')
    plan=dict(contract='sponge.hexane.cosine-pilot.v1',dt_ps=dt,h=h,**schedule,duration_ps=duration,
        acceleration_nm_ps2=acceleration,tau_ps=tau,thermostat_seed=seed,bath_temperature_k=m['temperature_k'],
        thermal_dof=3*n-len(state['pairs'])-4,model_sha256=water.digest(source/'model.json'),
        static_plan_sha256=water.digest(static/'plan.json'),static_summary_sha256=water.digest(static/'summary.json'),
        sponge_sha256=water.digest(sponge),probe_sha256=water.digest(probe),source_sha256=water.digest(__file__),
        dependencies={Path(p).name:water.digest(p) for p in (model.__file__,nve.__file__,prep.__file__,rattle.__file__,water.__file__,io.__file__,prep.cosine.__file__)},
        mdin_sha256=water.digest(output/'mdin.spg.toml'),initial_state_sha256=water.digest(output/'initial-state.npz'),
        physical_handoff_max_error_internal=handoff,formal_viscosity_acceptance=False,
        gates=dict(native_kinetic_rtol=2e-6,native_kinetic_atol=1e-5,canonical_endpoint_kinetic_atol=1e-8,
            length=2e-12,tangent_nm_ps=1e-9,native_endpoint_angstrom=3e-6,
            initial_pressure_pa='100 + 5e-4 maxabs(reference)',physical_handoff_internal=5e-7,
            initial_temperature_k=.01),
        energy_gate=None)
    water.save(output/'plan.json',plan);started=time.perf_counter()
    try:
        wall=water.execute([sponge,'-mdin','mdin.spg.toml'],output,timeout=14400)
        water.save(output/'execution.json',dict(engine_wall_seconds=wall))
        return audit(source,static,output,m,state,plan,wall,started)
    except Exception as error:
        water.save(output/'failure.json',dict(error=str(error),plan_sha256=water.digest(output/'plan.json'),wall_seconds=time.perf_counter()-started))
        raise


def audit(source,static,output,m,state,plan,wall,started):
    for path,key in ((source/'model.json','model_sha256'),(static/'plan.json','static_plan_sha256'),
                     (static/'summary.json','static_summary_sha256'),(output/'mdin.spg.toml','mdin_sha256'),
                     (output/'initial-state.npz','initial_state_sha256')):
        if water.digest(path)!=plan[key]:raise ValueError('audit identity: '+str(path))
    d,metadata=prep.cosine.load_profile(output/'viscosity.h5')
    expected=np.arange(0,plan['steps']+1,plan['sample_interval'])
    np.testing.assert_array_equal(d['step'],expected)
    np.testing.assert_allclose(d['time'],expected*plan['dt_ps'],rtol=1e-12,atol=1e-12)
    ratio=plan['stride']//plan['sample_interval'];sampled={k:v[::ratio] for k,v in d.items()}
    steps=expected[::ratio];count=len(steps);n=m['atoms'];mass=state['masses'];box=m['box_angstrom'];h=plan['h']
    with np.load(output/'initial-state.npz',allow_pickle=False) as initial:
        q0=initial['canonical_position'];s0=initial['canonical_staggered_velocity'];f0=initial['native_force']
        external=initial['external_force']
    q=np.concatenate((state['positions'][None],io.native_frames(output/'crd.dat',n,count-1)))
    s=np.concatenate((s0.astype(np.float32).astype(float)[None],io.native_frames(output/'vel.dat',n,count-1)))
    force=io.native_frames(output/'force.dat',n,count)
    np.testing.assert_array_equal(force[0],f0)
    obs=io.read_observables(output/'observables.h5',steps)
    tdof=plan['thermal_dof']
    com=np.column_stack([sampled['com_v'+axis] for axis in 'xyz'])*10/water.SP_TIME
    full=.5*float(np.float32(tdof+1)*np.float32(water.KB))*sampled['kinetic_temperature']+.5*mass.sum()*np.sum(com**2,axis=1)
    thermal=.5*float(np.float32(tdof)*np.float32(water.KB))*sampled['thermal_temperature']
    frames=[prep.frame(x,v,f,mass,box,h) for x,v,f in zip(q,s,force)]
    independent=np.array([.5*np.sum(mass[:,None]*v*v) for v,_ in frames])
    joint=np.array([k for _,k in frames])
    np.testing.assert_allclose(independent,full,rtol=2e-6,atol=1e-5)
    np.testing.assert_allclose(joint,thermal,rtol=2e-6,atol=1e-5)
    length=float(d['constraint_length_residual'].max());tangent=float(d['constraint_tangent_residual'].max())
    if length>2e-12 or tangent>1e-9:raise ValueError('constraint residual')
    with h5py.File(output/'checkpoint.spgr.h5') as f:
        if f['parameters/sponge/output/status'][()]!=b'finalized':raise ValueError('checkpoint incomplete')
        group=f['parameters/restart/integrator_state'];values=group['double']
        if set(values)!={'constrained_flow_position','constrained_flow_velocity','constrained_flow_drift_energy'} or any(x.dtype!=np.dtype('float64') for x in values.values()):
            raise ValueError('canonical checkpoint state')
        qd=values['constrained_flow_position'][()].reshape(n,3);sd=values['constrained_flow_velocity'][()].reshape(n,3)
        identity=prep.cosine.text(group['constrained_flow_configuration'][()])
        if (not identity.startswith('rattle_flow_double_v2;') or
                'energy_virial_every_step_v1' not in identity.split(';') or
                float(re.search(r';dt=([^;]+)',identity).group(1))!=h):
            raise ValueError('constrained flow policy identity')
        if int(group['step'][()])!=plan['steps']-1 or abs(float(group['time'][()])-plan['duration_ps'])>1e-10:
            raise ValueError('checkpoint time/step')
        ledger=f['parameters/restart/coupling']
        if (int(ledger['steps'][()])!=plan['steps'] or
                abs(float(ledger['time'][()])-plan['duration_ps'])>1e-10 or
                float(ledger['work'][()])!=d['coupling_work'][-1] or
                float(ledger['heat'][()])!=d['thermostat_heat'][-1]):
            raise ValueError('checkpoint work/heat/clock closure')
        drift=values['constrained_flow_drift_energy'][()]
        if drift.shape!=(1,) or not np.isfinite(drift[0]) or drift[0]!=d['constraint_drift_energy'][-1]:
            raise ValueError('checkpoint drift diagnostic closure')
    delta=q[-1]-qd;delta-=box*np.floor(delta/box+.5)
    np.testing.assert_allclose(delta,0,rtol=0,atol=3e-6)
    vd,kd=prep.frame(qd,sd,force[-1],mass,box,h)
    np.testing.assert_allclose([.5*np.sum(mass[:,None]*vd**2),kd],[full[-1],thermal[-1]],rtol=0,atol=1e-8)
    lm=np.loadtxt(static/'lammps/forces.dump',skiprows=9)[:,1:]+external
    reference=nve.reference(q0,s0,lm,mass,box,h)
    static_result=json.loads((static/'summary.json').read_text())
    pressure=np.array(static_result['reference_pressure_pa'])+np.array([(reference['kinetic']+reference['constraint'])[a,b] for a,b in water.AXES])/box**3*water.PA
    pressure0=np.array([d[name][0] for name in water.NAMES]);pressure_error=float(abs(pressure0-pressure).max())
    if pressure_error>100+5e-4*abs(pressure).max():raise ValueError('independent initial total-force constraint stress')
    initial_temperature_error=float(d['kinetic_temperature'][0]-reference['temperature'])
    if abs(initial_temperature_error)>.01:raise ValueError('independent initial kinetic temperature')
    shifted=obs['eff_pot']-np.array([nve.cutoff_offset(x,m,state) for x in q])+full
    work=sampled['coupling_work']-sampled['coupling_work'][0];heat=sampled['thermostat_heat']-sampled['thermostat_heat'][0]
    residual=shifted-shifted[0]-work-heat
    if not np.all(np.isfinite(residual)):raise ValueError('nonfinite energy diagnostic')
    np.savetxt(output/'audit.csv',np.column_stack((steps*plan['dt_ps'],shifted,full,thermal,work,heat,sampled['constraint_drift_energy'],residual)),delimiter=',',comments='',header='time_ps,E_shifted,Kfull,Kthermal,work,heat,drift_kinetic_change,deltaE_minus_work_heat')
    np.savez(output/'final-state.npz',positions=qd,staggered_velocity=sd,physical_velocity=vd,force=force[-1])
    result=dict(status='passed_cosine_numerical_pilot_not_transport_acceptance',plan_sha256=water.digest(output/'plan.json'),
        engine_wall_seconds=wall,total_wall_seconds=time.perf_counter()-started,metadata=metadata,
        length_residual=length,tangent_residual_nm_ps=tangent,initial_pressure_error_pa=pressure_error,
        initial_reference_pressure_pa=pressure.tolist(),initial_sponge_pressure_pa=pressure0.tolist(),
        initial_temperature_error_k=initial_temperature_error,
        physical_handoff_max_error_internal=plan['physical_handoff_max_error_internal'],
        full_kinetic_max_error=float(abs(independent-full).max()),thermal_kinetic_max_error=float(abs(joint-thermal).max()),
        canonical_endpoint_kinetic_error=float(.5*np.sum(mass[:,None]*vd**2)-full[-1]),
        mean_thermal_temperature_k=float(d['thermal_temperature'].mean()),
        work_kcal_mol=float(work[-1]),heat_kcal_mol=float(heat[-1]),energy_max_residual_kcal_mol=float(abs(residual).max()),
        formal_viscosity_acceptance=False,limitations=['No energy gate, steady-state, linear-response or transport acceptance.',
            'Shared initial physical velocity with NVE does not imply identical later thermodynamic states.',
            'Static LAMMPS potential pressure plus independent total-force reaction validates the first frame only.'],
        raw_sha256={p.name:water.digest(p) for p in output.iterdir() if p.suffix in ('.h5','.dat','.npz','.txt')})
    water.save(output/'summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','static','output','sponge','probe'):parser.add_argument('--'+name,required=True)
    for name,default in (('duration',1.),('dt',.0005),('acceleration',.2),('tau',.1),('native-ps',.01)):
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--seed',type=int,default=79101);parser.add_argument('--sample-interval',type=int,default=1)
    args=vars(parser.parse_args());result=run(**args)
    print(json.dumps({k:result[k] for k in ('status','engine_wall_seconds',
        'initial_pressure_error_pa','physical_handoff_max_error_internal','work_kcal_mol',
        'heat_kcal_mol','energy_max_residual_kcal_mol')},indent=2))
