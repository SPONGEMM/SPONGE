#!/usr/bin/env python3
"""Validate integrated constrained GK and its offline observer on rigid rotors.

No liquid viscosity is computed. Inputs, HDF5 stress and all native trajectories
are preserved in a new output directory.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import shutil
import subprocess
import time
import numpy as np
import h5py

TIME=float(np.float32(20.455))
AXES=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rot(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def run_case(root,sponge,observer,body,algorithm,h_request):
    mass=np.array([2.,3.] if body=='dimer' else [16.,1.,1.])
    position=np.array([[0.,0.,0.],[1.,0.,0.]] if body=='dimer' else [[0.,0.,0.],[.6,.8,0.],[-.6,.8,0.]])
    position-=np.average(position,axis=0,weights=mass)
    pairs=[(0,1,1.)] if body=='dimer' else [(0,1,1.),(0,2,1.),(1,2,1.2)]
    omega=2.;dt_ps=h_request/TIME;h=float(np.float32(dt_ps)*np.float32(TIME));steps=round(4/h)
    # Exact chord incoming velocity for a SHAKE free planar rotation. Its
    # tangent projection at q0 equals omega cross q0. Do not input tangent
    # velocity as a leapfrog half-step and silently compare another energy.
    theta=np.arcsin(h*omega)
    half=(position-position@rot(-theta).T)/h
    q0=np.asarray(position+[8,8,8],dtype=np.float32).astype(float)
    s0=np.asarray(half,dtype=np.float32).astype(float)
    root.mkdir()
    for name,values in [('mass',mass),('charge',mass*0)]:
        (root/f'{name}.txt').write_text(str(len(mass))+'\n'+'\n'.join(map(str,values))+'\n')
    for name,values in [('coordinate',q0),('velocity',s0)]:
        (root/f'{name}.txt').write_text(str(len(mass))+'\n'+'\n'.join(' '.join(f'{v:.17g}' for v in row) for row in values)+('\n32 32 32\n90 90 90\n' if name=='coordinate' else '\n'))
    (root/'constraint.txt').write_text(str(len(pairs))+'\n'+'\n'.join(f'{a} {b} {r}' for a,b,r in pairs)+'\n')
    config=dict(md_name='constraint_observer_probe',mode='nve',pbc=True,dt=dt_ps,step_limit=steps,cutoff=8.,
        print_zeroth_frame=True,write_mdout_interval=steps,write_information_interval=steps,
        write_trajectory_interval=1,write_restart_file_interval=0,crd='crd.dat',vel='vel.dat',frc='frc.dat',constrain_mode=algorithm,
        settle_disable=algorithm=='SHAKE')
    config.update({'PM.MPI_size':1,'constrain.in_file':'constraint.txt'})
    if algorithm=='SHAKE':config.update({'SHAKE.iteration_numbers':100,'SHAKE.tolerance':1e-6})
    config.update({f'{key}_in_file':f'{key}.txt' for key in ('mass','charge','coordinate','velocity')})
    write_engine_input(root/'mdin.spg.toml', '\n'.join(f'{k}={json.dumps(v)}' for k,v in config.items())+'\n[viscosity]\nmethod="green_kubo"\n')
    save(root/'plan.json',dict(body=body,algorithm=algorithm,h_internal=h,dt_ps=dt_ps,steps=steps,omega=omega,
        sample_phase='q_n and incoming s_n; zero unconstrained force',input_sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
    start=time.perf_counter()
    with (root/'run.log').open('w') as log:
        run=subprocess.run([str(sponge),'-mdin','mdin.spg.toml'],cwd=root,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=60)
    save(root/'execution.json',dict(returncode=run.returncode,wall_seconds=time.perf_counter()-start))
    if run.returncode:raise RuntimeError(f'MD failed: {root}/run.log')
    n=len(mass)
    q=np.concatenate(([q0],np.fromfile(root/'crd.dat',np.float32).reshape(-1,n,3).astype(float)))
    half=np.concatenate(([s0],np.fromfile(root/'vel.dat',np.float32).reshape(-1,n,3).astype(float)))
    force=np.fromfile(root/'frc.dat',np.float32).reshape(-1,n,3).astype(float)
    if q.shape!=(steps+1,n,3) or half.shape!=q.shape or force.shape!=q.shape or np.any(force!=0):
        raise ValueError('native frame/force mismatch')
    header=f'{n} {len(pairs)} {steps+1}\n'+' '.join(map(str,mass))+'\n'+'\n'.join(f'{a} {b} {r}' for a,b,r in pairs)+'\n'
    text=io.StringIO();text.write(header)
    for x,v in zip(q,half):
        text.write(f'{h:.17g} 32 32 32 0.0001\n')
        for row in [x,v,np.zeros_like(v)]:np.savetxt(text,row,fmt='%.17g')
        text.write('0 0 0 0 0 0\n')
    (root/'observer-input.txt').write_text(text.getvalue())
    result=subprocess.run([str(observer),'--stream'],input=text.getvalue(),capture_output=True,text=True,timeout=60)
    (root/'observer.log').write_text(result.stderr)
    if result.returncode:raise RuntimeError(f'observer failed: {result.stderr}')
    (root/'observer-output.txt').write_text(result.stdout);out=np.loadtxt(io.StringIO(result.stdout),ndmin=2)
    # Compare the actual HDF5 stream to the offline observer fed from native
    # files, then retain the independent continuous-rotor convergence check.
    pressure_to_pa=4184/6.02214076e23/1e-30
    with h5py.File(root/'viscosity.h5') as file:
        decode=lambda x: x.decode() if isinstance(x,bytes) else str(x)
        metadata={key:decode(value[()]) for key,value in file['parameters/sponge/provenance'].items()}
        if (metadata.get('observable_contract')!='sponge.observables.physical_state.v1' and metadata.get('viscosity_contract')!='sponge.green_kubo.constrained_nve.v1') or int(metadata['constraint_count'])!=len(pairs):
            raise ValueError('integrated observer contract/topology mismatch')
        if decode(file['parameters/sponge/output/status'][()])!='finalized':
            raise ValueError('integrated stress is not finalized')
        values=file['observables/all']
        pressure=np.column_stack([values[f'p{x}/value'][()] for x in ('xx','yy','zz','xy','xz','yz')])
        np.testing.assert_allclose(pressure,out[:,8:14]*pressure_to_pa,rtol=1e-11,atol=1e-5)
        np.testing.assert_allclose(values['temperature/value'][()],2*out[:,0]/(out[:,1].astype(np.float32)*np.float32(.00198716)).astype(float),rtol=1e-11)
        np.testing.assert_array_equal(values['step'][()],np.arange(steps+1))
        integration_error=float(np.max(abs(pressure-out[:,8:14]*pressure_to_pa)))
    save(root/'h5-contract.json',metadata)
    actual_v=out[:,17:17+3*n].reshape(-1,n,3)
    # Analytic continuous free rotation; no trajectory fitting or replacing
    # omega by the observed angular velocity.
    reference_q=np.array([position@rot(omega*k*h).T for k in range(steps+1)])
    reference_v=np.stack((-omega*reference_q[:,:,1],omega*reference_q[:,:,0],np.zeros((steps+1,n))),axis=-1)
    reference_k=.5*np.sum(mass[None,:,None]*reference_v**2,axis=(1,2))
    reference_w=-omega**2*np.einsum('i,tia,tib->tab',mass,reference_q,reference_q)
    reference_pressure=(np.einsum('i,tia,tib->tab',mass,reference_v,reference_v)+reference_w)/32**3
    expected=np.column_stack([reference_pressure[:,a,b] for a,b in AXES]);measured=out[:,8:14]
    centered_q=q-np.average(q,axis=1,weights=mass)[:,None,:]
    angular=np.sum(mass[None,:,None]*np.cross(centered_q,actual_v),axis=1)
    expected_l=np.sum(mass[:,None]*np.cross(position,reference_v[0]),axis=0)
    record=dict(body=body,algorithm=algorithm,h_internal=h,steps=steps,
        pressure_rms_error=float(np.sqrt(np.mean((measured-expected)**2))),
        position_max_error=float(np.max(abs(centered_q-reference_q))),
        kinetic_relative_error=float(np.max(abs(out[:,0]/reference_k-1))),
        angular_relative_error=float(np.max(np.linalg.norm(angular-expected_l,axis=1))/np.linalg.norm(expected_l)),
        max_relative_length_error=float(out[:,14].max()),max_tangent_residual=float(out[:,15].max()),max_acceleration_residual=float(out[:,16].max()),
        thermal_dof=int(out[0,1]),h5_offline_max_pressure_difference_pa=integration_error,not_a_viscosity_result=True)
    np.savez(root/'reference.npz',position=reference_q,velocity=reference_v,kinetic=reference_k,pressure=expected)
    save(root/'summary.json',record);return record

def run(sponge,observer,output):
    sponge=Path(sponge).resolve();observer=Path(observer).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    sources=['SPONGE/constrain/tangent_frame.h','SPONGE/constrain/tangent_frame.cpp',
        'SPONGE/observables/stress_math.h','SPONGE/observables/stress_math.cpp',
        'SPONGE/external_force/drive_config.h','SPONGE/utils/float_classification.hpp',
        'tests/control/test_constraint_stress.cpp','benchmarks/viscosity/constraint_observer_probe.py',
        'SPONGE/MD_core/energy_accounting.h','SPONGE/MD_core/energy_accounting.cpp',
        'SPONGE/MD_core/initial_state.cpp','SPONGE/io/output_session.cpp','SPONGE/main.cpp',
        'cmake/targets/SPONGE.cmake','scripts/analyze_green_kubo.py',
        'tests/nonequilibrium/test_green_kubo_runtime.py','tests/nonequilibrium/test_green_kubo_analysis.py',
        'tests/nonequilibrium/test_coupling_restart.py','tests/nonequilibrium/test_constraint_observer_runtime.py']
    for source in sources:
        dest=output/'source'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/source,dest)
    plan=dict(contract='sponge.constraint-observer-probe.v2',h_internal_requested=[.08,.04,.02],duration_internal=4.,
        bodies=['dimer','triangle'],algorithms=['SETTLE','SHAKE'],sponge_sha256=sha(sponge),observer_sha256=sha(observer),script_sha256=sha(__file__),
        source_sha256={source:sha(output/'source'/source) for source in sources},
        source_snapshot_note='Current source snapshot; executable hashes are recorded separately. Build provenance must be retained by the caller.',
        gates=dict(pressure_error_reduction_interval=[2.8,5.2],max_relative_length_error=1e-4,kinetic_relative_error=.003,angular_relative_error=.003,tangent_residual=1e-10,acceleration_residual=1e-9),
        scope='Integrated constrained GK on free planar rigid rotors; not a liquid viscosity or experimental validation')
    save(output/'plan.json',plan);records=[];start=time.perf_counter()
    try:
        for body in plan['bodies']:
            for algorithm in plan['algorithms']:
                for h in plan['h_internal_requested']:
                    records.append(run_case(output/f'{body}-{algorithm}-{h}',sponge,observer,body,algorithm,h))
        ratios=[]
        for body in plan['bodies']:
            for algorithm in plan['algorithms']:
                rows=[r for r in records if r['body']==body and r['algorithm']==algorithm]
                for a,b in zip(rows,rows[1:]):
                    ratio=a['pressure_rms_error']/b['pressure_rms_error'];ratios.append(dict(body=body,algorithm=algorithm,coarse_h=a['h_internal'],ratio=ratio))
        passed=all(2.8<=r['ratio']<=5.2 for r in ratios) and all(r['max_relative_length_error']<1e-4 and r['kinetic_relative_error']<.003 and r['angular_relative_error']<.003 and r['max_tangent_residual']<1e-10 and r['max_acceleration_residual']<1e-9 and r['thermal_dof']==(2 if r['body']=='dimer' else 3) for r in records)
        summary=dict(status='passed_bounded_rotor_gates' if passed else 'failed_bounded_rotor_gates',cases=records,pressure_error_reduction=ratios,wall_seconds=time.perf_counter()-start,formal_viscosity_acceptance=False)
        save(output/'summary.json',summary);return summary
    except Exception as error:
        save(output/'failure.json',dict(error=str(error),completed_cases=records,wall_seconds=time.perf_counter()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sponge',required=True);p.add_argument('--observer',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=run(a.sponge,a.observer,a.output);print(json.dumps(r,indent=2));raise SystemExit(0 if r['status'].startswith('passed') else 1)
