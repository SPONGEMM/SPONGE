#!/usr/bin/env python3
"""Matched SPC/E constrained dynamics and continuous-stress reconstruction.

SPONGE RATTLE and unmodified LAMMPS rigid/nve/small integrate the same rigid
three-site water model. They are different finite-step integration methods.

LAMMPS fix virial is retained as a discrete diagnostic, never substituted for
the independent continuous reaction stress. This bounded runner is not a
viscosity convergence or equilibrium certificate.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import sys
import time

import h5py
import numpy as np
import water_model as water
import water_pilot as pilot
import hexane_rattle
import hexane_pilot
import lj_gk_compare
from hexane_rattle import project
from hexane_pilot import identity
from lj_gk_compare import LM_TIME, LM_KB

sys.path.insert(0, str(water.ROOT/'scripts'))
import analyze_green_kubo as gk


def reference(q, velocity, force, state, model, h=0):
    return water.constrained_reference(q, velocity, force, state['masses'],
        state['pairs'], model['box_angstrom'], h, np.zeros((3, 3)))


def write_sponge(directory, source, model, qraw, incoming, steps, dt, grid):
    directory.mkdir()
    water.array_file(directory/'coordinate.txt', qraw, model['box_angstrom'])
    water.array_file(directory/'velocity.txt', incoming)
    cfg = dict(md_name='spce_rattle_pair', mode='nve', pbc=True, dt=dt,
        step_limit=steps, cutoff=model['cutoff_angstrom'], skin=1.,
        print_zeroth_frame=True, write_mdout_interval=1, write_information_interval=1,
        write_trajectory_interval=1, write_restart_file_interval=steps,
        crd='crd.dat', vel='vel.dat', frc='force.dat',
        output_h5_observable_path='observables.h5', constrain_mode='RATTLE',
        output_h5_restart_path='checkpoint.spgr.h5',
        output_h5_restart_topology_hash=identity(model['input_sha256']),
        output_h5_restart_atom_order_hash=identity(model['atoms']),
        output_h5_restart_protocol_hash=identity(dict(dt=dt,grid=grid,mode='RATTLE')))
    cfg.update({'PM.MPI_size':1, 'PM.fftx':grid, 'PM.ffty':grid, 'PM.fftz':grid,
                'PM.Direct_Tolerance':1e-10, 'constrain.in_file':str(source/'constraint.txt')})
    for key in ('mass','charge','LJ','exclude','residue'):
        cfg[key+'_in_file'] = str(source/(key.lower()+'.txt'))
    cfg.update(coordinate_in_file=str(directory/'coordinate.txt'), velocity_in_file=str(directory/'velocity.txt'))
    write_engine_input(directory/'mdin.spg.toml', '\n'.join(f'{k}={json.dumps(v)}' for k,v in cfg.items())+
        '\n[viscosity]\nmethod="green_kubo"\nsample_interval=1\n')


def write_lammps(directory, source, model, q, u, steps, h):
    directory.mkdir()
    prefix, body = (source/'lammps.data').read_text().split('Atoms # full\n\n')
    atoms, suffix = body.split('\nBonds\n')
    rows = [line.split() for line in atoms.splitlines() if line.strip()]
    if len(rows) != len(q) or [int(r[0]) for r in rows] != list(range(1,len(q)+1)):
        raise ValueError('LAMMPS atom order')
    # read_data infers image flags from out-of-box coordinates. Keep each water
    # whole before it remaps atoms, so rigid/small sees the correct inertia.
    whole = whole_water(q, model['box_angstrom'])
    atoms = '\n'.join(' '.join(row[:4])+' '+' '.join(f'{v:.17g}' for v in xyz)
                      for row,xyz in zip(rows,whole))
    velocities = '\n'.join(str(i+1)+' '+' '.join(f'{v/LM_TIME:.17g}' for v in row) for i,row in enumerate(u))
    (directory/'model.data').write_text(prefix+'Atoms # full\n\n'+atoms+'\n\nBonds\n'+suffix+
                                      '\nVelocities\n\n'+velocities+'\n')
    water.lammps_input(directory,source,model,'full')
    text = (directory/'in.lammps').read_text().split('compute p all')[0]
    text = text.replace(str(source/'lammps.data'),'model.data')
    # Potential forces are stored separately from the rigid constraint virial.
    # Potential pressure explicitly excludes fixes and kinetic contributions.
    text += f'''timestep {h*LM_TIME:.17g}
compute thermal all temp/com
compute potential_pressure all pressure NULL pair bond angle dihedral improper kspace
compute discrete_pressure all pressure thermal
variable potential equal pe
variable kinetic equal ke
fix raw all store/force
fix constraints all rigid/nve/small molecule
fix samples all ave/time 1 1 1 c_thermal v_potential v_kinetic c_potential_pressure[1] c_potential_pressure[2] c_potential_pressure[3] c_potential_pressure[4] c_potential_pressure[5] c_potential_pressure[6] c_discrete_pressure[1] c_discrete_pressure[2] c_discrete_pressure[3] c_discrete_pressure[4] c_discrete_pressure[5] c_discrete_pressure[6] file stress.dat format " %.17g"
dump state all custom 1 state.dump id x y z vx vy vz f_raw[1] f_raw[2] f_raw[3]
dump_modify state sort id format float %.17g
thermo {max(1,steps//10)}
thermo_style custom step c_thermal pe ke
thermo_modify norm no format float %.17g
run {steps}
'''
    (directory/'in.lammps').write_text(text)


def whole_water(q, box):
    """Choose coherent periodic images without changing physical geometry."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) % 3 or not np.all(np.isfinite(q)):
        raise ValueError('water coordinate shape/values')
    if not np.isfinite(box) or box <= 0:
        raise ValueError('positive finite box required')
    x = q.copy().reshape(-1, 3, 3)
    x[:, 1:] -= box * np.rint((x[:, 1:] - x[:, :1]) / box)
    return x.reshape(-1, 3)


def read_dump(path, atoms, steps, box):
    values=[]
    with Path(path).open() as file:
        for step in range(steps+1):
            if file.readline().strip() != 'ITEM: TIMESTEP' or int(file.readline()) != step:
                raise ValueError('LAMMPS dump step')
            if file.readline().strip() != 'ITEM: NUMBER OF ATOMS' or int(file.readline()) != atoms:
                raise ValueError('LAMMPS dump atom count')
            if file.readline().strip() != 'ITEM: BOX BOUNDS pp pp pp': raise ValueError('LAMMPS box header')
            cell=np.array([list(map(float,file.readline().split())) for _ in range(3)])
            np.testing.assert_allclose(cell,[[0,box]]*3,rtol=0,atol=1e-12)
            if file.readline().strip() != 'ITEM: ATOMS id x y z vx vy vz f_raw[1] f_raw[2] f_raw[3]':
                raise ValueError('LAMMPS dump fields')
            frame=np.array([list(map(float,file.readline().split())) for _ in range(atoms)])
            if frame.shape != (atoms,10) or not np.all(np.isfinite(frame)) or not np.array_equal(frame[:,0],np.arange(1,atoms+1)):
                raise ValueError('LAMMPS dump values/order')
            values.append(frame[:,1:])
        if file.read().strip(): raise ValueError('extra LAMMPS frames')
    data=np.array(values)
    return data[:,:,:3],data[:,:,3:6]*LM_TIME,data[:,:,6:]


def run(source, preparation, output, sponge, lammps, probe, steps=100, dt=.00025, grid=128):
    source,preparation,output,sponge,lammps,probe = [Path(p).resolve() for p in (source,preparation,output,sponge,lammps,probe)]
    m,state=water.load_model(source)
    if not 1<=steps<=1000 or not np.isfinite(dt) or not 0<dt<=.0005 or grid not in (128,256):
        raise ValueError('bounded steps/dt/grid required')
    report=json.loads((preparation/'summary.json').read_text()); prior=json.loads((preparation/'plan.json').read_text())
    if (report['status']!='completed_liquid_pilot_not_transport_acceptance' or
        report['plan_sha256']!=water.digest(preparation/'plan.json') or
        prior['model_sha256']!=water.digest(source/'model.json') or
        report['physical_final_sha256']!=water.digest(preparation/'physical-final.npz')):
        raise ValueError('prepared snapshot identity')
    output.mkdir(parents=True,exist_ok=False)
    initial=np.load(preparation/'physical-final.npz'); box=m['box_angstrom'];mass=state['masses'];n=m['atoms']
    qraw=np.asarray(initial['positions']%box,np.float32).astype(float);qraw[qraw>=box]=0
    q,u=project(probe,qraw,initial['physical_velocity'],mass,state['pairs'],box,output/'projection')
    u-=np.average(u,weights=mass,axis=0)
    h=water.internal_timestep(dt)
    plan=dict(contract='sponge.spce.constrained-pair.v1',steps=steps,dt_ps=dt,h=h,grid=grid,
        integrators=dict(sponge='RATTLE',lammps='rigid/nve/small molecule'),
        lammps_dt_ps=h*LM_TIME/1000,nominal_time_relative_difference=h*LM_TIME/(1000*dt)-1,
        model_sha256=water.digest(source/'model.json'),preparation_summary_sha256=water.digest(preparation/'summary.json'),
        snapshot_sha256=water.digest(preparation/'physical-final.npz'),sponge_sha256=water.digest(sponge),
        lammps_sha256=water.digest(lammps),probe_sha256=water.digest(probe),source_sha256=water.digest(__file__),
        source_dependencies={Path(p).name:water.digest(p) for p in
            (water.__file__,pilot.__file__,gk.__file__,hexane_rattle.__file__,hexane_pilot.__file__,lj_gk_compare.__file__)},
        gates=dict(length=2e-10,tangent=1e-9,sp_native_kinetic_rtol=2e-6,sp_native_kinetic_atol=1e-5,
                   lm_kinetic_rtol=2e-10,initial_force_max=.002,initial_temperature_k=.01,
                   initial_pressure='100+5e-4 maxabs(reference)'),
        energy_and_long_trajectory_comparison='diagnostic only, no posthoc threshold',formal_viscosity_acceptance=False)
    water.save(output/'plan.json',plan);started=time.perf_counter()
    try:
        force_dir=output/'force-probe'
        write_sponge(force_dir,source,m,qraw,np.zeros_like(q),1,dt,grid)
        probe_wall=water.execute([sponge,'-mdin','mdin.spg.toml'],force_dir,timeout=1800)
        force=pilot.native_frames(force_dir/'force.dat',n,2)[0]
        incoming=np.asarray(u-.5*h*force/mass[:,None],np.float32).astype(float)
        qcheck,s=project(probe,qraw,incoming,mass,state['pairs'],box,output/'handoff')
        np.testing.assert_array_equal(q,qcheck)
        actual=reference(q,s,force,state,m,h)['velocity']
        if np.max(abs(actual-u))>5e-7: raise ValueError('physical initial handoff')
        np.savez(output/'initial-state.npz',raw=qraw,canonical=q,staggered=s,target=u,physical=actual,force=force)
        sp,lm=output/'sponge',output/'lammps'
        write_sponge(sp,source,m,qraw,incoming,steps,dt,grid)
        write_lammps(lm,source,m,q,actual,steps,h)
        water.save(output/'input-identities.json',{str(p.relative_to(output)):water.digest(p) for p in output.rglob('*')
                   if p.is_file() and p.suffix in ('.toml','.data','.txt','.npz')})
        walls=dict(force_probe=probe_wall,sponge=water.execute([sponge,'-mdin','mdin.spg.toml'],sp,timeout=1800),
                   lammps=water.execute([lammps,'-in','in.lammps'],lm,timeout=1800))
        water.save(output/'execution.json',walls)
        for name, sha in json.loads((output/'input-identities.json').read_text()).items():
            if water.digest(output/name) != sha: raise ValueError('input changed during execution: '+name)
        result=audit(output,m,state,plan)
        result.update(engine_wall_seconds=walls,total_wall_seconds=time.perf_counter()-started,plan_sha256=water.digest(output/'plan.json'),
                      raw_sha256={str(p.relative_to(output)):water.digest(p) for p in output.rglob('*') if p.is_file() and p.name not in ('summary.json','failure.json')})
        water.save(output/'summary.json',result);return result
    except Exception as error:
        water.save(output/'failure.json',dict(error=str(error),plan_sha256=water.digest(output/'plan.json'),wall_seconds=time.perf_counter()-started))
        raise


def audit(output,m,state,plan):
    n=m['atoms'];mass=state['masses'];box=m['box_angstrom'];steps=plan['steps'];h=plan['h'];dof=3*n-len(state['pairs'])-3
    initial=np.load(output/'initial-state.npz');sp=output/'sponge';lm=output/'lammps'
    q0=initial['canonical'];qn=np.asarray(q0%box,np.float32).astype(float);qn[qn>=box]=0
    q=np.concatenate((qn[None],pilot.native_frames(sp/'crd.dat',n,steps)))
    s=np.concatenate((initial['staggered'].astype(np.float32).astype(float)[None],pilot.native_frames(sp/'vel.dat',n,steps)))
    force=pilot.native_frames(sp/'force.dat',n,steps+1);np.testing.assert_array_equal(force[0],initial['force'])
    times,stress,temp,volume,meta=gk.load_stress(sp/'viscosity.h5')
    np.testing.assert_allclose(times,np.arange(steps+1)*plan['dt_ps'],rtol=1e-12,atol=1e-12)
    if not meta['constraint_solver'].startswith('RATTLE|') or int(meta['thermal_dof'])!=dof: raise ValueError('SPONGE stress contract')
    with h5py.File(sp/'viscosity.h5') as file:
        g=file['observables/all'];sp_pressure=np.column_stack([g[k+'/value'][()] for k in water.NAMES])
        com=np.column_stack([g['com_v'+a+'/value'][()] for a in 'xyz'])*10/water.SP_TIME
        canonical_length=float(g['constraint_length_residual/value'][()].max())
        canonical_tangent=float(g['constraint_tangent_residual/value'][()].max())
    if canonical_length>2e-12 or canonical_tangent>1e-9: raise ValueError('SPONGE canonical constraints')
    sp_k=.5*float(np.float32(dof)*np.float32(water.KB))*temp+.5*mass.sum()*np.sum(com*com,axis=1)
    refs=[reference(x,v,f,state,m,h) for x,v,f in zip(q,s,force)]
    sp_native_k=np.array([.5*np.sum(mass[:,None]*r['velocity']**2) for r in refs])
    np.testing.assert_allclose(sp_native_k,sp_k,rtol=2e-6,atol=1e-5)
    lq,lu,lf=read_dump(lm/'state.dump',n,steps,box)
    delta=lq[0]-q0;delta-=box*np.rint(delta/box)
    np.testing.assert_allclose(delta,0,rtol=0,atol=2e-12)
    np.testing.assert_allclose(lu[0],initial['physical'],rtol=0,atol=2e-12)
    rows=np.loadtxt(lm/'stress.dat',comments='#',ndmin=2)
    if rows.shape!=(steps+1,16) or not np.array_equal(rows[:,0],np.arange(steps+1)) or not np.all(np.isfinite(rows)):
        raise ValueError('LAMMPS stress shape/cadence')
    lr=[reference(x,v,f,state,m) for x,v,f in zip(lq,lu,lf)]
    length=max(r['max_relative_length_error'] for r in lr)
    tangent=max(float(np.max(abs(r['velocity']-v))) for r,v in zip(lr,lu))
    if length>2e-10 or tangent>1e-9:
        raise ValueError(f'LAMMPS constraint accuracy: length={length:.17g}, tangent={tangent:.17g}')
    lk=.5*np.einsum('i,tij,tij->t',mass,lu,lu)
    np.testing.assert_allclose(lk,rows[:,3],rtol=2e-10,atol=1e-10)
    lt=np.array([r['temperature'] for r in lr])
    lm_from_k=np.array([np.trace(r['kinetic'])/(dof*LM_KB) for r in lr])
    np.testing.assert_allclose(lm_from_k,rows[:,1],rtol=2e-10,atol=1e-8)
    tail=-2*np.pi/3*m['molecules']**2*m['effective_c6']/m['cutoff_angstrom']**3/box**3
    tail_p=np.array([1,1,1,0,0,0])*2*tail/box**3*water.PA
    lp=rows[:,4:10]*101325+tail_p
    lp+=np.array([[r['pressure'][a,b] for a,b in water.AXES] for r in lr])
    force_error=float(abs(force[0]-lf[0]).max());pressure_error=float(abs(sp_pressure[0]-lp[0]).max())
    if force_error>.002 or pressure_error>100+5e-4*abs(lp[0]).max(): raise ValueError('paired initial force/stress')
    if abs(temp[0]-lt[0])>.01: raise ValueError('paired initial temperature')
    obs=pilot.read_observables(sp/'observables.h5',np.arange(steps+1))
    se=obs['eff_pot']-np.array([pilot.energy_shift(x,state,m) for x in q])+sp_k
    # LAMMPS Ewald has a different split parameter; retain unshifted full
    # electrostatics. Correct only the common LJ discontinuity for its energy.
    rc=m['cutoff_angstrom']; c=m['effective_c12']/rc**12-m['effective_c6']/rc**6
    oi,oj=np.triu_indices(m['molecules'],1)
    offsets=[]
    for x in lq:
        d=x[::3][oi]-x[::3][oj];d-=box*np.rint(d/box)
        offsets.append(c*np.count_nonzero(np.sum(d*d,axis=1)<rc**2))
    le=rows[:,2]-offsets+tail+lk
    np.savez(output/'continuous-stress.npz',sponge_time_ps=times,lammps_time_ps=np.arange(steps+1)*plan['lammps_dt_ps'],
        sponge_pressure_pa=sp_pressure,lammps_pressure_pa=lp,lammps_discrete_pressure_pa=rows[:,10:16]*101325+tail_p,
        sponge_temperature_k=temp,lammps_temperature_sponge_units_k=lt,volume_nm3=volume,
        sponge_full_energy=se,lammps_full_energy=le)
    np.savetxt(output/'energy.csv',np.column_stack((times,se,le,sp_k,lk)),delimiter=',',comments='',
               header='sponge_time_ps,sponge_energy,lammps_energy,sponge_kinetic,lammps_kinetic')
    return dict(status='passed_paired_constrained_numerical_audit_not_transport_acceptance',duration_ps=steps*plan['dt_ps'],
        integrators=plan['integrators'],
        initial_force_max_error=force_error,initial_pressure_max_error_pa=pressure_error,
        initial_sponge_pressure_pa=sp_pressure[0].tolist(),initial_lammps_pressure_pa=lp[0].tolist(),
        initial_temperature_error_k=float(temp[0]-lt[0]),canonical_length_residual=canonical_length,
        canonical_tangent_nm_ps=canonical_tangent,lammps_length_residual=length,lammps_velocity_projection_max=tangent,
        sponge_native_kinetic_max_error=float(abs(sp_native_k-sp_k).max()),
        sponge_energy_max_excursion=float(abs(se-se[0]).max()),lammps_energy_max_excursion=float(abs(le-le[0]).max()),
        sponge_mean_temperature=float(temp.mean()),lammps_mean_temperature_sponge_units=float(lt.mean()),
        discrete_continuous_pressure_max_difference_pa=float(abs(rows[:,10:16]*101325+tail_p-lp).max()),
        formal_viscosity_acceptance=False,
        limitations=['Native SPONGE force evaluates float publication; LAMMPS evaluates double q.',
          'LAMMPS rigid/nve/small is a different integrator of the same three-site rigid water model; not a RATTLE implementation match.',
          'Continuous constraint reconstruction uses an independent NumPy solve; LAMMPS discrete fix virial retained separately.',
          'Different electrostatic solvers and finite-dt trajectories; no pointwise long-trajectory equivalence claim.',
          'Shared prepared structural start; no equilibrium, GK integral or experimental acceptance.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','preparation','output','sponge','lammps','probe'):p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=100);p.add_argument('--dt',type=float,default=.00025);p.add_argument('--grid',type=int,default=128)
    args=p.parse_args();print(json.dumps(run(**vars(args)),indent=2))
