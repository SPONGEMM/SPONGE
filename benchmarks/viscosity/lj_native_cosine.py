#!/usr/bin/env python3
"""Native LAMMPS cosine/Nose-Hoover comparison against a completed SPONGE run.

Reuses an immutable paired initial state, not its custom reference integrator.
The thermostats differ: this diagnoses native-method agreement and actual
thermal states, and is not an identical-integrator or zero-shear certificate.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from lj_model import digest, execute
from lj_gk_compare import LM_TIME, LM_KB, SP_KB
from lj_cosine_compare import cosine


def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def input_text(prior):
    p=read(prior/'plan.json');m=p['model'];ref=p['reference'];n=m['atoms']
    prefix=(prior/'lammps/in.lammps').read_text().split('fix integrate all python/move')[0]
    if 'fix integrate all python/move' not in (prior/'lammps/in.lammps').read_text():raise ValueError('unexpected prior input')
    damp=p['tau_sponge_ps']/p['dt_sponge_ps']*ref['dt_fs']
    text=prefix+f'''compute flow all viscosity/cos
fix drive all accelerate/cos {ref['acceleration_angstrom_fs2']:.17g}
fix integrate all nvt temp {ref['temperature_k']:.17g} {ref['temperature_k']:.17g} {damp:.17g}
fix_modify integrate temp flow
variable cs atom cos(2*PI*z/lz)
variable cs2 atom v_cs*v_cs
variable vc atom vx*v_cs
variable v2 atom vx*vx+vy*vy+vz*vz
compute moments all reduce ave v_cs v_cs2 vx vy vz v_vc v_v2
variable variance equal c_moments[2]-c_moments[1]*c_moments[1]
variable covariance equal c_moments[6]-c_moments[3]*c_moments[1]
variable comfourier equal 2*v_covariance
variable fit equal v_covariance/v_variance
variable thermal equal {n*m['mass_da']*LM_TIME**2/((3*n-4)*LM_KB):.17g}*(c_moments[7]-c_moments[3]^2-c_moments[4]^2-c_moments[5]^2-v_covariance^2/v_variance)
fix sample all ave/time {p['sample_interval']} 1 {p['sample_interval']} c_flow c_flow[7] v_comfourier v_fit v_thermal c_moments[3] c_moments[4] c_moments[5] c_moments[1] v_variance file native.dat format " %.17g"
dump audit all custom {p['sample_interval']*100} audit.dump id x y z vx vy vz
dump_modify audit sort id format float %.17g
thermo {p['sample_interval']*100}
thermo_style custom step c_flow c_flow[7] v_thermal pe ke
thermo_modify norm no format float %.17g
run {p['steps']}
'''
    return text


def audit_dump(path,rows,p):
    """Independent raw-velocity formulas, including native and joint temperatures."""
    m=p['model'];n=m['atoms'];box=m['box_angstrom'];errors=np.zeros(5);count=0
    with Path(path).open() as stream:
        while True:
            line=stream.readline()
            if not line:break
            if line.strip()!='ITEM: TIMESTEP':raise ValueError('dump header')
            step=int(stream.readline())
            if stream.readline().strip()!='ITEM: NUMBER OF ATOMS' or int(stream.readline())!=n:raise ValueError('dump atoms')
            if stream.readline().strip()!='ITEM: BOX BOUNDS pp pp pp':raise ValueError('dump box')
            bounds=np.array([list(map(float,stream.readline().split())) for _ in range(3)])
            np.testing.assert_allclose(bounds,[[0,box]]*3,rtol=0,atol=1e-12)
            if stream.readline().strip()!='ITEM: ATOMS id x y z vx vy vz':raise ValueError('dump columns')
            frame=np.array([list(map(float,stream.readline().split())) for _ in range(n)])
            if not np.isfinite(frame).all() or not np.array_equal(frame[:,0],np.arange(1,n+1)):raise ValueError('dump values/order')
            q,v=frame[:,1:4],frame[:,4:7];c=np.cos(2*np.pi*q[:,2]/box);com=v.mean(axis=0)
            var=np.mean((c-c.mean())**2);cov=np.mean((v[:,0]-com[0])*(c-c.mean()))
            fourier=2*np.mean(v[:,0]*c);fit=cov/var
            residual=v-com;residual[:,0]-=fit*(c-c.mean())
            joint=np.sum(residual**2)*m['mass_da']*LM_TIME**2/((3*n-4)*LM_KB)
            raw=v.copy();raw[:,0]-=fourier*c
            native=np.sum(raw**2)*m['mass_da']*LM_TIME**2/((3*n-3)*LM_KB)
            i=step//p['sample_interval']
            if step!=count*p['sample_interval']*100 or rows[i,0]!=step:raise ValueError('dump cadence')
            expected=np.array([native,fourier,2*cov,fit,joint])
            errors=np.maximum(errors,abs(expected-rows[i,1:6]));count+=1
    if count!=p['steps']//(p['sample_interval']*100)+1:raise ValueError('dump coverage')
    if np.any(errors>np.array([1e-7,1e-12,1e-12,1e-12,1e-7])):raise ValueError('native independent observer mismatch '+repr(errors))
    return dict(frames=count,max_errors_nativeT_fourier_comfourier_fit_jointT=errors.tolist())


def block_report(step,vel,temp,p,acceleration):
    m=p['model'];start=p['start_star'];width=p['block_star'];end=p['steps']*m['dt_star']
    begin=round(start/m['dt_star']);block=round(width/m['dt_star']);finish=p['steps']
    if (abs(begin*m['dt_star']-start)>1e-10 or abs(block*m['dt_star']-width)>1e-10
            or (finish-begin)%block):raise ValueError('complete block alignment')
    density=m['atoms']*m['mass_da']*1.66053906660/m['box_angstrom']**3
    prefactor=density*acceleration/(20*np.pi/m['box_angstrom'])**2
    records=[]
    for first in range(begin,finish,block):
        select=(step>=first)&(step<first+block)
        if select.sum()!=block//p['sample_interval']:raise ValueError('block sample count')
        records.append(dict(start_star=first*m['dt_star'],fluidity_per_mpa_s=(vel[select].mean(axis=0)/prefactor).tolist(),
            temperature_k=float(temp[select].mean())))
    means=np.mean([r['fluidity_per_mpa_s'] for r in records],axis=0)
    return dict(start_star=start,end_star=end,blocks=records,fluidity_per_mpa_s=means.tolist(),
        viscosity_mpa_s=[float(1/x) if x>0 else None for x in means],temperature_k=float(np.mean([r['temperature_k'] for r in records])))


def run(prior,output,lammps):
    prior,output,lammps=map(lambda x:Path(x).resolve(),(prior,output,lammps))
    p=read(prior/'plan.json');ex=read(prior/'execution.json')
    if p['contract']!='sponge.lj.paired-cosine-projected-middle.v1' or ex['status']!='engines_completed':raise ValueError('completed source pair required')
    if ex['plan_sha256']!=digest(prior/'plan.json'):raise ValueError('prior plan hash')
    for name,sha in (p['input_sha256']|ex['raw_output_sha256']).items():
        path=(prior/name).resolve()
        if not path.is_relative_to(prior) or digest(path)!=sha:raise ValueError('prior data changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(prior/'lammps/model.data',output/'model.data')
    (output/'in.lammps').write_text(input_text(prior))
    plan=dict(contract='sponge.lj.native-cosine-comparison.v1',prior=str(prior),prior_plan_sha256=digest(prior/'plan.json'),
        prior_execution_sha256=digest(prior/'execution.json'),sponge_profile_sha256=digest(prior/'sponge/viscosity.h5'),
        lammps_sha256=digest(lammps),source_sha256=digest(__file__),
        input_sha256={n:digest(output/n) for n in ('model.data','in.lammps')},
        reference='unmodified native LAMMPS nvt + viscosity/cos + accelerate/cos',
        comparison='same model, initial state and drive; different bath/integrator',formal_transport_acceptance=False)
    save(output/'plan.json',plan);started=time.perf_counter()
    try:
        wall=execute([lammps,'-in','in.lammps'],output,timeout=600)
        rows=np.loadtxt(output/'native.dat',ndmin=2)
        expected=np.arange(0,p['steps']+1,p['sample_interval'])
        if rows.shape!=(len(expected),11) or not np.isfinite(rows).all() or not np.array_equal(rows[:,0],expected):raise ValueError('native output shape/cadence')
        independent=audit_dump(output/'audit.dump',rows,p)
        sp,_=cosine.load_profile(prior/'sponge/viscosity.h5')
        np.testing.assert_array_equal(sp['step'],expected)
        velocities=np.column_stack([sp['velocity_'+e] for e in cosine.ESTIMATORS])
        np.testing.assert_allclose(rows[0,2:5]*100,velocities[0],rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(rows[0,5]*LM_KB/SP_KB,sp['thermal_temperature'][0],rtol=2e-5,atol=1e-5)
        for name,sha in plan['input_sha256'].items():
            if digest(output/name)!=sha:raise ValueError('input changed')
        report=dict(status='completed_native_thermostat_diagnostic',plan_sha256=digest(output/'plan.json'),
            estimators=list(cosine.ESTIMATORS),independent_observer=independent,
            sponge=block_report(expected,velocities,sp['thermal_temperature'],p,p['acceleration_nm_ps2']),
            lammps=block_report(expected,rows[:,2:5]*100,rows[:,5]*LM_KB/SP_KB,p,p['reference']['acceleration_nm_ps2']),
            native_temperature_k=float(rows[expected>=round(p['start_star']/p['model']['dt_star']),1].mean()),
            raw_sha256={n:digest(output/n) for n in ('native.dat','audit.dump','run.log')},
            engine_wall_seconds=wall,total_wall_seconds=time.perf_counter()-started,formal_transport_acceptance=False,
            limitations=['Native cosine bias differs from joint COM/cos projection; no same-thermostat assertion.',
                'Native nvt is Nose-Hoover; SPONGE is projected Middle Langevin. Compare measured thermal states.',
                'Single case estimates have no independence/steady-state/zero-drive/size acceptance.',
                'Prior SPONGE data reused; new native run is not a new independent paired preparation.'])
        save(output/'summary.json',report);return report
    except Exception as e:
        save(output/'failure.json',dict(error=repr(e),wall_seconds=time.perf_counter()-started));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('prior','output','lammps'):parser.add_argument('--'+key,required=True)
    a=parser.parse_args();r=run(**vars(a));print(json.dumps(r,indent=2))
