#!/usr/bin/env python3
"""Whole-replica conditional GK diagnostics for the frozen hexane campaign.

Never concatenate trajectories or treat stress components as independent
replicas. Finite-window output is not an equilibrium/transport acceptance.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import hexane_model as model
import water_model as water
import lj_model as lj

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import analyze_green_kubo as gk


def reduce_replicas(numerators, temperatures, draws=10000, seed=78101):
    """Numerator is eta*T, with shape (replicas, jointly resampled settings)."""
    numerators,temperatures=np.asarray(numerators,float),np.asarray(temperatures,float)
    if (numerators.ndim!=2 or numerators.shape!=temperatures.shape or
            numerators.shape[0]<2 or numerators.shape[1]<1 or
            not np.all(np.isfinite(numerators)) or not np.all(np.isfinite(temperatures)) or
            np.any(temperatures<=0)):
        raise ValueError('finite matched replica/settings arrays and positive temperatures required')
    if isinstance(draws,bool) or not isinstance(draws,int) or not 100<=draws<=100000:
        raise ValueError('bounded integer bootstrap draws required')
    indices=np.random.default_rng(seed).integers(len(numerators),size=(draws,len(numerators)))
    samples=numerators[indices].mean(axis=1)/temperatures[indices].mean(axis=1)
    estimate=numerators.mean(axis=0)/temperatures.mean(axis=0)
    return estimate,np.quantile(samples,[.025,.975],axis=0),samples,indices


def replica_curves(times, stress, temperature, volume, settings):
    """Analyze every declared setting, preserving its full mean correlation."""
    records=[];curves={}
    for start in settings['start_ps']:
        for decimation in settings['decimation']:
            if isinstance(decimation,bool) or not isinstance(decimation,int) or decimation<1:
                raise ValueError('positive integer decimation required')
            for demean in settings['demean']:
                if not isinstance(demean,bool):raise ValueError('boolean demeaning required')
                values=[np.asarray(x)[::decimation] for x in (times,stress,temperature,volume)]
                report,lags,corr,integral=gk.analyze(*values,start,max(settings['windows_ps']),
                    settings['block_ps'],demean,100,settings['replica_bootstrap_seed'])
                key=f'start-{start:g}-decimation-{decimation}-demean-{int(demean)}'
                curves[key]=np.column_stack((lags,corr,integral*1000))
                for window in settings['windows_ps']:
                    lag=round(window/report['sample_ps'])
                    if lag<1 or lag>=len(lags) or abs(lags[lag]-window)>1e-9:
                        raise ValueError('requested window must align with sampled lags')
                    eta=integral[lag]*1000
                    records.append(dict(setting=key,window_ps=window,temperature_k=report['temperature_k'],
                        volume_nm3=report['volume_nm3'],component_mpa_s=eta.tolist(),
                        viscosity_mpa_s=float(eta.mean()),numerator_mpa_s_k=float(eta.mean()*report['temperature_k']),
                        blocks=report['blocks'],used_frames=report['used_frames'],
                        first_time_ps=report['first_time_ps'],last_time_ps=report['last_time_ps'],
                        stress_mean_pa=report['stress_mean_pa'],stress_rms_pa=report['stress_rms_pa']))
    return records,curves


def verify_raw(root, hashes):
    """Require all engine/audit products, including every consumed HDF5 file."""
    root=Path(root).resolve()
    required={'viscosity.h5','observables.h5','checkpoint.spgr.h5',
              'initial-state.npz','final-state.npz','crd.dat','vel.dat','force.dat'}
    actual={p.name for p in root.iterdir() if p.suffix in ('.h5','.dat','.npz','.txt')}
    if not required.issubset(hashes) or set(hashes)!=actual:
        raise ValueError('incomplete raw manifest')
    for name,digest in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or water.digest(path)!=digest:
            raise ValueError('raw identity '+name)


def load_pressure(path,times):
    with h5py.File(path) as file:
        columns=[]
        for axis in 'xyz':
            name=f'p{axis}{axis}'
            dataset=file[f'observables/all/{name}/value']
            values=np.asarray(dataset,dtype=float)
            if values.shape!=times.shape or not np.all(np.isfinite(values)):
                raise ValueError('invalid diagonal pressure '+name)
            if gk.text(dataset.attrs.get('unit',''))!='Pa':
                raise ValueError('invalid diagonal pressure unit '+name)
            columns.append(values)
    return np.column_stack(columns)


def verify_force_policy(path):
    with h5py.File(path) as file:
        identity=gk.text(file['parameters/restart/integrator_state/constrained_nve_configuration'][()])
    fields=identity.split(';')
    if fields[0]!='rattle_nve_double_v3' or 'energy_virial_every_step_v1' not in fields:
        raise ValueError('checkpoint force policy identity')


def summarize(campaign,output):
    campaign,output=Path(campaign).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('new output directory required')
    plan=json.loads((campaign/'plan.json').read_text())
    execution=json.loads((campaign/'execution.json').read_text())
    if (plan['contract']!='sponge.hexane.gk-replicas.v1' or
            execution['stage']!='completed_sampling_pending_analysis' or
            execution['completed']!=plan['replicas'] or (campaign/'failure.json').exists() or
            execution['plan_sha256']!=water.digest(campaign/'plan.json')):
        raise ValueError('complete unchanged campaign required')
    for name,digest in plan['source_hashes'].items():
        path=(campaign/'sources'/name).resolve()
        if not path.is_relative_to(campaign/'sources') or water.digest(path)!=digest:
            raise ValueError('frozen source identity '+name)
    dependencies={str(Path(module.__file__).resolve().relative_to(Path(__file__).resolve().parents[2])):
                  water.digest(module.__file__) for module in (model,water,lj,gk)}
    for name,digest in dependencies.items():
        if plan['source_hashes'].get(name)!=digest:
            raise ValueError('analysis dependency differs from frozen campaign '+name)
    started=time.perf_counter();records=[];all_curves={};summaries=[];physical=None;common_times=None
    for replica in plan['replicas']:
        root=campaign/'runs'/replica
        summary=json.loads((root/'summary.json').read_text());run=json.loads((root/'plan.json').read_text())
        if ((root/'failure.json').exists() or summary['status']!='passed_rattle_nve_numerical_pilot_not_transport_acceptance'
                or summary['plan_sha256']!=water.digest(root/'plan.json') or
                run['duration_ps']!=plan['duration_ps'] or run['dt_ps']!=plan['dt_ps'] or
                run['sample_interval']!=plan['sample_interval'] or
                run['sponge_sha256']!=plan['binaries']['bin/SPONGE']):
            raise ValueError('completed run identity '+replica)
        verify_raw(root,summary['raw_sha256'])
        verify_force_policy(root/'checkpoint.spgr.h5')
        source=campaign/'inputs'/replica/'model';m,state=model.load(source)
        if water.digest(source/'model.json')!=run['model_sha256']:raise ValueError('model identity')
        signature={k:m[k] for k in ('atoms','molecules','box_angstrom','cutoff_angstrom','actual_density_kg_m3',
            'coefficients','angle_k','theta0_radian','torsion_coefficients','bond_length','thermal_dof')}
        signature['topology']={k:v for k,v in m['input_sha256'].items()
            if k not in ('coordinate.txt','velocity.txt','state.npz','lammps.data')}
        if physical is None:physical=signature
        elif physical!=signature:raise ValueError('different physical models cannot be pooled')
        data=gk.load_stress(root/'viscosity.h5');times,stress,temp,volume,metadata=data
        if common_times is None:common_times=times
        else:np.testing.assert_array_equal(common_times,times)
        np.testing.assert_allclose(times,np.arange(0,run['steps']+1,run['sample_interval'])*run['dt_ps'],rtol=1e-12,atol=1e-12)
        rows,curves=replica_curves(times,stress,temp,volume,plan['analysis'])
        records.append(rows);all_curves[replica]=curves
        pressure=load_pressure(root/'viscosity.h5',times)
        states=[]
        for start in plan['analysis']['start_ps']:
            selected=times>=start
            states.append(dict(start_ps=start,mean_temperature_k=float(temp[selected].mean()),
                temperature_sd_k=float(temp[selected].std(ddof=1)),mean_pressure_diagonal_pa=pressure[selected].mean(axis=0).tolist()))
        summaries.append(dict(replica=replica,states=states,engine_wall_seconds=summary['engine_wall_seconds'],
            total_wall_seconds=summary['total_wall_seconds'],energy_max_excursion_kcal_mol=summary['energy_max_excursion_kcal_mol'],
            energy_slope_kcal_mol_ps=summary['energy_slope_kcal_mol_ps'],metadata=metadata,
            model_sha256=run['model_sha256'],run_summary_sha256=water.digest(root/'summary.json'),estimates=rows))
    layout=[(r['setting'],r['window_ps']) for r in records[0]]
    if any([(r['setting'],r['window_ps']) for r in rows]!=layout for rows in records):raise ValueError('setting layout')
    numerators=np.array([[r['numerator_mpa_s_k'] for r in rows] for rows in records])
    temperatures=np.array([[r['temperature_k'] for r in rows] for rows in records])
    estimate,interval,samples,indices=reduce_replicas(numerators,temperatures,
        plan['analysis']['replica_bootstrap_draws'],plan['analysis']['replica_bootstrap_seed'])
    aggregate=[dict(setting=key,window_ps=window,viscosity_mpa_s=float(estimate[i]),
        conditional_replica_95_interval_mpa_s=interval[:,i].tolist(),
        mean_of_individually_normalized_mpa_s=float((numerators[:,i]/temperatures[:,i]).mean()),
        temperature_k=float(temperatures[:,i].mean()),
        relative_interval_halfwidth=float((interval[1,i]-interval[0,i])/(2*abs(estimate[i]))) if estimate[i]!=0 else None)
        for i,(key,window) in enumerate(layout)]
    output.mkdir(parents=True)
    for replica,curves in all_curves.items():
        (output/replica).mkdir()
        for key,array in curves.items():
            np.savetxt(output/replica/(key+'.csv'),array,delimiter=',',comments='',
                header='lag_ps,c_xy_pa2,c_xz_pa2,c_yz_pa2,eta_xy_mpa_s,eta_xz_mpa_s,eta_yz_mpa_s')
    np.savez(output/'bootstrap.npz',indices=indices,viscosity_mpa_s=samples)
    result=dict(status='conditional_finite_window_diagnostics_not_transport_acceptance',
        plan_sha256=water.digest(campaign/'plan.json'),analysis_sha256=water.digest(__file__),
        gk_analysis_sha256=water.digest(gk.__file__),dependencies=dependencies,
        physical_model=physical,replicas=summaries,aggregate=aggregate,
        sampling_wall_seconds=execution['wall_seconds'],analysis_wall_seconds=time.perf_counter()-started,
        formal_viscosity_acceptance=False,cost_to_precision=None,
        limitations=['Shared structural ancestry; four whole-replica bootstrap units, not tensor components.',
            'Same indices preserve covariance across windows, demeaning, discard and decimation settings; no inner block resampling.',
            'Pooled raw integral divided by pooled measured T is diagnostic if replica thermodynamic states differ.',
            'Intervals exclude equilibrium, cutoff-tail, finite-size, timestep, sampling and force-field biases.',
            'No interval-width stopping or automatic equivalence/experimental acceptance.'])
    water.save(output/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=summarize(args.campaign,args.output)
    print(json.dumps(dict(status=result['status'],settings=len(result['aggregate'])),indent=2))
