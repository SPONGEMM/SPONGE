#!/usr/bin/env python3
"""Preregistered whole-replica reduction of a completed hexane cosine campaign.

Signed fluidity is the primary sampled quantity. No wrong-sign sample is
discarded; inverse samples including infinities are retained in NPZ. A
reference ratio interval is unavailable if any inverse draw is nonpositive
or nonfinite. These are conditional diagnostics, not transport acceptance.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import experimental_reference as reference
import hexane_model as model
import lj_model
import water_model as water

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import analyze_cosine as cosine


def read(path):
    return json.loads(Path(path).read_text())


def quantile(values):
    return np.quantile(values, [.025, .975], axis=0).tolist()


def analysis_sources():
    return {**{('scripts/' if m is cosine else 'benchmarks/viscosity/')+Path(m.__file__).name:
        Path(m.__file__).resolve() for m in (reference,model,lj_model,water,cosine)},
        'benchmarks/viscosity/hexane_cosine_analysis.py':Path(__file__).resolve()}


def freeze(root):
    """Write a separate analysis snapshot without changing the sampling plan."""
    root=Path(root).resolve();plan=read(root/'plan.json');sources=analysis_sources()
    if (root/'analysis-readiness.json').exists() or (root/'analysis-sources').exists():
        raise ValueError('analysis snapshot already exists; never overwrite it')
    hashes={name:water.digest(path) for name,path in sources.items()}
    for name,sha in hashes.items():
        if name!='benchmarks/viscosity/hexane_cosine_analysis.py' and plan['frozen_sha256'].get('sources/'+name)!=sha:
            raise ValueError('analysis sampling dependency differs: '+name)
    for name,source in sources.items():
        dest=root/'analysis-sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    execution=read(root/'execution.json')
    ready=dict(plan_sha256=water.digest(root/'plan.json'),source_sha256=hashes,
        frozen_utc=datetime.now(timezone.utc).isoformat(),sampling_status_at_freeze=execution['status'])
    (root/'analysis-readiness.json').write_text(json.dumps(ready,indent=2)+'\n')
    verify_analysis_sources(root,plan)
    return ready


def verify_analysis_sources(root,plan):
    ready=read(root/'analysis-readiness.json');sources=analysis_sources()
    if ready['plan_sha256']!=water.digest(root/'plan.json') or set(ready['source_sha256'])!=set(sources):
        raise ValueError('analysis readiness plan/source set mismatch')
    for name,actual in sources.items():
        sha=ready['source_sha256'][name]
        if water.digest(actual)!=sha or water.digest(root/'analysis-sources'/name)!=sha:
            raise ValueError('frozen analysis source changed: '+name)
        if name!='benchmarks/viscosity/hexane_cosine_analysis.py' and plan['frozen_sha256'].get('sources/'+name)!=sha:
            raise ValueError('analysis sampling dependency differs: '+name)
    return ready['source_sha256']


def edge_difference(values):
    """Equal-duration first/last block groups, omitting an odd middle block."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('two or more finite blocks required')
    half = len(values)//2
    return float(values[-half:].mean()-values[:half].mean())


def reduce(fluidity, temperature, pressure, density, draws=10000, seed=80301):
    """Arrays are (whole replica, setting); same indices across every column."""
    f, t, p = [np.asarray(v, dtype=float) for v in (fluidity, temperature, pressure)]
    if (f.ndim != 2 or f.shape[0] < 4 or not f.shape[1] or t.shape != f.shape or p.shape != f.shape
            or not all(np.isfinite(v).all() for v in (f,t,p)) or np.any(t <= 0)
            or isinstance(draws,bool) or int(draws) != draws or draws < 100):
        raise ValueError('finite matching replica-by-setting arrays and >=100 draws required')
    indices = np.random.default_rng(seed).integers(0,len(f),(int(draws),len(f)))
    fs, ts, ps = [v[indices].mean(axis=1) for v in (f,t,p)]
    # Keep signed/zero draws. NPZ records their IEEE reciprocals and a mask;
    # JSON reports null instead of manufacturing a finite confidence interval.
    with np.errstate(divide='ignore',invalid='raise'):
        eta = 1/fs
    refs = np.array([[reference.n_hexane_viscosity_pa_s(float(x),density)*1000 for x in row] for row in ts])
    ratio = eta/refs
    out = []
    for j in range(f.shape[1]):
        mean = float(f[:,j].mean()); limits = np.quantile(fs[:,j],[.025,.975])
        resolved = mean > 0 and limits[0] > 0
        point_ref = reference.reference('n_hexane',float(t[:,j].mean()),density)
        all_positive = bool(np.all(fs[:,j] > 0))
        out.append(dict(fluidity_per_mpa_s=mean,fluidity_95_interval=limits.tolist(),
            viscosity_mpa_s=1/mean if mean>0 else None,
            viscosity_95_interval_mpa_s=[1/limits[1],1/limits[0]] if resolved else None,
            nonpositive_draws=int(np.count_nonzero(fs[:,j]<=0)),zero_draws=int(np.count_nonzero(fs[:,j]==0)),
            status='conditional_replica_estimate' if resolved else 'unresolved_response',
            temperature_k=float(t[:,j].mean()),temperature_95_interval_k=quantile(ts[:,j]),
            pressure_mpa=float(p[:,j].mean()),pressure_95_interval_mpa=quantile(ps[:,j]),
            reference=point_ref,reference_viscosity_mpa_s=point_ref['viscosity_pa_s']*1000,
            ratio_to_reference=1/mean/(point_ref['viscosity_pa_s']*1000) if mean>0 else None,
            ratio_to_reference_95_interval=quantile(ratio[:,j]) if resolved and all_positive else None,
            all_draws_positive=all_positive,formal_transport_acceptance=False))
    return out,dict(indices=indices,fluidity_per_mpa_s=fs,temperature_k=ts,pressure_mpa=ps,
        viscosity_mpa_s=eta,reference_viscosity_mpa_s=refs,ratio_to_reference=ratio,
        positive_finite_inverse=np.isfinite(eta)&(eta>0))


def verify_case(root, rep, drive, plan, completed):
    name=rep['id']+'/'+drive['id']; case=root/'runs'/name; source=root/'inputs'/rep['id']/'model'
    summary=read(case/'summary.json'); cp=read(case/'plan.json'); static=root/'static'/rep['id']
    if (summary['status']!='passed_cosine_numerical_pilot_not_transport_acceptance'
            or summary['formal_viscosity_acceptance'] is not False
            or summary['plan_sha256']!=water.digest(case/'plan.json')
            or completed[name]['summary_sha256']!=water.digest(case/'summary.json')):
        raise ValueError('completed case identity: '+name)
    expected=dict(duration_ps=plan['duration_ps'],dt_ps=plan['dt_ps'],tau_ps=plan['tau_ps'],
        stride=round(plan['native_ps']/plan['dt_ps']),sample_interval=plan['sample_interval'],
        steps=plan['steps_per_case'],thermostat_seed=rep['thermostat_seeds'][drive['id']],
        acceleration_nm_ps2=drive['acceleration_nm_ps2'],model_sha256=water.digest(source/'model.json'),
        sponge_sha256=water.digest(root/'SPONGE'),probe_sha256=water.digest(root/'test_constrained_nve'),
        static_plan_sha256=water.digest(static/'plan.json'),static_summary_sha256=water.digest(static/'summary.json'),
        source_sha256=water.digest(root/'sources/benchmarks/viscosity/hexane_cosine.py'),
        initial_state_sha256=water.digest(case/'initial-state.npz'),mdin_sha256=water.digest(case/'mdin.spg.toml'))
    if any(cp.get(k)!=v for k,v in expected.items()):raise ValueError('case condition mismatch: '+name)
    for filename,sha in summary['raw_sha256'].items():
        if Path(filename).name!=filename or water.digest(case/filename)!=sha:
            raise ValueError('case raw identity: '+name+'/'+filename)
    d,metadata=cosine.load_profile(case/'viscosity.h5')
    np.testing.assert_array_equal(d['step'],np.arange(0,cp['steps']+1,cp['sample_interval']))
    np.testing.assert_allclose(d['time'],d['step']*cp['dt_ps'],rtol=1e-12,atol=1e-12)
    if not metadata['constraint_configuration'].startswith('rattle_flow_double_v2;'):
        raise ValueError('flow version mismatch')
    m,state=model.load(source)
    if cp['bath_temperature_k']!=m['temperature_k']:raise ValueError('bath mismatch')
    np.testing.assert_allclose(d['density']*1000,m['actual_density_kg_m3'],rtol=1e-7)
    np.testing.assert_allclose(d['acceleration'],drive['acceleration_nm_ps2'],rtol=1e-12)
    with np.load(case/'initial-state.npz') as initial:
        np.testing.assert_array_equal(initial['canonical_position'],state['canonical_position'])
        np.testing.assert_array_equal(initial['target_physical_velocity'],state['physical_velocity'])
    return d,summary


def extract(data, start, plan, estimator):
    report,profile=cosine.analyze(data,start,plan['block_ps'],estimator,
        plan['block_bootstrap_draws'],plan['block_bootstrap_seed'])
    expected=round((plan['duration_ps']-start)/plan['block_ps'])
    if report['blocks']!=expected or report['dropped_tail_frames']!=1:
        raise ValueError('predeclared complete block/tail count mismatch')
    blocks=report['block_records']; pressures=[]
    for block in blocks:
        chosen=(data['time']>=block['first_time_ps'])&(data['time']<=block['last_time_ps'])
        if np.count_nonzero(chosen)!=report['samples_per_block']:raise ValueError('block alignment')
        pressures.append(float(((data['pxx']+data['pyy']+data['pzz'])[chosen]/3e6).mean()))
    prefactor=float(data['density'][0]*abs(data['acceleration'][0])/data['wave_number'][0]**2)
    signed=np.sign(data['acceleration'][0])*np.array([b['amplitude_nm_per_ps'] for b in blocks])
    temperature=np.array([b['thermal_temperature_k'] for b in blocks])
    return dict(prefactor=prefactor,fluidity_per_mpa_s=float(signed.mean()/prefactor),
        temperature_k=float(temperature.mean()),pressure_mpa=float(np.mean(pressures)),
        edge_duration_ps=(len(blocks)//2)*plan['block_ps'],
        last_minus_first_fluidity=edge_difference(signed)/prefactor,
        last_minus_first_temperature_k=edge_difference(temperature),
        last_minus_first_pressure_mpa=edge_difference(pressures),
        pressure_blocks_mpa=pressures,report=report,spatial_profile=profile)


def run(root,output):
    started=time.perf_counter();root=Path(root).resolve();output=Path(output).resolve()
    plan=read(root/'plan.json');execution=read(root/'execution.json')
    if (plan['contract']!='sponge.hexane.cosine-replicas.v1'
            or execution['status']!='completed_sampling_pending_analysis'
            or execution['plan_sha256']!=water.digest(root/'plan.json') or execution['failures'] or execution['active']):
        raise ValueError('completed frozen campaign required')
    for name,sha in plan['frozen_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or water.digest(path)!=sha:raise ValueError('frozen file changed: '+name)
    if plan['estimators']!=list(cosine.ESTIMATORS) or len(plan['drives'])!=2:
        raise ValueError('all declared estimators and two drives required')
    ids=[r['id'] for r in plan['replicas']]
    if len(ids)<4 or len(set(ids))!=len(ids):raise ValueError('four unique whole replicas required')
    completed={r['case']:r for r in execution['results']}
    names={r['id']+'/'+d['id'] for r in plan['replicas'] for d in plan['drives']}
    if set(completed)!=names or len(execution['results'])!=len(names):raise ValueError('missing/duplicate completed case')
    dependencies=verify_analysis_sources(root,plan)
    settings=[dict(start_ps=start,estimator=e,drive=d['id']) for start in plan['starts_ps']
              for e in plan['estimators'] for d in plan['drives']]
    replicas=[];seen=set();density=None
    for rep in plan['replicas']:
        source=root/'inputs'/rep['id']/'model';m,_=model.load(source)
        identity=water.digest(source/'state.npz')
        if identity in seen:raise ValueError('duplicate prepared state')
        seen.add(identity)
        if density is not None and density!=m['actual_density_kg_m3']:raise ValueError('density mismatch')
        density=m['actual_density_kg_m3'];data={};audits={}
        for drive in plan['drives']:
            data[drive['id']],audits[drive['id']]=verify_case(root,rep,drive,plan,completed)
        entries=[extract(data[s['drive']],s['start_ps'],plan,s['estimator']) for s in settings]
        replicas.append(dict(id=rep['id'],state_sha256=identity,entries=entries,audits=audits))
    arrays=[np.array([[e[key] for e in r['entries']] for r in replicas])
            for key in ('fluidity_per_mpa_s','temperature_k','pressure_mpa')]
    estimates,samples=reduce(*arrays,density,plan['replica_bootstrap_draws'],plan['replica_bootstrap_seed'])
    pairs=[]
    for j in range(0,len(settings),2):
        lo,hi=samples['fluidity_per_mpa_s'][:,j:j+2].T
        good=all(estimates[k]['all_draws_positive'] and estimates[k]['fluidity_95_interval'][0]>0 for k in (j,j+1))
        means=[estimates[k]['fluidity_per_mpa_s'] for k in (j,j+1)]
        with np.errstate(divide='ignore',invalid='ignore'):ratios=lo/hi
        samples['paired_fluidity_ratio_'+str(j//2)]=ratios
        pairs.append(dict(start_ps=settings[j]['start_ps'],estimator=settings[j]['estimator'],
            low_drive=settings[j]['drive'],high_drive=settings[j+1]['drive'],
            fluidity_ratio_low_over_high=means[0]/means[1] if min(means)>0 else None,
            fluidity_ratio_95_interval=quantile(ratios) if good else None,
            all_draws_positive=good,linear_response_accepted=False))
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'bootstrap.npz',**samples)
    result=dict(status='conditional_four_replica_two_drive_diagnostics',formal_transport_acceptance=False,
        plan_sha256=water.digest(root/'plan.json'),execution_sha256=water.digest(root/'execution.json'),
        analysis_readiness_sha256=water.digest(root/'analysis-readiness.json'),
        source_sha256=water.digest(__file__),dependencies=dependencies,density_kg_m3=density,
        bootstrap_sha256=water.digest(output/'bootstrap.npz'),statistical_unit=plan['statistical_unit'],
        replicas=replicas,settings=[{**s,**e} for s,e in zip(settings,estimates)],paired_drives=pairs,
        costs=dict(campaign_wall_seconds=execution['total_wall_seconds'],
            summed_engine_seconds=sum(r['engine_wall_seconds'] for r in execution['results']),
            analysis_wall_seconds=time.perf_counter()-started,cost_to_precision=None),
        limitations=['Four shared-ancestry preparations; independent RNG does not prove equilibrium independence.',
            'All windows and estimators retained; no steady-state, zero-drive, size, timestep or thermostat acceptance.',
            'GK actual temperatures differ; shared initial q/u is not matched-ensemble equivalence.',
            'Reference correlation uses same T/rho but no EOS, pressure/phase check or reference uncertainty.',
            'Zero/negative fluidity draws retained; inverse draws may be nonfinite in NPZ, never dropped.',
            'No energy error acceptance threshold; numerical audit pass is not scientific acceptance.'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--output');p.add_argument('--freeze',action='store_true')
    a=p.parse_args()
    if a.freeze:
        if a.output:p.error('--freeze does not accept --output')
        r=freeze(a.campaign);print(json.dumps(r,indent=2))
    else:
        if not a.output:p.error('--output required for analysis')
        r=run(a.campaign,a.output)
        print(json.dumps(dict(status=r['status'],settings=len(r['settings']),costs=r['costs']),indent=2))
