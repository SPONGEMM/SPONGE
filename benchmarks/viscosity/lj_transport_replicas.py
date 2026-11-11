#!/usr/bin/env python3
"""Paired GK/RNEMD replicas reusing a verified independent cosine campaign.

All comparisons are conditional diagnostics, not zero-drive equivalence.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import lj_cosine_replicas as origin
import lj_gk_compare as gk
import lj_rnemd_compare as rn
from lj_cosine_scan import exact_steps
from lj_model import ROOT,digest
from lj_gk_compare import write_json,SP_KB,LM_KB

CONTRACT='sponge.lj.transport-replicas.v1'
ENGINES=('sponge','lammps')


def validate_spec(s,dt):
    if set(s)!={'contract','duration_star','start_star','block_star','gk_sample_star','gk_audit_star','gk_windows_star',
                'reference_gk_window_star','rnemd_exchange','rnemd_bins','rnemd_exclusions','bootstrap_draws','bootstrap_seed'} or s['contract']!=CONTRACT:
        raise ValueError('unknown or missing transport specification fields')
    for k in ('duration_star','block_star','gk_sample_star','gk_audit_star','start_star'):
        v=s[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 or (k!='start_star' and v==0):
            raise ValueError('invalid time specification')
    steps=exact_steps(s['duration_star'],dt);stride=exact_steps(s['gk_sample_star'],dt);audit=exact_steps(s['gk_audit_star'],dt)
    if steps%audit or audit%stride or steps//audit>10000 or steps//stride>2000000: raise ValueError('invalid GK cadence/bound')
    windows=s['gk_windows_star']
    if not isinstance(windows,list) or not 1<=len(windows)<=8 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in windows) or len(set(windows))!=len(windows):
        raise ValueError('unique positive GK windows required')
    if s['reference_gk_window_star'] not in windows: raise ValueError('declare reference window from full grid')
    lag=exact_steps(max(windows),s['gk_sample_star']);block=exact_steps(s['block_star'],s['gk_sample_star'])
    if block<2*lag+1: raise ValueError('GK block too short for correlation window')
    for key in ('rnemd_exchange','rnemd_bins','bootstrap_draws','bootstrap_seed'):
        if type(s[key]) is not int or not 0<=s[key]<=2**31-1: raise ValueError('invalid integer specification')
    exchange=s['rnemd_exchange'];bins=s['rnemd_bins'];exclusions=s['rnemd_exclusions']
    if exchange<1 or steps%exchange or steps//exchange>500000 or bins<6 or bins>10000 or bins%2: raise ValueError('invalid RNEMD cadence/bins')
    if not isinstance(exclusions,list) or not exclusions or any(type(e) is not int or e<0 or bins//2-1-2*e<2 for e in exclusions) or len(set(exclusions))!=len(exclusions): raise ValueError('invalid RNEMD exclusions')
    for spacing in (s['gk_sample_star'],exchange*dt):
        if s['start_star']: exact_steps(s['start_star'],spacing)
        exact_steps(s['block_star'],spacing)
    blocks=exact_steps(s['duration_star']-s['start_star'],s['block_star'])
    if blocks<2 or blocks%2: raise ValueError('even complete block count required')
    if not 1000<=s['bootstrap_draws']<=100000: raise ValueError('invalid bootstrap size')
    return dict(steps=steps,stride=stride,audit_stride=audit)


def extract_gk(directory):
    p=json.loads((directory/'plan.json').read_text());r=json.loads((directory/'summary.json').read_text())
    result={}
    with np.load(directory/'canonical-series.npz') as data:
        for i,window in enumerate(p['windows_star']):
            label=f'gk-{window:g}';result[label]={}
            for e in ENGINES:
                record=r['analyses'][e][i];first=record['skipped_initial_frames'];n=record['samples_per_block'];b=record['blocks'];end=first+b*n
                temperature=data[e+'_temperature_k'][first:end];values=np.array(record['block_viscosity_pa_s'])*1000
                if b%2 or len(temperature)!=n*b or not np.isfinite(values).all(): raise ValueError('invalid GK sufficient statistics')
                if not np.isclose(temperature.mean(),record['temperature_k'],rtol=1e-13): raise ValueError('GK temperature mismatch')
                result[label][e]=dict(kind='gk',numerator_mpa_s_k=float(values.mean()*record['temperature_k']),
                    temperature_k=record['temperature_k'],temperature_scale=LM_KB/SP_KB if e=='lammps' else 1.,
                    per_replica_viscosity_mpa_s=float(values.mean()),
                    components_numerator_mpa_s_k=(np.array(record['component_viscosity_pa_s'])*1000*record['temperature_k']).tolist(),
                    first_numerator=float(values[:b//2].mean()*record['temperature_k']),second_numerator=float(values[b//2:].mean()*record['temperature_k']),
                    first_temperature=float(temperature[:len(temperature)//2].mean()),second_temperature=float(temperature[len(temperature)//2:].mean()),
                    blocks=b,samples_per_block=n,volume_nm3=record['volume_nm3'],actual_window_ps=record['actual_window_ps'],
                    sample_weight_duration_ps=len(temperature)*record['sample_ps'])
    return result


def extract_rn(directory, sample_interval=None):
    """Read original samples, optionally decimate a shared trajectory in memory.

    No interpolation, new MD or changed source plan. All extensive quantities
    and cumulative impulses use the same selected frames.
    """
    p=json.loads((directory/'plan.json').read_text());result={}
    if any(type(p.get(k)) is not int or p[k]<1 for k in ('steps','sample_interval')) or p['steps']%p['sample_interval']:
        raise ValueError('invalid original RNEMD steps/cadence')
    spacing=p['sample_interval'] if sample_interval is None else sample_interval
    if type(spacing) is not int or spacing<1 or spacing%p['sample_interval'] or p['steps']%spacing:
        raise ValueError('analysis cadence must be an integer multiple of source cadence and divide steps')
    if sample_interval is not None:
        for key in ('start_star','block_star'):
            count=p[key]/p['model']['dt_star']/spacing
            if not np.isclose(count,round(count),rtol=0,atol=1e-8):
                raise ValueError('analysis boundaries must align with decimated samples')
    for e in ENGINES:
        with np.load(directory/f'{e}-canonical-profile.npz') as data:
            step=np.arange(0,p['steps']+1,p['sample_interval'])
            if data['time'].shape!=step.shape or not np.allclose(data['time'],step*p[f'dt_{e}_ps'],rtol=1e-12,atol=1e-12):
                raise ValueError('RNEMD canonical time differs from declared cadence')
            if 'step' in data and not np.array_equal(data['step'],step): raise ValueError('RNEMD canonical step mismatch')
            for key in ('area','length','momentum','thermal_temperature','kinetic_temperature'):
                if data[key].shape!=step.shape or not np.isfinite(data[key]).all(): raise ValueError('invalid RNEMD scalar')
            for key in ('area','length'):
                if np.any(data[key]<=0) or not np.allclose(data[key],data[key][0],rtol=1e-12): raise ValueError('RNEMD geometry varies')
            for key in ('bin_mass','bin_momentum'):
                if data[key].shape!=(len(step),p['bins']) or not np.isfinite(data[key]).all(): raise ValueError('invalid RNEMD profile')
            if np.any(data['bin_mass']<0) or np.any(data['momentum']<0) or np.any(np.diff(data['momentum'])<0): raise ValueError('invalid RNEMD mass/impulse')
            if sample_interval is not None:
                factor=spacing//p['sample_interval'];step=step[::factor]
                data={k:data[k][::factor] for k in ('time','area','length','momentum','thermal_temperature','kinetic_temperature','bin_mass','bin_momentum')}
            first=int(np.searchsorted(step,round(p['start_star']/p['model']['dt_star'])))
            intervals=round(p['block_star']/p['model']['dt_star']/spacing)
            if intervals<1: raise ValueError('analysis block shorter than sampling interval')
            blocks=(len(step)-1-first)//intervals;end=first+blocks*intervals
            if blocks<2 or blocks%2: raise ValueError('invalid RNEMD analysis blocks')
            times=data['time'];dt=times[1]-times[0];area=float(data['area'][0]);length=float(data['length'][0])
            weights=np.ones(end-first+1);weights[[0,-1]]=.5
            def sufficient(a,b):
                w=np.ones(b-a+1);w[[0,-1]]=.5
                mass=w@data['bin_mass'][a:b+1]/(b-a);momentum=w@data['bin_momentum'][a:b+1]/(b-a)
                impulse=float(data['momentum'][b]-data['momentum'][a])
                flux=impulse*rn.rn.DA_KG*1e3/(2*area*1e-18*(times[b]-times[a])*1e-12)
                return dict(mass=mass.tolist(),momentum=momentum.tolist(),flux_pa=flux,
                            temperature_k=float(w@data['thermal_temperature'][a:b+1]/(b-a)))
            whole=sufficient(first,end);middle=first+(end-first)//2
            for exclude in p['exclusions']:
                label=f'rnemd-exclude-{exclude}';result.setdefault(label,{})[e]=dict(kind='rnemd',**whole,
                    first=sufficient(first,middle),second=sufficient(middle,end),length_nm=length,exclude=exclude,
                    temperature_scale=LM_KB/SP_KB if e=='lammps' else 1.,blocks=blocks,
                    intervals_per_block=intervals,duration_ps=float(times[end]-times[first]),
                    kinetic_temperature_k=float(weights@data['kinetic_temperature'][first:end+1]/(end-first)))
                if sample_interval is not None:
                    result[label][e].update(analysis_sample_interval=spacing,source_sample_interval=p['sample_interval'])
    return result


def rn_response(mass,momentum,flux,length,exclude):
    """Vectorized pooled-profile signed branch fluidity, in (mPa s)^-1."""
    mass=np.atleast_2d(mass);momentum=np.atleast_2d(momentum);flux=np.atleast_1d(flux)
    bins=mass.shape[1];half=bins//2;z=(np.arange(bins)+.5)*length/bins
    velocity=np.divide(momentum,mass,out=np.full_like(momentum,np.nan,dtype=float),where=mass>0)
    responses=[]
    for sign,indices in ((1,np.arange(1+exclude,half-exclude)),(-1,np.arange(half+1+exclude,bins-exclude))):
        if len(indices)<2: raise ValueError('insufficient fit bins')
        inverse=np.linalg.pinv(np.column_stack((z[indices],np.ones(len(indices)))))
        slopes=velocity[:,indices]@inverse[0]
        responses.append(np.divide(sign*slopes*1e9,flux,out=np.full_like(slopes,np.nan),where=flux>0))
    return np.stack(responses,axis=1)


def quantile(values): return np.quantile(values,[.025,.975],axis=0).tolist()


def estimate(entries,indices):
    kind=entries[0]['kind'];n=len(entries)
    def arr(k): return np.asarray([x[k] for x in entries],dtype=float)
    def mean_draw(x): return x.mean(axis=0),x[indices].mean(axis=1)
    t,td=mean_draw(arr('temperature_k'));scale=entries[0]['temperature_scale']
    extra={};valid=True
    if kind=='gk':
        for key in ('blocks','samples_per_block','volume_nm3','actual_window_ps','sample_weight_duration_ps'):
            if any(x[key]!=entries[0][key] for x in entries): raise ValueError('incompatible GK replica windows')
        numerator,nd=mean_draw(arr('numerator_mpa_s_k'));point=numerator/t;draws=nd/td
        first=arr('first_numerator').mean()/arr('first_temperature').mean();second=arr('second_numerator').mean()/arr('second_temperature').mean()
        extra=dict(mean_of_separately_normalized_replicas_mpa_s=float(arr('per_replica_viscosity_mpa_s').mean()),
            component_viscosity_mpa_s=(arr('components_numerator_mpa_s_k').mean(axis=0)/t).tolist(),
            second_minus_first_viscosity_mpa_s=float(second-first))
        interval=quantile(draws)
    elif kind=='rnemd':
        for key in ('length_nm','exclude','blocks','intervals_per_block','duration_ps'):
            if any(x[key]!=entries[0][key] for x in entries): raise ValueError('incompatible RNEMD replicas')
        mass,md=mean_draw(arr('mass'));momentum,pd=mean_draw(arr('momentum'));flux,fd=mean_draw(arr('flux_pa'))
        point_response=rn_response(mass,momentum,flux,entries[0]['length_nm'],entries[0]['exclude'])[0]
        responses=rn_response(md,pd,fd,entries[0]['length_nm'],entries[0]['exclude'])
        f=point_response.mean();df=responses.mean(axis=1)
        point=1/f if np.isfinite(point_response).all() and np.all(point_response>0) else None
        draws=np.divide(1.,df,out=np.full_like(df,np.nan),where=(responses>0).all(axis=1)&np.isfinite(responses).all(axis=1))
        valid=np.isfinite(responses).all()
        branch_ci=quantile(responses) if valid else None
        common_ci=quantile(df) if valid else None
        interval=[1/common_ci[1],1/common_ci[0]] if valid and common_ci[0]>0 and min(branch_ci[0])>0 else None
        extra=dict(flux_pa=float(flux),branch_fluidity_per_mpa_s=[float(x) if np.isfinite(x) else None for x in point_response],
            branch_fluidity_95_interval=branch_ci,fluidity_95_interval=common_ci,
            nonpositive_branch_draws=np.sum(responses<=0,axis=0).tolist(),undefined_draws=int(np.count_nonzero(~np.isfinite(responses).all(axis=1))),
            aggregate_mass_da=mass.tolist(),aggregate_momentum_da_nm_ps=momentum.tolist())
        halves=[]
        for half in ('first','second'):
            hs=[x[half] for x in entries]
            v=rn_response(np.mean([x['mass'] for x in hs],axis=0),np.mean([x['momentum'] for x in hs],axis=0),np.mean([x['flux_pa'] for x in hs]),entries[0]['length_nm'],entries[0]['exclude'])[0]
            halves.append(float(v.mean()) if np.isfinite(v).all() else None)
        extra['first_second_fluidity_per_mpa_s']=halves
    elif kind=='cosine':
        f,df=mean_draw(arr('fluidity_per_mpa_s'));ci=quantile(df)
        point=1/f if f>0 else None;draws=np.divide(1.,df,out=np.full_like(df,np.nan),where=df>0)
        interval=[1/ci[1],1/ci[0]] if ci[0]>0 else None
        extra=dict(fluidity_per_mpa_s=float(f),fluidity_95_interval=ci,nonpositive_draws=int(np.count_nonzero(df<=0)))
    else: raise ValueError('unknown estimator')
    report=dict(kind=kind,viscosity_mpa_s=float(point) if point is not None else None,
        replica_bootstrap_95_interval_mpa_s=interval,temperature_sponge_k=float(t*scale),
        temperature_95_interval_k=quantile(td*scale),status='conditional_diagnostic' if point is not None and point>0 and interval is not None else 'unresolved_response',**extra)
    return report,draws,td*scale


def reduce_replicas(records,spec):
    n=len(records)
    if n<4 or len({r['id'] for r in records})!=n: raise ValueError('four distinct replicas required')
    methods=list(records[0]['statistics'])
    if any(list(r['statistics'])!=methods for r in records): raise ValueError('method matrix differs across replicas')
    indices=np.random.default_rng(spec['bootstrap_seed']).integers(n,size=(spec['bootstrap_draws'],n))
    results={};draws={};temps={}
    for method in methods:
        results[method]={}
        for e in ENGINES:
            results[method][e],draws[method,e],temps[method,e]=estimate([r['statistics'][method][e] for r in records],indices)
    comparisons=[]
    def compare(a,b):
        da,db=draws[a],draws[b];pa=results[a[0]][a[1]]['viscosity_mpa_s'];pb=results[b[0]][b[1]]['viscosity_mpa_s']
        valid=np.isfinite(da).all() and np.isfinite(db).all() and np.all(da>0) and np.all(db>0)
        comparisons.append(dict(numerator=list(a),denominator=list(b),ratio=pa/pb if pa is not None and pb is not None and pa>0 and pb>0 else None,
            paired_ratio_95_interval=quantile(da/db) if valid else None,
            paired_temperature_difference_95_interval_k=quantile(temps[a]-temps[b]),formal_equivalence_accepted=False))
    for method in methods: compare((method,'sponge'),(method,'lammps'))
    reference=f"gk-{spec['reference_gk_window_star']:g}"
    for e in ENGINES:
        for method in methods:
            if method!=reference: compare((method,e),(reference,e))
    return results,comparisons


def summarize(output):
    output=Path(output).resolve();plan=json.loads((output/'plan.json').read_text());execution=json.loads((output/'execution.json').read_text())
    if plan['contract']!=CONTRACT or digest(output/'plan.json')!=execution['plan_sha256']: raise ValueError('campaign plan changed')
    origin.verify_files(output/'source',plan['source_sha256'])
    if execution['status']!='completed': raise ValueError('incomplete campaign')
    upstream=json.loads((output/'source/upstream-summary.json').read_text());spec=plan['specification'];records=[]
    model=upstream['physical_condition_identity']['fields'];options=validate_spec(spec,model['dt_star'])
    ids=[r['id'] for r in upstream['replicas']]
    if len(ids)<4 or len(set(ids))!=len(ids) or set(execution['replicas'])!=set(ids) or len({r['seed_sha256'] for r in upstream['replicas']})!=len(ids):
        raise ValueError('independent source replica identities differ from execution')
    costs=dict(sponge_seconds=0.,lammps_seconds=0.,pair_analysis_seconds=0.)
    for old in upstream['replicas']:
        name=old['id'];entry=execution['replicas'][name];directory=output/name
        if entry['status']!='completed': raise ValueError('incomplete replica')
        origin.verify_files(directory,entry['files_sha256']);stats={};audits={}
        for method in ('gk','rnemd'):
            path=directory/method;p=json.loads((path/'plan.json').read_text());r=json.loads((path/'summary.json').read_text());run=json.loads((path/'execution.json').read_text())
            if digest(path/'plan.json')!=run.get('plan_sha256'): raise ValueError('method execution plan changed')
            origin.verify_files(path,p['input_sha256'] | run['raw_output_sha256'])
            if run['status']!='engines_completed' or run['analysis_status']!='completed' or run['executable_sha256']!=plan['executable_sha256']: raise ValueError('incomplete/different engine')
            if p['model_sha256']!=old['model_sha256'] or p['seed_sha256']!=old['seed_sha256'] or origin.identity(p['model'])!=upstream['physical_condition_identity']: raise ValueError('model/seed mismatch')
            expected=dict(steps=options['steps'],start_star=spec['start_star'],block_star=spec['block_star'])
            expected.update(dict(sample_interval=options['stride'],audit_interval=options['audit_stride'],windows_star=spec['gk_windows_star']) if method=='gk' else dict(exchange_interval=spec['rnemd_exchange'],bins=spec['rnemd_bins'],exclusions=spec['rnemd_exclusions']))
            if any(p[k]!=v for k,v in expected.items()): raise ValueError('method condition differs from plan')
            if not all(r['initial_state_gate'].values()): raise ValueError('initial state gate failed')
            stats.update(extract_gk(path) if method=='gk' else extract_rn(path));audits[method]=r['energy_audit']
            for e in ENGINES: costs[e+'_seconds']+=run['engines'][e]['wall_seconds']
            costs['pair_analysis_seconds']+=run['analysis_wall_seconds']
        for estimator in origin.paired.cosine.ESTIMATORS:
            stats['cosine-'+estimator]={e:dict(kind='cosine',fluidity_per_mpa_s=old['statistics'][e][estimator]['fluidity_per_mpa_s'],
                temperature_k=old['statistics'][e][estimator]['temperature_sponge_k'],temperature_scale=1.) for e in ENGINES}
        records.append(dict(id=name,model_sha256=old['model_sha256'],seed_sha256=old['seed_sha256'],statistics=stats,energy_audits=audits))
    estimates,comparisons=reduce_replicas(records,spec)
    costs.update(measured_campaign_wall_seconds=execution['wall_seconds'],aggregation_wall_seconds=execution.get('aggregation_wall_seconds'),
        upstream_shared_equilibration_and_cosine_campaign_seconds=upstream['costs']['measured_campaign_wall_seconds'],
        upstream_aggregation_seconds=upstream['costs']['aggregation_wall_seconds'],formal_cost_to_precision=None,
        semantics='current engine/analysis times are nested subsets of campaign wall time; shared upstream campaign counted once')
    report=dict(contract=CONTRACT,status='paired_cross_method_replica_diagnostics',plan_sha256=digest(output/'plan.json'),replicas=records,
        estimates=estimates,comparisons=comparisons,costs=costs,
        warnings=['Whole-replica common bootstrap indices preserve engine, method and window covariance. No inner block resampling.',
        'GK pools unnormalized correlation integrals before dividing by pooled measured temperature; separately normalized mean remains diagnostic.',
        'RNEMD pools extensive profiles and flux before fitting both signed branches; no mean of inverse gradients.',
        'All predeclared windows/exclusions/estimators retained. Few replicas give only approximate percentile intervals.',
        'No state-matching, stationarity, long-wave, weak-drive, timestep or formal equivalence acceptance.',
        'Thermal temperature definitions differ between global COM, binned flow and cosine projection; finite-system effects remain.',
        'Cosine reference includes custom Python dynamics; GK/RNEMD use native LAMMPS integration. Cost is not native-engine throughput.',
        'Energy audit is same-time shifted potential plus full COM-free kinetic energy, not just exchange conservation error. No dt-convergence gate.'])
    write_json(output/'summary.json',report);return report


def run_campaign(spec,upstream,output,sponge,lammps):
    upstream=Path(upstream).resolve();old=origin.summarize(upstream);options=validate_spec(spec,old['physical_condition_identity']['fields']['dt_star'])
    binaries=dict(sponge=Path(sponge).resolve(),lammps=Path(lammps).resolve());hashes={e:digest(p) for e,p in binaries.items()}
    old_plan=json.loads((upstream/'plan.json').read_text())
    if hashes!=old_plan['executable_sha256']: raise ValueError('cross-method binaries must match source cosine campaign')
    if any(spec[k]!=old_plan['specification'][k] for k in ('duration_star','start_star','block_star')): raise ValueError('nominal production/analysis duration must match source cosine')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'source').mkdir()
    modules=('lj_transport_replicas.py','lj_cosine_replicas.py','lj_cosine_scan.py','lj_cosine_compare.py','lj_gk_compare.py','lj_rnemd_compare.py','lj_model.py','lj_pilot.py')
    for name in modules: shutil.copy2(Path(__file__).with_name(name),output/'source'/name)
    for name in ('analyze_green_kubo.py','analyze_rnemd.py','analyze_cosine.py'): shutil.copy2(ROOT/'scripts'/name,output/'source'/name)
    for name in ('plan','execution','summary'): shutil.copy2(upstream/(name+'.json'),output/'source'/('upstream-'+name+'.json'))
    (output/'source/changes.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    plan=dict(contract=CONTRACT,specification=spec,upstream_path=str(upstream),source_sha256=origin.manifest(output/'source'),
        executable_sha256=hashes,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),formal_equivalence_eligible=False)
    write_json(output/'plan.json',plan);execution=dict(status='running',replicas={},plan_sha256=digest(output/'plan.json'));write_json(output/'execution.json',execution)
    start=time.perf_counter()
    for rep in old['replicas']:
        name=rep['id'];directory=output/name;directory.mkdir();entry=dict(status='running',stage='copy_frozen_inputs');execution['replicas'][name]=entry;begin=time.perf_counter()
        write_json(output/'execution.json',execution)
        try:
            if {e:digest(p) for e,p in binaries.items()}!=hashes: raise ValueError('executable changed')
            shutil.copytree(upstream/name/'model',directory/'model');(directory/'seed').mkdir()
            for file in ('structural_seed.npz','summary.json'): shutil.copy2(upstream/name/'equilibration'/file,directory/'seed'/file)
            entry['stage']='gk';write_json(output/'execution.json',execution)
            gk.run_pair(directory/'model',directory/'seed/structural_seed.npz',directory/'gk',**binaries,**options,
                start_star=spec['start_star'],block_star=spec['block_star'],windows_star=spec['gk_windows_star'])
            entry['stage']='rnemd';write_json(output/'execution.json',execution)
            rn.run_pair(directory/'model',directory/'seed/structural_seed.npz',directory/'rnemd',**binaries,steps=options['steps'],
                exchange=spec['rnemd_exchange'],bins=spec['rnemd_bins'],start_star=spec['start_star'],block_star=spec['block_star'],exclusions=spec['rnemd_exclusions'])
            if {e:digest(p) for e,p in binaries.items()}!=hashes: raise ValueError('executable changed')
            entry.update(status='completed',wall_seconds=time.perf_counter()-begin,files_sha256=origin.manifest(directory))
        except Exception as error:
            entry.update(status='failed',error=str(error),wall_seconds=time.perf_counter()-begin,files_sha256=origin.manifest(directory))
            execution.update(status='failed',wall_seconds=time.perf_counter()-start);write_json(output/'execution.json',execution);raise
        write_json(output/'execution.json',execution);print('Completed '+name,flush=True)
    execution.update(status='completed',wall_seconds=time.perf_counter()-start);write_json(output/'execution.json',execution);begin=time.perf_counter()
    try: result=summarize(output)
    except Exception as error:
        execution.update(aggregation_status='failed',aggregation_error=str(error));write_json(output/'execution.json',execution);raise
    execution.update(aggregation_status='completed',aggregation_wall_seconds=time.perf_counter()-begin);write_json(output/'execution.json',execution)
    result['costs']['aggregation_wall_seconds']=execution['aggregation_wall_seconds'];write_json(output/'summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','upstream','output','sponge','lammps','summarize'): parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.summarize: result=summarize(args.summarize)
    else:
        if any(getattr(args,k) is None for k in ('spec','upstream','output','sponge','lammps')): parser.error('all run paths required')
        result=run_campaign(json.loads(args.spec.read_text()),args.upstream,args.output,args.sponge,args.lammps)
    print(json.dumps(dict(status=result['status'],replicas=len(result['replicas']),costs=result['costs']),indent=2))
