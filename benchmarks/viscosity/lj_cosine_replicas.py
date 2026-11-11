#!/usr/bin/env python3
"""Independent-initialization paired cosine campaign and cluster bootstrap.

The statistical unit is one independently initialized/equilibrated replica.
Resample its two engines and all estimators together. No stationarity or
zero-drive/long-wave convergence is inferred from successful execution.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

import lj_model
import lj_pilot
import lj_cosine_compare as paired
from lj_cosine_scan import validate_spec as validate_scan
from lj_gk_compare import write_json, LM_KB, SP_KB

CONTRACT = 'sponge.lj.cosine-replicas.v1'
ENGINES = ('sponge', 'lammps')
MODEL_KEYS = {'cells','density','temperature','epsilon','sigma','mass','cutoff','dt'}
SOURCES = ('lj_cosine_replicas.py','lj_cosine_scan.py','lj_cosine_compare.py',
           'lj_model.py','lj_pilot.py','lj_gk_compare.py','lammps_projected_middle.py')


def identity(model):
    # Deliberately includes integration and thermodynamic target as well as
    # Hamiltonian: matching this is stronger than matching forces alone.
    if model['contract'] != 'sponge.lj.transport-model.v1':
        raise ValueError('unknown model contract')
    fields = {k:v for k,v in model.items() if k not in ('seed','initialization','input_sha256')}
    encoded = json.dumps(fields,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(fields=fields,sha256=hashlib.sha256(encoded).hexdigest())


def validate_spec(spec):
    keys = {'contract','model','equilibration_steps','duration_star','sample_star',
            'start_star','block_star','bins','acceleration','tau_star','replicas',
            'bootstrap_draws','bootstrap_seed'}
    if set(spec) != keys or spec['contract'] != CONTRACT:
        raise ValueError('unknown or missing replica specification fields')
    model = spec['model']
    if set(model) != MODEL_KEYS or type(model['cells']) is not int or not 3 <= model['cells'] <= 12:
        raise ValueError('explicit bounded model parameters required')
    for key in MODEL_KEYS-{'cells'}:
        v=model[key]
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) or v<=0:
            raise ValueError('invalid model parameter: '+key)
    n=4*model['cells']**3
    if 2*model['cutoff'] >= (n/model['density'])**(1/3):
        raise ValueError('cutoff must be below half box')
    eq=spec['equilibration_steps']
    if type(eq) is not int or not 1000<=eq<=1000000 or eq%100:
        raise ValueError('bounded equilibration steps divisible by 100 required')
    for key in ('bootstrap_draws','bootstrap_seed'):
        v=spec[key]
        if type(v) is not int or not 0<=v<=2**31-1:
            raise ValueError('invalid bootstrap parameter')
    if not 1000<=spec['bootstrap_draws']<=100000:
        raise ValueError('need 1000..100000 bootstrap draws')
    replicas=spec['replicas']
    if not isinstance(replicas,list) or not 4<=len(replicas)<=32:
        raise ValueError('need 4..32 predeclared replicas')
    ids=set();seeds=set()
    for rep in replicas:
        if set(rep) != {'id','initialization_seed','equilibration_seed','production_seed'}:
            raise ValueError('unknown or missing replica fields')
        name=rep['id']
        if not isinstance(name,str) or not re.fullmatch('[a-z][a-z0-9_-]{0,63}',name) or name in ids:
            raise ValueError('unique safe replica ids required')
        ids.add(name)
        rng=[rep[k] for k in ('initialization_seed','equilibration_seed','production_seed')]
        if any(type(v) is not int or not 0<=v<2**31-1 for v in rng):
            raise ValueError('invalid replica RNG seed')
        rng.append(rep['production_seed']+1)
        if len(set(rng))!=4 or seeds.intersection(rng):
            raise ValueError('initialization, equilibration and both production RNG seeds must be unique')
        seeds.update(rng)
    scan={k:spec[k] for k in ('duration_star','sample_star','start_star','block_star','bins')}
    scan.update(contract='sponge.lj.cosine-sensitivity.v1',cases=[dict(id='condition',
        acceleration=spec['acceleration'],tau_star=spec['tau_star'],dt_star=model['dt'],seed=0)])
    condition=validate_scan(scan)[0]
    samples=round(spec['block_star']/spec['sample_star'])
    first=int(np.ceil(spec['start_star']/spec['sample_star']-1e-9))
    blocks=(condition['steps']//condition['stride']+1-first)//samples
    if blocks%2: raise ValueError('even complete block count required for equal-duration half diagnostics')
    return condition


def extract_replica(directory):
    directory=Path(directory)
    plan=json.loads((directory/'paired/plan.json').read_text())
    report=json.loads((directory/'paired/summary.json').read_text())
    result={}
    for engine in ENGINES:
        with np.load(directory/f'paired/{engine}-canonical-profile.npz') as data:
            prefactor=float(data['density'][0]*abs(data['acceleration'][0])/data['wave_number'][0]**2)
            sign=float(np.sign(data['acceleration'][0]))
        records=report['analyses'][engine]; result[engine]={}
        if [r['estimator'] for r in records]!=list(paired.cosine.ESTIMATORS):
            raise ValueError('missing or reordered estimators')
        for record in records:
            amplitude=np.array([r['amplitude_nm_per_ps'] for r in record['block_records']])
            temperature=np.array([r['thermal_temperature_k'] for r in record['block_records']])
            if len(amplitude)<2 or not np.isfinite(amplitude).all() or not np.isfinite(temperature).all():
                raise ValueError('invalid replica sufficient statistics')
            if not np.isclose(amplitude.mean(),record['raw_mean_amplitude_nm_per_ps'],rtol=1e-12,atol=1e-15):
                raise ValueError('amplitude/block mismatch')
            # Matched numerical kBT; preserve each engine actual time/A labels.
            t_scale=LM_KB/SP_KB if engine=='lammps' else 1.
            split=len(amplitude)//2
            result[engine][record['estimator']]=dict(
                fluidity_per_mpa_s=float(sign*amplitude.mean()/prefactor),
                signed_amplitude_nm_ps=float(sign*amplitude.mean()),prefactor=prefactor,
                temperature_sponge_k=float(temperature.mean()*t_scale),
                first_half_fluidity_per_mpa_s=float(sign*amplitude[:split].mean()/prefactor),
                second_half_fluidity_per_mpa_s=float(sign*amplitude[split:].mean()/prefactor),
                first_half_temperature_sponge_k=float(temperature[:split].mean()*t_scale),
                second_half_temperature_sponge_k=float(temperature[split:].mean()*t_scale),
                blocks=len(amplitude),samples_per_block=record['samples_per_block'],
                sample_weight_duration_ps=record['sample_weight_duration_ps'],
                actual_duration_ps=plan['steps']*plan[f'dt_{engine}_ps'])
    return result


def reduce_replicas(records, bootstrap, seed):
    """Equal-duration replicas. Average response/known prefactor, then invert.

    One cluster index matrix is shared by all engines and estimators. Never
    concatenate within-trajectory blocks into a fictitious independent sample.
    """
    if type(bootstrap) is not int or not 1000<=bootstrap<=100000 or type(seed) is not int or not 0<=seed<=2**31-1:
        raise ValueError('invalid bootstrap parameters')
    if len(records)<4 or len({r['id'] for r in records})!=len(records):
        raise ValueError('at least four distinct replica ids required')
    indices=np.random.default_rng(seed).integers(0,len(records),(bootstrap,len(records)))
    def ci(values): return np.quantile(values,[.025,.975]).tolist()
    results={}
    for estimator in paired.cosine.ESTIMATORS:
        entries={e:[r['statistics'][e][estimator] for r in records] for e in ENGINES}
        for e in ENGINES:
            for key in ('prefactor','blocks','samples_per_block','sample_weight_duration_ps','actual_duration_ps'):
                if any(r[key]!=entries[e][0][key] for r in entries[e]):
                    raise ValueError('incompatible within-engine replica condition: '+key)
        means={};draws={};out={};temp_draws={}
        for e in ENGINES:
            f=np.array([r['fluidity_per_mpa_s'] for r in entries[e]])
            t=np.array([r['temperature_sponge_k'] for r in entries[e]])
            drift=np.array([r['second_half_fluidity_per_mpa_s']-r['first_half_fluidity_per_mpa_s'] for r in entries[e]])
            tdrift=np.array([r['second_half_temperature_sponge_k']-r['first_half_temperature_sponge_k'] for r in entries[e]])
            if not all(np.isfinite(v).all() for v in (f,t,drift,tdrift)):
                raise ValueError('nonfinite replica statistic')
            means[e]=float(f.mean());draws[e]=f[indices].mean(axis=1)
            limits=ci(draws[e]);resolved=means[e]>0 and limits[0]>0
            temp_draws[e]=t[indices].mean(axis=1)
            out[e]=dict(fluidity_per_mpa_s=means[e],fluidity_95_interval=limits,
                viscosity_mpa_s=1/means[e] if means[e]>0 else None,
                replica_bootstrap_95_interval_mpa_s=[1/limits[1],1/limits[0]] if resolved else None,
                nonpositive_bootstrap_draws=int(np.count_nonzero(draws[e]<=0)),
                temperature_sponge_k=float(t.mean()),temperature_95_interval_k=ci(temp_draws[e]),
                second_minus_first_fluidity_per_mpa_s=float(drift.mean()),
                second_minus_first_fluidity_95_interval=ci(drift[indices].mean(axis=1)),
                second_minus_first_temperature_k=float(tdrift.mean()),
                second_minus_first_temperature_95_interval_k=ci(tdrift[indices].mean(axis=1)),
                replica_values=entries[e],status='conditional_replica_estimate' if resolved else 'unresolved_response')
        # Do not discard a single wrong-sign draw to manufacture a ratio CI.
        # Ratio is a secondary comparison; require every sampled denominator
        # positive, and both mean-response intervals resolved.
        ratio_ok=all(np.all(draws[e]>0) and out[e]['fluidity_95_interval'][0]>0 for e in ENGINES)
        out['paired_comparison']=dict(
            viscosity_ratio_sponge_over_lammps=means['lammps']/means['sponge'] if all(v>0 for v in means.values()) else None,
            viscosity_ratio_95_interval=ci(draws['lammps']/draws['sponge']) if ratio_ok else None,
            temperature_difference_sponge_minus_lammps_k=out['sponge']['temperature_sponge_k']-out['lammps']['temperature_sponge_k'],
            temperature_difference_95_interval_k=ci(temp_draws['sponge']-temp_draws['lammps']),
            formal_equivalence_accepted=False)
        results[estimator]=out
    return results


def manifest(directory):
    return {str(p.relative_to(directory)):lj_model.digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def verify_files(directory, hashes):
    for name,expected in hashes.items():
        if lj_model.digest(directory/name)!=expected:
            raise ValueError('campaign artifact changed: '+name)


def summarize(output):
    output=Path(output).resolve()
    plan=json.loads((output/'plan.json').read_text()); execution=json.loads((output/'execution.json').read_text())
    condition=validate_spec(plan['specification'])
    if plan['contract']!=CONTRACT or lj_model.digest(output/'plan.json')!=execution['plan_sha256']:
        raise ValueError('campaign plan changed')
    verify_files(output/'source',plan['source_sha256'])
    if execution['status']!='completed': raise ValueError('incomplete replica campaign')
    records=[];physical=None;state_hashes=set();costs=dict(initialization_seconds=0.,equilibration_total_seconds=0.,equilibration_engine_seconds=0.,sponge_seconds=0.,lammps_seconds=0.,pair_analysis_seconds=0.,production_total_seconds=0.)
    spec=plan['specification']
    for rep in spec['replicas']:
        name=rep['id'];entry=execution['replicas'].get(name)
        if not entry or entry['status']!='completed': raise ValueError('incomplete replica: '+name)
        directory=output/name;verify_files(directory,entry['files_sha256'])
        m=json.loads((directory/'model/model.json').read_text())
        current=identity(m)
        if physical is None: physical=current
        if physical!=current: raise ValueError('replica physical conditions differ')
        params=spec['model']
        expected_model=dict(seed=rep['initialization_seed'],cells=params['cells'],
            dt_star=params['dt'],temperature_star=params['temperature'],
            requested_density_star=params['density'],epsilon_kcal_mol=params['epsilon'],
            sigma_angstrom=params['sigma'],mass_da=float(np.float32(params['mass'])),
            cutoff_angstrom=float(np.float32(params['cutoff']*params['sigma'])))
        if any(m[k]!=v for k,v in expected_model.items()):
            raise ValueError('model differs from predeclared specification')
        for path in ('model/state.npz','equilibration/structural_seed.npz'):
            value=lj_model.digest(directory/path)
            if value in state_hashes: raise ValueError('duplicated initial or equilibrated state')
            state_hashes.add(value)
        pilot=json.loads((directory/'equilibration/summary.json').read_text())
        rp=json.loads((directory/'paired/plan.json').read_text())
        run=json.loads((directory/'paired/execution.json').read_text())
        if pilot['thermostat_seed']!=rep['equilibration_seed'] or pilot['steps']!=spec['equilibration_steps'] or pilot['executable_sha256']!=plan['executable_sha256']['sponge']:
            raise ValueError('equilibration differs from declared campaign')
        if rp['model_sha256']!=lj_model.digest(directory/'model/model.json') or rp['seed_sha256']!=pilot['seed_sha256']:
            raise ValueError('paired source mismatch')
        if run['executable_sha256']!=plan['executable_sha256'] or run['analysis_status']!='completed' or run['status']!='engines_completed':
            raise ValueError('paired engine/analysis incomplete or binary changed')
        expected=dict(steps=condition['steps'],sample_interval=condition['stride'],bins=spec['bins'],
            acceleration_nm_ps2=spec['acceleration'],start_star=spec['start_star'],block_star=spec['block_star'],
            tau_sponge_ps=float(np.float32(spec['tau_star']*m['tau_ps'])),
            rng_seeds=dict(sponge_philox=rep['production_seed'],reference_pcg64=rep['production_seed']+1))
        if any(rp[k]!=v for k,v in expected.items()): raise ValueError('production condition differs from campaign')
        # Also recheck frozen state/source contracts independently of manifest.
        paired.frozen_inputs(directory/'model',directory/'equilibration/structural_seed.npz')
        statistics=extract_replica(directory)
        records.append(dict(id=name,seeds=rep,model_sha256=rp['model_sha256'],seed_sha256=rp['seed_sha256'],
            equilibration_diagnostics=pilot,statistics=statistics))
        costs['initialization_seconds']+=entry['stage_seconds']['initialization']
        costs['equilibration_total_seconds']+=entry['stage_seconds']['equilibration']
        costs['production_total_seconds']+=entry['stage_seconds']['production']
        costs['equilibration_engine_seconds']+=pilot['wall_seconds']
        for e in ENGINES: costs[e+'_seconds']+=run['engines'][e]['wall_seconds']
        costs['pair_analysis_seconds']+=run['analysis_wall_seconds']
    results=reduce_replicas(records,spec['bootstrap_draws'],spec['bootstrap_seed'])
    costs['campaign_recording_overhead_seconds']=execution['wall_seconds']-sum(costs[k] for k in ('initialization_seconds','equilibration_total_seconds','production_total_seconds'))
    costs.update(component_semantics='stage totals are disjoint; engine and pair-analysis times are nested diagnostic subsets, not additive to stage totals',
                 measured_campaign_wall_seconds=execution['wall_seconds'],
                 aggregation_wall_seconds=execution.get('aggregation_wall_seconds'),prior_development_and_sensitivity_seconds=None,
                 formal_cost_to_precision=None)
    report=dict(status='independent_initialization_replica_diagnostics',contract=CONTRACT,
        plan_sha256=lj_model.digest(output/'plan.json'),physical_condition_identity=physical,
        replicas=records,estimates=results,costs=costs,
        statistical_unit='independent initialization and equilibration; engines and estimators paired by replica id',
        warnings=['Independent random initializations do not prove stationary liquid sampling or removal of lattice memory.',
            'Equal-duration replica mean responses divided by the known engine prefactor are pooled before inversion; no averaging of inverse responses.',
            'Whole replicas are jointly resampled across engines/estimators; no inner block resampling and no dropped wrong-sign draws.',
            'Percentile intervals with few replicas are approximate and conditional; first/second-half diagnostics do not certify stationarity.',
            'All three estimators retained. No finite-size, weak-drive, timestep, thermostat, or formal equivalence acceptance.',
            'Temperature is reported in matched SPONGE kBT units; actual engine time labels retain historical conversion differences.',
            'Costs include these initializations/equilibration/production/analysis only; Python reference overhead prevents native performance ranking.'])
    write_json(output/'summary.json',report)
    return report


def run_campaign(spec,output,sponge,lammps):
    condition=validate_spec(spec)
    binaries={e:Path(p).resolve() for e,p in zip(ENGINES,(sponge,lammps))}
    binary_hashes={e:lj_model.digest(p) for e,p in binaries.items()}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'source').mkdir()
    for name in SOURCES: shutil.copy2(Path(__file__).with_name(name),output/'source'/name)
    shutil.copy2(paired.cosine.__file__,output/'source/analyze_cosine.py')
    (output/'source/changes.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=lj_model.ROOT))
    plan=dict(contract=CONTRACT,specification=spec,executable_sha256=binary_hashes,
        source_sha256=manifest(output/'source'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=lj_model.ROOT,text=True).strip(),
        formal_equivalence_eligible=False)
    write_json(output/'plan.json',plan)
    execution=dict(status='running',replicas={},plan_sha256=lj_model.digest(output/'plan.json'))
    write_json(output/'execution.json',execution);start=time.perf_counter()
    for rep in spec['replicas']:
        name=rep['id'];directory=output/name;directory.mkdir();begin=time.perf_counter()
        entry=dict(status='running',stage='initialization',stage_seconds={});execution['replicas'][name]=entry
        write_json(output/'execution.json',execution)
        try:
            if {e:lj_model.digest(p) for e,p in binaries.items()}!=binary_hashes: raise ValueError('executable changed before replica')
            stage_begin=time.perf_counter()
            lj_model.prepare(directory/'model',seed=rep['initialization_seed'],**spec['model'])
            entry['stage_seconds']['initialization']=time.perf_counter()-stage_begin
            entry['stage']='equilibration';write_json(output/'execution.json',execution)
            stage_begin=time.perf_counter()
            lj_pilot.equilibrate(directory/'model',directory/'equilibration',binaries['sponge'],
                                 steps=spec['equilibration_steps'],seed=rep['equilibration_seed'])
            entry['stage_seconds']['equilibration']=time.perf_counter()-stage_begin
            entry['stage']='production';write_json(output/'execution.json',execution)
            stage_begin=time.perf_counter()
            paired.run_pair(directory/'model',directory/'equilibration/structural_seed.npz',directory/'paired',**binaries,
                steps=condition['steps'],stride=condition['stride'],bins=spec['bins'],acceleration=spec['acceleration'],
                tau_star=spec['tau_star'],seed=rep['production_seed'],start_star=spec['start_star'],block_star=spec['block_star'])
            entry['stage_seconds']['production']=time.perf_counter()-stage_begin
            if {e:lj_model.digest(p) for e,p in binaries.items()}!=binary_hashes: raise ValueError('executable changed during replica')
            entry.update(status='completed',wall_seconds=time.perf_counter()-begin,files_sha256=manifest(directory))
        except Exception as error:
            entry.update(status='failed',error=str(error),wall_seconds=time.perf_counter()-begin,files_sha256=manifest(directory))
            execution.update(status='failed',wall_seconds=time.perf_counter()-start)
            write_json(output/'execution.json',execution);raise
        write_json(output/'execution.json',execution);print('Completed '+name,flush=True)
    execution.update(status='completed',wall_seconds=time.perf_counter()-start)
    write_json(output/'execution.json',execution)
    analysis_start=time.perf_counter()
    try: result=summarize(output)
    except Exception as error:
        execution.update(aggregation_status='failed',aggregation_error=str(error))
        write_json(output/'execution.json',execution);raise
    execution.update(aggregation_status='completed',aggregation_wall_seconds=time.perf_counter()-analysis_start)
    write_json(output/'execution.json',execution)
    result['costs']['aggregation_wall_seconds']=execution['aggregation_wall_seconds']
    write_json(output/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','output','sponge','lammps','summarize'): parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.summarize: report=summarize(args.summarize)
    else:
        if any(getattr(args,k) is None for k in ('spec','output','sponge','lammps')): parser.error('all run paths required')
        report=run_campaign(json.loads(args.spec.read_text()),args.output,args.sponge,args.lammps)
    print(json.dumps(dict(status=report['status'],replicas=len(report['replicas']),costs=report['costs']),indent=2))
