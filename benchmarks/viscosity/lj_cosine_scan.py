#!/usr/bin/env python3
"""Predeclare finite-duration cosine sensitivity cases; never pool conditions.

This driver preserves per-case provenance, failures and measured cost. It does
not infer stationarity, independence, zero-drive/long-wave convergence or
formal equivalence from a scan.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

import lj_cosine_compare as paired
from lj_model import ROOT, digest
from lj_gk_compare import write_json, LM_KB, SP_KB

CONTRACT = 'sponge.lj.cosine-sensitivity.v1'


def exact_steps(duration, dt):
    ratio = duration/dt
    if not np.isfinite(ratio) or ratio < 1 or ratio > 2**31-1 or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8):
        raise ValueError('duration/sample interval must be a positive integral number of steps')
    return int(round(ratio))


def validate_spec(spec):
    if set(spec) != {'contract','duration_star','sample_star','start_star','block_star','bins','cases'} or spec['contract'] != CONTRACT:
        raise ValueError('unknown or missing scan specification fields')
    for key in ('duration_star','sample_star','start_star','block_star'):
        value=spec[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value < 0 or (key!='start_star' and value==0):
            raise ValueError('invalid scan time: '+key)
    if type(spec['bins']) is not int or not 4 <= spec['bins'] <= 10000:
        raise ValueError('invalid bin count')
    cases=spec['cases']
    if not isinstance(cases,list) or not 1 <= len(cases) <= 64:
        raise ValueError('need 1..64 explicitly declared cases')
    seen=set(); result=[]
    for case in cases:
        if set(case) != {'id','acceleration','tau_star','dt_star','seed'}:
            raise ValueError('unknown or missing case fields')
        name=case['id']
        if not isinstance(name,str) or not re.fullmatch('[a-z][a-z0-9_-]{0,63}',name) or name in seen:
            raise ValueError('unique safe case id required')
        seen.add(name)
        for key in ('acceleration','tau_star','dt_star'):
            value=case[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value==0 or (key!='acceleration' and value<0):
                raise ValueError('invalid case parameter: '+key)
        if type(case['seed']) is not int or not 0 <= case['seed'] <= 2**31-1:
            raise ValueError('invalid thermostat seed')
        steps=exact_steps(spec['duration_star'],case['dt_star'])
        stride=exact_steps(spec['sample_star'],case['dt_star'])
        if steps % stride or (steps//stride+1)*spec['bins'] > 2000000:
            raise ValueError('invalid cadence or profile too large')
        block=exact_steps(spec['block_star'],spec['sample_star'])
        first=int(np.ceil(spec['start_star']/spec['sample_star']-1e-9))
        if (steps//stride+1-first)//block < 2:
            raise ValueError('at least two complete analysis blocks required')
        result.append(dict(case,steps=steps,stride=stride))
    return result


def rows_for_case(case, plan, report):
    rows=[]
    for engine, analyses in report['analyses'].items():
        for record in analyses:
            eta=record['viscosity_mpa_s']; ci=record['conditional_bootstrap_95_interval_mpa_s']
            target=plan['temperature_sponge_k'] if engine=='sponge' else plan['reference']['temperature_k']
            t=record['mean_thermal_temperature_k']
            rows.append(dict(case_id=case['id'], engine=engine, estimator=record['estimator'],
                acceleration_nm_ps2=plan['acceleration_nm_ps2'] if engine=='sponge' else plan['reference']['acceleration_nm_ps2'],
                tau_star=case['tau_star'],dt_star=case['dt_star'],seed=plan['rng_seeds'][{'sponge':'sponge_philox','lammps':'reference_pcg64'}[engine]],
                atoms=plan['model']['atoms'],density_star=plan['model']['actual_density_star'],
                wave_number_nm_inv=2*np.pi/(plan['model']['box_angstrom']/10),
                target_temperature_k=target,thermal_temperature_k=t,
                actual_duration_ps=plan['steps']*plan[f'dt_{engine}_ps'],
                actual_sample_ps=plan['sample_interval']*plan[f'dt_{engine}_ps'],
                temperature_in_sponge_kbt_units=t*(LM_KB/SP_KB if engine=='lammps' else 1),
                temperature_departure_percent=100*(t/target-1),
                temperature_block_standard_error_k=record['temperature_block_standard_error_k'],
                viscosity_mpa_s=eta,conditional_ci_low_mpa_s=ci[0] if ci else None,
                conditional_ci_high_mpa_s=ci[1] if ci else None,
                relative_ci_halfwidth=(ci[1]-ci[0])/(2*eta) if ci and eta and eta>0 else None,
                response_per_acceleration_ps=record['mean_response_per_acceleration_ps'],
                cosine_mode_peak_shear_rate_ps_inv=record['cosine_mode_peak_shear_rate_ps_inv'],
                profile_relative_residual=record['spatial_fit']['relative_residual_rms'],
                first_sample_ps=record['first_time_ps'],last_sample_ps=record['last_time_ps'],
                sample_weight_duration_ps=record['sample_weight_duration_ps'],
                ledger_endpoint_duration_ps=record['ledger_endpoint_duration_ps'],
                blocks=record['blocks'],work_kcal_mol=record['work_in_window_kcal_per_mol'],
                heat_kcal_mol=record['heat_in_window_kcal_per_mol'],
                engine_wall_seconds=report['execution']['engines'][engine]['wall_seconds'],
                status=record['status']))
    return rows


def summarize(output):
    output=Path(output).resolve()
    plan=json.loads((output/'plan.json').read_text())
    execution=json.loads((output/'execution.json').read_text())
    if plan['contract']!=CONTRACT or digest(output/'plan.json')!=execution['plan_sha256']:
        raise ValueError('scan plan changed')
    for name,expected in plan['source_sha256'].items():
        if digest(output/'source'/name)!=expected: raise ValueError('archived source changed: '+name)
    if digest(output/'source/analyze_cosine.py')!=plan['analyzer_sha256']:
        raise ValueError('archived analyzer changed')
    if digest(output/'source/seed-summary.json')!=plan['seed_summary_sha256']:
        raise ValueError('archived seed summary changed')
    rows=[]; engine_costs=dict(sponge=0.,lammps=0.); analysis_cost=0.
    for case in plan['cases']:
        name=case['id']; completed=execution['cases'].get(name)
        if not completed or completed['status']!='completed':
            raise ValueError('incomplete scan case: '+name)
        directory=output/name
        for path,expected in completed['result_sha256'].items():
            if digest(directory/path)!=expected: raise ValueError('scan result changed: '+name+'/'+path)
        run_plan=json.loads((directory/'plan.json').read_text())
        report=json.loads((directory/'summary.json').read_text())
        run_execution=json.loads((directory/'execution.json').read_text())
        if run_execution['executable_sha256']!=plan['executable_sha256']:
            raise ValueError('case executable differs from declared scan')
        if digest(directory/'plan.json')!=run_execution['plan_sha256']:
            raise ValueError('case plan changed')
        for path,expected in (run_plan['input_sha256'] | run_execution['raw_output_sha256']).items():
            if digest(directory/path)!=expected: raise ValueError('case input/output changed: '+name+'/'+path)
        if run_plan['model_sha256']!=plan['model_sha256'] or run_plan['seed_sha256']!=plan['seed_sha256']:
            raise ValueError('case source differs from scan')
        if run_plan['rng_seeds']!={'sponge_philox':case['seed'],'reference_pcg64':case['seed']+1} or run_plan['tau_sponge_ps']!=float(np.float32(case['tau_star']*run_plan['model']['tau_ps'])):
            raise ValueError('case thermostat differs from declared scan')
        if any(run_plan[key]!=case[value] for key,value in [('acceleration_nm_ps2','acceleration'),('steps','steps'),('sample_interval','stride')]) or run_plan['model']['dt_star']!=case['dt_star']:
            raise ValueError('case condition differs from declared scan')
        rows.extend(rows_for_case(case,run_plan,report))
        for engine in engine_costs: engine_costs[engine]+=run_execution['engines'][engine]['wall_seconds']
        analysis_cost+=run_execution['analysis_wall_seconds']
    result=dict(status='sensitivity_diagnostics_only',cases=len(plan['cases']),rows=rows,
        plan_sha256=digest(output/'plan.json'),
        costs=dict(engine_seconds=engine_costs,pair_analysis_seconds=analysis_cost,
            measured_scan_wall_seconds=execution.get('wall_seconds'),
            upstream_equilibration_seconds=plan['upstream_equilibration_seconds'],
            prior_model_setup_and_development_seconds=None,formal_cost_to_precision=None),
        statistical_unit='one trajectory per case and engine; common structural origin; no independent-replica aggregation',
        warnings=['All conditions and all three estimators retained, including unresolved intervals.',
            'Equal nominal reduced duration/cadence; actual engine time labels differ slightly by historical unit conversion, recorded in each case.',
            'Changing RNG seed alone does not prove loss of initial-state memory or independent stationary replicas.',
            'Temperature block SE and viscosity intervals are conditional on adequate block independence.',
            'No zero-drive, long-wave, timestep, thermostat or 5 percent equivalence acceptance inferred.',
            'LAMMPS timings include custom Python integration/measurement; not native throughput.',
            'Reported scan cost is observed cost, not total cost to a validated viscosity precision.'])
    write_json(output/'summary.json',result)
    with (output/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


def run_campaign(spec,model,seed_file,output,sponge,lammps):
    cases=validate_spec(spec)
    model,seed_file=Path(model).resolve(),Path(seed_file).resolve()
    paired.frozen_inputs(model,seed_file)
    seed_report=json.loads((seed_file.parent/'summary.json').read_text())
    binaries=dict(sponge=Path(sponge).resolve(),lammps=Path(lammps).resolve())
    hashes={e:digest(p) for e,p in binaries.items()}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source_names=['lj_cosine_scan.py','lj_cosine_compare.py','lj_gk_compare.py','lj_model.py','lammps_projected_middle.py']
    (output/'source').mkdir()
    for name in source_names:shutil.copy2(Path(__file__).with_name(name),output/'source'/name)
    shutil.copy2(paired.cosine.__file__,output/'source/analyze_cosine.py')
    shutil.copy2(seed_file.parent/'summary.json',output/'source/seed-summary.json')
    (output/'source/changes.diff').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
    plan=dict(contract=CONTRACT,specification=spec,cases=cases,
        model_path=str(model),seed_path=str(seed_file),model_sha256=digest(model/'model.json'),seed_sha256=digest(seed_file),
        seed_summary_sha256=digest(seed_file.parent/'summary.json'),upstream_equilibration_seconds=seed_report.get('wall_seconds'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={name:digest(Path(__file__).with_name(name)) for name in source_names},
        analyzer_sha256=digest(paired.cosine.__file__),executable_sha256=hashes,
        formal_equivalence_eligible=False)
    write_json(output/'plan.json',plan)
    execution=dict(status='running',cases={},plan_sha256=digest(output/'plan.json'))
    write_json(output/'execution.json',execution);started=time.perf_counter()
    for case in cases:
        name=case['id'];begin=time.perf_counter()
        execution['cases'][name]=dict(status='running');write_json(output/'execution.json',execution)
        try:
            paired.run_pair(model,seed_file,output/name,**binaries,
                steps=case['steps'],stride=case['stride'],bins=spec['bins'],
                acceleration=case['acceleration'],tau_star=case['tau_star'],seed=case['seed'],dt_star=case['dt_star'],
                start_star=spec['start_star'],block_star=spec['block_star'])
        except Exception as error:
            execution['cases'][name]=dict(status='failed',wall_seconds=time.perf_counter()-begin,error=str(error))
            execution.update(status='failed',wall_seconds=time.perf_counter()-started)
            write_json(output/'execution.json',execution);raise
        execution['cases'][name]=dict(status='completed',wall_seconds=time.perf_counter()-begin,
            result_sha256={p:digest(output/name/p) for p in ('plan.json','execution.json','summary.json')})
        write_json(output/'execution.json',execution)
        print('Completed '+name,flush=True)
    execution.update(status='completed',wall_seconds=time.perf_counter()-started)
    write_json(output/'execution.json',execution)
    return summarize(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize',type=Path)
    for name in ('spec','model','seed-file','output','sponge','lammps'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.summarize: result=summarize(args.summarize)
    else:
        if any(getattr(args,k) is None for k in ('spec','model','seed_file','output','sponge','lammps')):parser.error('all run paths required')
        result=run_campaign(json.loads(args.spec.read_text()),args.model,args.seed_file,args.output,args.sponge,args.lammps)
    print(json.dumps(dict(status=result['status'],cases=result['cases'],costs=result['costs']),indent=2))
