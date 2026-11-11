#!/usr/bin/env python3
"""Paired cosine pilot with native LAMMPS forces and a matched reference bath.

The reference bath is an explicit Python extension, not native viscosity/cos
thermostatting. It has the same projected splitting but independent noise.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import shutil
import subprocess
import sys
import time

import numpy as np

from lj_model import ROOT, digest, execute, pair_reference
from lj_gk_compare import frozen_inputs, write_json, LM_TIME, SP_TIME, LM_KB, SP_KB
sys.path.insert(0, str(ROOT/'scripts'))
import analyze_cosine as cosine


def driven_initial_state(model, q, centered, h, acceleration):
    force = pair_reference(q, model['box_angstrom'], model['c12'], model['c6'], model['cutoff_angstrom'])[1]
    a_internal = acceleration*10/SP_TIME**2
    force[:, 0] += model['mass_da']*a_internal*np.cos(2*np.pi*q[:, 2]/model['box_angstrom'])
    half = (centered-.5*h*force/model['mass_da']).astype(np.float32).astype(float)
    return half, half+.5*h*force/model['mass_da'], a_internal/LM_TIME**2


def prepare(model_dir, seed_file, output, steps=50000, stride=10, bins=20,
            acceleration=.1, temperature=None, tau_star=1., seed=19575,
            start_star=20., block_star=10., dt_star=None):
    model, q, _, centered, h = frozen_inputs(model_dir, seed_file)
    source_dt_star = model['dt_star']
    # Validate the immutable source model/seed first. The shared centered
    # initial state remains the one canonicalized at its source timestep.
    # Only the new experiment's integration increment changes.
    if dt_star is not None:
        if isinstance(dt_star, bool) or not np.isfinite(dt_star) or dt_star <= 0:
            raise ValueError('integration dt_star must be finite and positive')
        model = dict(model, dt_star=float(dt_star), dt_ps=float(dt_star)*model['tau_ps'])
        h = float(np.float32(np.float32(model['dt_ps'])*np.float32(SP_TIME)))
        if not np.isfinite(h) or h <= 0:
            raise ValueError('integration dt is not representable')
    if any(not isinstance(v, (int, np.integer)) for v in (steps, stride, bins, seed)) or steps < 1 or stride < 1 or steps % stride or bins < 4 or bins > 10000 or seed < 0 or seed > 2**31-1:
        raise ValueError('invalid steps, stride, bins or seed')
    if steps//stride*bins > 2000000:
        raise ValueError('pilot profile size exceeds bound')
    if not np.isfinite(acceleration) or acceleration == 0 or not np.isfinite(tau_star) or tau_star <= 0:
        raise ValueError('nonzero finite acceleration and positive tau required')
    temperature = float(np.float32(model['target_temperature_k'] if temperature is None else temperature))
    if not np.isfinite(temperature) or temperature < 0 or not np.isfinite(start_star) or start_star < 0 or not np.isfinite(block_star) or block_star <= 0:
        raise ValueError('invalid bath or analysis window')
    spacing = stride*model['dt_star']; block = int(np.floor(block_star/spacing+1e-9))
    begin = int(np.ceil(start_star/spacing-1e-9))
    if block < 1 or (steps//stride+1-begin)//block < 2:
        raise ValueError('need at least two complete analysis blocks')
    half, centered, lm_acceleration = driven_initial_state(model, q, centered, h, acceleration)
    tau = float(np.float32(tau_star*model['tau_ps']))
    gamma = np.float32(np.float32(1)/np.float32(tau))/np.float32(SP_TIME)
    damping = float(np.exp(np.float32(-gamma*np.float32(h))))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    sp, lm = output/'sponge', output/'lammps'; sp.mkdir(); lm.mkdir()
    n, box = model['atoms'], model['box_angstrom']
    for name, values in (('coordinate', q), ('velocity', half)):
        tail = f'{box:.17g} {box:.17g} {box:.17g}\n90 90 90\n' if name == 'coordinate' else ''
        (sp/f'{name}.txt').write_text(str(n)+'\n'+'\n'.join(' '.join(f'{v:.17g}' for v in row) for row in values)+'\n'+tail)
    config = dict(md_name='lj_cosine_pair', mode='nvt', pbc=True, dt=model['dt_ps'], step_limit=steps,
        target_temperature=temperature, cutoff=model['cutoff_angstrom'], skin=.3*model['sigma_angstrom'],
        print_zeroth_frame=True, write_mdout_interval=stride, write_information_interval=max(stride, 1000//stride*stride),
        write_trajectory_interval=steps, write_restart_file_interval=0, crd='crd.dat', vel='vel.dat',
        output_h5_observable_path='observables.h5')
    config['PM.MPI_size'] = 1
    for name in ('mass', 'charge'): config[f'{name}_in_file'] = str(Path(model_dir).resolve()/f'{name}.txt')
    config.update(coordinate_in_file='coordinate.txt', velocity_in_file='velocity.txt', LJ_in_file=str(Path(model_dir).resolve()/'lj.txt'))
    write_engine_input(sp/'mdin.spg.toml', '\n'.join(f'{key}={json.dumps(value)}' for key, value in config.items())+
        f'\n[thermostat]\nmode="middle_langevin"\ntau={tau:.17g}\nseed={seed}\n'+
        f'\n[viscosity]\nmethod="cosine"\nacceleration={acceleration:.17g}\nbins={bins}\nsample_interval={stride}\n')
    lines = ['Paired cosine structural seed', '', f'{n} atoms', '1 atom types', '',
        *[f'0 {box:.17g} {axis}lo {axis}hi' for axis in 'xyz'], '', 'Masses', '', f"1 {model['mass_da']:.17g}", '', 'Atoms # atomic', '']
    lines.extend(f'{i+1} 1 '+' '.join(f'{v:.17g}' for v in row) for i, row in enumerate(q))
    lines.extend(['', 'Velocities', ''])
    lines.extend(f'{i+1} '+' '.join(f'{v/LM_TIME:.17g}' for v in row) for i, row in enumerate(centered))
    (lm/'model.data').write_text('\n'.join(lines)+'\n')
    shutil.copy2(Path(__file__).with_name('lammps_projected_middle.py'), lm/'lammps_projected_middle.py')
    reference = dict(atoms=n, mass_da=model['mass_da'], box_angstrom=box, dt_fs=h*LM_TIME,
        damping=damping, temperature_k=temperature*SP_KB/LM_KB, seed=seed+1,
        acceleration_angstrom_fs2=lm_acceleration, acceleration_nm_ps2=lm_acceleration*1e5,
        bins=bins, sample_interval=stride, steps=steps)
    write_json(lm/'projected_middle.json', reference)
    (lm/'in.lammps').write_text(f'''units real
atom_style atomic
boundary p p p
read_data model.data
pair_style lj/cut {model['cutoff_angstrom']:.17g}
pair_coeff 1 1 {model['epsilon_kcal_mol']:.17g} {model['sigma_angstrom']:.17g}
pair_modify shift yes tail no
neighbor {model['sigma_angstrom']*.3:.17g} bin
neigh_modify every 1 delay 0 check yes
timestep {h*LM_TIME:.17g}
fix integrate all python/move lammps_projected_middle.ProjectedMiddle
fix force all accelerate/cos {lm_acceleration:.17g}
compute native all viscosity/cos
fix native_sample all ave/time {stride} 1 {stride} c_native c_native[7] file native-cosine.dat format " %.17g"
thermo {stride}
thermo_style custom step c_native c_native[7]
thermo_modify norm no format float %.17g
run {steps}
''')
    plan = dict(contract='sponge.lj.paired-cosine-projected-middle.v1', model=model, steps=steps,
        sample_interval=stride, bins=bins, acceleration_nm_ps2=acceleration,
        integration=dict(source_dt_star=source_dt_star, effective_dt_star=model['dt_star'],
            override_requested=dt_star is not None, dt_internal=h,
            initial_state='source-validated centered state; driven half-step rebuilt for effective dt'),
        rng_seeds=dict(sponge_philox=seed, reference_pcg64=seed+1),
        temperature_sponge_k=temperature, tau_sponge_ps=tau, damping=damping,
        dt_sponge_ps=model['dt_ps'], dt_lammps_ps=h*LM_TIME/1000,
        start_star=start_star, block_star=block_star, estimators=list(cosine.ESTIMATORS),
        model_path=str(Path(model_dir).resolve()), seed_path=str(Path(seed_file).resolve()),
        model_sha256=digest(Path(model_dir)/'model.json'), seed_sha256=digest(seed_file),
        input_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=digest(__file__), analyzer_sha256=digest(cosine.__file__),
        reference=reference, experiment_kind='zero-noise-splitting-gate' if temperature == 0 else 'thermal-pilot',
        formal_equivalence_eligible=False,
        deterministic_tolerances=dict(amplitude_nm_ps=2e-5, temperature_k=.01, work_kcal_mol=2e-5, heat_kcal_mol=1e-3),
        conventions=['B/2 A/2 O_projected A/2 B/2; COM3+cos; 3N-4 thermal DOF',
            'Native LAMMPS LJ and accelerate/cos forces; custom Python QR-projected integrator, NOT native viscosity/cos thermostat',
            'Force and numerical kBT matched via explicit time/Boltzmann conversions; physical A/time labels retain conversion differences',
            'Reference PCG64 noise independent of SPONGE Philox; no stochastic trajectory identity claim'])
    write_json(output/'plan.json', plan)
    return plan


def analyze_run(output):
    output = Path(output).resolve(); plan = json.loads((output/'plan.json').read_text())
    execution = json.loads((output/'execution.json').read_text())
    if plan['contract'] != 'sponge.lj.paired-cosine-projected-middle.v1' or execution.get('status') != 'engines_completed' or any(execution.get('engines',{}).get(e,{}).get('status') != 'completed' for e in ('sponge','lammps')):
        raise ValueError('unsupported plan or incomplete engines')
    if digest(output/'plan.json') != execution['plan_sha256']: raise ValueError('plan changed')
    for name, expected in (plan['input_sha256'] | execution['raw_output_sha256']).items():
        if digest(output/name) != expected: raise ValueError(f'input/output changed: {name}')
    sp, metadata = cosine.load_profile(output/'sponge/viscosity.h5')
    with np.load(output/'lammps/canonical-profile.npz') as file: lm = {k: file[k] for k in file.files}
    complete = json.loads((output/'lammps/reference-completion.json').read_text())
    expected_steps = np.arange(0, plan['steps']+1, plan['sample_interval'])
    if complete['status'] != 'finalized' or complete['steps'] != plan['steps'] or complete['frames'] != len(expected_steps):
        raise ValueError('reference not finalized')
    for engine, data in dict(sponge=sp,lammps=lm).items():
        if not np.array_equal(data['step'],expected_steps) or not np.allclose(data['time'],expected_steps*plan[f'dt_{engine}_ps'],rtol=1e-12,atol=1e-12):
            raise ValueError('profile phase/cadence mismatch')
    native = np.loadtxt(output/'lammps/native-cosine.dat',ndmin=2)
    if native.shape != (len(expected_steps),3) or not np.array_equal(native[:,0],expected_steps) or not np.all(np.isfinite(native)):
        raise ValueError('invalid native cosine diagnostic')
    native_agreement = bool(np.allclose(native[:,2]*100,lm['velocity_fourier'],rtol=1e-10,atol=1e-12))
    version = next((s for s in (output/'lammps/run.log').read_text().splitlines() if s.startswith('LAMMPS (')), 'unknown')
    initial = dict(expected_version_banner=version.startswith('LAMMPS (29 Aug 2024'), native_fourier_matches_independent_observer=native_agreement,
        population=bool(np.array_equal(sp['bin_count'][0],lm['bin_count'][0])),
        mass=bool(np.allclose(sp['bin_mass'][0],lm['bin_mass'][0],rtol=1e-12,atol=1e-12)),
        momentum=bool(np.allclose(sp['bin_momentum'][0],lm['bin_momentum'][0],rtol=2e-5,atol=2e-5)),
        geometry=bool(np.allclose(sp['length'],lm['length'],rtol=1e-12,atol=0)),
        centered_temperature=bool(np.isclose(sp['thermal_temperature'][0],lm['thermal_temperature'][0]*LM_KB/SP_KB,rtol=2e-5,atol=1e-5)),
        amplitudes=bool(all(np.isclose(sp['velocity_'+e][0],lm['velocity_'+e][0],rtol=2e-5,atol=2e-6) for e in cosine.ESTIMATORS)))
    write_json(output/'initial-state-gate.json',dict(checks=initial,version=version))
    if not all(initial.values()): raise ValueError('cosine initial/native-observer gate failed')
    reports={}; states={}
    for engine,data in dict(sponge=sp,lammps=lm).items():
        np.savez(output/f'{engine}-canonical-profile.npz',**data)
        tau=plan[f'dt_{engine}_ps']/plan['model']['dt_star']
        reports[engine]=[]
        for estimator in plan['estimators']:
            if plan['experiment_kind'] == 'zero-noise-splitting-gate':
                reports[engine].append(dict(estimator=estimator, status='not_a_stationary_viscosity_experiment'))
                continue
            result,profile=cosine.analyze(data,plan['start_star']*tau,plan['block_star']*tau,estimator)
            reports[engine].append(result)
            with (output/f'{engine}-{estimator}-profile.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(profile[0]));writer.writeheader();writer.writerows(profile)
        if plan['experiment_kind'] == 'thermal-pilot':
            measured = reports[engine][0]['mean_thermal_temperature_k']
            target = plan['temperature_sponge_k'] if engine == 'sponge' else plan['reference']['temperature_k']
            states[engine] = dict(target_temperature_k=target, measured_thermal_temperature_k=measured,
                temperature_departure_percent=100*(measured/target-1),
                mean_com_nm_ps=reports[engine][0]['mean_com_nm_per_ps'],
                interpretation='Actual thermal state must be matched before cross-method viscosity comparison; no acceptance threshold inferred from this pilot.')
    differences={name:float(np.max(np.abs(sp[name]-lm[name]))) for name in ('velocity_fourier','velocity_fit','coupling_work')}
    if "thermostat_heat" in sp and "thermostat_heat" in lm:
        differences["thermostat_heat"] = float(np.max(np.abs(sp["thermostat_heat"]-lm["thermostat_heat"])))
    deterministic = None
    if plan['experiment_kind'] == 'zero-noise-splitting-gate':
        limits = plan.get('deterministic_tolerances', dict(amplitude_nm_ps=2e-5, temperature_k=.01, work_kcal_mol=2e-5))
        temperature_error = float(np.max(np.abs(sp['thermal_temperature']-lm['thermal_temperature']*LM_KB/SP_KB)))
        checks = dict(amplitude=all(differences[key] <= limits['amplitude_nm_ps'] for key in ('velocity_fourier','velocity_fit')),
            temperature=temperature_error <= limits['temperature_k'], work=differences['coupling_work'] <= limits['work_kcal_mol'])
        if "heat_kcal_mol" in limits:
            checks["heat"] = "thermostat_heat" in differences and differences["thermostat_heat"] <= limits["heat_kcal_mol"]
        deterministic = dict(checks=checks, tolerances=limits, temperature_error_k=temperature_error,
            thresholds_in_original_plan='deterministic_tolerances' in plan, status='passed' if all(checks.values()) else 'failed')
    report=dict(status='paired_pilot_only', analyses=reports, initial_state_gate=initial,
        reference_completion=complete, sponge_provenance=metadata, execution=execution, state_diagnostics=states,
        series_max_absolute_differences=differences, deterministic_gate=deterministic,
        analysis_script_sha256=digest(__file__),analyzer_sha256=digest(cosine.__file__),
        warnings=['No convergence, steady-state, weak-drive, finite-size or equivalence acceptance.',
            'Same QR-projected splitting in the reference; it is a benchmark extension, not a native LAMMPS thermostat.',
            'Independent RNG algorithms: finite-temperature trajectory differences are expected, not a force-error metric.',
            'Zero-noise tests validate splitting only and have no stationary liquid viscosity interpretation.',
            'All three predeclared estimators retained; do not choose one by agreement.',
            'Reference Python overhead is included; observed speed is not a native LAMMPS performance ranking.'])
    write_json(output/'summary.json',report);return report


def run_pair(model_dir,seed_file,output,sponge,lammps,**options):
    prepare(model_dir,seed_file,output,**options); output=Path(output).resolve()
    binaries=dict(sponge=Path(sponge).resolve(),lammps=Path(lammps).resolve())
    reference_python_home = str(binaries['lammps'].parent.parent)
    execution=dict(status='running',engines={},sequential=True,cpu_threads=1,reference_python_home=reference_python_home,plan_sha256=digest(output/'plan.json'),
        executable_sha256={e:digest(p) for e,p in binaries.items()})
    write_json(output/'execution.json',execution)
    for engine,args in dict(sponge=['-mdin','mdin.spg.toml'],lammps=['-in','in.lammps']).items():
        started = time.perf_counter()
        try:
            command = [binaries[engine], *args]
            if engine == 'lammps': command = ['env', f'PYTHONHOME={reference_python_home}', 'PYTHONNOUSERSITE=1', *command]
            elapsed=execute(command,output/engine,timeout=900)
            execution['engines'][engine]=dict(status='completed',wall_seconds=elapsed)
        except Exception as error:
            execution['status']='failed';execution['engines'][engine]=dict(status='failed',error=str(error),wall_seconds=time.perf_counter()-started,
                available_files_sha256={p.name:digest(p) for p in (output/engine).iterdir() if p.is_file()})
            write_json(output/'execution.json',execution);raise
        execution['engines'][engine]['available_files_sha256'] = {p.name:digest(p) for p in (output/engine).iterdir() if p.is_file()}
        write_json(output/'execution.json',execution)
    execution.update(status='engines_completed',raw_output_sha256={name:digest(output/name) for name in (
        'sponge/viscosity.h5','lammps/canonical-profile.npz','lammps/reference-completion.json','lammps/native-cosine.dat','lammps/run.log')})
    write_json(output/'execution.json',execution); begin=time.perf_counter()
    try: result=analyze_run(output)
    except Exception as error:
        execution.update(analysis_status='failed',analysis_error=str(error),analysis_wall_seconds=time.perf_counter()-begin)
        write_json(output/'execution.json',execution);raise
    execution.update(analysis_status='completed',analysis_wall_seconds=time.perf_counter()-begin)
    write_json(output/'execution.json',execution);result['execution']=execution;write_json(output/'summary.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','seed-file','output','sponge','lammps'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=50000);parser.add_argument('--stride',type=int,default=10)
    parser.add_argument('--bins',type=int,default=20);parser.add_argument('--acceleration',type=float,default=.1)
    parser.add_argument('--dt-star',type=float);parser.add_argument('--temperature',type=float);parser.add_argument('--tau-star',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=19575);parser.add_argument('--start-star',type=float,default=20.)
    parser.add_argument('--block-star',type=float,default=10.);args=parser.parse_args()
    result=run_pair(args.model,args.seed_file,args.output,args.sponge,args.lammps,steps=args.steps,stride=args.stride,
        bins=args.bins,acceleration=args.acceleration,temperature=args.temperature,tau_star=args.tau_star,
        seed=args.seed,start_star=args.start_star,block_star=args.block_star,dt_star=args.dt_star)
    print(json.dumps({e:[{k:r[k] for k in ('estimator','status','viscosity_mpa_s') if k in r} for r in rows] for e,rows in result['analyses'].items()},indent=2))
