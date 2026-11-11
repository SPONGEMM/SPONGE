#!/usr/bin/env python3
"""Paired LJ Green--Kubo pilots with common analysis and sparse NVE energy audit.

This is a reproducible comparison, not automatic equilibration or equivalence.
It preserves unsuccessful/uncertain results and costs before postprocessing.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import subprocess
import sys
import time

import h5py
import numpy as np

from lj_model import ROOT, digest, execute, pair_reference
sys.path.insert(0, str(ROOT/'scripts'))
import analyze_green_kubo as gk

LM_TIME = 48.88821291  # stable_29Aug2024 real units
SP_TIME = float(np.float32(20.455))
SP_KB = float(np.float32(.00198716))
LM_KB = .0019872067
ATM_PA = 101325.


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def frozen_inputs(model_dir, seed_file):
    model_dir, seed_file = Path(model_dir).resolve(), Path(seed_file).resolve()
    model = json.loads((model_dir/'model.json').read_text())
    if model['contract'] != 'sponge.lj.transport-model.v1':
        raise ValueError('unsupported model contract')
    for name, expected in model['input_sha256'].items():
        if digest(model_dir/name) != expected:
            raise ValueError(f'model input changed: {name}')
    seed_report = json.loads((seed_file.parent/'summary.json').read_text())
    if digest(seed_file) != seed_report['seed_sha256'] or digest(model_dir/'model.json') != seed_report['model_sha256']:
        raise ValueError('seed/model identity mismatch')
    with np.load(seed_file) as seed:
        q = np.asarray(seed['positions'], dtype=np.float32).astype(float)
        u = np.asarray(seed['centered_velocities'], dtype=float)
        box = float(seed['box'])
    if q.shape != (model['atoms'], 3) or u.shape != q.shape or not np.all(np.isfinite(q)) or not np.all(np.isfinite(u)) or box != model['box_angstrom']:
        raise ValueError('invalid structural seed arrays or box')
    # Deliberately define a new NVE experiment with zero COM; no RNG continuation.
    u = u-u.mean(axis=0)
    _, force, _, _ = pair_reference(q, box, model['c12'], model['c6'], model['cutoff_angstrom'])
    h = float(np.float32(np.float32(model['dt_ps'])*np.float32(SP_TIME)))
    half = (u-.5*h*force/model['mass_da']).astype(np.float32).astype(float)
    # Both start from the integer velocity represented by the actual native file.
    centered = half+.5*h*force/model['mass_da']
    return model, q, half, centered, h


def prepare(model_dir, seed_file, output, steps, stride, audit_stride, start_star, block_star, windows_star):
    model, q, half, centered, h = frozen_inputs(model_dir, seed_file)
    if min(steps, stride, audit_stride) < 1 or steps % audit_stride or audit_stride % stride:
        raise ValueError('positive intervals required; audit divides steps and is a multiple of stride')
    if steps//audit_stride > 10000 or steps//stride > 2000000:
        raise ValueError('pilot output exceeds explicit frame bounds; increase intervals or split runs')
    if not np.isfinite(start_star) or start_star < 0 or not np.isfinite(block_star) or block_star <= 0 or not windows_star or not all(np.isfinite(w) and w > 0 for w in windows_star):
        raise ValueError('invalid analysis design')
    # Validate the analysis design before starting either expensive engine.
    spacing_star = stride*model['dt_star']
    lag = int(np.floor(max(windows_star)/spacing_star+1e-9))
    block = int(np.floor(block_star/spacing_star+1e-9))
    begin = int(np.ceil(start_star/spacing_star-1e-9))
    if lag < 1 or block < 2*lag+1 or (steps//stride+1-begin)//block < 2:
        raise ValueError('design needs >=2 blocks with block samples >=2*max_lag+1')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sp, lm = output/'sponge', output/'lammps'
    sp.mkdir(); lm.mkdir()
    n, box = model['atoms'], model['box_angstrom']
    for name, array in (('coordinate', q), ('velocity', half)):
        tail = f'{box:.17g} {box:.17g} {box:.17g}\n90 90 90\n' if name == 'coordinate' else ''
        (sp/f'{name}.txt').write_text(str(n)+'\n'+'\n'.join(' '.join(f'{x:.17g}' for x in row) for row in array)+'\n'+tail)
    config = dict(md_name='lj_gk_pair', mode='nve', pbc=True, dt=model['dt_ps'], step_limit=steps,
                  cutoff=model['cutoff_angstrom'], skin=.3*model['sigma_angstrom'], print_zeroth_frame=True,
                  write_mdout_interval=audit_stride, write_information_interval=audit_stride,
                  write_trajectory_interval=audit_stride, write_restart_file_interval=0,
                  crd='crd.dat', vel='vel.dat', output_h5_observable_path='observables.h5')
    config['PM.MPI_size'] = 1
    for name in ('mass', 'charge'):
        config[f'{name}_in_file'] = str(Path(model_dir).resolve()/f'{name}.txt')
    config.update(coordinate_in_file='coordinate.txt', velocity_in_file='velocity.txt', LJ_in_file=str(Path(model_dir).resolve()/'lj.txt'))
    write_engine_input(sp/'mdin.spg.toml', '\n'.join(f'{k}={json.dumps(v)}' for k, v in config.items())+
                                   f'\n[viscosity]\nmethod="green_kubo"\nsample_interval={stride}\n')
    lines = ['Frozen LJ paired GK structural seed', '', f'{n} atoms', '1 atom types', '',
             *[f'0 {box:.17g} {axis}lo {axis}hi' for axis in 'xyz'], '', 'Masses', '',
             f"1 {model['mass_da']:.17g}", '', 'Atoms # atomic', '']
    lines.extend(f'{i+1} 1 '+' '.join(f'{x:.17g}' for x in row) for i, row in enumerate(q))
    lines.extend(['', 'Velocities', ''])
    lines.extend(f'{i+1} '+' '.join(f'{x/LM_TIME:.17g}' for x in row) for i, row in enumerate(centered))
    (lm/'model.data').write_text('\n'.join(lines)+'\n')
    # shift=yes changes reported energy only, preserving the frozen force model.
    # Explicit pressure compute uses the COM-free kinetic tensor.
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
compute kinetic all temp/com
compute pressure all pressure kinetic
variable volume equal vol
variable potential equal pe
variable kinetic_energy equal 0.5*{3*n-3}*{LM_KB:.17g}*c_kinetic
fix integrate all nve
fix samples all ave/time {stride} 1 {stride} c_kinetic v_volume c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6] v_potential v_kinetic_energy file stress.dat format " %.17g"
thermo {audit_stride}
thermo_style custom step c_kinetic pe v_kinetic_energy
thermo_modify norm no format float %.17g
run {steps}
''')
    plan = dict(contract='sponge.lj.paired-gk-pilot.v1', status='prepared', model=model,
                model_path=str(Path(model_dir).resolve()), model_sha256=digest(Path(model_dir)/'model.json'),
                seed_path=str(Path(seed_file).resolve()), seed_sha256=digest(seed_file),
                steps=steps, sample_interval=stride, audit_interval=audit_stride,
                start_star=start_star, block_star=block_star, windows_star=windows_star,
                dt_sponge_ps=model['dt_ps'], dt_lammps_ps=h*LM_TIME/1000,
                dt_internal=h, nominal_time_relative_difference=h*LM_TIME/(1000*model['dt_ps'])-1,
                temperature_constants=dict(sponge_kb_kcal_mol_k=SP_KB, lammps_kb_kcal_mol_k=LM_KB),
                intended_equivalence_ratio=[.95, 1.05], formal_equivalence_eligible=False,
                initialization='shared q and centered velocity, COM removed before native quantization; structural start, no thermostat/RNG continuation',
                energy_convention='LAMMPS cutoff-shifted with tail off; SPONGE removes C6 tail and subtracts active_pair_count*U(rc) at sparse coordinate audits',
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                script_sha256=digest(__file__), analyzer_sha256=digest(gk.__file__))
    plan['input_sha256'] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'plan.json', plan)
    return plan


def read_lammps(path, steps, stride, dt_ps):
    rows = np.loadtxt(path, comments='#', ndmin=2)
    expected = np.arange(0, steps+1, stride)
    # ave/time with Nrepeat=1 emits step zero for this pinned version.
    if rows.shape != (len(expected), 11) or not np.all(np.isfinite(rows)) or not np.array_equal(rows[:, 0], expected):
        raise ValueError('LAMMPS stress stream shape/cadence/incomplete run mismatch')
    if np.any(rows[:, 1] <= 0) or not np.allclose(rows[:, 2], rows[0, 2], rtol=1e-12, atol=0) or rows[0, 2] <= 0:
        raise ValueError('invalid LAMMPS temperature/volume')
    return dict(step=expected, time=expected*dt_ps, temperature=rows[:, 1], volume=rows[:, 2]/1000,
                pressure=rows[:, 3:9]*ATM_PA, potential_shifted=rows[:, 9], kinetic=rows[:, 10])


def cutoff_audit(q, box, cutoff):
    count, ambiguous = 0, 0
    nearest = float("inf")
    for i in range(len(q)-1):
        delta = q[i]-q[i+1:]
        delta -= box*np.rint(delta/box)
        r = np.sqrt(np.sum(delta*delta, axis=1))
        count += np.count_nonzero(r < cutoff)
        distances = np.abs(r-cutoff)
        nearest = min(nearest, float(distances.min()))
        ambiguous += np.count_nonzero(distances <= 2e-5)
    return int(count), nearest, int(ambiguous)


def energy_diagnostics(steps, potential, kinetic, epsilon, atoms):
    total = np.asarray(potential)+kinetic
    if len(total) < 3 or not np.all(np.isfinite(total)):
        raise ValueError('invalid energy audit')
    scaled = (total-total[0])/(atoms*epsilon)
    slope = np.polyfit(steps, scaled, 1)[0]
    return dict(samples=len(total), first_step=int(steps[0]), last_step=int(steps[-1]),
                energy_change_per_atom_epsilon=float(scaled[-1]),
                energy_range_per_atom_epsilon=float(np.ptp(scaled)),
                linear_drift_per_step_per_atom_epsilon=float(slope),
                status='diagnostic_only_no_timestep_convergence_gate')


def analyze_run(output):
    output = Path(output).resolve()
    plan = json.loads((output/'plan.json').read_text())
    execution = json.loads((output/'execution.json').read_text())
    if execution['status'] != 'engines_completed':
        raise ValueError('both engines must complete before paired analysis')
    if 'plan_sha256' in execution and digest(output/'plan.json')!=execution['plan_sha256']: raise ValueError('run plan changed')
    for name, expected in (plan['input_sha256'] | execution.get('raw_output_sha256',{})).items():
        if digest(output/name) != expected: raise ValueError('run input/output changed')
    model, n = plan['model'], plan['model']['atoms']
    sp, lm = output/'sponge', output/'lammps'
    times, stress, temperature, volume, provenance = gk.load_stress(sp/'viscosity.h5')
    expected = np.arange(0, plan['steps']+1, plan['sample_interval'])
    if len(times) != len(expected) or not np.allclose(times, expected*plan['dt_sponge_ps'], rtol=1e-12, atol=1e-12):
        raise ValueError('SPONGE sample endpoint mismatch')
    lammps = read_lammps(lm/'stress.dat', plan['steps'], plan['sample_interval'], plan['dt_lammps_ps'])
    np.savez(output/'canonical-series.npz', step=expected, sponge_time_ps=times, sponge_stress_pa=stress,
             sponge_temperature_k=temperature, lammps_time_ps=lammps['time'],
             lammps_stress_pa=lammps['pressure'][:, 3:], lammps_temperature_k=lammps['temperature'], volume_nm3=volume)
    audit_steps = np.arange(0, plan['steps']+1, plan['audit_interval'])
    q0 = np.loadtxt(sp/'coordinate.txt', skiprows=1, max_rows=n)
    coords = np.fromfile(sp/'crd.dat', dtype=np.float32).reshape(-1, n, 3).astype(float)
    if len(coords) != len(audit_steps)-1 or not np.all(np.isfinite(coords)):
        raise ValueError('sparse native coordinate audit mismatch')
    audits = np.array([cutoff_audit(q, model['box_angstrom'], model['cutoff_angstrom']) for q in [q0, *coords]])
    counts = audits[:, 0]
    with h5py.File(sp/'observables.h5') as file:
        if gk.text(file['parameters/sponge/output/status'][()]) != 'finalized':
            raise ValueError('energy observable file not finalized')
        group = file['observables/all']
        if not np.array_equal(group['step'][()], audit_steps):
            raise ValueError('energy audit cadence mismatch')
        potential = group['eff_pot/value'][()]
    cutoff_value = model['c12']/model['cutoff_angstrom']**12-model['c6']/model['cutoff_angstrom']**6
    tail = -2*np.pi*n*n*model['c6']/(3*model['box_angstrom']**3*model['cutoff_angstrom']**3)
    shifted = potential-tail-counts*cutoff_value
    select = plan['audit_interval']//plan['sample_interval']
    kinetic = .5*(3*n-3)*SP_KB*temperature[::select]
    sp_energy = energy_diagnostics(audit_steps, shifted, kinetic, model['epsilon_kcal_mol'], n)
    lm_energy = energy_diagnostics(audit_steps, lammps['potential_shifted'][::select], lammps['kinetic'][::select], model['epsilon_kcal_mol'], n)
    np.savetxt(output/'energy-audit.csv', np.column_stack((audit_steps, counts, potential, shifted, kinetic,
                lammps['potential_shifted'][::select], lammps['kinetic'][::select], audits[:, 1:])), delimiter=',',
                header='step,active_pairs,sponge_raw_potential,sponge_shifted_potential,sponge_centered_com_free_ke,lammps_shifted_potential,lammps_com_free_ke,nearest_cutoff_angstrom,ambiguous_pairs_within_2e-5_angstrom', comments='')
    sp_energy['ambiguous_coordinate_frames'] = int(np.count_nonzero(audits[:, 2]))
    sp_energy['max_cutoff_count_uncertainty_per_atom_epsilon'] = float(audits[:, 2].max()*abs(cutoff_value)/(n*model['epsilon_kcal_mol']))
    version = next((line for line in (lm/'run.log').read_text().splitlines() if line.startswith('LAMMPS (')), 'unknown')
    initial_gate = dict(
        pinned_lammps_version=version.startswith('LAMMPS (29 Aug 2024'),
        shear_stress=bool(np.allclose(stress[0], lammps['pressure'][0, 3:], rtol=2e-4, atol=20)),
        shifted_potential=bool(np.isclose(shifted[0], lammps['potential_shifted'][0], rtol=2e-6, atol=2e-5)),
        centered_kinetic=bool(np.isclose(kinetic[0], lammps['kinetic'][0], rtol=2e-6, atol=1e-6)))
    write_json(output/'initial-state-gate.json', dict(checks=initial_gate,
        lammps_version=version, shear_difference_pa=(stress[0]-lammps['pressure'][0, 3:]).tolist(),
        shifted_potential_difference_kcal_mol=float(shifted[0]-lammps['potential_shifted'][0]),
        centered_kinetic_difference_kcal_mol=float(kinetic[0]-lammps['kinetic'][0]),
        tolerance_source='existing LJ static float pressure/energy tolerances; kinetic rtol=2e-6, atol=1e-6 kcal/mol'))
    if not all(initial_gate.values()):
        raise ValueError('paired initial-state gate failed; inspect initial-state-gate.json')
    analyses = {}
    for engine, arrays in dict(sponge=(times, stress, temperature, volume),
                               lammps=(lammps['time'], lammps['pressure'][:, 3:], lammps['temperature'], lammps['volume'])).items():
        dt = plan[f'dt_{engine}_ps']
        tau = dt/model['dt_star']
        records = []
        for index, window in enumerate(plan['windows_star']):
            report, lags, correlation, curves = gk.analyze(*arrays, plan['start_star']*tau,
                window*tau, plan['block_star']*tau)
            records.append(report)
            np.savetxt(output/f'{engine}-window-{index}.csv', np.column_stack((lags, correlation, curves)),
                       delimiter=',', header='lag_ps,cxy_pa2,cxz_pa2,cyz_pa2,eta_xy_pa_s,eta_xz_pa_s,eta_yz_pa_s', comments='')
        analyses[engine] = records
    comparison = []
    for index, window in enumerate(plan['windows_star']):
        a, b = analyses['sponge'][index], analyses['lammps'][index]
        comparison.append(dict(window_star=window,
            observed_ratio=a['viscosity_pa_s']/b['viscosity_pa_s'] if b['viscosity_pa_s'] != 0 else None,
            difference_pa_s=a['viscosity_pa_s']-b['viscosity_pa_s'], status='inconclusive',
            reason='one shared-seed pair; stationarity, truncation bias, independent replicas and precision not established'))
    report = dict(status='paired_pilot_only', analyses=analyses, comparison=comparison,
                  energy_audit=dict(sponge=sp_energy, lammps=lm_energy), initial_state_gate=initial_gate, sponge_provenance=provenance,
                  execution=execution,
                  warnings=['Per-engine bootstrap intervals are conditional on independent blocks; no cross-engine equivalence pass is assigned.',
                            'NVE state temperature is measured, not forced to the input thermostat target.',
                            'The two engines share the initial state; they are not independent replicas of one ensemble.',
                            'The full window grid is retained; selecting a favorable integration window would be invalid.',
                            'Sparse NVE energy audit subtracts the cutoff energy from every interacting pair; coordinates within 2e-5 Angstrom of cutoff flag count ambiguity. This is not a timestep convergence test.',
                            'Costs include these runs and analysis; upstream equilibration and all sensitivity scans are still needed for cost-to-precision.'])
    write_json(output/'summary.json', report)
    return report


def run_pair(model_dir, seed_file, output, sponge, lammps, steps=50000, stride=2, audit_stride=1000,
             start_star=20., block_star=10., windows_star=(.5, 1., 2.)):
    plan = prepare(model_dir, seed_file, output, steps, stride, audit_stride, start_star, block_star, list(windows_star))
    output = Path(output).resolve()
    binaries = dict(sponge=Path(sponge).resolve(), lammps=Path(lammps).resolve())
    execution = dict(status='running', engines={}, cpu_threads=1, sequential=True,plan_sha256=digest(output/'plan.json'),
                     executable_sha256={key: digest(path) for key, path in binaries.items()})
    write_json(output/'execution.json', execution)
    for engine, command in dict(sponge=[binaries['sponge'], '-mdin', 'mdin.spg.toml'],
                                 lammps=[binaries['lammps'], '-in', 'in.lammps']).items():
        begin=time.perf_counter()
        try:
            wall = execute(command, output/engine, timeout=900)
            execution['engines'][engine] = dict(status='completed', wall_seconds=wall)
        except Exception as error:
            execution['status'] = 'failed'
            execution['engines'][engine] = dict(status='failed', error=str(error),wall_seconds=time.perf_counter()-begin)
            write_json(output/'execution.json', execution)
            raise
        write_json(output/'execution.json', execution)
    execution['status'] = 'engines_completed'
    execution['raw_output_sha256']={name:digest(output/name) for name in ('sponge/viscosity.h5','sponge/observables.h5','sponge/crd.dat','lammps/stress.dat','lammps/run.log')}
    write_json(output/'execution.json', execution)
    begin = time.perf_counter()
    try:
        result = analyze_run(output)
    except Exception as error:
        execution['analysis_status'] = 'failed'
        execution['analysis_error'] = str(error)
        execution['analysis_wall_seconds'] = time.perf_counter()-begin
        write_json(output/'execution.json', execution)
        raise
    execution['analysis_status'] = 'completed'
    execution['analysis_wall_seconds'] = time.perf_counter()-begin
    write_json(output/'execution.json', execution)
    result['execution'] = execution
    write_json(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--seed-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sponge', type=Path, required=True)
    parser.add_argument('--lammps', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--sample-interval', type=int, default=2)
    parser.add_argument('--audit-interval', type=int, default=1000)
    parser.add_argument('--start-star', type=float, default=20.)
    parser.add_argument('--block-star', type=float, default=10.)
    parser.add_argument('--windows-star', type=float, nargs='+', default=[.5, 1., 2.])
    args = parser.parse_args()
    result = run_pair(args.model, args.seed_file, args.output, args.sponge, args.lammps, args.steps,
                      args.sample_interval, args.audit_interval, args.start_star, args.block_star, args.windows_star)
    print(json.dumps(dict(status=result['status'], comparison=result['comparison'], energy_audit=result['energy_audit']), indent=2))
