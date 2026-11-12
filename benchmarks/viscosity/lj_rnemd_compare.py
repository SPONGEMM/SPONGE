#!/usr/bin/env python3
"""Same-model RNEMD pilots; explicitly aligned pre-exchange profile/ledger phase."""
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

from lj_model import ROOT, digest, execute
from lj_gk_compare import frozen_inputs, write_json, LM_TIME, LM_KB, SP_KB, cutoff_audit, energy_diagnostics
sys.path.insert(0, str(ROOT/'scripts'))
import analyze_rnemd as rn


def lammps_profile(path, ledger_path, *, steps, stride, bins, atoms, mass, box, dt_ps,
                   ledger_stride=None, kinetic_path=None):
    """Read norm-none instantaneous mass/momentum/kinetic sums, not mean(v)*mass."""
    if any(not isinstance(v, (int, np.integer)) for v in (steps, stride, bins, atoms)) or steps < 1 or stride < 1 or steps % stride or bins < 4 or bins % 2 or atoms < 2:
        raise ValueError('invalid integer profile geometry/cadence')
    if any(not np.isfinite(v) or v <= 0 for v in (mass, box, dt_ps)):
        raise ValueError('positive finite mass, box and dt required')
    lines = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith('#')]
    expected = np.arange(0, steps+1, stride)
    if len(lines) != len(expected)*(bins+1):
        raise ValueError('LAMMPS profile frame count mismatch')
    frames = []
    for frame, step in enumerate(expected):
        offset = frame*(bins+1)
        header = np.array(lines[offset], dtype=float)
        if header.shape != (3,) or not np.array_equal(header, [step, bins, atoms]):
            raise ValueError('LAMMPS profile step/bin/population mismatch')
        values = np.array(lines[offset+1:offset+bins+1], dtype=float)
        if values.shape != (bins, 8) or not np.all(np.isfinite(values)):
            raise ValueError('LAMMPS profile field mismatch')
        if not np.array_equal(values[:, 0], np.arange(1, bins+1)) or not np.allclose(values[:, 1], (np.arange(bins)+.5)/bins, atol=1e-12, rtol=0):
            raise ValueError('LAMMPS bin geometry/order mismatch')
        count, masses = values[:, 2], values[:, 3]
        if np.any(count < 0) or np.any(count != np.floor(count)) or count.sum() != atoms or not np.allclose(masses, count*mass, rtol=1e-12, atol=1e-12):
            raise ValueError('LAMMPS bin mass/count mismatch')
        if np.any(values[count == 0, 3:] != 0) or np.any(values[:, 7] < 0):
            raise ValueError('invalid empty-bin content or kinetic sum')
        frames.append(values)
    frames = np.array(frames)
    ledger_stride = stride if ledger_stride is None else ledger_stride
    if isinstance(ledger_stride,bool) or not isinstance(ledger_stride,(int,np.integer)) or ledger_stride<1 or steps%ledger_stride:
        raise ValueError('invalid ledger cadence')
    ledger_steps=np.arange(0,steps+1,ledger_stride)
    ledger = np.loadtxt(ledger_path, comments='#', ndmin=2)
    if ledger.shape != (len(ledger_steps), 3) or not np.all(np.isfinite(ledger)) or not np.array_equal(ledger[:, 0], ledger_steps):
        raise ValueError('LAMMPS impulse/temperature cadence mismatch')
    if kinetic_path is None:
        if ledger_stride != stride: raise ValueError('separate kinetic samples required for sparse impulse ledger')
        kinetic_temperature=ledger[:,2]
    else:
        kinetic=np.loadtxt(kinetic_path,comments='#',ndmin=2)
        if kinetic.shape!=(len(expected),2) or not np.isfinite(kinetic).all() or not np.array_equal(kinetic[:,0],expected) or np.any(kinetic[:,1]<=0):
            raise ValueError('LAMMPS kinetic temperature cadence mismatch')
        kinetic_temperature=kinetic[:,1]
        common,ki,li=np.intersect1d(expected,ledger_steps,return_indices=True)
        if not np.allclose(kinetic_temperature[ki],ledger[li,2],rtol=1e-12,atol=1e-12):
            raise ValueError('LAMMPS shared-step kinetic temperature mismatch')
    # real units: Da Angstrom/fs -> Da nm/ps. Native fix viscosity scalar
    # accumulates momentum LOST by middle, opposite to SPONGE's convention.
    # A ledger at exchange step k excludes exchange k. Its value holds for
    # all samples AFTER exchange k-exchange_interval THROUGH step k.
    # Read the next-or-equal exchange checkpoint; never linearly interpolate.
    momentum = -ledger[np.searchsorted(ledger_steps,expected,side='left'), 1]*100
    if ledger[0,1] != 0 or np.any(np.diff(ledger[:,1]) > 0) or momentum[0] != 0 or np.any(np.diff(momentum) < 0) or np.any(ledger[:, 2] <= 0):
        raise ValueError('wrong-sign cumulative impulse or invalid temperature')
    masses, px = frames[:, :, 3], frames[:, :, 4]
    total_mass = masses.sum(axis=1)
    transverse = frames[:, :, 5:7].sum(axis=1)
    flow_energy = np.divide(px*px, masses, out=np.zeros_like(px), where=masses > 0).sum(axis=1)
    thermal_twice = frames[:, :, 7].sum(axis=1)-flow_energy-(transverse*transverse).sum(axis=1)/total_mass
    if np.any(thermal_twice < -1e-12):
        raise ValueError('negative residual kinetic energy')
    thermal = np.maximum(thermal_twice, 0)*LM_TIME**2/((3*atoms-(masses > 0).sum(axis=1)-2)*LM_KB)
    if not np.all(np.isfinite(thermal)):
        raise ValueError('nonfinite thermal temperature')
    return dict(time=expected*dt_ps, step=expected, bin_count=frames[:, :, 2], bin_mass=masses,
                bin_momentum=px*100, momentum=momentum,
                area=np.full(len(expected), box**2/100), length=np.full(len(expected), box/10),
                kinetic_temperature=kinetic_temperature, thermal_temperature=thermal)


def prepare(model_dir, seed_file, output, steps=50000, exchange=100, bins=20,
            start_star=20., block_star=20., exclusions=(0, 1), stride=None, audit_stride=None):
    model, q, half, centered, h = frozen_inputs(model_dir, seed_file)
    stride = exchange if stride is None else stride
    audit_stride = exchange if audit_stride is None else audit_stride
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
           for v in (steps, exchange, bins, stride, audit_stride)):
        raise ValueError('positive integer steps, exchange, bins and sampling cadences required')
    if steps % stride or steps % audit_stride or audit_stride % stride:
        raise ValueError('steps must align with both cadences; energy audit must align with profiles')
    if steps < 1 or exchange < 1 or steps % exchange or bins < 6 or bins % 2 or bins > 10000:
        raise ValueError('steps must be a multiple of positive exchange; even bins >=6 required')
    if steps//stride > 500000:
        raise ValueError('pilot profile frame bound exceeded')
    if not exclusions or any(e < 0 or int(e) != e or bins//2-1-2*e < 2 for e in exclusions):
        raise ValueError('need at least two fitted bins per branch for every exclusion')
    if not np.isfinite(start_star) or start_star < 0 or not np.isfinite(block_star) or block_star <= 0:
        raise ValueError('invalid analysis window')
    spacing = stride*model['dt_star']
    intervals = int(np.floor(block_star/spacing+1e-9))
    begin = int(np.ceil(start_star/spacing-1e-9))
    if intervals < 1 or (steps//stride-begin)//intervals < 2:
        raise ValueError('at least two complete profile blocks required')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    sp, lm = output/'sponge', output/'lammps'; sp.mkdir(); lm.mkdir()
    n, box = model['atoms'], model['box_angstrom']
    for name, values in (('coordinate', q), ('velocity', half)):
        tail = f'{box:.17g} {box:.17g} {box:.17g}\n90 90 90\n' if name == 'coordinate' else ''
        (sp/f'{name}.txt').write_text(str(n)+'\n'+'\n'.join(' '.join(f'{v:.17g}' for v in row) for row in values)+'\n'+tail)
    config = dict(md_name='lj_rnemd_pair', mode='nve', pbc=True, dt=model['dt_ps'], step_limit=steps,
                  cutoff=model['cutoff_angstrom'], skin=.3*model['sigma_angstrom'], print_zeroth_frame=True,
                  write_mdout_interval=audit_stride, write_information_interval=max(exchange, 1000//exchange*exchange),
                  write_trajectory_interval=audit_stride, write_restart_file_interval=0,
                  crd='crd.dat', vel='vel.dat', output_h5_observable_path='observables.h5')
    config['PM.MPI_size'] = 1
    for name in ('mass', 'charge'):
        config[f'{name}_in_file'] = str(Path(model_dir).resolve()/f'{name}.txt')
    config.update(coordinate_in_file='coordinate.txt', velocity_in_file='velocity.txt', LJ_in_file=str(Path(model_dir).resolve()/'lj.txt'))
    write_engine_input(sp/'mdin.spg.toml', '\n'.join(f'{key}={json.dumps(value)}' for key, value in config.items())+
        f'\n[viscosity]\nmethod="rnemd"\nexchange_interval={exchange}\nbins={bins}\nsample_interval={stride}\n')
    lines = ['Paired RNEMD structural seed', '', f'{n} atoms', '1 atom types', '',
             *[f'0 {box:.17g} {axis}lo {axis}hi' for axis in 'xyz'], '', 'Masses', '', f"1 {model['mass_da']:.17g}", '', 'Atoms # atomic', '']
    lines.extend(f'{i+1} 1 '+' '.join(f'{v:.17g}' for v in row) for i, row in enumerate(q))
    lines.extend(['', 'Velocities', ''])
    lines.extend(f'{i+1} '+' '.join(f'{v/LM_TIME:.17g}' for v in row) for i, row in enumerate(centered))
    (lm/'model.data').write_text('\n'.join(lines)+'\n')
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
fix integrate all nve
compute kinetic all temp/com
compute layers all chunk/atom bin/1d z lower {1/bins:.17g} units reduced
variable px atom mass*vx
variable py atom mass*vy
variable pz atom mass*vz
variable mv2 atom mass*(vx*vx+vy*vy+vz*vz)
fix profile all ave/chunk {stride} 1 {stride} layers mass v_px v_py v_pz v_mv2 norm none file profile.dat format " %.17g"
variable impulse equal f_exchange
fix ledger all ave/time {exchange} 1 {exchange} v_impulse c_kinetic file momentum.dat format " %.17g"
fix temperatures all ave/time {stride} 1 {stride} c_kinetic file kinetic.dat format " %.17g"
variable potential equal pe
variable ke_com equal 0.5*{3*n-3}*{LM_KB:.17g}*c_kinetic
fix energy all ave/time {audit_stride} 1 {audit_stride} v_potential v_ke_com file energy.dat format " %.17g"
fix exchange all viscosity {exchange} x z {bins} swap 1 vtarget INF
thermo {exchange}
thermo_style custom step c_kinetic f_exchange pe
thermo_modify norm no format float %.17g
run {steps}
''')
    plan = dict(contract='sponge.lj.paired-rnemd-pilot.v1', model=model, steps=steps,
                exchange_interval=exchange, sample_interval=stride, audit_interval=audit_stride, bins=bins,
                impulse_ledger_interval=exchange, kinetic_file='kinetic.dat',
                energy_audit_contract='cutoff-shifted-centered-com-free.v1',
                start_star=start_star, block_star=block_star, exclusions=list(exclusions),
                dt_sponge_ps=model['dt_ps'], dt_lammps_ps=h*LM_TIME/1000,
                model_sha256=digest(Path(model_dir)/'model.json'), seed_sha256=digest(seed_file),
                model_path=str(Path(model_dir).resolve()), seed_path=str(Path(seed_file).resolve()),
                input_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                script_sha256=digest(__file__), analyzer_sha256=digest(rn.__file__),
                sampling_phase='pre-exchange integer-time velocities; profile and ledger fixes BEFORE LAMMPS exchange fix',
                impulse_conversion='SPONGE middle gain positive; -100 * LAMMPS f_exchange in Da Angstrom/fs -> Da nm/ps',
                comparison='pilot only; same geometry and signed candidates; LAMMPS floating vtarget=1e10 and local-order ties differ from SPONGE global-ID ties',
                formal_equivalence_eligible=False)
    write_json(output/'plan.json', plan)
    return plan


def audit_energy(output, plan, sponge_profile):
    """Compare same-time total energy, distinct from exchange-only error ledger."""
    model=plan['model']; n=model['atoms']; stride=plan.get('audit_interval',plan['sample_interval'])
    profile_stride=plan['sample_interval']
    if stride % profile_stride: raise ValueError('energy audit must align with profiles')
    steps=np.arange(0,plan['steps']+1,stride)
    q0=np.loadtxt(output/'sponge/coordinate.txt',skiprows=1,max_rows=n)
    coords=np.fromfile(output/'sponge/crd.dat',dtype=np.float32).reshape(-1,n,3).astype(float)
    if len(coords)!=len(steps)-1 or not np.isfinite(coords).all():
        raise ValueError('RNEMD coordinate audit cadence mismatch')
    audit=np.array([cutoff_audit(q,model['box_angstrom'],model['cutoff_angstrom']) for q in [q0,*coords]])
    with h5py.File(output/'sponge/observables.h5') as file:
        if file['parameters/sponge/output/status'][()]!=b'finalized' or not np.array_equal(file['observables/all/step'][()],steps):
            raise ValueError('RNEMD energy observable completion/cadence mismatch')
        potential=file['observables/all/eff_pot/value'][()]
    cutoff=model['c12']/model['cutoff_angstrom']**12-model['c6']/model['cutoff_angstrom']**6
    tail=-2*np.pi*n*n*model['c6']/(3*model['box_angstrom']**3*model['cutoff_angstrom']**3)
    shifted=potential-tail-audit[:,0]*cutoff
    temperatures=np.asarray(sponge_profile['kinetic_temperature'])
    if temperatures.shape != (plan['steps']//profile_stride+1,):
        raise ValueError('RNEMD kinetic temperature cadence mismatch')
    kinetic=.5*float(np.float32((3*n-3)*SP_KB))*temperatures[::stride//profile_stride]
    lm=np.loadtxt(output/'lammps/energy.dat',comments='#',ndmin=2)
    if lm.shape!=(len(steps),3) or not np.isfinite(lm).all() or not np.array_equal(lm[:,0],steps):
        raise ValueError('RNEMD LAMMPS energy audit cadence mismatch')
    reports=dict(sponge=energy_diagnostics(steps,shifted,kinetic,model['epsilon_kcal_mol'],n),
                 lammps=energy_diagnostics(steps,lm[:,1],lm[:,2],model['epsilon_kcal_mol'],n))
    reports['sponge']['ambiguous_coordinate_frames']=int(np.count_nonzero(audit[:,2]))
    reports['sponge']['max_cutoff_count_uncertainty_per_atom_epsilon']=float(audit[:,2].max()*abs(cutoff)/(n*model['epsilon_kcal_mol']))
    reports['initial_energy_gate']=dict(shifted_potential=bool(np.isclose(shifted[0],lm[0,1],rtol=2e-6,atol=2e-5)),
        centered_kinetic=bool(np.isclose(kinetic[0],lm[0,2],rtol=2e-6,atol=1e-6)))
    np.savetxt(output/'energy-audit.csv',np.column_stack((steps,audit[:,0],potential,shifted,kinetic,lm[:,1:],audit[:,1:])),delimiter=',',
        header='step,active_pairs,sponge_raw_potential,sponge_shifted_potential,sponge_centered_com_free_ke,lammps_shifted_potential,lammps_com_free_ke,nearest_cutoff_angstrom,ambiguous_pairs',comments='')
    if not all(reports['initial_energy_gate'].values()): raise ValueError('RNEMD initial energy gate failed')
    return reports


def analyze_run(output):
    output = Path(output).resolve()
    plan = json.loads((output/'plan.json').read_text())
    execution = json.loads((output/'execution.json').read_text())
    if plan.get('contract') != 'sponge.lj.paired-rnemd-pilot.v1':
        raise ValueError('unsupported run plan')
    if execution.get('status') != 'engines_completed' or any(
            execution.get('engines', {}).get(engine, {}).get('status') != 'completed' for engine in ('sponge', 'lammps')):
        raise ValueError('both engines must complete')
    if 'plan_sha256' in execution and digest(output/'plan.json') != execution['plan_sha256']:
        raise ValueError('run plan changed since execution')
    for name, expected in execution.get('raw_output_sha256', {}).items():
        if digest(output/name) != expected: raise ValueError('raw output changed since execution')
    for name, expected in plan['input_sha256'].items():
        if digest(output/name) != expected: raise ValueError('run input changed')
    model = plan['model']
    sp, metadata = rn.load_profile(output/'sponge/viscosity.h5')
    expected_steps = np.arange(0, plan['steps']+1, plan['sample_interval'])
    if len(sp['time']) != len(expected_steps) or not np.allclose(sp['time'], expected_steps*plan['dt_sponge_ps'], rtol=1e-12, atol=1e-12):
        raise ValueError('SPONGE endpoint/cadence mismatch')
    lm = lammps_profile(output/'lammps/profile.dat', output/'lammps/momentum.dat', steps=plan['steps'], stride=plan['sample_interval'],
                         bins=plan['bins'], atoms=model['atoms'], mass=model['mass_da'], box=model['box_angstrom'], dt_ps=plan['dt_lammps_ps'],
                         ledger_stride=plan.get('impulse_ledger_interval'),kinetic_path=output/'lammps'/plan['kinetic_file'] if 'kinetic_file' in plan else None)
    version = next((line for line in (output/'lammps/run.log').read_text().splitlines() if line.startswith('LAMMPS (')), 'unknown')
    phase = (metadata.get('sampling_phase') == 'post_force_pre_exchange_pre_integrate'
             and metadata.get('velocity_time_layer') == 'force_centered') if 'observable_contract' in metadata else (
                 metadata.get('sampling_convention') == 'pre-exchange integer-time velocity; momentum gained by middle slab from prior exchanges; two periodic paths')
    initial = dict(expected_lammps_version_banner=version.startswith('LAMMPS (29 Aug 2024'),
                   phase=phase,
                   configuration=int(metadata['exchange_interval']) == plan['exchange_interval'] and int(metadata['bins']) == plan['bins'] and int(metadata['flow']) == 0 and int(metadata['gradient']) == 2,
                   geometry=bool(np.allclose(sp['area'], lm['area'], rtol=2e-7, atol=0) and np.allclose(sp['length'], lm['length'], rtol=2e-7, atol=0)),
                   initial_kinetic_temperature=bool(np.isclose(sp['kinetic_temperature'][0], lm['kinetic_temperature'][0]*LM_KB/float(np.float32(.00198716)), rtol=2e-5, atol=1e-5)),
                   population=bool(np.array_equal(sp['bin_count'][0], lm['bin_count'][0])),
                   mass=bool(np.allclose(sp['bin_mass'][0], lm['bin_mass'][0], rtol=1e-12, atol=1e-12)),
                   momentum=bool(np.allclose(sp['bin_momentum'][0], lm['bin_momentum'][0], rtol=2e-5, atol=2e-5)))
    write_json(output/'initial-state-gate.json', dict(checks=initial, lammps_version=version))
    if not all(initial.values()): raise ValueError('RNEMD initial state/profile gate failed')
    reports = {}
    for engine, data in dict(sponge=sp, lammps=lm).items():
        np.savez(output/f'{engine}-canonical-profile.npz', **data)
        tau = plan[f'dt_{engine}_ps']/model['dt_star']
        records = []
        for exclude in plan['exclusions']:
            try:
                result, profiles = rn.analyze(data, plan['start_star']*tau, plan['block_star']*tau, exclude)
                np.savetxt(output/f'{engine}-exclude-{exclude}-profiles.csv', np.array(profiles, dtype=float), delimiter=',',
                    header='block,bin,z_nm,mean_mass_da,mean_velocity_nm_per_ps', comments='')
            except ValueError as error:
                result = dict(status='unresolved_profile', reason=str(error), exclude_neighbors=exclude)
            records.append(result)
        reports[engine] = records
    energy=audit_energy(output,plan,sp) if plan.get('energy_audit_contract')=='cutoff-shifted-centered-com-free.v1' else None
    report = dict(status='paired_pilot_only', energy_audit=energy, analysis_script_sha256=digest(__file__), analyzer_sha256=digest(rn.__file__), execution_integrity_manifest_present='raw_output_sha256' in execution and 'plan_sha256' in execution, analyses=reports, initial_state_gate=initial, sponge_provenance=metadata, execution=execution,
        warnings=['No equivalence, steady-state, weak-drive or convergence claim; all predeclared exclusions retained.',
                  'One common structural seed is not independent replicas; NVE thermal temperatures can depend on imposed flow.',
                  'Aggregate profiles and impulse are resampled jointly; no per-block inverse-gradient averaging. Positive conditional intervals are not scientific acceptance.',
                  'LAMMPS exposes no matching swap/skip/conservation counters here; missing counters are not zero.',
                  'Cumulative momentum at a sampled exchange step excludes that step in BOTH engines; no absolute-value sign repair.',
                  'Version banner is a compatibility check, not proof that a binary was built from the separately inspected source.',
                  'Historical runs without execution-time output/plan hashes are marked explicitly; reanalysis records its current script hashes.',
                  'Costs omit upstream equilibration and further bias/precision scans.'])
    write_json(output/'summary.json', report)
    return report


def run_pair(model_dir, seed_file, output, sponge, lammps, **options):
    prepare(model_dir, seed_file, output, **options)
    output = Path(output).resolve()
    binaries = dict(sponge=Path(sponge).resolve(), lammps=Path(lammps).resolve())
    execution = dict(status='running', engines={}, sequential=True, cpu_threads=1, plan_sha256=digest(output/'plan.json'),
        executable_sha256={name: digest(path) for name, path in binaries.items()})
    write_json(output/'execution.json', execution)
    for engine, args in dict(sponge=['-mdin', 'mdin.spg.toml'], lammps=['-in', 'in.lammps']).items():
        begin=time.perf_counter()
        try:
            elapsed = execute([binaries[engine], *args], output/engine, timeout=900)
            execution['engines'][engine] = dict(status='completed', wall_seconds=elapsed)
        except Exception as error:
            execution['status'] = 'failed'; execution['engines'][engine] = dict(status='failed', error=str(error),wall_seconds=time.perf_counter()-begin)
            write_json(output/'execution.json', execution); raise
        write_json(output/'execution.json', execution)
    execution['status'] = 'engines_completed'
    execution['raw_output_sha256'] = {name: digest(output/name) for name in ('sponge/viscosity.h5', 'lammps/profile.dat', 'lammps/momentum.dat', 'lammps/kinetic.dat', 'lammps/run.log','lammps/energy.dat','sponge/observables.h5','sponge/crd.dat')}
    write_json(output/'execution.json', execution)
    begin = time.perf_counter()
    try:
        result = analyze_run(output)
    except Exception as error:
        execution.update(analysis_status='failed', analysis_error=str(error), analysis_wall_seconds=time.perf_counter()-begin)
        write_json(output/'execution.json', execution); raise
    execution.update(analysis_status='completed', analysis_wall_seconds=time.perf_counter()-begin)
    write_json(output/'execution.json', execution); result['execution'] = execution
    write_json(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'seed-file', 'output', 'sponge', 'lammps'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--exchange', type=int, default=100)
    parser.add_argument('--stride', type=int, help='profile/impulse sampling steps; default: exchange interval')
    parser.add_argument('--audit-stride', type=int, help='energy/coordinate audit steps; default: exchange interval')
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--start-star', type=float, default=20.)
    parser.add_argument('--block-star', type=float, default=20.)
    parser.add_argument('--exclude-neighbors', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    result = run_pair(args.model, args.seed_file, args.output, args.sponge, args.lammps,
        steps=args.steps, exchange=args.exchange, bins=args.bins, start_star=args.start_star,
        block_star=args.block_star, exclusions=args.exclude_neighbors, stride=args.stride, audit_stride=args.audit_stride)
    print(json.dumps({engine: [{key: record[key] for key in ('status', 'viscosity_pa_s', 'reason') if key in record} for record in records]
                      for engine, records in result['analyses'].items()}, indent=2))
