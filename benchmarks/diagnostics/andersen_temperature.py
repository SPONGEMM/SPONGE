import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--head-binary', type=Path, required=True)
parser.add_argument('--base-binary', type=Path)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--plan', choices=['local', 'mac', 'mac-long'], required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(args.source))
from benchmarks.validation.thermostat.tests.utils import write_thermostat_mdin, parse_temperature_series, evaluate_temperature_block_average

if args.plan == 'local':
    plan = [
        dict(label='head-long-2fs-s2026', binary='head', threads=1, dt=0.002, ps=40, seed=2026),
        dict(label='head-2fs-s2027', binary='head', threads=1, dt=0.002, ps=12, seed=2027),
        dict(label='head-2fs-s2028', binary='head', threads=1, dt=0.002, ps=12, seed=2028),
        dict(label='head-1fs-s2026', binary='head', threads=1, dt=0.001, ps=12, seed=2026),
        dict(label='head-half-fs-s2026', binary='head', threads=1, dt=0.0005, ps=12, seed=2026),
    ]
elif args.plan == 'mac':
    assert args.base_binary is not None
    plan = []
    for threads in [1, 3]:
        for binary in ['base', 'head']:
            plan.append(dict(label=f'{binary}-short-t{threads}', binary=binary, threads=threads, dt=0.002, ps=2, seed=2026))
    for binary in ['base', 'head']:
        plan.append(dict(label=f'{binary}-long-2fs-s2026', binary=binary, threads=1, dt=0.002, ps=12, seed=2026))
    plan += [
        dict(label='head-1fs-s2026', binary='head', threads=1, dt=0.001, ps=12, seed=2026),
        dict(label='head-half-fs-s2026', binary='head', threads=1, dt=0.0005, ps=12, seed=2026),
        dict(label='head-2fs-s2027', binary='head', threads=1, dt=0.002, ps=12, seed=2027),
        dict(label='head-2fs-s2028', binary='head', threads=1, dt=0.002, ps=12, seed=2028),
    ]
else:
    plan = [
        dict(label='head-2fs-40ps-s2026', binary='head', threads=1, dt=0.002, ps=40, seed=2026),
        dict(label='head-2fs-40ps-s2027', binary='head', threads=1, dt=0.002, ps=40, seed=2027),
        dict(label='head-half-fs-40ps-s2026', binary='head', threads=1, dt=0.0005, ps=40, seed=2026),
        dict(label='zero-pair-2fs-40ps-s2026', binary='head', threads=1, dt=0.002, ps=40, seed=2026, interaction='zero-pair'),
    ]
binaries = {'head': args.head_binary.resolve()}
if args.base_binary:
    binaries['base'] = args.base_binary.resolve()
protocol = dict(plan=plan, binaries={k: dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k, v in binaries.items()},
    thermostat='andersen', thermostat_tau_ps=0.1, target_temperature_K=300, constraint='SETTLE',
    analysis_windows=[dict(prefix_ps=2, burn_ps=0.04, block_ps=0.2), dict(prefix_ps=12, burn_ps=4, block_ps=0.5), dict(prefix_ps=40, burn_ps=4, block_ps=2)],
    collision_phase_note='printed step 2 is the first state after collision at loop step 1; subsequent collision states repeat every round(tau/dt) steps',
    policy='diagnostic only; no acceptance threshold/seed selection; all scheduled results retained; no production code changes')
(args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
results = []
for cfg in plan:
    case = args.output / cfg['label']
    shutil.copytree(args.source / 'benchmarks/validation/thermostat/statics/tip3p_water', case)
    if cfg.get('interaction') == 'zero-pair':
        charge = case / 'tip3p_charge.txt'
        atom_count = int(charge.read_text().split()[0])
        charge.write_text(str(atom_count) + '\n' + '0\n' * atom_count)
        lj = case / 'tip3p_LJ.txt'
        tokens = lj.read_text().split()
        type_count = int(tokens[1])
        coefficient_count = type_count * (type_count + 1)
        tokens[2:2+coefficient_count] = ['0'] * coefficient_count
        lj.write_text(' '.join(tokens[:2]) + '\n' + '\n'.join(tokens[2:]) + '\n')
    steps = round(cfg['ps'] / cfg['dt'])
    write_thermostat_mdin(case, step_limit=steps, dt=cfg['dt'], cutoff=8.0, thermostat='andersen', constrain_mode='SETTLE', target_temperature=300.0, thermostat_tau=0.1, thermostat_seed=cfg['seed'], initial_velocity_seed=cfg['seed'], default_in_file_prefix='tip3p', md_name='Andersen predeclared diagnostic')
    env = dict(os.environ, OMP_NUM_THREADS=str(cfg['threads']), MKL_NUM_THREADS=str(cfg['threads']), OPENBLAS_NUM_THREADS='1')
    started = time.time()
    with (case / 'execution.log').open('w') as log:
        run = subprocess.run([str(binaries[cfg['binary']]), '-mdin', 'mdin.spg.toml'], cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    result = dict(**cfg, exit_code=run.returncode, seconds=time.time()-started)
    if run.returncode == 0:
        temperatures = parse_temperature_series(case / 'mdout.txt')
        assert len(temperatures) == steps + 1
        result['mdout_sha256'] = hashlib.sha256((case / 'mdout.txt').read_bytes()).hexdigest()
        if cfg.get('interaction') == 'zero-pair':
            lines = (case / 'mdout.txt').read_text().splitlines()
            headers = lines[0].split()
            column = headers.index('eff_pot')
            result['max_abs_eff_pot'] = max(abs(float(line.split()[column])) for line in lines[1:] if line.strip())
        result['windows'] = []
        for window in protocol['analysis_windows']:
            if window['prefix_ps'] > cfg['ps']:
                continue
            sample = temperatures[:round(window['prefix_ps'] / cfg['dt'])+1]
            stats = evaluate_temperature_block_average(sample, burn_in=round(window['burn_ps']/cfg['dt']), block_size=round(window['block_ps']/cfg['dt']))
            result['windows'].append(dict(**window, stats=stats))
        period = round(0.1 / cfg['dt'])
        start = round(4 / cfg['dt']) if cfg['ps'] >= 12 else 20
        phase = [[] for _ in range(period)]
        for step in range(start, len(temperatures)):
            phase[(step-2) % period].append(temperatures[step])
        result['collision_phase'] = [dict(offset_steps=i, offset_ps=i*cfg['dt'], count=len(values), mean=statistics.fmean(values), std=statistics.stdev(values) if len(values)>1 else 0) for i, values in enumerate(phase)]
    results.append(result)
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='collision_phase'}), flush=True)
assert all(r['exit_code'] == 0 for r in results)
