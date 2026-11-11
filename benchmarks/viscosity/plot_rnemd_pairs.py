#!/usr/bin/env python3
"""Plot all declared paired RNEMD pilot fits; no equivalence or convergence claim."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(paths, output):
    output = Path(output)
    if any(output.with_suffix(suffix).exists() for suffix in ('.png', '.svg', '.json')):
        raise ValueError('plot output already exists')
    if not paths:
        raise ValueError('at least one run required')
    fig, axes = plt.subplots(3, len(paths), figsize=(6*len(paths), 10), squeeze=False)
    colors = dict(sponge='#176b9d', lammps='#bf581c')
    provenance = []
    viscosity_upper = 0.
    for col, path in enumerate(map(Path, paths)):
        plan = json.loads((path/'plan.json').read_text())
        report = json.loads((path/'summary.json').read_text())
        provenance.append(dict(path=str(path.resolve()), summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest()))
        profile_ax, eta_ax, temp_ax = axes[:, col]
        profile_ax.set_title(f"Exchange every {plan['exchange_interval']} steps; N={plan['model']['atoms']}")
        n_exclusions = len(plan['exclusions'])
        labels = []
        styles = {value: style for value, style in zip(plan['exclusions'], ['-', '--', ':', '-.']*(1+len(plan['exclusions'])//4))}
        for engine, records in report['analyses'].items():
            color = colors[engine]
            first = next((r for r in records if 'aggregate_profile' in r), None)
            if first is not None:
                profile = first['aggregate_profile']
                z = np.array(profile['z_nm']); velocity = np.array(profile['velocity_nm_per_ps'], dtype=float)
                length = z[-1]+z[0]
                profile_ax.plot(z/length, velocity, 'o', ms=4, color=color, label=engine.upper())
                for record in records:
                    if 'aggregate_fits' not in record:
                        continue
                    style = styles[record['exclude_neighbors']]
                    for fit in record['aggregate_fits']:
                        x = z[fit['bins']]
                        profile_ax.plot(x/length, fit['slope_per_ps']*x+fit['intercept_nm_per_ps'], style, color=color, alpha=.85)
            for i, record in enumerate(records):
                y = i + (0 if engine == 'sponge' else n_exclusions)
                labels.append((y, f"{engine.upper()}, exclude {record['exclude_neighbors']}"))
                eta = record.get('viscosity_pa_s')
                interval = record.get('conditional_bootstrap_95_interval_pa_s')
                if eta is not None:
                    viscosity_upper = max(viscosity_upper, eta*1000)
                    eta_ax.plot(eta*1000, y, 'o', color=color)
                if interval is not None:
                    viscosity_upper = max(viscosity_upper, interval[1]*1000)
                    eta_ax.hlines(y, interval[0]*1000, interval[1]*1000, color=color, lw=2)
                else:
                    eta_ax.text(.02, y, 'unresolved interval', transform=eta_ax.get_yaxis_transform(), va='center', fontsize=8, color=color)
            with np.load(path/f'{engine}-canonical-profile.npz') as data:
                tau = plan[f'dt_{engine}_ps']/plan['model']['dt_star']
                temp_ax.plot(data['time']/tau, data['thermal_temperature'], lw=.65, alpha=.65, color=color, label=engine.upper())
        profile_ax.set(xlabel='z / L', ylabel='Mean flow velocity (nm/ps)')
        profile_ax.axvspan(0, 1/plan['bins'], color='gray', alpha=.15)
        profile_ax.axvspan(.5, .5+1/plan['bins'], color='gray', alpha=.15)
        profile_ax.legend(fontsize=8)
        profile_ax.text(.02, .97, '; '.join(f'{style} exclude {value}' for value, style in styles.items()), transform=profile_ax.transAxes, va='top', fontsize=8)
        eta_ax.set_yticks([v[0] for v in labels], [v[1] for v in labels], fontsize=8)
        eta_ax.set(xlabel='Viscosity (mPa s); conditional 95% interval', ylim=(2*n_exclusions-.5, -.5))
        temp_ax.axvline(plan['start_star'], ls='--', color='black', lw=1)
        temp_ax.set(xlabel='Time / LJ time', ylabel='Flow-subtracted temperature (K)')
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    if viscosity_upper > 0:
        for ax in axes[1]: ax.set_xlim(0, viscosity_upper*1.08)
    fig.suptitle('RNEMD paired pilots — same initial state, not independent replicas', fontsize=14)
    fig.text(.5, .012, 'Conditional intervals assume independent blocks. No steady-state, weak-drive, equivalence or target-precision acceptance.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=160)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(inputs=provenance, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.runs, args.output)
