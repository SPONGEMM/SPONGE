#!/usr/bin/env python3
"""Plot all predeclared GK windows and sparse energy audits for paired pilots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directories, output):
    if not directories:
        raise ValueError('at least one pair directory required')
    output = Path(output)
    if output.with_suffix('.png').exists() or output.with_suffix('.svg').exists():
        raise FileExistsError('choose a new figure prefix')
    fig, axes = plt.subplots(len(directories), 3, figsize=(14, 3.8*len(directories)), squeeze=False)
    colors = dict(sponge='#1565c0', lammps='#d55e00')
    for row, directory in enumerate(map(Path, directories)):
        report = json.loads((directory/'summary.json').read_text())
        plan = json.loads((directory/'plan.json').read_text())
        if report['status'] != 'paired_pilot_only':
            raise ValueError('unsupported paired summary')
        windows = np.array(plan['windows_star'])
        index = int(np.argmax(windows))
        for engine, color in colors.items():
            curve = np.loadtxt(directory/f'{engine}-window-{index}.csv', delimiter=',', skiprows=1)
            tau = plan[f'dt_{engine}_ps']/plan['model']['dt_star']
            axes[row, 0].plot(curve[:, 0]/tau, curve[:, 4:7].mean(axis=1)*1000, label=engine.upper(), color=color)
            records = report['analyses'][engine]
            means = np.array([record['viscosity_pa_s'] for record in records])*1000
            intervals = np.array([record['conditional_bootstrap_95_interval_pa_s'] for record in records])*1000
            offset = -.025 if engine == 'sponge' else .025
            # Explicit segments also represent intervals not centered on the point.
            axes[row, 1].vlines(windows+offset, intervals[:, 0], intervals[:, 1], color=color, lw=2)
            axes[row, 1].scatter(windows+offset, means, color=color, s=22)
        audit = np.loadtxt(directory/'energy-audit.csv', delimiter=',', skiprows=1)
        scale = plan['model']['atoms']*plan['model']['epsilon_kcal_mol']
        for engine, columns in [('sponge', (3, 4)), ('lammps', (5, 6))]:
            energy = audit[:, columns[0]]+audit[:, columns[1]]
            axes[row, 2].plot(audit[:, 0]*plan['model']['dt_star'], (energy-energy[0])/scale,
                              color=colors[engine], lw=1, label=engine.upper())
        ambiguous = audit[:, 8] > 0
        if np.any(ambiguous):
            energy = audit[:, 3]+audit[:, 4]
            axes[row, 2].scatter(audit[ambiguous, 0]*plan['model']['dt_star'],
                                 (energy[ambiguous]-energy[0])/scale, facecolors='none', edgecolors='black',
                                 marker='o', s=38, label='cutoff count ambiguous')
        axes[row, 0].set(xlabel='Correlation lag / LJ time', ylabel='Running viscosity (mPa s)',
                         title=f'{directory.name}: running integral')
        axes[row, 1].set(xlabel='Integration window / LJ time', ylabel='Viscosity (mPa s)',
                         title='All windows; conditional 95% block intervals')
        axes[row, 2].set(xlabel='Elapsed LJ time', ylabel=r'$[E(t)-E(0)]/(N\epsilon)$',
                         title='Shifted-potential + centered kinetic energy')
        axes[row, 0].legend(frameon=False, fontsize=8)
        axes[row, 2].legend(frameon=False, fontsize=7)
        for axis in axes[row]:
            axis.grid(alpha=.2)
            axis.spines[['right', 'top']].set_visible(False)
    fig.suptitle('Paired LJ Green–Kubo pilots — convergence and 5% equivalence NOT established', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=160)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='new figure prefix, without extension')
    args = parser.parse_args()
    plot(args.pairs, args.output)
