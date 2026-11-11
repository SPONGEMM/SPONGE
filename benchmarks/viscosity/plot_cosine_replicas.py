#!/usr/bin/env python3
"""Plot archived independent-initialization replica diagnostics, not acceptance."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lj_cosine_replicas import summarize, ENGINES


def plot(directory, output):
    report=summarize(directory)
    estimators=list(report['estimates']);replicas=report['replicas']
    colors=dict(sponge='#0072B2',lammps='#D55E00')
    labels=dict(sponge='SPONGE',lammps='LAMMPS + projected Python bath')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for e,offset in zip(ENGINES,(-.10,.10)):
        for j,estimator in enumerate(estimators):
            record=report['estimates'][estimator][e]
            eta=record['viscosity_mpa_s'];interval=record['replica_bootstrap_95_interval_mpa_s']
            if eta is not None:
                axes[0,0].plot(j+offset,eta,'o',color=colors[e],label=labels[e] if j==0 else None)
                if interval:
                    axes[0,0].vlines(j+offset,*interval,color=colors[e],lw=2)
                else:
                    axes[0,0].annotate('unbounded',(j+offset,eta),fontsize=8)
        x=np.arange(len(replicas))+offset
        records=[r['statistics'][e]['fourier'] for r in replicas]
        axes[0,1].plot(x,[r['fluidity_per_mpa_s'] for r in records],'o-',color=colors[e],label=labels[e],lw=.8)
        axes[1,0].plot(x,[r['temperature_sponge_k'] for r in records],'o-',color=colors[e],lw=.8)
        for j,estimator in enumerate(estimators):
            record=report['estimates'][estimator][e]
            axes[1,1].plot(j+offset,record['second_minus_first_fluidity_per_mpa_s'],'o',color=colors[e])
            axes[1,1].vlines(j+offset,*record['second_minus_first_fluidity_95_interval'],color=colors[e],lw=2)
    for ax in axes.flat:
        ax.grid(alpha=.2,axis='y');ax.spines[['top','right']].set_visible(False)
    axes[0,0].set(title='Pooled response then inversion',ylabel='Viscosity (mPa s)',xticks=range(len(estimators)),xticklabels=estimators)
    bounds=[v for est in report['estimates'].values() for e in ENGINES
            for v in (est[e]['replica_bootstrap_95_interval_mpa_s'] or [])]
    if bounds:
        low,high=min(bounds),max(bounds);span=max(high-low,abs(high)*.1)
        axes[0,0].set_ylim(low-.1*span,high+.4*span)
    axes[0,0].legend(fontsize=8,loc='upper right')
    axes[0,1].set(title='Fourier fluidity by independent initialization',ylabel='Fluidity (mPa s)⁻¹',xticks=range(len(replicas)),xticklabels=[r['id'] for r in replicas])
    axes[1,0].set(title='Production thermal temperature',ylabel='Temperature in SPONGE kBT units (K)',xticks=range(len(replicas)),xticklabels=[r['id'] for r in replicas])
    target=report['physical_condition_identity']['fields']['target_temperature_k']
    axes[1,0].axhline(target,color='black',ls='--',lw=1,label='target');axes[1,0].legend(fontsize=8)
    axes[1,1].set(title='Second half minus first half: pooled fluidity',ylabel='Fluidity difference (mPa s)⁻¹',xticks=range(len(estimators)),xticklabels=estimators)
    axes[1,1].axhline(0,color='black',ls='--',lw=1)
    # The drive is a condition, not a plot default.
    plan=json.loads((Path(directory)/'plan.json').read_text())
    a=plan['specification']['acceleration']
    fig.suptitle(f'{len(replicas)} independently initialized paired replicas | A = {a:g} nm/ps²\nWhole-replica 95% percentile intervals; stationarity / zero-drive / long-wave limits unverified',fontsize=13)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=180);fig.savefig(output.with_suffix('.svg'));plt.close(fig)
    return output.with_suffix('.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(plot(args.directory,args.output))
