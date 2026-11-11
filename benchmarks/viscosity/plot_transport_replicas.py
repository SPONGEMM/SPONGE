#!/usr/bin/env python3
"""Finite-condition transport comparison; all declared variants retained."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lj_transport_replicas import summarize,ENGINES


def plot(directory,output):
    r=summarize(directory);methods=list(r['estimates'])
    labels=[m.replace('gk-','GK ').replace('rnemd-exclude-','RN excl ').replace('cosine-','cos ').replace('com_fourier','COM F').replace('fourier','F') for m in methods]
    color=dict(sponge='#0072B2',lammps='#D55E00')
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for e,offset in zip(ENGINES,(-.1,.1)):
        for j,m in enumerate(methods):
            v=r['estimates'][m][e];p=v['viscosity_mpa_s'];ci=v['replica_bootstrap_95_interval_mpa_s']
            if p is not None:
                axes[0,0].plot(j+offset,p,'o',color=color[e],label=e.upper() if j==0 else None)
                if ci: axes[0,0].vlines(j+offset,*ci,color=color[e],lw=1.5)
            if ci is None: axes[0,0].annotate('unresolved',(j+offset,.02),xycoords=('data','axes fraction'),rotation=90,fontsize=7,color=color[e])
    axes[0,0].set(title='All retained finite-condition viscosity estimates',ylabel='Viscosity (mPa s)')
    axes[0,0].legend(fontsize=9,loc='upper left')
    pairs=[x for x in r['comparisons'] if x['numerator'][1]=='sponge' and x['denominator'][1]=='lammps']
    for j,p in enumerate(pairs):
        if p['ratio'] is not None: axes[0,1].plot(j,p['ratio'],'o',color='#6B4C9A')
        ci=p['paired_ratio_95_interval']
        if ci: axes[0,1].vlines(j,*ci,color='#6B4C9A',lw=1.5)
    axes[0,1].axhspan(.95,1.05,color='gray',alpha=.15,label='±5% comparison band; no acceptance')
    axes[0,1].axhline(1,color='black',ls='--',lw=.8)
    axes[0,1].set(title='Paired SPONGE / LAMMPS viscosity ratio',ylabel='Ratio')
    axes[0,1].legend(fontsize=8,loc='upper left')
    for ax in axes[0]:ax.set_xticks(range(len(methods)),labels,rotation=35,ha='right')
    plan=json.loads((Path(directory)/'plan.json').read_text())['specification']
    selected=[f"gk-{plan['reference_gk_window_star']:g}",f"rnemd-exclude-{plan['rnemd_exclusions'][0]}",'cosine-fourier']
    for e,offset in zip(ENGINES,(-.1,.1)):
        for j,m in enumerate(selected):
            v=r['estimates'][m][e]
            axes[1,0].plot(j+offset,v['temperature_sponge_k'],'o',color=color[e])
            axes[1,0].vlines(j+offset,*v['temperature_95_interval_k'],color=color[e],lw=1.5)
    source=json.loads((Path(directory)/'source/upstream-summary.json').read_text())
    target=source['physical_condition_identity']['fields']['target_temperature_k']
    axes[1,0].axhline(target,color='black',ls='--',lw=.8,label='initial target')
    axes[1,0].set(xticks=range(3),xticklabels=['GK global COM','RN binned flow','cosine projection'],
        title='Actual temperature; definitions differ by method',ylabel='Temperature in SPONGE kBT units (K)')
    axes[1,0].legend(fontsize=9)
    for j,(method,e) in enumerate((m,e) for m in ('gk','rnemd') for e in ENGINES):
        vals=[rep['energy_audits'][method][e]['energy_change_per_atom_epsilon'] for rep in r['replicas']]
        x=j+np.linspace(-.12,.12,len(vals));axes[1,1].scatter(x,vals,color=color[e],s=25)
    axes[1,1].axhline(0,color='black',lw=.8,ls='--')
    axes[1,1].set(xticks=range(4),xticklabels=['GK SP','GK LM','RN SP','RN LM'],
        title='Same-time total-energy change in each replica',ylabel='(E_final − E_initial) / (N ε)')
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"{len(r['replicas'])} shared independent initializations | whole-replica 95% percentile intervals\nState, drive, window and size convergence remain unverified; no equivalence claim",fontsize=14)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output.with_suffix('.png'),dpi=180);fig.savefig(output.with_suffix('.svg'));plt.close(fig)
    return output.with_suffix('.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(plot(a.directory,a.output))
