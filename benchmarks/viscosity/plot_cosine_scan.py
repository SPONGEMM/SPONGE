#!/usr/bin/env python3
"""Plot every predeclared sensitivity case without pooling or acceptance claims."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directory,output):
    directory,output=Path(directory),Path(output)
    if any(output.with_suffix(ext).exists() for ext in ('.png','.svg','.json')):
        raise ValueError('plot output exists')
    summary=json.loads((directory/'summary.json').read_text())
    plan=json.loads((directory/'plan.json').read_text())
    if summary['status']!='sensitivity_diagnostics_only' or summary['plan_sha256']!=hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest():
        raise ValueError('unsupported or mismatched scan summary')
    cases=plan['cases'];rows=summary['rows'];index={c['id']:i for i,c in enumerate(cases)}
    colors=dict(sponge='#176b9d',lammps='#b65a1a')
    markers=dict(fourier='o',com_fourier='s',fit='^')
    offsets=dict(fourier=-.05,com_fourier=0,fit=.05)
    fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
    eta,temp,response,shear,cost,precision=axes.flat
    for row in rows:
        engine=row['engine'];est=row['estimator'];color=colors[engine]
        y=index[row['case_id']]+(-.15 if engine=='sponge' else .15)+offsets[est]
        label=engine.upper()+' '+est if index[row['case_id']]==0 else None
        if row['viscosity_mpa_s'] is not None:
            eta.plot(row['viscosity_mpa_s'],y,markers[est],color=color,ms=4,label=label)
        low,high=row['conditional_ci_low_mpa_s'],row['conditional_ci_high_mpa_s']
        if low is not None:eta.hlines(y,low,high,color=color,alpha=.65,lw=1)
        else:eta.text(.98,y,'unresolved',transform=eta.get_yaxis_transform(),ha='right',va='center',fontsize=6,color=color)
        response.plot(row['response_per_acceleration_ps'],y,markers[est],color=color,ms=4)
        if est=='fourier':
            se=row['temperature_block_standard_error_k']
            temp.errorbar(row['thermal_temperature_k']-row['target_temperature_k'],y,xerr=se,fmt='o',color=color,ms=4,capsize=2)
            shear.plot(row['cosine_mode_peak_shear_rate_ps_inv'],y,'o',color=color,ms=4)
            cost.barh(y,row['engine_wall_seconds'],height=.22,color=color,alpha=.8,label=engine.upper() if index[row['case_id']]==0 else None)
        if row['relative_ci_halfwidth'] is not None:
            precision.scatter(row['engine_wall_seconds'],100*row['relative_ci_halfwidth'],marker=markers[est],color=color,s=25)
            if est=='fourier':precision.annotate(str(index[row['case_id']]+1),(row['engine_wall_seconds'],100*row['relative_ci_halfwidth']),xytext=(4,4),textcoords='offset points',fontsize=7,color=color)
    labels=[f"{i+1}. A={c['acceleration']:g}, tau*={c['tau_star']:g}, dt*={c['dt_star']:g}" for i,c in enumerate(cases)]
    for ax in (eta,temp,response,shear,cost):
        ax.set_yticks(range(len(cases)),labels,fontsize=8);ax.set_ylim(len(cases)-.5,-.5);ax.grid(axis='x',alpha=.2)
    eta.set(xlabel='Viscosity (mPa s)',title='All estimators; conditional 95% intervals')
    temp.axvline(0,color='black',ls='--',lw=.8);temp.set(xlabel='Measured thermal T minus bath T (K)',title='Temperature: mean and block SE')
    response.set(xlabel='Mean response / acceleration (ps)',title='Drive-normalized response; all estimators')
    shear.set(xlabel='k × |mean fitted amplitude| (ps⁻¹)',title='Cosine-mode peak shear rate')
    cost.set(xlabel='Measured engine wall time (s)',title='Run cost; reference includes Python');cost.legend(fontsize=8)
    precision.axhline(5,color='gray',ls='--',lw=.8)
    precision.set(xlabel='Measured engine wall time (s)',ylabel='Conditional relative CI half-width (%)',title='Observed precision/cost; bias not accepted')
    precision.grid(alpha=.2)
    handles,labels=eta.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.035),ncol=6,fontsize=9)
    fig.suptitle('Cosine sensitivity: common structural origin; no pooling or formal equivalence\nA in nm/ps²; tau* and dt* in LJ units',fontsize=15,y=1.10)
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('.png','.svg'):fig.savefig(output.with_suffix(ext),dpi=160,bbox_inches='tight')
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(summary_sha256=hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),plan_sha256=summary['plan_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),display='Every predeclared case and estimator; numbers denote cases; 5 percent line is a future precision target, not acceptance.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.directory,a.output)
