#!/usr/bin/env python3
"""Show all cosine pilot estimators, spatial fits, thermal state and COM drift."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(paths, output):
    output=Path(output)
    if any(output.with_suffix(ext).exists() for ext in ('.png','.svg','.json')):
        raise ValueError('plot output exists')
    fig,axes=plt.subplots(4,len(paths),figsize=(6.5*len(paths),12),squeeze=False)
    colors=dict(sponge='#176b9d',lammps='#bf581c'); inputs=[]; max_eta=0.
    for col,path in enumerate(map(Path,paths)):
        plan=json.loads((path/'plan.json').read_text());report=json.loads((path/'summary.json').read_text())
        if plan['experiment_kind']!='thermal-pilot': raise ValueError('only thermal pilots have viscosity plots')
        inputs.append(dict(path=str(path.resolve()),summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest()))
        profile_ax,eta_ax,temp_ax,com_ax=axes[:,col]
        profile_ax.set_title(f"A = {plan['acceleration_nm_ps2']} nm/ps²; N = {plan['model']['atoms']}")
        labels=[]
        for engine,records in report['analyses'].items():
            color=colors[engine]; first=records[0]
            with (path/f"{engine}-{first['estimator']}-profile.csv").open() as file: rows=list(csv.DictReader(file))
            def values(name): return np.array([float(row[name]) if row[name] else np.nan for row in rows])
            z=values('center_nm'); offset=first['spatial_fit']['offset_nm_per_ps']
            velocity=values('velocity_nm_per_ps')-offset
            fit=first['spatial_fit']['cosine_nm_per_ps']*values('mean_cosine')+first['spatial_fit']['sine_nm_per_ps']*values('mean_sine')
            profile_ax.plot(z/(plan['model']['box_angstrom']/10),velocity,'o',ms=4,color=color,label=engine.upper())
            profile_ax.plot(z/(plan['model']['box_angstrom']/10),fit,'-',color=color)
            for index,record in enumerate(records):
                y=index+(0 if engine=='sponge' else len(records));labels.append((y,f"{engine.upper()} {record['estimator']}"))
                eta=record['viscosity_mpa_s'];interval=record['conditional_bootstrap_95_interval_mpa_s']
                if eta is not None: eta_ax.plot(eta,y,'o',color=color);max_eta=max(max_eta,eta)
                if interval is not None:
                    eta_ax.hlines(y,*interval,color=color,lw=2);max_eta=max(max_eta,interval[1])
                else: eta_ax.text(.02,y,'unresolved interval',transform=eta_ax.get_yaxis_transform(),va='center',fontsize=8)
            with np.load(path/f'{engine}-canonical-profile.npz') as data:
                time=data['time']/(plan[f'dt_{engine}_ps']/plan['model']['dt_star'])
                temp_ax.plot(time,data['thermal_temperature'],color=color,lw=.45,alpha=.35)
                com_ax.plot(time,data['com_vx'],color=color,lw=.8,label=engine.upper())
            temp_ax.axvline(plan['start_star'],ls='--',color='gray',lw=.6)
            com_ax.axvline(plan['start_star'],ls='--',color='gray',lw=.6)
        profile_ax.set(xlabel='z / L',ylabel='Flow minus fitted offset (nm/ps)');profile_ax.legend(fontsize=8)
        eta_ax.set_yticks([y for y,_ in labels],[label for _,label in labels],fontsize=8)
        eta_ax.set(xlabel='Viscosity (mPa s); conditional 95% interval',ylim=(len(labels)-.5,-.5))
        temp_ax.axhline(plan['temperature_sponge_k'],ls='--',color='black',lw=1,label='SPONGE bath setpoint')
        temp_ax.legend(fontsize=8)
        temp_ax.set(xlabel='Time / LJ time',ylabel='Projected thermal temperature (K)')
        com_ax.set(xlabel='Time / LJ time',ylabel='Flow-direction COM (nm/ps)')
        for ax in axes[:,col]: ax.grid(alpha=.2)
    if max_eta>0:
        for ax in axes[1]: ax.set_xlim(0,max_eta*1.08)
    fig.suptitle('Cosine pilots: SPONGE vs native LAMMPS forces + custom projected Middle',fontsize=14)
    fig.text(.5,.012,'Same structural seed; independent RNGs. Conditional block intervals; no weak-drive, long-wave or equivalence acceptance.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96));output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('.png','.svg'): fig.savefig(output.with_suffix(ext),dpi=160)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(inputs=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.runs,args.output)
