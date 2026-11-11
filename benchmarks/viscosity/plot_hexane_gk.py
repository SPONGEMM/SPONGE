#!/usr/bin/env python3
"""Plot every predeclared GK window; no best-window or convergence selection."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(analysis,output):
    analysis,output=Path(analysis).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('new plot directory required')
    source=analysis/'summary.json';report=json.loads(source.read_text())
    if report['status']!='conditional_finite_window_diagnostics_not_transport_acceptance':
        raise ValueError('complete conditional GK analysis required')
    replicas=report['replicas'];aggregate=report['aggregate']
    keys=sorted({row['setting'] for row in aggregate})
    parsed={}
    for key in keys:
        match=re.fullmatch(r'start-([\d.]+)-decimation-(\d+)-demean-([01])',key)
        if not match:raise ValueError('unknown setting '+key)
        parsed[key]=(float(match[1]),int(match[2]),int(match[3]))
    starts=sorted({v[0] for v in parsed.values()});decimations=sorted({v[1] for v in parsed.values()})
    curves={};input_hashes={'summary.json':digest(source)}
    for replica in replicas:
        name=replica['replica'];curves[name]={}
        for key in keys:
            path=analysis/name/(key+'.csv')
            data=np.loadtxt(path,delimiter=',',skiprows=1)
            if data.ndim!=2 or data.shape[1]!=7 or not np.all(np.isfinite(data)) or np.any(np.diff(data[:,0])<=0):
                raise ValueError('invalid integral curve '+str(path))
            estimates=[row for row in replica['estimates'] if row['setting']==key]
            if not estimates:raise ValueError('missing replica estimates')
            for row in estimates:
                index=np.argmin(abs(data[:,0]-row['window_ps']))
                if abs(data[index,0]-row['window_ps'])>1e-9:raise ValueError('missing integral lag')
                np.testing.assert_allclose(data[index,4:7],row['component_mpa_s'],rtol=1e-12,atol=1e-14)
            curves[name][key]=(data,estimates[0]['temperature_k'])
            input_hashes[str(path.relative_to(analysis))]=digest(path)
    output.mkdir(parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    colors=plt.get_cmap('tab10').colors
    for column,demean in enumerate((0,1)):
        chosen=[key for key,value in parsed.items() if value==(starts[0],decimations[0],demean)]
        if len(chosen)!=1:raise ValueError('missing predeclared first-start/finest-sampling curve')
        key=chosen[0];ax=axes[0,column];numerators=[];temperatures=[];lags=None
        for i,replica in enumerate(replicas):
            data,temperature=curves[replica['replica']][key]
            if lags is not None:np.testing.assert_array_equal(lags,data[:,0])
            lags=data[:,0];eta=data[:,4:7].mean(axis=1)
            ax.plot(lags,eta,color=colors[i%len(colors)],alpha=.7,lw=1,label=replica['replica'])
            numerators.append(eta*temperature);temperatures.append(temperature)
        ax.plot(lags,np.mean(numerators,axis=0)/np.mean(temperatures),color='black',lw=2,label='pooled integral / T')
        ax.axhline(0,color='.7',lw=.7)
        ax.set(title=f'{"Block-demeaned" if demean else "Raw"} stress: discard {starts[0]:g} ps, decimation {decimations[0]}',
               xlabel='Correlation lag (ps)',ylabel='Integrated viscosity (mPa s)')
        ax.legend(fontsize=8,ncol=3)
        ax=axes[1,column]
        for key in sorted(keys,key=lambda k:parsed[k]):
            start,decimation,d=parsed[key]
            if d!=demean:continue
            rows=sorted((row for row in aggregate if row['setting']==key),key=lambda row:row['window_ps'])
            x=np.array([row['window_ps'] for row in rows]);y=np.array([row['viscosity_mpa_s'] for row in rows])
            interval=np.array([row['conditional_replica_95_interval_mpa_s'] for row in rows])
            # Draw endpoints directly: a percentile interval need not contain
            # the plug-in estimate, so matplotlib's nonnegative yerr is unsafe.
            color=colors[starts.index(start)%len(colors)]
            shift=np.exp(.035*(decimations.index(decimation)-(len(decimations)-1)/2))
            ax.vlines(x*shift,interval[:,0],interval[:,1],color=color,alpha=.45,lw=1)
            ax.plot(x*shift,y,marker=('o','s','^')[decimations.index(decimation)%3],
                    linestyle=('-','--',':')[decimations.index(decimation)%3],color=color,ms=4,lw=1,
                    label=f'discard {start:g} ps, decim. {decimation}')
        ax.axhline(0,color='.7',lw=.7);ax.set_xscale('log')
        windows=sorted({r['window_ps'] for r in aggregate})
        ax.set_xticks(windows,[f'{v:g}' for v in windows])
        ax.set(xlabel='Declared correlation window (ps; markers offset for legibility)',
               ylabel='Finite-window viscosity (mPa s)',title='All declared settings; conditional 95% replica intervals')
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,fontsize=9,frameon=False)
    fig.suptitle('Hexane GK: finite-window diagnostics, not transport acceptance',fontsize=15)
    fig.text(.5,.018,f'{len(replicas)} replicas share structural ancestry. Intervals exclude equilibrium, timestep, size and force-field biases.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.95))
    for suffix in ('png','pdf'):fig.savefig(output/f'integrals-and-windows.{suffix}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4.5));x=np.arange(len(replicas))
    for j,start in enumerate(starts):
        states=[next(s for s in r['states'] if s['start_ps']==start) for r in replicas]
        axes[0].plot(x,[s['mean_temperature_k'] for s in states],'o-',label=f'discard {start:g} ps')
        axes[1].plot(x,[np.mean(s['mean_pressure_diagonal_pa'])/1e6 for s in states],'o-',label=f'discard {start:g} ps')
    engine=np.array([r['engine_wall_seconds'] for r in replicas]);total=np.array([r['total_wall_seconds'] for r in replicas])
    axes[2].bar(x,engine,color='.5',label='engine')
    axes[2].bar(x,total,facecolor='none',edgecolor='black',label='engine + run audit')
    axes[2].set_ylim(0,1.25*max(total))
    for ax in axes:ax.set_xticks(x,[r['replica'] for r in replicas]);ax.legend(fontsize=8)
    axes[0].set(ylabel='Measured mean temperature (K)',title='Actual replica states')
    axes[1].set(ylabel='Trace(pressure) / 3 (MPa)',title='Fixed density does not fix pressure')
    axes[2].set(ylabel='Wall time (s)',title='Measured elapsed time; shared host')
    fig.suptitle('State and cost diagnostics; no cost-to-precision claim',fontsize=14)
    fig.text(.5,.015,f'Campaign wall: {report["sampling_wall_seconds"]:.1f} s; analysis: {report["analysis_wall_seconds"]:.2f} s. '
             'Nested engine/run times must not be added twice.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93))
    for suffix in ('png','pdf'):fig.savefig(output/f'states-and-cost.{suffix}',dpi=180)
    plt.close(fig)
    record=dict(status='plotted_conditional_diagnostics',source_sha256=digest(__file__),inputs=input_hashes,
                outputs={p.name:digest(p) for p in output.iterdir()},formal_viscosity_acceptance=False)
    (output/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=plot(args.analysis,args.output)
    print(json.dumps(dict(status=result['status'],outputs=list(result['outputs'])),indent=2))
