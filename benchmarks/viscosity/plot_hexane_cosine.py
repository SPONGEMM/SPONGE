#!/usr/bin/env python3
"""All declared cosine settings, states and conditional GK comparison.

This renderer requires completed frozen analysis and replays its saved
bootstrap arrays. Plots never select a preferred estimator/window or infer
transport convergence from a small interval.
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hexane_cosine_analysis as analysis

COLORS=dict(fourier='#2463a5',com_fourier='#bc6021',fit='#218066')
MARKERS=dict(fourier='o',com_fourier='s',fit='^')


def load(root):
    root=Path(root).resolve();plan=analysis.read(root/'plan.json')
    result=analysis.read(root/'analysis/summary.json')
    execution=analysis.read(root/'execution.json')
    dependencies=analysis.verify_analysis_sources(root,plan)
    if (result['status']!='conditional_four_replica_two_drive_diagnostics'
            or result['formal_transport_acceptance'] is not False
            or result['plan_sha256']!=analysis.water.digest(root/'plan.json')
            or result['execution_sha256']!=analysis.water.digest(root/'execution.json')
            or result['analysis_readiness_sha256']!=analysis.water.digest(root/'analysis-readiness.json')
            or result['source_sha256']!=dependencies['benchmarks/viscosity/hexane_cosine_analysis.py']
            or result['dependencies']!=dependencies
            or result['bootstrap_sha256']!=analysis.water.digest(root/'analysis/bootstrap.npz')):
        raise ValueError('completed frozen analysis identity mismatch')
    if execution['status']!='completed_sampling_pending_analysis' or execution['failures'] or execution['active']:
        raise ValueError('completed sampling required')
    expected=[dict(start_ps=s,estimator=e,drive=d['id']) for s in plan['starts_ps']
              for e in plan['estimators'] for d in plan['drives']]
    if len(result['settings'])!=len(expected) or any(any(r[k]!=v for k,v in e.items()) for r,e in zip(result['settings'],expected)):
        raise ValueError('all declared settings required in order')
    if [r['id'] for r in result['replicas']]!=[r['id'] for r in plan['replicas']]:
        raise ValueError('replica order mismatch')
    for rep in result['replicas']:
        for drive in plan['drives']:
            if rep['audits'][drive['id']]!=analysis.read(root/'runs'/rep['id']/drive['id']/'summary.json'):
                raise ValueError('case audit differs from analysis')
    if (result['costs']['campaign_wall_seconds']!=execution['total_wall_seconds'] or
            result['costs']['summed_engine_seconds']!=sum(r['engine_wall_seconds'] for r in execution['results'])):
        raise ValueError('cost differs from execution journal')
    values=[]
    for key in ('fluidity_per_mpa_s','temperature_k','pressure_mpa'):
        values.append(np.array([[e[key] for e in r['entries']] for r in result['replicas']]))
    reduced,samples=analysis.reduce(*values,result['density_kg_m3'],plan['replica_bootstrap_draws'],plan['replica_bootstrap_seed'])
    for expected,item in zip(reduced,result['settings']):
        if expected!={k:item[k] for k in expected}:raise ValueError('reduced estimate mismatch')
    with np.load(root/'analysis/bootstrap.npz') as saved:
        for key,value in samples.items():np.testing.assert_array_equal(saved[key],value)
        for i in range(len(result['paired_drives'])):
            j=2*i
            with np.errstate(divide='ignore',invalid='ignore'):
                ratios=samples['fluidity_per_mpa_s'][:,j]/samples['fluidity_per_mpa_s'][:,j+1]
            np.testing.assert_array_equal(saved['paired_fluidity_ratio_'+str(i)],ratios)
            row=result['paired_drives'][i]
            good=all(reduced[k]['all_draws_positive'] and reduced[k]['fluidity_95_interval'][0]>0 for k in (j,j+1))
            means=[reduced[k]['fluidity_per_mpa_s'] for k in (j,j+1)]
            point_ratio=means[0]/means[1] if min(means)>0 else None
            if (row['start_ps']!=result['settings'][j]['start_ps'] or row['estimator']!=result['settings'][j]['estimator']
                    or row['fluidity_ratio_low_over_high']!=point_ratio or row['all_draws_positive']!=good
                    or row['fluidity_ratio_95_interval']!=(analysis.quantile(ratios) if good else None)):
                raise ValueError('paired interval mismatch')
    gkroot=Path(plan['source_gk_archive']);manifest=analysis.read(gkroot/'manifest.json')
    if analysis.water.digest(gkroot/'manifest.json')!=plan['source_gk_manifest_sha256']:
        raise ValueError('closed GK manifest mismatch')
    consumed='analysis/summary.json'
    entry=next(r for r in manifest['files'] if r['path']==consumed)
    if analysis.water.digest(gkroot/consumed)!=entry['sha256']:raise ValueError('GK summary changed')
    gk=analysis.read(gkroot/consumed)
    if gk['status']!='conditional_finite_window_diagnostics_not_transport_acceptance' or len(gk['aggregate'])!=48:
        raise ValueError('complete GK comparison required')
    return plan,result,gk,dict(analysis_summary_sha256=analysis.water.digest(root/'analysis/summary.json'),
        bootstrap_sha256=result['bootstrap_sha256'],gk_summary_sha256=entry['sha256'],
        gk_manifest_sha256=plan['source_gk_manifest_sha256'])


def point(ax,x,value,interval,color,marker,label=None):
    """Segments accept percentile intervals not containing the point estimate."""
    if value is None:
        ax.plot([x],[.04],marker='x',color=color,transform=ax.get_xaxis_transform(),clip_on=False,label=label)
        return
    ax.plot([x],[value],marker=marker,ls='none',color=color,
        markerfacecolor=color if interval is not None else 'none',label=label)
    if interval is not None:
        ax.plot([x,x],interval,color=color,lw=1.1)
        ax.plot([x,x],interval,color=color,marker='_',ls='none')


def style(ax):
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)


def draw(result,gk,plan,output,prefix=''):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    starts=plan['starts_ps'];drives=plan['drives'];settings=result['settings']
    if len(starts)!=2 or len(drives)!=2:raise ValueError('renderer expects two declared windows and drives')
    paths=[]
    def finish(fig,name,title):
        fig.suptitle(prefix+title,fontsize=13)
        fig.text(.5,.015,'Conditional whole-replica intervals; no steady-state, weak-drive, size or experimental acceptance.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.045,1,.94))
        for suffix in ('png','pdf'):
            path=output/(name+'.'+suffix);fig.savefig(path,dpi=180);paths.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for row,start in enumerate(starts):
        for j,estimator in enumerate(plan['estimators']):
            for i,drive in enumerate(drives):
                s=next(s for s in settings if s['start_ps']==start and s['estimator']==estimator and s['drive']==drive['id'])
                x=i+(j-1)*.09
                point(axes[row,0],x,s['viscosity_mpa_s'],s['viscosity_95_interval_mpa_s'],COLORS[estimator],MARKERS[estimator],estimator if i==0 else None)
                point(axes[row,1],x,s['ratio_to_reference'],s['ratio_to_reference_95_interval'],COLORS[estimator],MARKERS[estimator])
            pair=next(p for p in result['paired_drives'] if p['start_ps']==start and p['estimator']==estimator)
            point(axes[row,2],j,pair['fluidity_ratio_low_over_high'],pair['fluidity_ratio_95_interval'],COLORS[estimator],MARKERS[estimator])
        for col in range(2):
            axes[row,col].set_xticks([0,1],[str(d['acceleration_nm_ps2']) for d in drives])
            axes[row,col].set_xlabel('Drive A (nm/ps²)');axes[row,col].set_xlim(-.3,1.3)
        axes[row,0].set_ylabel('Viscosity (mPa s)');axes[row,0].legend(fontsize=8)
        axes[row,1].axhline(1,color='black',lw=.8,ls='--');axes[row,1].set_ylabel('Viscosity / same-T,rho reference')
        axes[row,2].axhline(1,color='black',lw=.8,ls='--');axes[row,2].set_ylabel('Fluidity: low drive / high drive')
        axes[row,2].set_xticks(range(3),plan['estimators'],rotation=15);axes[row,2].set_xlim(-.4,2.4)
        for ax in axes[row]:ax.set_title(f'Discard {start:g} ps');style(ax)
    finish(fig,'response-and-reference','Hexane cosine: all 12 declared settings\nOpen marker: unresolved interval; x: unresolved point')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for col,start in enumerate(starts):
        for ri,rep in enumerate(result['replicas']):
            for row,key in enumerate(('temperature_k','pressure_mpa')):
                values=[]
                for drive in drives:
                    entries=[e for e,s in zip(rep['entries'],settings) if s['start_ps']==start and s['drive']==drive['id']]
                    if any(e[key]!=entries[0][key] for e in entries):raise ValueError('state varies across estimators')
                    values.append(entries[0][key])
                axes[row,col].plot(np.arange(2)+(ri-1.5)*.025,values,'o-',label=rep['id'],alpha=.85)
        for row in range(2):
            axes[row,col].set_xticks([0,1],[str(d['acceleration_nm_ps2']) for d in drives]);axes[row,col].set_xlabel('Drive A (nm/ps²)')
            axes[row,col].set_title(f'Discard {start:g} ps; per-replica means');style(axes[row,col])
        axes[0,col].set_ylabel('Thermal temperature (K)');axes[0,col].legend(ncol=2)
        axes[1,col].set_ylabel('Mean trace(P)/3 (MPa)');axes[1,col].axhline(0,color='black',lw=.8,ls='--')
    finish(fig,'states','Actual thermal state of each replica\nFixed density does not imply experimental pressure')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    palette=['#2463a5','#bc6021'];markers={1:'o',2:'s',4:'^'}
    for col,demean in enumerate((0,1)):
        keys=sorted({s['setting'] for s in gk['aggregate']})
        for key in keys:
            match=re.fullmatch(r'start-([\d.]+)-decimation-(\d+)-demean-([01])',key)
            if not match:raise ValueError('unrecognized GK setting')
            start,dec,dm=float(match[1]),int(match[2]),int(match[3])
            if dm!=demean:continue
            rows=[s for s in gk['aggregate'] if s['setting']==key]
            color=palette[starts.index(start)]
            for index,s in enumerate(rows):
                offset=(dec-2)*.025+(starts.index(start)-.5)*.015
                point(axes[0,col],s['window_ps']+offset,s['viscosity_mpa_s'],s['conditional_replica_95_interval_mpa_s'],color,markers[dec],f'discard {start:g}; stride ×{dec}' if index==0 else None)
        axes[0,col].set_title('GK: '+('block-demeaned' if demean else 'raw correlations'))
        axes[0,col].set_xlabel('Finite integration window (ps)');axes[0,col].set_ylabel('Viscosity (mPa s)')
        axes[0,col].legend(fontsize=7,ncol=2);style(axes[0,col])
    for j,s in enumerate(settings):
        estimator=s['estimator'];x=j
        point(axes[1,0],x,s['viscosity_mpa_s'],s['viscosity_95_interval_mpa_s'],COLORS[estimator],MARKERS[estimator])
    axes[1,0].set_xticks(range(len(settings)),[f"{s['start_ps']:g}/{s['drive'][1:]}" for s in settings],rotation=55,fontsize=7)
    axes[1,0].set_xlabel('Discard ps / A; Fourier, COM Fourier, fit within each window')
    axes[1,0].set_ylabel('Viscosity (mPa s)');axes[1,0].set_title('Cosine: all 12 settings');style(axes[1,0])
    comparison_axes=(axes[0,0],axes[0,1],axes[1,0])
    low=min(ax.get_ylim()[0] for ax in comparison_axes)
    high=max(ax.get_ylim()[1] for ax in comparison_axes)
    for ax in comparison_axes:ax.set_ylim(low,high)
    costs=result['costs'];axes[1,1].axis('off')
    gkt=gk['sampling_wall_seconds'];gke=sum(r['engine_wall_seconds'] for r in gk['replicas'])
    text=(f'GK: {gkt:.1f} s campaign; {gke:.1f} s summed engines\n'
          f"Cosine: {costs['campaign_wall_seconds']:.1f} s campaign;\n"
          f"             {costs['summed_engine_seconds']:.1f} s summed engines\n\n"
          'Different concurrency and shared-host load.\nCost to specified precision: not established.\n\n'
          'GK: NVE; replica temperatures differ.\nCosine: flow-aware NVT; actual T shown separately.\n'
          'Same prepared q/u does not imply equal ensembles.\n\n'
          'All 48 GK + 12 cosine conditions retained.\nNo method-equivalence or experimental pass claimed.')
    axes[1,1].text(.02,.98,text,va='top',fontsize=10,linespacing=1.6)
    finish(fig,'all-method-settings','Conditional comparison across methods\nState and ensemble mismatch prevents an equivalence claim')
    return paths


def run(root,output):
    root=Path(root).resolve();plan,result,gk,inputs=load(root)
    paths=draw(result,gk,plan,output)
    provenance=dict(source_sha256=analysis.water.digest(__file__),inputs=inputs,
        figures={p.name:analysis.water.digest(p) for p in paths},formal_transport_acceptance=False)
    (Path(output)/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.campaign,args.output),indent=2))
