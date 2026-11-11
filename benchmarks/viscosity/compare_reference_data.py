#!/usr/bin/env python3
"""Reference-vs-measurement provenance report; contains no simulated viscosity."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import experimental_reference as ref

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    data=Path(__file__).resolve().parent/'data';source=data/'n_hexane_klein2019.csv'
    with source.open() as f:rows=list(csv.DictReader(f))
    results=[]
    for row in rows:
        t=float(row['temperature_k']);rho=float(row['liquid_density_kg_m3']);measured=float(row['viscosity_mpa_s'])
        predicted=ref.n_hexane_viscosity_pa_s(t,rho)*1000
        results.append(dict(temperature_k=t,density_kg_m3=rho,measured_mpa_s=measured,
            correlation_mpa_s=predicted,correlation_minus_measured_percent=100*(predicted/measured-1),
            measurement_relative_expanded_uncertainty_percent=float(row['relative_expanded_uncertainty_percent']),
            correlation_liquid_relative_expanded_uncertainty_percent=2. if t<=450 else 6.,
            uncertainty_assignment='Conditional on the source saturation-liquid interpretation; no separate EOS/pressure check.',
            state_path=row['state_path'],not_an_md_result=True))
    with (output/'hexane-reference-comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    t=np.array([r['temperature_k'] for r in results]);exp=np.array([r['measured_mpa_s'] for r in results]);corr=np.array([r['correlation_mpa_s'] for r in results]);u=np.array([r['measurement_relative_expanded_uncertainty_percent'] for r in results]);uc=np.array([r['correlation_liquid_relative_expanded_uncertainty_percent'] for r in results])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axes[0].errorbar(t,exp,yerr=exp*u/100,fmt='o',ms=4,capsize=3,label='Klein2019 SLS (k=2)')
    axes[0].plot(t,corr,'-s',ms=3,label='2013 correlation at tabulated T,density')
    axes[0].fill_between(t,corr*(1-uc/100),corr*(1+uc/100),alpha=.15,color='C1',label='2013 liquid expanded uncertainty')
    axes[0].set(xlabel='Temperature (K)',ylabel='Viscosity (mPa s)',title='n-Hexane: two reference sources')
    bias=100*(corr/exp-1)
    axes[1].plot(t,bias,'o-',color='C2',label='(correlation / SLS -1) x100')
    axes[1].fill_between(t,-u,u,alpha=.18,color='C0',label='SLS relative uncertainty (k=2)')
    axes[1].plot(t,uc,':',color='C1',label='2013 liquid uncertainty bounds');axes[1].plot(t,-uc,':',color='C1')
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xlabel='Temperature (K)',ylabel='Relative difference (%)',title='Uncertainties shown separately, not combined')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Reference preparation only - no SPONGE/MD prediction\nSaturation approximation with helium; not constant total pressure 0.1 MPa',fontsize=12)
    fig.savefig(output/'reference-comparison.png',dpi=180);fig.savefig(output/'reference-comparison.svg');plt.close(fig)
    shutil.copy2(source,output/source.name);shutil.copy2(source.with_suffix('.metadata.json'),output/source.with_suffix('.metadata.json').name)
    for p in (Path(__file__),Path(ref.__file__)):shutil.copy2(p,output/p.name)
    record=dict(status='reference_data_prepared_no_md_comparison',rows=results,
        source_sha256={p.name:sha(p) for p in (source,source.with_suffix('.metadata.json'),Path(__file__),Path(ref.__file__))},
        source_urls=[ref.WATER_URL,ref.HEXANE_URL,'https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=928174'],
        water_example=ref.reference('water',298.15,998),
        warnings=['Correlation and original experiment are different reference types. No MD curve or force-field accuracy claim.',
            'Density values are literature inputs to the SLS analysis. Applying the2013 correlation there is a T,rho comparison.',
            'No interpolation to298.15/328.15K or conversion to a0.1MPa isobar. No EOS is provided.',
            'Uncertainty components may correlate across temperatures and sources; they are not combined as independent noise.'])
    (output/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=build(a.output);print(json.dumps({'status':r['status'],'points':len(r['rows'])}))
