#!/usr/bin/env python3
"""Published T,rho viscosity correlations, not MD or raw measurements.

No equation of state or phase validation is supplied here. A finite result
does not establish that a T,rho pair is a stable state in the published domain.
Water: IAPWS R12-08, equations 10-12, mu2=1 outside the critical region.
n-Hexane: Michailidou et al. (2013), equations 1-8, DOI 10.1063/1.4818980.
"""
import argparse
import json
import math
from numbers import Real

WATER_URL='https://iapws.org/technical-guidance/release/viscosity.download'
HEXANE_URL='https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=913970'
WATER_H=(1.67752,2.20462,.6366564,-.241605)
WATER_HIJ=((0,0,.520094),(1,0,.0850895),(2,0,-1.08374),(3,0,-.289555),
    (0,1,.222531),(1,1,.999115),(2,1,1.88797),(3,1,1.26613),(5,1,.120573),
    (0,2,-.281378),(1,2,-.906851),(2,2,-.772479),(3,2,-.489837),(4,2,-.257040),
    (0,3,.161913),(1,3,.257399),(0,4,-.0325372),(3,4,.0698452),(4,5,.00872102),
    (3,6,-.00435673),(5,6,-.000593264))
HEXANE_B=(-19.572881,219.73999,-1015.3226,2471.0125,-3375.1717,2491.6597,-787.26086,14.085455,-.34664158)
HEXANE_C=(2.53402335,-9.724061002,.469437316,158.5571631,72.42916856,10.60751253,8.628373915,-6.61346441,-2.212724566)


def _state(temperature_k,density_kg_m3,low,high):
    for v in (temperature_k,density_kg_m3):
        if isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v):
            raise ValueError('finite numerical temperature and density required')
    if not low<=temperature_k<=high or density_kg_m3<0:
        raise ValueError('temperature outside implemented range or negative density')
    return float(temperature_k),float(density_kg_m3)


def water_viscosity_pa_s(temperature_k,density_kg_m3):
    """Noncritical IAPWS correlation; caller must establish fluid/EOS validity."""
    t,rho=_state(temperature_k,density_kg_m3,273.16,1173.15)
    if 645.91<=t<=650.77 and 245.8<=rho<=405.3:
        raise ValueError('critical enhancement required; simplified water reference unavailable')
    tr=t/647.096;dr=rho/322.
    dilute=100*math.sqrt(tr)/sum(h/tr**i for i,h in enumerate(WATER_H))
    exponent=dr*sum(h*(1/tr-1)**i*(dr-1)**j for i,j,h in WATER_HIJ)
    try:eta=dilute*math.exp(exponent)*1e-6
    except OverflowError as error:raise ValueError('reference overflow; inspect state/domain') from error
    if not math.isfinite(eta) or eta<=0:raise ValueError('nonpositive/nonfinite reference; inspect state/domain')
    return eta


def n_hexane_viscosity_pa_s(temperature_k,density_kg_m3):
    """2013 correlation with omitted critical enhancement, as in the source."""
    t,rho=_state(temperature_k,density_kg_m3,177.83,600.)
    ts=t/378.4;logts=math.log(ts);molar_mass=.08617536;sigma_m=.6334e-9
    collision=math.exp(.18760-.48430*logts+.04477*logts**2)
    eta0=.021357*math.sqrt(molar_mass*1000*t)/(.6334**2*collision) # microPa s
    bstar=sum(HEXANE_B[i]*ts**(-.25*i) for i in range(7))+HEXANE_B[7]*ts**-2.5+HEXANE_B[8]*ts**-5.5
    # B_eta has molar-volume units. Convert MASS density to molar density.
    initial=eta0*bstar*6.02214076e23*sigma_m**3*rho/molar_mass
    tr=t/507.82;dr=rho/233.182;c=HEXANE_C
    denominator=c[5]+c[6]*tr+c[7]*dr+dr**2+c[8]*dr*tr
    if denominator<=0:raise ValueError('residual denominator outside admissible correlation branch')
    residual=dr**(2/3)*math.sqrt(tr)*(c[0]/tr+c[1]/(c[2]+tr+c[3]*dr**2)+c[4]*(1+dr)/denominator)
    eta=(eta0+initial+residual)*1e-6
    if not math.isfinite(eta) or eta<=0:raise ValueError('nonpositive/nonfinite reference; inspect state/domain')
    return eta


def reference(fluid,temperature_k,density_kg_m3):
    if fluid not in ('water','n_hexane'):raise ValueError('unsupported fluid')
    fn=water_viscosity_pa_s if fluid=='water' else n_hexane_viscosity_pa_s
    return dict(fluid=fluid,temperature_k=temperature_k,density_kg_m3=density_kg_m3,
        viscosity_pa_s=fn(temperature_k,density_kg_m3),reference_type='experiment_based_reference_correlation',
        source_url=WATER_URL if fluid=='water' else HEXANE_URL,
        equation_of_state_used=False,pressure_mpa=None,phase_and_pressure_validated=False,
        source_domain_note='Stable fluid; IAPWS pressure-dependent temperature limits and melting boundary require a separate EOS check.' if fluid=='water' else 'Triple point to600K, pressures up to100MPa; not checked by this T,rho evaluator.',
        critical_enhancement='omitted; water critical rectangle rejected' if fluid=='water' else 'omitted in source correlation',
        relative_expanded_uncertainty=None,
        uncertainty_note='State/phase dependent; no automatic assignment from T,rho alone. Reference uncertainties may be correlated across states.',
        not_an_md_result=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fluid',choices=['water','n_hexane'],required=True)
    p.add_argument('--temperature',type=float,required=True);p.add_argument('--density',type=float,required=True)
    a=p.parse_args();print(json.dumps(reference(a.fluid,a.temperature,a.density),indent=2,allow_nan=False))
