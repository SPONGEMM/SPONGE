#!/usr/bin/env python3
"""Fixed-bond TraPPE-UA candidate inputs and same-state SPONGE/LAMMPS gate.

The ordered initial structure is not an equilibrated liquid. Model density,
transport convergence, cutoff effects and experimental agreement are separate.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import time
import numpy as np
from lj_model import digest, execute
from water_model import array_file, save, read_sponge, PA, AXES

KB_PARAMETER = .00198720425864083
SIGMA = np.array([3.75, 3.95])
EPSILON_K = np.array([98., 46.])
MASS = np.array([15.035, 14.027], dtype=np.float32).astype(float)
ANGLE_K = float(np.float32(31250*KB_PARAMETER))
THETA = float(np.float32(np.deg2rad(114)))
TORSION = (np.array([355.03, -68.19, 791.32])*KB_PARAMETER).astype(np.float32).astype(float)
LENGTH = float(np.float32(1.54))
MODEL_FILES = {'angle.txt', 'charge.txt', 'constraint.txt', 'coordinate.txt',
               'dihedral.txt', 'exclude.txt', 'lammps.data', 'lj.txt', 'mass.txt',
               'residue.txt', 'state.npz', 'velocity.txt'}


def coefficients():
    values=[]
    for i,j in ((0,0),(1,0),(1,1)):
        sigma=(SIGMA[i]+SIGMA[j])/2
        epsilon=np.sqrt(EPSILON_K[i]*EPSILON_K[j])*KB_PARAMETER
        raw=np.array([4*epsilon*sigma**12,4*epsilon*sigma**6],dtype=np.float32)
        effective=(raw*np.array([12,6],dtype=np.float32)).astype(float)/[12,6]
        values.append(dict(types=[i,j],raw=raw.astype(float).tolist(),effective=effective.tolist(),
                           epsilon=effective[1]**2/(4*effective[0]),sigma=(effective[0]/effective[1])**(1/6)))
    return values


def prepare(directory, cells=(3,6,6), density=659.29, temperature=293.15, cutoff=14.):
    if len(cells)!=3 or any(isinstance(c,bool) or int(c)!=c or c<1 for c in cells):
        raise ValueError('three positive integer cell counts required')
    if any(not np.isfinite(v) or v<=0 for v in (density,temperature,cutoff)):
        raise ValueError('finite positive state and cutoff required')
    molecules=int(np.prod(cells)); n=6*molecules
    if molecules<2: raise ValueError('at least two molecules required for liquid fixture')
    types=np.tile([0,1,1,1,1,0],molecules); mass=MASS[types]
    box=float(np.float32((mass.sum()*1.66053906660e3/density)**(1/3)))
    cutoff=float(np.float32(cutoff))
    if 2*cutoff>=box: raise ValueError('cutoff must be less than half actual box')
    # Planar all-trans chain, long axis x. Fixed lattice is declared, not hidden
    # behind a claim of random liquid packing. Each bond angle is 114 degrees.
    alpha=(np.pi-np.deg2rad(114))/2
    body=np.array([[i*LENGTH*np.cos(alpha),(i%2)*LENGTH*np.sin(alpha),0.] for i in range(6)])
    body-=np.average(body,weights=mass[:6],axis=0)
    spacing=box/np.array(cells)
    if np.ptp(body[:,0])>=spacing[0]: raise ValueError('chain spans its x cell; choose fewer x cells')
    positions=[]
    for i,j,k in np.ndindex(*cells):
        local=body.copy()
        # Alternating zigzag planes reduce an artificial common dipole-like
        # shape direction but do not make the chain-axis distribution isotropic.
        if (j+k)%2: local[:,1]*=-1
        positions.extend((local+(np.array([i,j,k])+.5)*spacing)%box)
    q=np.asarray(positions,dtype=np.float32).astype(float)
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    pairs=np.array([(b+i,b+i+1,LENGTH) for b in range(0,n,6) for i in range(5)])
    angles=[(b+i,b+i+1,b+i+2) for b in range(0,n,6) for i in range(4)]
    torsions=[(b+i,b+i+1,b+i+2,b+i+3) for b in range(0,n,6) for i in range(3)]
    excluded=[[j for j in range(i+1,min(n,6*(i//6+1))) if j-i<=3] for i in range(n)]
    for name,value in [('coordinate',q),('mass',mass),('charge',np.zeros(n)),('velocity',np.zeros_like(q))]:
        array_file(directory/f'{name}.txt',value,box if name=='coordinate' else None)
    coef=coefficients()
    (directory/'lj.txt').write_text(f'{n} 2\n'+'\n'.join(f'{x:.17g}' for x in np.array([v['raw'] for v in coef]).T.ravel())+'\n'+'\n'.join(map(str,types))+'\n')
    (directory/'exclude.txt').write_text(f'{n} {sum(map(len,excluded))}\n'+''.join(' '.join(map(str,[len(row),*row]))+'\n' for row in excluded))
    (directory/'residue.txt').write_text(f'{n} {molecules}\n'+'6\n'*molecules)
    (directory/'constraint.txt').write_text(str(len(pairs))+'\n'+''.join(f'{int(i)} {int(j)} {r:.17g}\n' for i,j,r in pairs))
    (directory/'angle.txt').write_text(str(len(angles))+'\n'+''.join(' '.join(map(str,a))+f' {ANGLE_K:.17g} {THETA:.17g}\n' for a in angles))
    (directory/'dihedral.txt').write_text(str(len(torsions)*3)+'\n'+''.join(' '.join(map(str,a))+f' {i+1} {c:.17g} {np.pi if i==1 else 0:.17g}\n' for a in torsions for i,c in enumerate(TORSION)))
    lines=['Fixed-bond TraPPE-UA candidate; ordered initial state','',f'{n} atoms',f'{len(pairs)} bonds',f'{len(angles)} angles',f'{len(torsions)} dihedrals','2 atom types','1 bond types','1 angle types','1 dihedral types','']
    lines += [f'0 {box:.17g} {a}lo {a}hi' for a in 'xyz']
    lines += ['','Masses','',f'1 {MASS[0]:.17g}',f'2 {MASS[1]:.17g}','','Atoms # full','']
    lines += [f'{i+1} {i//6+1} {types[i]+1} 0 '+' '.join(f'{v:.17g}' for v in row) for i,row in enumerate(q)]
    for title,rows in [('Bonds',pairs[:,:2].astype(int)),('Angles',angles),('Dihedrals',torsions)]:
        lines += ['',title,'']+[f'{i+1} 1 '+' '.join(str(int(v)+1) for v in a) for i,a in enumerate(rows)]
    (directory/'lammps.data').write_text('\n'.join(lines)+'\n')
    np.savez(directory/'state.npz',positions=q,masses=mass,pairs=pairs,types=types)
    nearest=float('inf')
    for i in range(n):
        js=np.arange(i+1,n);keep=(js//6!=i//6)|(js-i>3);js=js[keep]
        d=q[i]-q[js];d-=box*np.rint(d/box)
        if len(d): nearest=min(nearest,float(np.sqrt(np.min(np.sum(d*d,axis=1)))))
    result=dict(contract='sponge.hexane.fixed-bond-candidate.v1',molecules=molecules,atoms=n,cells=list(cells),box_angstrom=box,cutoff_angstrom=cutoff,requested_density_kg_m3=density,actual_density_kg_m3=mass.sum()*1.66053906660e3/box**3,temperature_k=temperature,coefficients=coef,angle_k=ANGLE_K,theta0_radian=THETA,torsion_coefficients=TORSION.tolist(),bond_length=LENGTH,parameter_boltzmann_kcal_mol_k=KB_PARAMETER,thermal_dof=3*n-len(pairs)-4,minimum_included_pair_distance_angstrom=nearest,initialization='ordered all-trans chain-axis-x lattice with alternating zigzag planes; not equilibrated liquid',model='TraPPE-UA candidate; 355.03 K torsion first term, not GOMC rounded 355.0 K',exclusions='graph distance <=3 excluded; 1-5,2-6,1-6 retained fully',mixing='Lorentz-Berthelot before native coefficient quantization',tail='SPONGE C6-only ordered Ntype*Ntype correction; not an untruncated liquid',sources=['https://doi.org/10.1021/jp972543+','https://www.osti.gov/servlets/purl/759882'],source_note='author preprint directly verifies LJ, mixing, exclusions and fixed bond; 1998 torsion coefficient from indexed text and later SI, not a locally retrieved original PDF',input_sha256={f.name:digest(f) for f in sorted(directory.iterdir()) if f.is_file()})
    save(directory/'model.json',result);return result


def load(directory):
    directory=Path(directory).resolve();m=json.loads((directory/'model.json').read_text())
    if m['contract']!='sponge.hexane.fixed-bond-candidate.v1':raise ValueError('model contract')
    if set(m['input_sha256']) != MODEL_FILES:raise ValueError('model manifest file set')
    for name,sha in m['input_sha256'].items():
        if Path(name).name!=name or digest(directory/name)!=sha:raise ValueError('model input hash: '+name)
    state=np.load(directory/'state.npz')
    n=m['atoms']
    if n != 6*m['molecules'] or state['positions'].shape != (n,3) or state['masses'].shape != (n,) or state['pairs'].shape != (5*m['molecules'],3):
        raise ValueError('model shapes')
    if any(not np.all(np.isfinite(state[k])) for k in state.files) or np.any(state['masses']<=0):
        raise ValueError('nonfinite model or nonpositive mass')
    return m,state


def write_input(directory,source,model,steps=1,dt=0,stride=1,extra=None,viscosity='method="green_kubo"'):
    directory,source=Path(directory),Path(source).resolve()
    cfg=dict(md_name='hexane',mode='nve',pbc=True,dt=dt,step_limit=steps,cutoff=model['cutoff_angstrom'],skin=1.,print_zeroth_frame=True,write_mdout_interval=stride,write_information_interval=stride,write_trajectory_interval=stride,write_restart_file_interval=0,crd='crd.dat',vel='vel.dat',frc='force.dat',output_h5_observable_path='observables.h5')
    cfg['PM.MPI_size']=1
    for key in ('mass','charge','coordinate','velocity','LJ','exclude','residue','angle','dihedral'):
        cfg[key+'_in_file']=str(source/f'{key.lower()}.txt')
    cfg.update(extra or {})
    write_engine_input(directory/'mdin.spg.toml', '\n'.join(f'{k}={json.dumps(v)}' for k,v in cfg.items())+'\n[viscosity]\n'+viscosity+'\n')


def tail(model,state):
    count=np.bincount(state['types'],minlength=2);c6=np.zeros((2,2))
    for pair in model['coefficients']:
        i,j=pair['types'];c6[i,j]=c6[j,i]=pair['effective'][1]
    return -2*np.pi/3*(count@c6@count)/model['cutoff_angstrom']**3/model['box_angstrom']**3


def static(source,output,sponge,lammps,*,rattle=False):
    source,output,sponge,lammps=[Path(p).resolve() for p in (source,output,sponge,lammps)]
    model,state=load(source);output.mkdir(parents=True,exist_ok=False)
    sp,lm=output/'sponge',output/'lammps';sp.mkdir();lm.mkdir()
    if rattle:
        if any(k not in state for k in ('canonical_position', 'runtime_input_position')):
            raise ValueError('RATTLE prepared geometry required')
        # Observe the first force-time frame, before the one integration step.
        # Both production and this probe project exactly the original input.
        from water_model import array_file
        array_file(sp/'input-coordinate.txt', state['runtime_input_position'], model['box_angstrom'])
        array_file(sp/'input-velocity.txt', np.zeros_like(state['positions']))
        native = np.asarray(state['canonical_position'] % model['box_angstrom'], np.float32).astype(float)
        native[native >= model['box_angstrom']] = 0
        np.testing.assert_array_equal(native, state['positions'])
        write_input(sp,source,model,dt=.0005,extra={'constrain_mode':'RATTLE',
            'constrain.in_file':str(source/'constraint.txt'),
            'coordinate_in_file':str(sp/'input-coordinate.txt'),
            'velocity_in_file':str(sp/'input-velocity.txt')})
    else:
        write_input(sp,source,model)
    pairs='\n'.join(f"pair_coeff {min(v['types'])+1} {max(v['types'])+1} {v['epsilon']:.17g} {v['sigma']:.17g}" for v in model['coefficients'])
    pressure=' '.join('$(c_p['+str(i)+']:%.17g)' for i in range(1,7));cs=model['torsion_coefficients']
    (lm/'in.lammps').write_text(f'''units real
atom_style full
boundary p p p
read_data {source/'lammps.data'}
bond_style zero
bond_coeff 1 {model['bond_length']:.17g}
angle_style harmonic
angle_coeff 1 {model['angle_k']:.17g} {np.rad2deg(model['theta0_radian']):.17g}
dihedral_style opls
dihedral_coeff 1 {2*cs[0]:.17g} {2*cs[1]:.17g} {2*cs[2]:.17g} 0
pair_style lj/cut {model['cutoff_angstrom']:.17g}
{pairs}
pair_modify shift no tail no
special_bonds lj/coul 0 0 0 angle no dihedral no
neighbor 1.0 bin
compute p all pressure NULL virial
thermo_style custom step pe c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6]
thermo_modify norm no format float %.17g
run 0
write_dump all custom forces.dump id fx fy fz modify sort id format float %.17g
print "$(pe:%.17g) {pressure}" file static.dat
''')
    save(output/'plan.json',dict(coordinate_sha256=digest(source/'coordinate.txt'),state_sha256=digest(source/'state.npz'),mdin_sha256=digest(sp/'mdin.spg.toml'),model_sha256=digest(source/'model.json'),source_sha256=digest(__file__),sponge_sha256=digest(sponge),lammps_sha256=digest(lammps),rattle_initial_frame=rattle,scope=('RATTLE initial native force; independent NumPy kinetic/constraint subtraction for potential pressure' if rattle else 'unconstrained static input; legacy molecule reimaging may round coordinates'),gates={'force':'2e-3 + 3e-4 maxabs reference','energy':'.02 + 2e-5 abs reference','pressure_pa':'30 + 5e-4 maxabs reference'}))
    start=time.perf_counter();walls={'sponge':execute([sponge,'-mdin','mdin.spg.toml'],sp),'lammps':execute([lammps,'-in','in.lammps'],lm)}
    sf,p,e,_=read_sponge(sp,model['atoms']);lv=np.loadtxt(lm/'static.dat');lf=np.loadtxt(lm/'forces.dump',skiprows=9)
    correction = np.zeros(6)
    if rattle:
        from hexane_nve import reference
        from water_model import internal_timestep, AXES
        ref = reference(state['canonical_position'], np.zeros_like(sf), sf,
                        state['masses'], model['box_angstrom'], internal_timestep(.0005))
        correction = np.array([(ref['kinetic']+ref['constraint'])[a,b] for a,b in AXES])/model['box_angstrom']**3*PA
        p -= correction
    if not np.array_equal(lf[:,0],np.arange(1,model['atoms']+1)):raise ValueError('atom order')
    et=tail(model,state);expected=lv[1:]*101325+np.array([1,1,1,0,0,0])*2*et/model['box_angstrom']**3*PA
    ef=float(np.max(abs(sf-lf[:,1:])));ee=float(e['eff_pot']-lv[0]-et);ep=float(np.max(abs(p-expected)))
    passed=bool(ef<=.002+3e-4*np.max(abs(lf[:,1:])) and abs(ee)<=.02+2e-5*abs(lv[0]+et) and ep<=30+5e-4*np.max(abs(expected)))
    result=dict(force_sha256=digest(sp/'force.dat'),plan_sha256=digest(output/'plan.json'),passed=passed,force_max_error=ef,energy_error=ee,pressure_max_error_pa=ep,reference_pressure_pa=expected.tolist(),sponge_pressure_pa=p.tolist(),subtracted_kinetic_constraint_pressure_pa=correction.tolist(),sponge_energy=e,tail_energy=et,engine_wall_seconds=walls,batch_wall_seconds=time.perf_counter()-start,formal_viscosity_acceptance=False)
    result['raw_sha256'] = {str(p.relative_to(output)):digest(p) for p in output.rglob('*')
                           if p.is_file() and p.name not in ('summary.json','plan.json')}
    save(output/'summary.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True);p.add_argument('--cells',type=int,nargs=3,default=[3,6,6]);p.add_argument('--density',type=float,default=659.29);p.add_argument('--temperature',type=float,default=293.15);p.add_argument('--cutoff',type=float,default=14.)
    p=sub.add_parser('static');p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--sponge',required=True);p.add_argument('--lammps',required=True)
    p.add_argument('--rattle',action='store_true',help='use the actual projected initial RATTLE force-time frame')
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.output,args.cells,args.density,args.temperature,args.cutoff)
    else:result=static(args.model,args.output,args.sponge,args.lammps,rattle=args.rattle)
    print(json.dumps(result,indent=2))
    if result.get('passed') is False:raise SystemExit(1)
