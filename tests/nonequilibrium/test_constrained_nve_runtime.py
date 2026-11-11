"""Actual RATTLE NVE, force-time GK and dynamic restart; not liquid acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import test_coupling_restart as base

h5py, np = base.h5py, base.np
sys.path.insert(0, str(base.ROOT/'scripts'))
import analyze_green_kubo as gk
TIME = float(np.float32(20.455))
OUT = None


class ConstrainedNVERuntime(unittest.TestCase):
    run_case = base.CouplingRestart.run_case

    def setUp(self):
        if OUT is None:
            temp = tempfile.TemporaryDirectory(prefix='sponge-rattle-nve-')
            self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        else:
            self.root = OUT/self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)

    def state(self, directory):
        with h5py.File(directory/'checkpoint.spgr.h5') as file:
            group = file['parameters/restart/integrator_state/double']
            self.assertEqual(set(group), {'constrained_nve_position','constrained_nve_velocity','constrained_nve_drift_energy'})
            for v in group.values(): self.assertEqual(v.dtype, np.dtype('float64'))
            return {key:value[()] for key,value in group.items()}

    def test_periodic_external_load_work_and_restart(self):
        settings={'pbc':True,'mode':'nve','constrain_mode':'RATTLE',
                  'PM.MPI_size':1,'LJ_in_file':'lj.txt','constrain.in_file':'constraint.txt'}
        files={'constraint.txt':'1\n0 1 4\n','lj.txt':'2 1\n0\n0\n0\n0\n'}
        q0=np.array([[10.,2.,2.],[6.,2.,2.]])
        coordinates='2\n10 2 2\n6 2 2\n32 32 32\n90 90 90\n'
        # Each step crosses more than one whole box. Minimum-image or wrapped
        # displacements cannot reproduce the mechanical path work.
        velocities='2\n20000 0 0\n20000 0 0\n'
        kwargs=dict(settings=settings,files=files,coordinates=coordinates,velocities=velocities)
        undriven=self.run_case('undriven',1,drive='',**kwargs)
        internal=np.fromfile(undriven/'frc.dat',np.float32).reshape(-1,2,3)[0]
        charge=np.loadtxt(undriven/'charge.txt',skiprows=1).astype(np.float32).astype(float)
        mass=np.array([1.,16.])
        force=np.array([[np.float32(100*6.02214076e23*1e-22/4184),0,0],[0,0,0]],float)
        acceleration=np.zeros((2,3));acceleration[:,0]=(mass*.25*10/TIME**2).astype(np.float32)
        electric=np.zeros((2,3));electric[:,2]=(charge/float(np.float32(18.2223))*float(np.float32(23.060548))/10*5).astype(np.float32)
        cases=[('force','type="force"\natoms=[0]\nforce=[100,0,0]\n',force),
               ('acceleration','type="acceleration"\nacceleration=[0.25,0,0]\n',acceleration),
               ('electric','type="electric_field"\nfield=[0,0,5]\n',electric)]
        for name,body,external in cases:
            drive='[[coupling]]\n'+body
            full=self.run_case(name,18,drive=drive,**kwargs)
            first=self.run_case(name+'-first',7,drive=drive,**kwargs)
            split=self.run_case(name+'-split',11,drive=drive,restart=first/'checkpoint.spgr.h5',**kwargs)
            final=self.state(full)
            for key,value in final.items():np.testing.assert_array_equal(value,self.state(split)[key])
            ledger=base.CouplingRestart.read_state(self,full)[0]
            self.assertEqual(ledger,base.CouplingRestart.read_state(self,split)[0])
            self.assertIn('rattle_path_work_unwrapped_owner_double_v1',ledger['configuration'])
            q=final['constrained_nve_position'].reshape(2,3)
            self.assertGreater(q[:,0].min(),800)
            expected_work=float(np.sum(external*(q-q0)))
            self.assertAlmostEqual(float(ledger['work']),expected_work,delta=2e-10)
            observed=np.fromfile(full/'frc.dat',np.float32).reshape(-1,2,3)[0]
            np.testing.assert_array_equal(observed,(internal+external).astype(np.float32))
            with h5py.File(full/'observables.h5') as f:
                self.assertEqual(float(np.ravel(f['observables/all/coupling_work/value'][()])[-1]),float(ledger['work']))
            if name=='acceleration':
                h=float(np.float32(base.DT*TIME));duration=18*h
                initial_com=np.average(q0,axis=0,weights=mass)
                expected_com=initial_com+np.array([20000,0,0])*duration+np.sum(external,axis=0)/mass.sum()*h*h*18*19/2
                np.testing.assert_allclose(np.average(q,axis=0,weights=mass),expected_com,rtol=0,atol=2e-8)
        # Generic stress remains available for driven dynamics; equilibrium
        # selection is enforced by the analysis workflow, outside the engine.
        driven=self.run_case('driven-stress',4,drive='[[coupling]]\n'+cases[0][1]+'[viscosity]\nmethod="green_kubo"\n',**kwargs)
        with self.assertRaisesRegex(ValueError,'driven'):
            gk.load_stress(driven/'viscosity.h5')

    def test_rattle_scheduled_field_ledger_identity(self):
        settings={'pbc':True,'mode':'nve','constrain_mode':'RATTLE',
                  'PM.MPI_size':1,'LJ_in_file':'lj.txt','constrain.in_file':'constraint.txt'}
        files={'constraint.txt':'1\n0 1 4\n','lj.txt':'2 1\n0\n0\n0\n0\n'}
        drive='[[coupling]]\ntype="electric_field"\nfield=[0,0,5]\nschedule={type="sine", period=0.004321987654}\n'
        kwargs=dict(drive=drive,settings=settings,files=files,coordinates='2\n2 2 2\n6 2 2\n32 32 32\n90 90 90\n')
        full=self.run_case('full',40,**kwargs)
        first=self.run_case('first',17,**kwargs)
        split=self.run_case('split',23,restart=first/'checkpoint.spgr.h5',**kwargs)
        for key,value in self.state(full).items():np.testing.assert_array_equal(value,self.state(split)[key])
        self.assertEqual(base.CouplingRestart.read_state(self,full)[0],base.CouplingRestart.read_state(self,split)[0])
        sparse=self.run_case('sparse',40,**{**kwargs,'settings':{**settings,'write_mdout_interval':7,
                            'write_information_interval':7,'write_trajectory_interval':7,'write_restart_file_interval':40}})
        for key,value in self.state(full).items():np.testing.assert_array_equal(value,self.state(sparse)[key])
        self.assertEqual(base.CouplingRestart.read_state(self,full)[0],base.CouplingRestart.read_state(self,sparse)[0])
        self.run_case('changed-field',1,restart=first/'checkpoint.spgr.h5',error='configuration',
                      **{**kwargs,'drive':drive.replace('[0,0,5]','[0,0,6]')})
        self.run_case('removed-field',1,restart=first/'checkpoint.spgr.h5',error='configuration missing or removed',**{**kwargs,'drive':''})

    def test_sinusoidal_acceleration_matches_analytic_com(self):
        # Free COM: x(t)-x(0)=10*A*(t/omega-sin(omega*t)/omega**2).
        # The constrained charged pair still has native PME internal forces.
        amplitude=10000.;period=.04;duration=.02;omega=2*np.pi/period
        drive=f'[[coupling]]\ntype="acceleration"\nacceleration=[{amplitude},0,0]\nschedule={{type="sine", period={period}}}\n'
        errors=[];energy_errors=[]
        for steps in (60,120,240):
            dt=duration/steps
            settings={'pbc':True,'mode':'nve','constrain_mode':'RATTLE','dt':dt,
                      'PM.MPI_size':1,'LJ_in_file':'lj.txt','constrain.in_file':'constraint.txt'}
            full=self.run_case(str(steps),steps,drive=drive,settings=settings,
                files={'constraint.txt':'1\n0 1 4\n','lj.txt':'2 1\n0\n0\n0\n0\n'},
                coordinates='2\n2 2 2\n6 2 2\n32 32 32\n90 90 90\n')
            state=self.state(full);mass=np.array([1.,16.])
            com=np.average(state['constrained_nve_position'].reshape(2,3),axis=0,weights=mass)
            exact=(2+16*6)/17+10*amplitude*(duration/omega-np.sin(omega*duration)/omega**2)
            errors.append(abs(com[0]-exact))
            # The imposed acceleration vanishes at both endpoints. Retain
            # the native total-force half kick for finite PME rounding.
            # This is COM kinetic energy, not total kinetic energy: internal
            # potential changes do not do COM work in the exact force model.
            velocity=np.average(state['constrained_nve_velocity'].reshape(2,3),axis=0,weights=mass)
            forces=np.fromfile(full/'frc.dat',np.float32).reshape(-1,2,3).astype(float)
            h=float(np.float32(dt*TIME))
            velocity+=.5*h*forces[-1].sum(axis=0)/mass.sum()
            initial_velocity=.5*h*forces[0].sum(axis=0)/mass.sum()
            work=float(base.CouplingRestart.read_state(self,full)[0]['work'])
            energy_errors.append(abs(.5*mass.sum()*(np.dot(velocity,velocity)-np.dot(initial_velocity,initial_velocity))-work))
        for ratio in (errors[0]/errors[1],errors[1]/errors[2]):
            self.assertGreater(ratio,3.8);self.assertLess(ratio,4.2)
        self.assertLess(max(energy_errors),.003)
        (self.root/'analytic-com.json').write_text(json.dumps(dict(steps=[60,120,240],
            absolute_position_errors_angstrom=errors,com_energy_work_errors_kcal_mol=energy_errors),indent=2)+'\n')

    def test_energy_output_cadence_preserves_force_path(self):
        sys.path.insert(0,str(base.ROOT/'benchmarks/viscosity'))
        import hexane_model
        source=self.root/'model'
        m=hexane_model.prepare(source,cells=(1,2,3),cutoff=5.)
        identity='sha256:'+hashlib.sha256(b'RATTLE nonzero LJ/angle/torsion cadence').hexdigest()
        outputs=[]
        for name,stride,sample,restart in [('dense',1,1,100),('sparse',10,5,20)]:
            d=self.root/name;d.mkdir()
            extra=dict(constrain_mode='RATTLE',write_restart_file_interval=restart,
                output_h5_restart_path='checkpoint.spgr.h5',
                output_h5_restart_topology_hash=identity,output_h5_restart_atom_order_hash=identity,
                output_h5_restart_protocol_hash=identity)
            extra['constrain.in_file']=str(source/'constraint.txt')
            hexane_model.write_input(d,source,m,100,.0005,stride,extra,
                f'method="green_kubo"\nsample_interval={sample}')
            result=subprocess.run([base.SPONGE,'-mdin','mdin.spg.toml'],cwd=d,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=40)
            (d/'run.log').write_text(result.stdout)
            self.assertEqual(result.returncode,0,result.stdout);outputs.append(d)
        for key,value in self.state(outputs[0]).items():
            np.testing.assert_array_equal(value,self.state(outputs[1])[key])
        with h5py.File(outputs[0]/'viscosity.h5') as a,h5py.File(outputs[1]/'viscosity.h5') as b:
            for name in ('pxy','pxz','pyz','temperature'):
                np.testing.assert_array_equal(a[f'observables/all/{name}/value'][()][::5],
                                              b[f'observables/all/{name}/value'][()])

    def test_pme_restart_cadence_and_rejection(self):
        settings = dict(pbc=True, mode='nve', constrain_mode='RATTLE')
        settings.update({'PM.MPI_size':1, 'LJ_in_file':'lj.txt', 'constrain.in_file':'constraint.txt'})
        files = {'constraint.txt':'1\n0 1 4\n', 'lj.txt':'2 1\n0\n0\n0\n0\n'}
        drive = '[viscosity]\nmethod="green_kubo"\n'
        kwargs = dict(drive=drive, settings=settings, files=files)
        coords = '2\n1 2 1\n3.4 2 4.2\n31.7 29.3 37.1\n90 90 90\n'
        velocity = '2\n0 .25 0\n0 -.015625 0\n'
        full = self.run_case('full',40,coordinates=coords,velocities=velocity,**kwargs)
        first = self.run_case('first',17,coordinates=coords,velocities=velocity,**kwargs)
        split = self.run_case('split',23,restart=first/'checkpoint.spgr.h5',**kwargs)
        a,b = self.state(full),self.state(split)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        with h5py.File(full/'viscosity.h5') as fa,h5py.File(split/'viscosity.h5') as fb:
            for key,node in fb['observables/all'].items():
                if isinstance(node,h5py.Group) and 'value' in node:
                    np.testing.assert_array_equal(fa[f'observables/all/{key}/value'][()][17:],node['value'][()])
        times,_,_,_,metadata = gk.load_stress(full/'viscosity.h5')
        self.assertTrue(metadata['constraint_solver'].startswith('RATTLE|rattle_pairs=1|'))
        self.assertEqual(float(metadata['constraint_length_tolerance']),2e-12)
        self.assertEqual(len(times),41)
        for name,extra in [('sparse',drive+'sample_interval=3\n'),('off','')]:
            other=self.run_case(name,40,coordinates=coords,velocities=velocity,**{**kwargs,'drive':extra})
            for key,value in self.state(other).items():np.testing.assert_array_equal(a[key],value)
            for file in ('crd.dat','vel.dat','frc.dat'):
                self.assertEqual((full/file).read_bytes(),(other/file).read_bytes())
        sparse_output = self.run_case('sparse-trajectory',40,coordinates=coords,velocities=velocity,
            **{**kwargs,'settings':{**settings,'write_trajectory_interval':7}})
        for key,value in self.state(sparse_output).items(): np.testing.assert_array_equal(a[key],value)
        for key in a:
            altered=self.root/f'tampered-{key}.h5';shutil.copy2(first/'checkpoint.spgr.h5',altered)
            with h5py.File(altered,'r+') as f:f[f'parameters/restart/integrator_state/double/{key}'][0] += .01
            self.run_case('reject-'+key,1,restart=altered,error='hash',**kwargs)
            missing=self.root/f'missing-{key}.h5';shutil.copy2(first/'checkpoint.spgr.h5',missing)
            with h5py.File(missing,'r+') as f:del f[f'parameters/restart/integrator_state/double/{key}']
            self.run_case('reject-missing-'+key,1,restart=missing,error='hash',**kwargs)
        unknown=self.root/'unknown-double.h5';shutil.copy2(first/'checkpoint.spgr.h5',unknown)
        with h5py.File(unknown,'r+') as f:
            f['parameters/restart/integrator_state/double'].create_dataset('unknown',data=[0.])
        self.run_case('reject-unknown-double',1,restart=unknown,error='hash',**kwargs)
        self.run_case('changed-dt',1,restart=first/'checkpoint.spgr.h5',error='mismatch',
                      **{**kwargs,'settings':{**settings,'dt':base.DT*1.01}})
        self.run_case('changed-length',1,restart=first/'checkpoint.spgr.h5',error='configuration mismatch',
                      **{**kwargs,'files':{**files,'constraint.txt':'1\n0 1 4.00001\n'}})
        self.run_case('different-algorithm',1,restart=first/'checkpoint.spgr.h5',error='matching RATTLE NVE state',
                      **{**kwargs,'settings':{**settings,'constrain_mode':'SHAKE','settle_disable':True}})
        old=self.run_case('old-shake',3,coordinates=coords,velocities=velocity,
                          **{**kwargs,'settings':{**settings,'constrain_mode':'SHAKE','settle_disable':True}})
        self.run_case('missing-state',1,restart=old/'checkpoint.spgr.h5',error='matching RATTLE NVE state',**kwargs)
        self.run_case('reject-regeneration',1,restart=first/'checkpoint.spgr.h5',error='cannot regenerate RATTLE',
                      **{**kwargs,'settings':{**settings,'initial_velocity.mode':'maxwell','initial_velocity.seed':76213}})
        self.run_case('reject-whole-output',1,coordinates=coords,
                      **{**kwargs,'settings':{**settings,'force_whole_output':True}},error='force_whole_output')
        self.run_case('reject-open',1,coordinates=coords.replace('31.7 29.3 37.1','1000 1000 1000'),
                      **{**kwargs,'drive':'','settings':{**settings,'pbc':False}},error='RATTLE requires')

    def test_periodic_double_state(self):
        settings=dict(pbc=True,mode='nve',constrain_mode='RATTLE')
        settings.update({'PM.MPI_size':1,'LJ_in_file':'lj.txt','constrain.in_file':'constraint.txt'})
        kwargs=dict(drive='',settings=settings,files={'constraint.txt':'1\n0 1 4\n','lj.txt':'2 1\n0\n0\n0\n0\n'})
        for sign in (-1,1):
            initial=dict(coordinates='2\n1 2 1\n3.4 2 4.2\n32 32 32\n90 90 90\n',
                         velocities=f'2\n{sign*1000} .25 0\n{sign*1000} -.015625 0\n')
            full=self.run_case(f'full-{sign}',120,**initial,**kwargs)
            first=self.run_case(f'first-{sign}',47,**initial,**kwargs)
            split=self.run_case(f'split-{sign}',73,restart=first/'checkpoint.spgr.h5',**kwargs)
            a,b=self.state(full),self.state(split)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            q=a['constrained_nve_position'].reshape(2,3)
            self.assertGreater(np.abs(q[:,0]).min(),256)
            native=np.fromfile(full/'crd.dat',np.float32).reshape(-1,2,3)[-1]
            delta=native-q%32;delta-=32*np.rint(delta/32)
            np.testing.assert_allclose(delta,0,rtol=0,atol=2e-6)

    def rotor(self,h):
        directory=self.root/str(h);directory.mkdir()
        mass=np.array([2.,3.]);q=np.array([[7.4,8,8],[8.4,8,8]],np.float32).astype(float)
        v=np.array([[0,-.48,0],[0,.32,0]],np.float32).astype(float)
        for key,values in [('mass',mass),('charge',[0.,0.])]:
            (directory/f'{key}.txt').write_text('2\n'+'\n'.join(map(str,values))+'\n')
        for key,values in [('coordinate',q),('velocity',v)]:
            (directory/f'{key}.txt').write_text('2\n'+'\n'.join(' '.join(map(str,r)) for r in values)+('\n32 32 32\n90 90 90\n' if key=='coordinate' else '\n'))
        (directory/'constraint.txt').write_text('1\n0 1 1\n')
        identity='sha256:'+hashlib.sha256(b'RATTLE unequal mass rotor fixture').hexdigest()
        steps=round(.4/h)
        cfg=dict(md_name='rotor',mode='nve',pbc=True,dt=h/TIME,step_limit=steps,cutoff=8,constrain_mode='RATTLE',
                 print_zeroth_frame=True,write_mdout_interval=1,write_information_interval=1,write_trajectory_interval=1,
                 write_restart_file_interval=1,output_h5_restart_path='checkpoint.spgr.h5',
                 output_h5_restart_topology_hash=identity,output_h5_restart_atom_order_hash=identity,
                 output_h5_restart_protocol_hash=identity,crd='crd.dat',vel='vel.dat',frc='frc.dat')
        cfg.update({'PM.MPI_size':1,'constrain.in_file':'constraint.txt'})
        cfg.update({f'{key}_in_file':f'{key}.txt' for key in ('mass','charge','coordinate','velocity')})
        write_engine_input(directory/'mdin.spg.toml', '\n'.join(f'{key}={json.dumps(value)}' for key,value in cfg.items())+'\n[viscosity]\nmethod="green_kubo"\n')
        result=subprocess.run([base.SPONGE,'-mdin','mdin.spg.toml'],cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                              env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},timeout=60)
        (directory/'run.log').write_text(result.stdout);self.assertEqual(result.returncode,0,result.stdout)
        state=self.state(directory);qn=state['constrained_nve_position'].reshape(2,3);sn=state['constrained_nve_velocity'].reshape(2,3)
        actual_h=float(np.float32(cfg['dt']*TIME));omega=v[1,1]-v[0,1]
        theta=steps*np.arcsin(actual_h*omega)
        angle=np.arctan2(qn[1,1]-qn[0,1],qn[1,0]-qn[0,0])
        self.assertAlmostEqual(angle,theta,delta=2e-10)
        com=np.average(q,weights=mass,axis=0)+steps*actual_h*np.average(v,weights=mass,axis=0)
        direction=np.array([np.cos(theta),np.sin(theta),0])
        np.testing.assert_allclose(qn,[com-.6*direction,com+.4*direction],rtol=0,atol=2e-10)
        np.testing.assert_allclose(.5*np.sum(mass[:,None]*sn**2),.5*np.sum(mass[:,None]*v**2),rtol=0,atol=1e-10)
        self.assertLess(abs(state['constrained_nve_drift_energy'][0]),1e-10)
        self.assertTrue(np.all(np.fromfile(directory/'frc.dat',np.float32)==0))
        with h5py.File(directory/'viscosity.h5') as file:
            obs=file['observables/all'];angles=np.arange(steps+1)*np.arcsin(actual_h*omega)
            r=np.column_stack((np.cos(angles),np.sin(angles),angles*0))
            u=np.column_stack((-np.sin(angles),np.cos(angles),angles*0))
            p=1.2*omega**2*(u[:,:,None]*u[:,None,:]-r[:,:,None]*r[:,None,:])/32**3*(4184/6.02214076e23/1e-30)
            for name,(i,j) in zip(('xx','yy','zz','xy','xz','yz'),((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))):
                np.testing.assert_allclose(obs[f'p{name}/value'][()],p[:,i,j],rtol=2e-10,atol=5e-5)
            self.assertLess(np.max(obs['constraint_length_residual/value'][()]),2e-12)
        gk.load_stress(directory/'viscosity.h5')
        error=abs(angle-steps*actual_h*omega)
        (directory/'summary.json').write_text(json.dumps(dict(dt_internal=actual_h,angle_error=error),indent=2)+'\n')
        return error

    def test_analytic_rotor(self):
        errors=[self.rotor(h) for h in (.04,.02,.01)]
        for a,b in zip(errors,errors[1:]):self.assertGreater(a/b,3.5);self.assertLess(a/b,4.5)

    def chain(self, name, steps, maxwell=False, stationary=False):
        directory = self.root/name
        directory.mkdir()
        mass = np.array([2., 3., 5.])
        q = np.array([[6., 7., 8.], [7., 7., 8.], [7., 7.6, 8.8]])
        velocity = np.array([[.2, .3, -.1], [-.4, .2, .3], [.1, -.2, .4]])
        if stationary:
            # All distances are exactly representable. The non-power-of-two
            # cell roundtrip used to change this in-box geometry before RATTLE.
            q = np.array([[24.426713943481445, 7., 8.],
                          [25.426713943481445, 7., 8.],
                          [25.426713943481445, 8., 8.]])
            velocity *= 0
        for key, values in [('mass', mass), ('charge', [0., 0., 0.])]:
            (directory/f'{key}.txt').write_text('3\n'+'\n'.join(map(str, values))+'\n')
        for key, values in [('coordinate', q), ('velocity', velocity)]:
            tail = '\n32 32 32\n90 90 90\n' if key == 'coordinate' else '\n'
            if stationary: tail = tail.replace('32 32 32', '28.619646072387695 29.3 37.1')
            (directory/f'{key}.txt').write_text('3\n'+'\n'.join(' '.join(map(str, row)) for row in values)+tail)
        (directory/'constraint.txt').write_text('2\n0 1 1\n1 2 1\n')
        # The unconstrained 0--2 spring produces a nonzero tangent acceleration.
        (directory/'bond.txt').write_text('3\n0 1 0 1\n1 2 0 1\n0 2 2 1\n')
        if stationary: (directory/'bond.txt').write_text('2\n0 1 0 1\n1 2 0 1\n')
        identity = 'sha256:'+hashlib.sha256(b'RATTLE three atom chain fixture').hexdigest()
        cfg = dict(md_name=name, mode='nve', pbc=True, dt=.0003, step_limit=steps,
                   cutoff=8, constrain_mode='RATTLE', print_zeroth_frame=True,
                   write_mdout_interval=1, write_information_interval=1,
                   write_trajectory_interval=1, write_restart_file_interval=1,
                   output_h5_restart_path='checkpoint.spgr.h5',
                   output_h5_restart_topology_hash=identity, output_h5_restart_atom_order_hash=identity,
                   output_h5_restart_protocol_hash=identity, crd='crd.dat', vel='vel.dat', frc='frc.dat')
        cfg.update({'PM.MPI_size':1, 'constrain.in_file':'constraint.txt'})
        cfg.update({f'{key}_in_file':f'{key}.txt' for key in ('mass', 'charge', 'coordinate')})
        if maxwell:
            cfg.update({'initial_velocity.mode':'maxwell', 'initial_velocity.seed':76213,
                        'target_temperature':300})
        else:
            cfg.update(velocity_in_file='velocity.txt', bond_in_file='bond.txt')
        write_engine_input(directory/'mdin.spg.toml', '\n'.join(f'{key}={json.dumps(value)}' for key, value in cfg.items())+
                                               '\n[viscosity]\nmethod="green_kubo"\n')
        result = subprocess.run([base.SPONGE, '-mdin', 'mdin.spg.toml'], cwd=directory,
                                env={**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=40)
        (directory/'run.log').write_text(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout)
        return directory, mass, float(np.float32(cfg['dt']*TIME))

    def test_initial_geometry_and_output_preserve_native_coordinates(self):
        directory, _, _ = self.chain('stationary', 3, stationary=True)
        original = np.loadtxt(directory/'coordinate.txt', skiprows=1, max_rows=3).astype(np.float32)
        q = self.state(directory)['constrained_nve_position'].reshape(3, 3)
        np.testing.assert_array_equal(q, original)
        native = np.fromfile(directory/'crd.dat', np.float32).reshape(-1, 3, 3)
        np.testing.assert_array_equal(native, np.broadcast_to(original, native.shape))
        self.assertTrue(np.all(np.fromfile(directory/'frc.dat', np.float32) == 0))

    def test_multibond_physical_velocity_and_six_stresses(self):
        # Independently solve the continuous Jacobian/acceleration equations at
        # the actual canonical endpoint. No C++ observer is used as oracle.
        for steps in (1, 20):
            directory, mass, h = self.chain(f'chain-{steps}', steps)
            state = self.state(directory)
            q = state['constrained_nve_position'].reshape(3, 3)
            s = state['constrained_nve_velocity'].reshape(3, 3)
            native = np.fromfile(directory/'crd.dat', np.float32).reshape(-1, 3, 3)[-1].astype(float)
            force = np.fromfile(directory/'frc.dat', np.float32).reshape(-1, 3, 3)[-1].astype(float)
            r = native[0]-native[2]
            reference_force = -4*(np.linalg.norm(r)-1)*r/np.linalg.norm(r)
            np.testing.assert_allclose(force, [reference_force, [0, 0, 0], -reference_force], rtol=2e-6, atol=2e-6)
            jac = np.zeros((2, 9)); vectors = []
            for row, (a, b) in enumerate(((0, 1), (1, 2))):
                vector = q[a]-q[b]; length = np.linalg.norm(vector)
                self.assertLess(abs(length-1), 2e-12)
                unit = vector/length; vectors.append(vector)
                jac[row, 3*a:3*a+3] = unit; jac[row, 3*b:3*b+3] = -unit
            inverse = np.repeat(1/mass, 3); metric = (jac*inverse)@jac.T
            np.testing.assert_allclose(jac@s.ravel(), 0, atol=2e-12)
            incoming = s.ravel()+.5*h*inverse*force.ravel()
            u = (incoming-inverse*(jac.T@np.linalg.solve(metric, jac@incoming))).reshape(3, 3)
            self.assertGreater(np.max(np.abs(u-s)), 1e-5)
            curvature = []
            for vector, (a, b) in zip(vectors, ((0, 1), (1, 2))):
                du = u[a]-u[b]; length = np.linalg.norm(vector); unit = vector/length
                curvature.append((du@du-(unit@du)**2)/length)
            multiplier = np.linalg.solve(metric, -jac@(inverse*force.ravel())-curvature)
            wc = sum(value*np.outer(vector, vector)/np.linalg.norm(vector)
                     for value, vector in zip(multiplier, vectors))
            thermal = u-np.average(u, weights=mass, axis=0)
            kinetic = np.einsum('i,ij,ik->jk', mass, thermal, thermal)
            pressure = (kinetic+wc+np.outer(r, force[0]))/32**3*(4184/6.02214076e23/1e-30)
            with h5py.File(directory/'viscosity.h5') as file:
                obs = file['observables/all']
                for name, (i, j) in zip(('xx','yy','zz','xy','xz','yz'), ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))):
                    # Native force/virial are float; the independent virial
                    # product is double, so allow their bounded rounding.
                    np.testing.assert_allclose(obs[f'p{name}/value'][-1], pressure[i,j], rtol=3e-6, atol=100)
                    np.testing.assert_allclose(obs[f'constraint_w{name}/value'][-1], wc[i,j], rtol=1e-10, atol=1e-11)
                temperature = np.trace(kinetic)/(4*float(np.float32(.00198716)))
                np.testing.assert_allclose(obs['temperature/value'][-1], temperature, rtol=1e-10)
                self.assertIn('rattle_pairs=2', gk.text(file['parameters/sponge/provenance/constraint_solver'][()]))

    def test_generated_velocity_is_normalized_after_projection(self):
        first, mass, _ = self.chain('generated', 1, maxwell=True)
        second, _, _ = self.chain('same-seed', 1, maxwell=True)
        a, b = self.state(first), self.state(second)
        for key in a: np.testing.assert_array_equal(a[key], b[key])
        s = a['constrained_nve_velocity'].reshape(3, 3)
        np.testing.assert_allclose(np.sum(mass[:, None]*s, axis=0), 0, atol=1e-14)
        kinetic = .5*np.sum(mass[:, None]*s*s)-a['constrained_nve_drift_energy'][0]
        self.assertAlmostEqual(kinetic, .5*4*float(np.float32(.00198716))*300, delta=1e-12)
        with h5py.File(first/'viscosity.h5') as file:
            self.assertAlmostEqual(file['observables/all/temperature/value'][0], 300, delta=1e-10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sponge',required=True);p.add_argument('--output')
    args,extra=p.parse_known_args();base.SPONGE=str(Path(args.sponge).resolve());OUT=Path(args.output).resolve() if args.output else None
    unittest.main(argv=[__file__,*extra])
