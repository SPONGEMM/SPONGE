"""Real constrained cosine trajectories and HDF5 restart; not liquid viscosity."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.viscosity.workflows.config import write_engine_input
import shutil
import subprocess
import sys
import tempfile
import unittest
import test_coupling_restart as base

h5py, np = base.h5py, base.np
TIME = float(np.float32(20.455))
OUT = None


class ConstrainedCosineRuntime(unittest.TestCase):
    run_case = base.CouplingRestart.run_case
    read_state = base.CouplingRestart.read_state

    def setUp(self):
        if OUT is None:
            temp = tempfile.TemporaryDirectory(prefix="sponge-constrained-cosine-")
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
        else:
            self.root = OUT / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)

    def typed_state(self, directory):
        with h5py.File(directory / "checkpoint.spgr.h5") as f:
            group = f["parameters/restart/integrator_state/double"]
            result = {k: v[()] for k, v in group.items()}
            for v in group.values():
                self.assertEqual(v.dtype, np.dtype("float64"))
            return result

    def test_nonzero_force_output_cadence(self):
        settings = dict(pbc=True, mode="nvt", thermostat="middle_langevin",
                        target_temperature=25, thermostat_tau=.013, thermostat_seed=1423,
                        constrain_mode="SHAKE", settle_disable=True)
        settings.update({"PM.MPI_size": 1, "LJ_in_file": "lj.txt",
                         "constrain.in_file": "constraint.txt"})
        kwargs = dict(drive='[viscosity]\nmethod="cosine"\nacceleration=0.25\n',
                      files={"constraint.txt": "1\n0 1 4\n",
                             "lj.txt": "2 1\n10000\n100\n0\n0\n"},
                      coordinates="2\n1 2 1\n3.4 2 4.2\n32 32 32\n90 90 90\n")
        dense = self.run_case("cadence-dense", 80, settings=settings, **kwargs)
        sparse = self.run_case("cadence-sparse", 80, settings={**settings,
            "write_mdout_interval": 17, "write_information_interval": 17,
            "print_virial": False}, **kwargs)
        self.assertGreater(np.max(np.abs(np.fromfile(dense/"frc.dat", np.float32))), 0.)
        for name in ("crd.dat", "vel.dat", "frc.dat"):
            np.testing.assert_array_equal(np.fromfile(dense/name, np.float32),
                                          np.fromfile(sparse/name, np.float32))
        for name, values in self.typed_state(dense).items():
            np.testing.assert_array_equal(values, self.typed_state(sparse)[name])
        self.assertEqual(self.read_state(dense)[0], self.read_state(sparse)[0])
        with h5py.File(dense/"viscosity.h5") as a, h5py.File(sparse/"viscosity.h5") as b:
            for name, node in a["observables/all"].items():
                if isinstance(node, h5py.Group) and "value" in node:
                    np.testing.assert_array_equal(node['value'][()], b[f'observables/all/{name}/value'][()])

    def test_hexane_energy_output_cadence(self):
        sys.path.insert(0, str(base.ROOT/'benchmarks/viscosity'))
        import hexane_model
        source = self.root/'model'
        m = hexane_model.prepare(source, cells=(1,2,3), cutoff=5.)
        identity = 'sha256:'+hashlib.sha256(b'constrained cosine LJ angle torsion cadence').hexdigest()
        outputs = []
        for name, mdout, stride, sample, checkpoint in [
                ('dense',1,1,1,100), ('sparse',10,1,1,100),
                ('all-sparse',10,10,5,20)]:
            directory = self.root/name; directory.mkdir()
            extra = dict(mode='nvt', thermostat='middle_langevin', thermostat_tau=.1,
                thermostat_seed=1423, target_temperature=m['temperature_k'],
                constrain_mode='SHAKE', settle_disable=True, write_mdout_interval=mdout,
                write_restart_file_interval=checkpoint, output_h5_restart_path='checkpoint.spgr.h5',
                output_h5_restart_topology_hash=identity, output_h5_restart_atom_order_hash=identity,
                output_h5_restart_protocol_hash=identity)
            extra['constrain.in_file'] = str(source/'constraint.txt')
            hexane_model.write_input(directory, source, m, 100, .0005, stride, extra,
                f'method="cosine"\nacceleration=0.25\nsample_interval={sample}')
            result = subprocess.run([base.SPONGE,'-mdin','mdin.spg.toml'],cwd=directory,
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=40)
            (directory/'run.log').write_text(result.stdout)
            self.assertEqual(result.returncode,0,result.stdout); outputs.append(directory)
        for other,stride,sample in ((outputs[1],1,1),(outputs[2],10,5)):
            for name in ('crd.dat','vel.dat','force.dat'):
                full=np.fromfile(outputs[0]/name,np.float32).reshape(-1,m['atoms'],3)
                selected=full[::stride] if name=='force.dat' else full[stride-1::stride]
                np.testing.assert_array_equal(selected,
                    np.fromfile(other/name,np.float32).reshape(-1,m['atoms'],3))
            for key,value in self.typed_state(outputs[0]).items():
                np.testing.assert_array_equal(value,self.typed_state(other)[key])
            self.assertEqual(self.read_state(outputs[0])[0],self.read_state(other)[0])
            with h5py.File(outputs[0]/'viscosity.h5') as a,h5py.File(other/'viscosity.h5') as b:
                for name,node in a['observables/all'].items():
                    if isinstance(node,h5py.Group) and 'value' in node:
                        np.testing.assert_array_equal(node['value'][()][::sample],b[f'observables/all/{name}/value'][()])
        with h5py.File(outputs[0]/'checkpoint.spgr.h5') as f:
            identity=base.text(f['parameters/restart/integrator_state/constrained_flow_configuration'])
        for field in ('rattle_flow_double_v2','raw_float_input_v1',
                      'output_preserves_native_v1','energy_virial_every_step_v1'):
            self.assertIn(field,identity.split(';'))

    def test_stationary_geometry_preserves_native_coordinates(self):
        q = np.array([[24.426713943481445,7.,8.],[25.426713943481445,7.,8.]],np.float32)
        coordinates = '2\n'+'\n'.join(' '.join(map(str,row)) for row in q)+'\n28.619646072387695 29.3 37.1\n90 90 90\n'
        settings = dict(pbc=True,mode='nvt',thermostat='middle_langevin',
            thermostat_tau=1e15,target_temperature=0,thermostat_seed=1423,
            constrain_mode='SHAKE',settle_disable=True)
        settings['constrain.in_file']='constraint.txt'
        directory=self.run_case('stationary',3,coordinates=coordinates,settings=settings,
            files={'charge.txt':'2\n0\n0\n','constraint.txt':'1\n0 1 1\n'},
            drive='[viscosity]\nmethod="cosine"\nacceleration=0\nflow="z"\ngradient="x"\n')
        np.testing.assert_array_equal(self.typed_state(directory)['constrained_flow_position'].reshape(2,3),q)
        native=np.fromfile(directory/'crd.dat',np.float32).reshape(-1,2,3)
        np.testing.assert_array_equal(native,np.broadcast_to(q,native.shape))
        self.assertTrue(np.all(np.fromfile(directory/'frc.dat',np.float32)==0))
        self.run_case('whole-output-rejected',1,coordinates=coordinates,
            settings={**settings,'force_whole_output':True},
            files={'charge.txt':'2\n0\n0\n','constraint.txt':'1\n0 1 1\n'},
            drive='[viscosity]\nmethod="cosine"\nacceleration=0\nflow="z"\ngradient="x"\n',
            error='force_whole_output')

    def test_restart_measurement_and_tamper(self):
        drive = '[viscosity]\nmethod="cosine"\nacceleration=0.25\n'
        settings = dict(pbc=True, mode="nvt", thermostat="middle_langevin",
                        target_temperature=25, thermostat_tau=.013, thermostat_seed=1423,
                        constrain_mode="SHAKE", settle_disable=True)
        settings.update({"PM.MPI_size": 1, "LJ_in_file": "lj.txt", "constrain.in_file": "constraint.txt"})
        files = {"constraint.txt": "1\n0 1 4\n", "lj.txt": "2 1\n0\n0\n0\n0\n"}
        kwargs = dict(drive=drive, settings=settings, files=files)
        coords = "2\n1 2 1\n3.4 2 4.2\n32 32 32\n90 90 90\n"
        full = self.run_case("full", 40, coordinates=coords, **kwargs)
        first = self.run_case("first", 17, coordinates=coords, **kwargs)
        split = self.run_case("split", 23, restart=first/"checkpoint.spgr.h5", **kwargs)
        a, b = self.read_state(full), self.read_state(split)
        self.assertEqual(a[0], b[0]); self.assertEqual(a[3], b[3])
        np.testing.assert_array_equal(a[1], b[1]); np.testing.assert_array_equal(a[2], b[2])
        sa, sb = self.typed_state(full), self.typed_state(split)
        for key in sa: np.testing.assert_array_equal(sa[key], sb[key])
        with h5py.File(full/"checkpoint.spgr.h5") as fa, h5py.File(split/"checkpoint.spgr.h5") as fb:
            np.testing.assert_array_equal(fa["parameters/restart/rng_state/middle_langevin/state_words"][()], fb["parameters/restart/rng_state/middle_langevin/state_words"][()])
        with h5py.File(full/"viscosity.h5") as fa, h5py.File(split/"viscosity.h5") as fb:
            self.assertEqual(base.text(fa["parameters/sponge/provenance/observable_contract"]), "sponge.observables.physical_state.v1")
            for key, node in fb["observables/all"].items():
                if isinstance(node, h5py.Group) and "value" in node:
                    np.testing.assert_array_equal(fa[f"observables/all/{key}/value"][()][17:], node["value"][()])
        sparse = self.run_case("sparse", 40, coordinates=coords,
                               **{**kwargs, "drive": drive+"bins=7\nsample_interval=3\n"})
        for key, value in self.typed_state(sparse).items(): np.testing.assert_array_equal(sa[key], value)
        self.assertEqual(a[0], self.read_state(sparse)[0])
        for key in sa:
            altered = self.root / f"tampered-{key}.h5"
            shutil.copy2(first/"checkpoint.spgr.h5", altered)
            with h5py.File(altered, "r+") as f:
                d = f[f"parameters/restart/integrator_state/double/{key}"]; d[0] += .01
            self.run_case("reject-"+key, 1, restart=altered, error="hash", **kwargs)
        self.run_case("changed_dt", 1, restart=first/"checkpoint.spgr.h5", error="mismatch",
                      **{**kwargs, "settings": {**settings, "dt": base.DT*1.01}})
        self.run_case("changed_length", 1, restart=first/"checkpoint.spgr.h5", error="configuration mismatch",
                      **{**kwargs, "files": {**files, "constraint.txt": "1\n0 1 4.00001\n"}})

    def test_native_precision_after_many_periodic_crossings(self):
        settings = dict(pbc=True, mode="nvt", thermostat="middle_langevin",
                        target_temperature=25, thermostat_tau=.013,
                        thermostat_seed=1423, constrain_mode="SHAKE",
                        settle_disable=True)
        settings.update({"PM.MPI_size": 1, "LJ_in_file": "lj.txt",
                         "constrain.in_file": "constraint.txt"})
        files = {"constraint.txt": "1\n0 1 4\n",
                 "lj.txt": "2 1\n0\n0\n0\n0\n"}
        kwargs = dict(drive='[viscosity]\nmethod="cosine"\nacceleration=0.25\n',
                      settings=settings, files=files)
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                initial = dict(coordinates="2\n1 2 1\n3.4 2 4.2\n32 32 32\n90 90 90\n",
                               velocities=f"2\n{sign*1000} 0 0\n{sign*1000} 0 0\n")
                full = self.run_case(f"fast-{sign}-full", 120, **initial, **kwargs)
                first = self.run_case(f"fast-{sign}-first", 47, **initial, **kwargs)
                split = self.run_case(f"fast-{sign}-split", 73,
                                      restart=first/"checkpoint.spgr.h5", **kwargs)
                a, b = self.typed_state(full), self.typed_state(split)
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(self.read_state(full)[0], self.read_state(split)[0])
                q = a["constrained_flow_position"].reshape(2, 3)
                self.assertGreater(np.abs(q[:, 0]).min(), 256)
                for directory in (full, split):
                    native = np.fromfile(directory/"crd.dat", np.float32).reshape(-1, 2, 3)[-1]
                    delta = native - (q % 32)
                    delta -= 32*np.rint(delta/32)
                    # One float32 box-scale rounding, independent of image count.
                    np.testing.assert_allclose(delta, 0, rtol=0, atol=2e-6)

    def free_rotor(self, name, h, temperature=0, tau=1e15, algorithm="SHAKE"):
        directory = self.root/name; directory.mkdir()
        mass = np.array([2.,3.]); q = np.array([[7.4,8,8],[8.4,8,8]],dtype=np.float32).astype(float)
        v = np.array([[0,-.48,0],[0,.32,0]],dtype=np.float32).astype(float)
        for key, values in (("mass",mass),("charge",[0.,0.])):
            (directory/f"{key}.txt").write_text("2\n"+"\n".join(map(str,values))+"\n")
        for key, values in (("coordinate",q),("velocity",v)):
            (directory/f"{key}.txt").write_text("2\n"+"\n".join(" ".join(map(str,row)) for row in values)+("\n32 32 32\n90 90 90\n" if key=="coordinate" else "\n"))
        (directory/"constraint.txt").write_text("1\n0 1 1\n")
        # Explicit synthetic metadata identity for this generated zero-force
        # model; checkpoint is inspected here, not loaded against another model.
        identity="sha256:"+hashlib.sha256(json.dumps({"mass":mass.tolist(),"charge":[0,0],"constraint":[0,1,1]}).encode()).hexdigest()
        steps=round(.4/h)
        cfg=dict(md_name=name,mode="nvt",pbc=True,thermostat="middle_langevin",target_temperature=temperature,
                 thermostat_tau=tau,thermostat_seed=581,dt=h/TIME,step_limit=steps,cutoff=8,
                 constrain_mode=algorithm,settle_disable=algorithm=="SHAKE",print_zeroth_frame=True,
                 write_mdout_interval=1,write_information_interval=1,write_trajectory_interval=1,
                 write_restart_file_interval=1,output_h5_restart_path="checkpoint.spgr.h5",
                 output_h5_restart_topology_hash=identity,output_h5_restart_atom_order_hash=identity,
                 output_h5_restart_protocol_hash=identity,crd="crd.dat",vel="vel.dat",frc="frc.dat")
        cfg.update({"PM.MPI_size":1,"constrain.in_file":"constraint.txt"})
        cfg.update({f"{key}_in_file":f"{key}.txt" for key in ("mass","charge","coordinate","velocity")})
        write_engine_input(directory/"mdin.spg.toml", "\n".join(f"{k}={json.dumps(val)}" for k,val in cfg.items())+'\n[viscosity]\nmethod="cosine"\nacceleration=0\nflow="z"\ngradient="x"\n')
        result=subprocess.run([base.SPONGE,"-mdin","mdin.spg.toml"],cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                              env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},timeout=60)
        (directory/"run.log").write_text(result.stdout)
        self.assertEqual(result.returncode,0,result.stdout)
        state=self.typed_state(directory);qn=state["constrained_flow_position"].reshape(2,3);sn=state["constrained_flow_velocity"].reshape(2,3)
        self.assertTrue(np.all(np.fromfile(directory/"frc.dat",np.float32)==0))
        self.assertLess(abs(np.linalg.norm(qn[0]-qn[1])-1),2e-12)
        self.assertLess(abs(np.dot(qn[0]-qn[1],sn[0]-sn[1])),1e-10)
        ledger=self.read_state(directory)[0]
        k0=.5*np.sum(mass[:,None]*v*v);kn=.5*np.sum(mass[:,None]*sn*sn)
        self.assertAlmostEqual(kn-k0,float(ledger["heat"])+float(state["constrained_flow_drift_energy"][0]),delta=1e-9)
        if temperature:
            self.assertNotEqual(float(ledger["heat"]),0)
            return
        # MD core converts parsed double ps to float internal time in one
        # expression; the legacy thermostat used two float roundings. Read
        # the explicit runtime identity and independently check that conversion.
        with h5py.File(directory/"checkpoint.spgr.h5") as f:
            identity=base.text(f["parameters/restart/integrator_state/constrained_flow_configuration"])
        actual_h=float(re.search(r";dt=([^;]+)",identity).group(1))
        self.assertEqual(actual_h,float(np.float32(cfg["dt"]*TIME)))
        omega=v[1,1]-v[0,1];theta=steps*2*np.arcsin(actual_h*omega/2)
        measured=np.arctan2(qn[1,1]-qn[0,1],qn[1,0]-qn[0,0])
        self.assertAlmostEqual(measured,theta,delta=2e-10)
        com=np.average(q,axis=0,weights=mass)+steps*actual_h*np.average(v,axis=0,weights=mass)
        expected=np.array([com-.6*np.array([np.cos(theta),np.sin(theta),0]),com+.4*np.array([np.cos(theta),np.sin(theta),0])])
        np.testing.assert_allclose(qn,expected,rtol=0,atol=2e-10)
        native=np.fromfile(directory/"crd.dat",np.float32).reshape(-1,2,3)[-1]
        np.testing.assert_allclose(native,expected,rtol=0,atol=2e-6)
        with h5py.File(directory/"viscosity.h5") as f:
            obs=f["observables/all"]
            np.testing.assert_array_equal(obs["step"][()],np.arange(steps+1))
            angles=np.arange(steps+1)*2*np.arcsin(actual_h*omega/2)
            r=np.column_stack([np.cos(angles),np.sin(angles),angles*0])
            tangent=np.column_stack([-np.sin(angles),np.cos(angles),angles*0])
            p=1.2*omega**2*(tangent[:,:,None]*tangent[:,None,:]-r[:,:,None]*r[:,None,:])/32**3*(4184/6.02214076e23/1e-30)
            for name,(i,j) in zip(("xx","yy","zz","xy","xz","yz"),((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))):
                np.testing.assert_allclose(np.ravel(obs[f"p{name}/value"][()]),p[:,i,j],rtol=2e-10,atol=5e-5)
        err=abs(measured-steps*actual_h*omega)
        (directory/"summary.json").write_text(json.dumps(dict(h=actual_h,angle_error=err,heat=float(ledger["heat"]),kinetic_delta=kn-k0),indent=2)+"\n")
        return err

    def test_analytic_rotor_and_energy_accounting(self):
        for algorithm in ("SHAKE","SETTLE"):
            errors=[self.free_rotor(f"{algorithm}-{h}",h,algorithm=algorithm) for h in (.04,.02,.01)]
            for a,b in zip(errors,errors[1:]):self.assertGreater(a/b,3.5);self.assertLess(a/b,4.5)
        self.free_rotor("noisy",.01,temperature=25,tau=.013)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--sponge",required=True);p.add_argument("--output")
    args,extra=p.parse_known_args();base.SPONGE=str(Path(args.sponge).resolve())
    OUT=Path(args.output).resolve() if args.output else None
    unittest.main(argv=[__file__,*extra])
