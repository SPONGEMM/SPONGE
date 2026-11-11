"""Independent projection, OU statistics and paired cosine input contracts."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock
try:
    import numpy as np
    import h5py
except ImportError:
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'benchmarks/viscosity'))
import lammps_projected_middle as reference
import lj_cosine_compare as paired
import lj_model as model
import lj_cosine_scan as scan
import lj_cosine_replicas as replicas


class CosineReference(unittest.TestCase):
    def state(self):
        rng = np.random.default_rng(718)
        return np.array([1.,2.,4.,7.,3.,9.]), rng.uniform(0,4,(6,3)), rng.normal(size=(6,3))

    def test_joint_projection_and_deterministic_decay(self):
        mass, q, v = self.state()
        bias, _, _ = reference.project(mass,q,v,4.)
        residual = v-bias
        cosine = np.cos(2*np.pi*q[:,2]/4.)
        np.testing.assert_allclose(np.sum(mass[:,None]*residual,axis=0),0,atol=2e-14)
        self.assertAlmostEqual(np.sum(mass*cosine*residual[:,0]),0,places=13)
        np.testing.assert_allclose(reference.project(mass,q,bias,4.)[0],bias,atol=1e-14)
        result = reference.ou_step(mass,q,v,4.,.8,0.,np.zeros_like(v))
        np.testing.assert_allclose(result,bias+.8*residual,atol=1e-14)
        np.testing.assert_allclose(reference.project(mass,q,result,4.)[0],bias,atol=1e-14)

    def test_noise_dof_and_collective_momentum(self):
        mass,q,v=self.state(); v*=0
        rng=np.random.default_rng(987); energy=[]
        kt, damping = .3,.7
        for _ in range(6000):
            result=reference.ou_step(mass,q,v,4.,damping,kt,rng.standard_normal(v.shape))
            energy.append(np.sum(mass[:,None]*result**2)/((1-damping**2)*kt))
            np.testing.assert_allclose(np.sum(mass[:,None]*result,axis=0),0,atol=2e-14)
        dof=3*len(mass)-4
        self.assertLess(abs(np.mean(energy)/dof-1),.02)
        self.assertLess(abs(np.var(energy)/(2*dof)-1),.06)

    def test_observer_raw_and_centered_estimators(self):
        mass,q,_=self.state(); cosine=np.cos(2*np.pi*q[:,2]/4.)
        v=np.tile([.03,-.04,.02],(len(mass),1)); v[:,0]+=.01*cosine
        record=reference.observe(mass,q,v,4.,8,1.,1.)
        self.assertAlmostEqual(record['velocity_fit'],1.)
        self.assertAlmostEqual(record['thermal_temperature'],0.)
        self.assertAlmostEqual(record['velocity_com_fourier'],2*record['variance_cosine']*record['velocity_fit'])
        self.assertAlmostEqual(record['velocity_fourier'],record['velocity_com_fourier']+2*record['mean_cosine']*record['com_vx'])
        self.assertAlmostEqual(record['bin_momentum'].sum()/mass.sum(),record['com_vx'])
        self.assertFalse(np.isclose(record['velocity_fourier'],record['velocity_fit']))
        for bad in (np.zeros_like(q),np.full_like(q,np.nan)):
            with self.assertRaises(ValueError): reference.project(mass,bad,v,4.)

    def test_driven_half_step_and_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); m=model.prepare(root/'model'); (root/'seed').mkdir()
            with np.load(root/'model/state.npz') as file:
                np.savez(root/'seed/state.npz',positions=file['positions'],centered_velocities=file['velocities'],box=m['box_angstrom'])
            paired.write_json(root/'seed/summary.json',dict(seed_sha256=model.digest(root/'seed/state.npz'),model_sha256=model.digest(root/'model/model.json')))
            m,q,_,u,h=paired.frozen_inputs(root/'model',root/'seed/state.npz')
            half,actual,lm_a=paired.driven_initial_state(m,q,u,h,.1)
            force=model.pair_reference(q,m['box_angstrom'],m['c12'],m['c6'],m['cutoff_angstrom'])[1]
            force[:,0]+=m['mass_da']*lm_a*paired.LM_TIME**2*np.cos(2*np.pi*q[:,2]/m['box_angstrom'])
            np.testing.assert_allclose(actual,half+.5*h*force/m['mass_da'],rtol=0,atol=2e-17)
            np.testing.assert_allclose(actual,u,rtol=0,atol=1e-8)
            plan=paired.prepare(root/'model',root/'seed/state.npz',root/'run',steps=100,stride=1,temperature=0,start_star=0,block_star=.05)
            self.assertEqual(plan['experiment_kind'],'zero-noise-splitting-gate')
            self.assertFalse(plan['formal_equivalence_eligible'])
            script=(root/'run/lammps/in.lammps').read_text()
            self.assertIn('fix force all accelerate/cos',script)
            self.assertIn('python/move',script)
            self.assertIn('c_native c_native[7]',script)
            for name, expected in plan['input_sha256'].items(): self.assertEqual(model.digest(root/'run'/name),expected)
            source_hash = model.digest(root/'model/model.json')
            seed_hash = model.digest(root/'seed/state.npz')
            smaller = paired.prepare(root/'model',root/'seed/state.npz',root/'small',steps=200,stride=2,
                                     temperature=0,start_star=0,block_star=.05,dt_star=.001)
            self.assertEqual(model.digest(root/'model/model.json'), source_hash)
            self.assertEqual(smaller['model_sha256'], source_hash)
            self.assertEqual(smaller['seed_sha256'], seed_hash)
            self.assertEqual(smaller['integration']['source_dt_star'], .002)
            self.assertEqual(smaller['integration']['effective_dt_star'], .001)
            self.assertAlmostEqual(smaller['dt_sponge_ps'], .5*plan['dt_sponge_ps'])
            self.assertAlmostEqual(smaller['reference']['dt_fs'], .5*plan['reference']['dt_fs'])
            self.assertAlmostEqual(smaller['damping']**2, plan['damping'], places=6)
            np.testing.assert_array_equal(np.loadtxt(root/'run/sponge/coordinate.txt',skiprows=1,max_rows=m['atoms']),
                                          np.loadtxt(root/'small/sponge/coordinate.txt',skiprows=1,max_rows=m['atoms']))
            for invalid in (0, -1, float('nan'), True):
                with self.assertRaisesRegex(ValueError,'dt_star'):
                    paired.prepare(root/'model',root/'seed/state.npz',root/'invalid',dt_star=invalid)
            self.assertFalse((root/'invalid').exists())

    def test_integration_hooks_work_and_heat_survive_atom_sorting(self):
        mass,q,v=self.state(); length=4.; dt=.02; acceleration=.3; damping=.7
        force=np.zeros_like(q); force[:,0]=mass*acceleration*np.cos(2*np.pi*q[:,2]/length)
        original_q=q.copy(); old_force=force.copy()
        half=v+.5*dt*force/mass[:,None]; midpoint=q+.5*dt*half
        c=np.cos(2*np.pi*midpoint[:,2]/length)
        design=np.column_stack((np.ones(len(mass)),c))
        bias=np.tile(np.average(half,axis=0,weights=mass),(len(mass),1))
        bias[:,0]=design@np.linalg.lstsq(design*np.sqrt(mass[:,None]),half[:,0]*np.sqrt(mass),rcond=None)[0]
        damped=bias+damping*(half-bias)
        expected_q=midpoint+.5*dt*damped
        new_force=np.zeros_like(q); new_force[:,0]=mass*acceleration*np.cos(2*np.pi*expected_q[:,2]/length)
        expected_v=damped+.5*dt*new_force/mass[:,None]
        expected_work=np.sum(.5*(old_force+new_force)*(expected_q-original_q))
        expected_heat=.5*np.sum(mass[:,None]*(damped*damped-half*half))
        integrator=reference.ProjectedMiddle.__new__(reference.ProjectedMiddle)
        integrator.config=dict(box_angstrom=length,damping=damping,temperature_k=0.,
            acceleration_angstrom_fs2=acceleration,sample_interval=10,steps=100)
        integrator.dt=dt;integrator.ftm2v=integrator.mvv2e=integrator.boltz=1.
        integrator.heat=integrator.work=0.;integrator.completed=0;integrator.rng=np.random.default_rng(123)
        arrays=[mass.copy(),q.copy(),v.copy(),force.copy(),np.arange(1,len(mass)+1)]
        integrator.arrays=lambda: arrays
        integrator.initial_integrate(0)
        order=np.array([3,1,5,0,2,4]); arrays[:]=[a[order].copy() for a in arrays]
        arrays[3][:]=new_force[order]
        integrator.final_integrate()
        np.testing.assert_allclose(arrays[1],expected_q[order],atol=1e-14)
        np.testing.assert_allclose(arrays[2],expected_v[order],atol=1e-14)
        self.assertAlmostEqual(integrator.work,expected_work,places=13)
        self.assertAlmostEqual(integrator.heat,expected_heat,places=13)

    def test_scan_fixes_duration_and_rejects_ambiguous_protocols(self):
        spec=dict(contract=scan.CONTRACT,duration_star=100.,sample_star=.02,
                  start_star=20.,block_star=10.,bins=20,
                  cases=[dict(id="base",acceleration=.05,tau_star=1.,dt_star=.002,seed=17),
                         dict(id="half",acceleration=.05,tau_star=1.,dt_star=.001,seed=17)])
        cases=scan.validate_spec(spec)
        self.assertEqual([c['steps'] for c in cases],[50000,100000])
        self.assertEqual([c['stride'] for c in cases],[10,20])
        self.assertEqual([c['steps']//c['stride'] for c in cases],[5000,5000])
        for bad in (dict(spec,extra=1),dict(spec,sample_star=.0015),dict(spec,cases=spec['cases']*2),
                    dict(spec,cases=[dict(spec['cases'][0],id='../escape')]),
                    dict(spec,cases=[dict(spec['cases'][0],tau_star=0)]),
                    dict(spec,start_star=99),dict(spec,bins=True)):
            with self.assertRaises(ValueError):scan.validate_spec(bad)

    def test_scan_failure_preserves_cost_and_stops_pending_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=model.prepare(root/'model');(root/'seed').mkdir()
            with np.load(root/'model/state.npz') as file:
                np.savez(root/'seed/state.npz',positions=file['positions'],centered_velocities=file['velocities'],box=m['box_angstrom'])
            paired.write_json(root/'seed/summary.json',dict(seed_sha256=model.digest(root/'seed/state.npz'),model_sha256=model.digest(root/'model/model.json')))
            binary=root/'exe';binary.write_text('mock executable')
            spec=dict(contract=scan.CONTRACT,duration_star=1.,sample_star=.02,start_star=0.,block_star=.2,bins=20,
                      cases=[dict(id=name,acceleration=.05,tau_star=1.,dt_star=.002,seed=17) for name in ('first','pending')])
            with mock.patch.object(scan.paired,'run_pair',side_effect=RuntimeError('injected')) as run:
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    scan.run_campaign(spec,root/'model',root/'seed/state.npz',root/'scan',binary,binary)
                self.assertEqual(run.call_count,1)
            execution=json.loads((root/'scan/execution.json').read_text())
            self.assertEqual(execution['status'],'failed')
            self.assertGreater(execution['cases']['first']['wall_seconds'],0)
            self.assertNotIn('pending',execution['cases'])
            with self.assertRaisesRegex(ValueError,'incomplete'):scan.summarize(root/'scan')
            self.assertTrue((root/'scan/source/lj_cosine_scan.py').exists())
            (root/'scan/source/lj_cosine_scan.py').write_text('modified')
            with self.assertRaisesRegex(ValueError,'source changed'):scan.summarize(root/'scan')

    def test_failed_reference_preserves_completed_engine_and_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'run'; binary=root/'lmp'; binary.write_text('test only')
            def prepare(*args,**kwargs):
                output.mkdir();(output/'sponge').mkdir();(output/'lammps').mkdir()
                (output/'plan.json').write_text('{}')
            def execute(command,directory,timeout):
                (directory/'run.log').write_text('synthetic execution fixture')
                if directory.name=='lammps': raise RuntimeError('injected reference failure')
                return .125
            with mock.patch.object(paired,'prepare',side_effect=prepare), mock.patch.object(paired,'execute',side_effect=execute):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    paired.run_pair(None,None,output,binary,binary)
            execution=json.loads((output/'execution.json').read_text())
            self.assertEqual(execution['status'],'failed')
            self.assertEqual(execution['engines']['sponge']['wall_seconds'],.125)
            self.assertGreater(execution['engines']['lammps']['wall_seconds'],0)
            for engine in ('sponge','lammps'):
                self.assertEqual(execution['engines'][engine]['available_files_sha256']['run.log'],model.digest(output/engine/'run.log'))


class ReplicaContracts(unittest.TestCase):
    def spec(self):
        return dict(contract=replicas.CONTRACT,
            model=dict(cells=3,density=.8,temperature=1.,epsilon=.238,sigma=3.4,mass=39.948,cutoff=2.5,dt=.002),
            equilibration_steps=1000,duration_star=2.,sample_star=.02,start_star=0.,block_star=.5,
            bins=20,acceleration=.025,tau_star=1.,bootstrap_draws=2000,bootstrap_seed=876,
            replicas=[dict(id='r'+str(i),initialization_seed=100+i*10,equilibration_seed=101+i*10,production_seed=102+i*10) for i in range(4)])

    def records(self, values):
        result=[]
        for i,value in enumerate(values):
            engines={}
            for engine,scale in [('sponge',1.),('lammps',2.)]:
                entry=dict(fluidity_per_mpa_s=value*scale,temperature_sponge_k=120.+value,
                    first_half_fluidity_per_mpa_s=value*scale-.1,second_half_fluidity_per_mpa_s=value*scale+.1,
                    first_half_temperature_sponge_k=120.+value,second_half_temperature_sponge_k=120.+value,
                    prefactor=.2,blocks=8,samples_per_block=20,sample_weight_duration_ps=10.,actual_duration_ps=12.)
                engines[engine]={e:entry.copy() for e in paired.cosine.ESTIMATORS}
            result.append(dict(id='r'+str(i),statistics=engines))
        return result

    def test_physical_identity_separates_random_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=model.prepare(root/'a',seed=12);b=model.prepare(root/'b',seed=13)
            self.assertNotEqual(model.digest(root/'a/model.json'),model.digest(root/'b/model.json'))
            self.assertNotEqual(model.digest(root/'a/state.npz'),model.digest(root/'b/state.npz'))
            self.assertEqual(replicas.identity(a),replicas.identity(b))
            self.assertNotEqual(replicas.identity(a),replicas.identity(dict(a,c6=a['c6']*1.01)))

    def test_paired_bootstrap_retains_covariance_and_inverts_after_pooling(self):
        records=self.records([1.,2.,4.,8.])
        report=replicas.reduce_replicas(records,2000,123)['fourier']
        self.assertAlmostEqual(report['sponge']['viscosity_mpa_s'],1/3.75)
        self.assertNotAlmostEqual(report['sponge']['viscosity_mpa_s'],np.mean([1.,.5,.25,.125]))
        # Perfectly correlated engine response scaling must yield zero ratio
        # uncertainty. Independently resampled engines fail this test.
        np.testing.assert_allclose(report['paired_comparison']['viscosity_ratio_95_interval'],[2.,2.],atol=0,rtol=0)
        np.testing.assert_allclose(report['paired_comparison']['temperature_difference_95_interval_k'],[0.,0.],atol=0,rtol=0)
        self.assertFalse(report['paired_comparison']['formal_equivalence_accepted'])
        report2=replicas.reduce_replicas(self.records([-8.,1.,2.,8.]),2000,123)['fourier']
        self.assertIsNone(report2['sponge']['replica_bootstrap_95_interval_mpa_s'])
        self.assertIsNone(report2['paired_comparison']['viscosity_ratio_95_interval'])
        self.assertGreater(report2['sponge']['nonpositive_bootstrap_draws'],0)
        records[1]['statistics']['sponge']['fourier']['sample_weight_duration_ps']=20.
        with self.assertRaisesRegex(ValueError,'incompatible'): replicas.reduce_replicas(records,2000,123)

    def test_unique_rng_roles_and_strict_spec(self):
        spec=self.spec();self.assertEqual(replicas.validate_spec(spec)['steps'],1000)
        bad=json.loads(json.dumps(spec));bad['replicas'][1]['initialization_seed']=bad['replicas'][0]['production_seed']+1
        for changed in (bad,dict(spec,extra=True),dict(spec,equilibration_steps=999),dict(spec,replicas=spec['replicas'][:3]),
                        dict(spec,sample_star=.0015),dict(spec,bootstrap_draws=20)):
            with self.assertRaises(ValueError): replicas.validate_spec(changed)

    def test_failed_equilibration_retains_stage_cost_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); binary=root/'binary';binary.write_text('synthetic executable')
            with mock.patch.object(replicas.lj_pilot,'equilibrate',side_effect=RuntimeError('injected equilibration failure')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    replicas.run_campaign(self.spec(),root/'campaign',binary,binary)
            execution=json.loads((root/'campaign/execution.json').read_text())
            self.assertEqual(execution['status'],'failed')
            self.assertEqual(list(execution['replicas']),['r0'])
            self.assertEqual(execution['replicas']['r0']['stage'],'equilibration')
            self.assertGreater(execution['replicas']['r0']['wall_seconds'],0)
            self.assertIn('model/model.json',execution['replicas']['r0']['files_sha256'])
            with self.assertRaisesRegex(ValueError,'incomplete'):replicas.summarize(root/'campaign')
            (root/'campaign/source/lj_pilot.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):replicas.summarize(root/'campaign')


if __name__ == '__main__': unittest.main()
