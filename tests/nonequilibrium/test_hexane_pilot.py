"""Independent geometry and dense-SVD checks for the six-site chain pilot."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import h5py
    import numpy as np
except ImportError:
    sys.exit(77)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/viscosity'))
import hexane_model as model
import hexane_pilot as pilot
import hexane_nve as nve
import hexane_rattle as rattle
import hexane_gk_analysis as replica_gk


class HexanePilot(unittest.TestCase):
    def test_gk_campaign_raw_pressure_and_force_identity_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('viscosity.h5','observables.h5','checkpoint.spgr.h5',
                         'initial-state.npz','final-state.npz','crd.dat','vel.dat','force.dat'):
                (root/name).write_bytes(b'fixture')
            hashes={p.name:replica_gk.water.digest(p) for p in root.iterdir()}
            replica_gk.verify_raw(root,hashes)
            missing=dict(hashes);del missing['viscosity.h5']
            with self.assertRaisesRegex(ValueError,'incomplete raw manifest'):
                replica_gk.verify_raw(root,missing)
            (root/'viscosity.h5').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'raw identity'):
                replica_gk.verify_raw(root,hashes)
            for unit,values,error in (('Pa',np.arange(4.),None),('bar',np.arange(4.),'unit'),
                                      ('Pa',np.arange(3.),'pressure'),('Pa',[1.,2.,3.,np.nan],'pressure')):
                with h5py.File(root/'viscosity.h5','w') as file:
                    for axis in 'xyz':
                        dataset=file.create_dataset(f'observables/all/p{axis}{axis}/value',data=values)
                        dataset.attrs['unit']=unit
                if error:
                    with self.assertRaisesRegex(ValueError,error):
                        replica_gk.load_pressure(root/'viscosity.h5',np.arange(4.))
                else:
                    np.testing.assert_array_equal(replica_gk.load_pressure(root/'viscosity.h5',np.arange(4.)),
                                                  np.tile(np.arange(4.)[:,None],(1,3)))
            for identity,valid in (('rattle_nve_double_v3;BA_full_drift;energy_virial_every_step_v1',True),
                                   ('rattle_nve_double_v2;BA_full_drift',False),
                                   ('rattle_nve_double_v3;BA_full_drift',False)):
                with h5py.File(root/'checkpoint.spgr.h5','w') as file:
                    file.create_dataset('parameters/restart/integrator_state/constrained_nve_configuration',data=identity)
                if valid:replica_gk.verify_force_policy(root/'checkpoint.spgr.h5')
                else:
                    with self.assertRaisesRegex(ValueError,'force policy'):
                        replica_gk.verify_force_policy(root/'checkpoint.spgr.h5')

    def test_replica_bootstrap_preserves_settings_and_temperature_normalization(self):
        numerator=np.array([[300.,600.],[1200.,2400.]])
        temperature=np.array([[300.,300.],[600.,600.]])
        estimate,ci,samples,indices=replica_gk.reduce_replicas(numerator,temperature,1000,19)
        np.testing.assert_allclose(estimate,[5/3,10/3])
        self.assertNotAlmostEqual(estimate[0],np.mean(numerator[:,0]/temperature[:,0]))
        np.testing.assert_array_equal(samples[:,1],2*samples[:,0])
        np.testing.assert_array_equal(ci[:,1],2*ci[:,0])
        np.testing.assert_array_equal(indices,replica_gk.reduce_replicas(numerator,temperature,1000,19)[3])
        negative=replica_gk.reduce_replicas(-numerator,temperature,1000,19)
        np.testing.assert_array_equal(negative[2],-samples)  # no positivity selection
        for a,b in ((numerator[:1],temperature[:1]),(numerator,temperature*0),
                    (numerator*np.nan,temperature),(numerator,temperature[:1])):
            with self.assertRaises(ValueError):replica_gk.reduce_replicas(a,b)

    def test_gk_windows_and_decimation_keep_known_si_integral(self):
        times=np.arange(401)*.05
        stress=np.broadcast_to(np.array([1.,2.,-3.])*1e6,(len(times),3))
        settings=dict(start_ps=[0.],decimation=[1,2],demean=[False,True],
            windows_ps=[.5,1.,2.],block_ps=10.,replica_bootstrap_seed=19)
        rows,_=replica_gk.replica_curves(times,stress,np.full(len(times),300.),np.full(len(times),10.),settings)
        self.assertEqual(len(rows),12)
        for row in rows:
            expected=np.array([1.,4.,9.])*1e12*10e-27/(replica_gk.gk.KB*300)*row['window_ps']*1e-12*1000
            if row['setting'].endswith('demean-1'):expected*=0
            np.testing.assert_allclose(row['component_mpa_s'],expected,rtol=1e-13,atol=1e-20)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'plan.json').write_text(json.dumps(dict(contract='sponge.hexane.gk-replicas.v1')))
            (root/'execution.json').write_text(json.dumps(dict(stage='sampling')))
            with self.assertRaisesRegex(ValueError,'complete unchanged campaign'):
                replica_gk.summarize(root,root/'analysis')
            self.assertFalse((root/'analysis').exists())

    def test_production_cadences_cover_same_physical_steps(self):
        schedule = rattle.sampling_schedule(100., .0005, .1, 10, 10.)
        self.assertEqual(schedule, dict(steps=200000,stride=200,sample_interval=10,checkpoint_interval=20000))
        stresses = np.arange(0,schedule['steps']+1,schedule['sample_interval'])
        native = np.arange(0,schedule['steps']+1,schedule['stride'])
        np.testing.assert_array_equal(stresses[::20],native)
        for args in ((100.,.0005,.1,3,10.), (100.,.0005,.1,True,10.),
                     (100.,.0005,.1,0,10.), (100.,.0005,.1,10,3.),
                     (100.,.0005,.0001,1,10.), (100.,0,.1,10,10.),
                     (float('nan'),.0005,.1,10,10.), (200.,.000001,.1,10,10.)):
            with self.assertRaises(ValueError): rattle.sampling_schedule(*args)

    def fixture(self, path):
        model.prepare(path, cells=(1, 2, 3), cutoff=3.)
        return model.load(path)

    def test_geometry_graph_and_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model'; m, state = self.fixture(path)
            q = state['positions'].reshape(-1, 6, 3)
            b = np.diff(q, axis=1)
            b -= m['box_angstrom']*np.rint(b/m['box_angstrom'])
            np.testing.assert_allclose(np.linalg.norm(b, axis=2), 1.54, rtol=0, atol=1e-6)
            angles = np.arccos(np.sum(-b[:, :-1]*b[:, 1:], axis=2)/np.linalg.norm(b[:, :-1], axis=2)/np.linalg.norm(b[:, 1:], axis=2))
            np.testing.assert_allclose(angles, np.deg2rad(114), rtol=0, atol=1e-6)
            # Independent graph shortest paths from constraints, not index gaps.
            n = m['atoms']; dist = np.full((n, n), np.inf); np.fill_diagonal(dist, 0)
            for i,j,_ in state['pairs']:
                dist[int(i),int(j)] = dist[int(j),int(i)] = 1
            for k in range(n): dist = np.minimum(dist, dist[:, k, None]+dist[None, k, :])
            rows = (path/'exclude.txt').read_text().splitlines()[1:]
            for i, row in enumerate(rows):
                fields = list(map(int, row.split()))
                self.assertEqual(fields[0], len(fields)-1)
                self.assertEqual(set(fields[1:]), set(np.flatnonzero((dist[i] > 0)&(dist[i] <= 3)&(np.arange(n)>i))))
            self.assertAlmostEqual(state['masses'].sum()/6.02214076e23*1e-3/(m['box_angstrom']*1e-10)**3, m['actual_density_kg_m3'], delta=1e-6)

    def test_full_dense_svd_thermal_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            m, st = self.fixture(Path(tmp)/'model'); q = st['positions']; mass = st['masses']; box=m['box_angstrom']
            rng=np.random.default_rng(81); half=rng.normal(size=q.shape); force=rng.normal(size=q.shape); h=.01
            n=len(q); jac=np.zeros((len(st['pairs']), 3*n)); root=np.sqrt(np.repeat(mass,3))
            for k,(i,j,_) in enumerate(st['pairs']):
                i,j=int(i),int(j); d=q[i]-q[j]; d-=box*np.rint(d/box)
                jac[k,3*i:3*i+3]=d; jac[k,3*j:3*j+3]=-d
            _,s,vh=np.linalg.svd(jac/root,full_matrices=True)
            self.assertGreater(s.min(),1e-4)
            null=vh[len(s):].T
            w=(half+.5*h*force/mass[:,None]).ravel()*root
            projected=null@(null.T@w)
            modes=np.zeros((3*n,4))
            for a in range(3):modes[a::3,a]=np.sqrt(mass)
            modes[0::3,3]=np.sqrt(mass)*np.cos(2*np.pi*q[:,2]/box)
            basis=np.linalg.qr(null@(null.T@modes))[0]
            thermal=projected-basis@(basis.T@projected)
            velocity, kinetic=pilot.frame(q,half,force,mass,box,h)
            np.testing.assert_allclose(velocity.ravel(),projected/root,rtol=0,atol=2e-14)
            self.assertAlmostEqual(kinetic,.5*thermal@thermal,delta=1e-10)
            np.testing.assert_allclose(jac@velocity.ravel(),0,atol=1e-14)

    def test_structure_periodic_image_invariance_and_trans(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); m,st=self.fixture(root/'model'); q=st['positions']; mass=st['masses']; box=m['box_angstrom']
            frames=np.repeat(q[None],11,axis=0); a=root/'a'; b=root/'b';a.mkdir();b.mkdir()
            first, order, gauche=pilot.structure(frames,mass,box,.1,a)
            shifted=frames+np.random.default_rng(9).integers(-3,4,size=frames.shape)*box
            second, other, other_gauche=pilot.structure(shifted,mass,box,.1,b)
            np.testing.assert_allclose(order,other,rtol=0,atol=1e-14)
            np.testing.assert_array_equal(gauche,np.zeros(11))
            np.testing.assert_array_equal(gauche,other_gauche)
            self.assertLess(first['latter_single_origin_msd_angstrom2'],1e-25)
            self.assertLess(second['latter_single_origin_msd_angstrom2'],1e-25)
            for name in ('rdf-com.csv','density-z.csv'):
                np.testing.assert_allclose(np.loadtxt(a/name,delimiter=',',skiprows=1),np.loadtxt(b/name,delimiter=',',skiprows=1),rtol=0,atol=1e-12)

    def test_manifest_missing_or_modified_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model';m,_=self.fixture(path)
            (path/'mass.txt').write_text('bad\n')
            with self.assertRaisesRegex(ValueError,'hash'):model.load(path)
            del m['input_sha256']['mass.txt']
            (path/'model.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'manifest'):model.load(path)

    def test_constraint_reaction_against_global_squared_distance_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,st=self.fixture(Path(tmp)/'model');q=st['positions'];mass=st['masses'];box=m['box_angstrom']
            rng=np.random.default_rng(82);half=rng.normal(size=q.shape);force=rng.normal(size=q.shape);h=.02
            n=len(q);pairs=st['pairs'];jac=np.zeros((len(pairs),3*n));displacements=[]
            for k,(a,b,_) in enumerate(pairs):
                a,b=int(a),int(b);d=q[a]-q[b];d-=box*np.rint(d/box);displacements.append(d)
                jac[k,3*a:3*a+3]=d;jac[k,3*b:3*b+3]=-d
            inverse=1/np.repeat(mass,3);metric=(jac*inverse)@jac.T
            u=(half+.5*h*force/mass[:,None]).ravel()
            u-=inverse*(jac.T@np.linalg.solve(metric,jac@u));u=u.reshape(n,3)
            curvature=np.array([np.sum((u[int(a)]-u[int(b)])**2) for a,b,_ in pairs])
            lambdas=np.linalg.solve(metric,-jac@(force.ravel()*inverse)-curvature)
            virial=sum(lam*np.outer(d,d) for lam,d in zip(lambdas,displacements))
            result=nve.reference(q,half,force,mass,box,h)
            np.testing.assert_allclose(result['velocity'],u,atol=2e-14,rtol=0)
            np.testing.assert_allclose(result['constraint'],virial,atol=1e-10,rtol=1e-12)
            boost=np.array([1.,-2.,3.]);shifted=q+rng.integers(-3,4,size=q.shape)*box
            other=nve.reference(shifted,half+boost,force,mass,box,h)
            np.testing.assert_allclose(other['kinetic'],result['kinetic'],atol=1e-10,rtol=1e-12)
            np.testing.assert_allclose(other['constraint'],result['constraint'],atol=1e-10,rtol=1e-12)

    def test_cutoff_offset_retains_one_five_intermolecular_excludes_one_four(self):
        m=dict(box_angstrom=100.,cutoff_angstrom=2.,coefficients=[dict(types=[0,0],effective=[64.,8.])])
        st=dict(types=np.zeros(12,dtype=int));q=np.arange(12)[:,None]*np.array([[5.,0,0]])
        # Atom 0 is near 3 (excluded), 4 (included), and 6 (other molecule).
        q[3]=[.5,0,0];q[4]=[1.,0,0];q[6]=[1.5,0,0]
        # Included: 0-4,0-6,3-6,4-6. 3-4 is bonded and excluded.
        self.assertEqual(nve.cutoff_offset(q,m,st),4*(64/2**12-8/2**6))


if __name__ == '__main__':
    unittest.main()
