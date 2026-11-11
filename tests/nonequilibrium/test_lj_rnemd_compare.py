"""LAMMPS profile units, normalization, phase adapter and integrity guards."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
try:
    import numpy as np
    import h5py
except ImportError:
    raise SystemExit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'benchmarks/viscosity'))
import lj_rnemd_compare as paired
import lj_transport_replicas as transport


def fixture(root):
    rows = []
    for step in range(4):
        rows.append(f'{step} 8 8')
        for b in range(8):
            # One particle per bin, vx=+/-0.01 and vy=+/-0.02 A/fs.
            vx = .01 if b < 4 else -.01
            vy = .02*(-1)**b
            if step >= 2 and b in (0, 4): vx *= -1
            rows.append(f'{b+1} {(b+.5)/8:.17g} 1 12 {12*vx:.17g} {12*vy:.17g} 0 {12*(vx*vx+vy*vy):.17g}')
    (root/'profile.dat').write_text('# synthetic sums\n'+'\n'.join(rows)+'\n')
    np.savetxt(root/'momentum.dat', [[0, 0, 300], [1, 0, 300], [2, -.24, 300], [3, -.24, 300]])
    return dict(steps=3, stride=1, bins=8, atoms=8, mass=12, box=32, dt_ps=.001)


class Adapter(unittest.TestCase):
    def test_sparse_pre_exchange_ledger_holds_between_pulses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);options=fixture(root)
            # Six sampled steps, exchanges at 2 and 4; the final exchange at
            # 6 is excluded from all outputs. Impulses are 0,0,24,48.
            lines=(root/'profile.dat').read_text().splitlines()[1:]
            first=lines[:9];rows=[]
            for step in range(7): rows.extend([f'{step} 8 8',*first[1:]])
            (root/'profile.dat').write_text('\n'.join(rows)+'\n')
            np.savetxt(root/'momentum.dat',[[0,0,300],[2,0,300],[4,-.24,300],[6,-.48,300]])
            np.savetxt(root/'kinetic.dat',[[i,300] for i in range(7)])
            data=paired.lammps_profile(root/'profile.dat',root/'momentum.dat',
                **dict(options,steps=6),ledger_stride=2,kinetic_path=root/'kinetic.dat')
            np.testing.assert_array_equal(data['momentum'],[0,0,0,24,24,48,48])
            with self.assertRaisesRegex(ValueError,'separate kinetic samples'):
                paired.lammps_profile(root/'profile.dat',root/'momentum.dat',**dict(options,steps=6),ledger_stride=2)

    def test_signed_impulse_norm_none_and_thermal_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); options = fixture(root)
            data = paired.lammps_profile(root/'profile.dat', root/'momentum.dat', **options)
            np.testing.assert_array_equal(data['momentum'], [0, 0, 24, 24])
            np.testing.assert_allclose(data['bin_momentum'][:, 0], [12, 12, -12, -12])
            np.testing.assert_array_equal(data['bin_mass'], np.full((4, 8), 12))
            # Per-bin flow removal leaves only alternating transverse velocity;
            # 3N - occupied - 2 = 14 thermal degrees of freedom.
            expected = (8*12*.02**2)*paired.LM_TIME**2/(14*paired.LM_KB)
            np.testing.assert_allclose(data['thermal_temperature'], expected, rtol=1e-12)
            np.testing.assert_allclose(data['area'], 10.24)
            np.testing.assert_allclose(data['length'], 3.2)

    def test_reanalysis_execution_and_integrity_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(contract='sponge.lj.paired-rnemd-pilot.v1', input_sha256={})
            (root/'plan.json').write_text(json.dumps(plan))
            completed = {engine: dict(status='completed') for engine in ('sponge', 'lammps')}
            for execution, message in (
                (dict(status='engines_completed'), 'both engines'),
                (dict(status='engines_completed', engines=completed, plan_sha256='bad'), 'run plan changed'),
                (dict(status='engines_completed', engines=completed, raw_output_sha256={'plan.json':'bad'}), 'raw output changed')):
                (root/'execution.json').write_text(json.dumps(execution))
                with self.assertRaisesRegex(ValueError, message): paired.analyze_run(root)

    def test_rejects_invalid_adapter_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); options = fixture(root)
            for key, value in (('mass', 0), ('mass', -1), ('box', 0), ('dt_ps', np.nan),
                               ('stride', 0), ('bins', 5), ('atoms', 1), ('steps', 2.5)):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    paired.lammps_profile(root/'profile.dat', root/'momentum.dat', **(options | {key: value}))

    def test_rejects_sign_normalization_geometry_and_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ('sign', 'mass', 'geometry', 'missing', 'negative_kinetic'):
                options = fixture(root)
                if mode == 'sign':
                    ledger = np.loadtxt(root/'momentum.dat'); ledger[:, 1] *= -1
                    np.savetxt(root/'momentum.dat', ledger)
                else:
                    lines = (root/'profile.dat').read_text().splitlines()
                    if mode == 'missing': lines.pop()
                    else:
                        fields = lines[2].split()
                        if mode == 'mass': fields[3] = '6'
                        if mode == 'geometry': fields[1] = '.2'
                        if mode == 'negative_kinetic': fields[7] = '-1'
                        lines[2] = ' '.join(fields)
                    (root/'profile.dat').write_text('\n'.join(lines)+'\n')
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    paired.lammps_profile(root/'profile.dat', root/'momentum.dat', **options)


class TransportReplicas(unittest.TestCase):
    def test_decimation_preserves_impulse_and_recomputes_extensive_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);step=np.arange(17);mass=np.ones((17,8));momentum=np.tile(np.arange(8),(17,1)).astype(float)
            momentum[1::2]*=3
            p=dict(steps=16,sample_interval=1,bins=8,start_star=0.,block_star=4.,model=dict(dt_star=1.),dt_sponge_ps=1.,dt_lammps_ps=1.,exclusions=[0])
            (root/'plan.json').write_text(json.dumps(p))
            for e in transport.ENGINES:
                np.savez(root/f'{e}-canonical-profile.npz',step=step,time=step.astype(float),area=np.ones(17),length=np.full(17,8.),momentum=np.maximum(0,(step-1)//4),thermal_temperature=np.full(17,120.),kinetic_temperature=np.full(17,125.),bin_mass=mass,bin_momentum=momentum)
            dense=transport.extract_rn(root,1)['rnemd-exclude-0']['sponge'];coarse=transport.extract_rn(root,2)['rnemd-exclude-0']['sponge']
            np.testing.assert_allclose(dense['momentum'],2*np.arange(8))
            np.testing.assert_allclose(coarse['momentum'],np.arange(8))
            self.assertEqual(dense['flux_pa'],coarse['flux_pa'])
            self.assertEqual(coarse['analysis_sample_interval'],2)
            self.assertEqual(coarse['intervals_per_block'],2)
            for bad in (True,0,-1,1.5,3,8):
                with self.subTest(bad=bad),self.assertRaises(ValueError):transport.extract_rn(root,bad)
            p['sample_interval']=2
            for e in transport.ENGINES:
                with np.load(root/f'{e}-canonical-profile.npz') as f:data={k:f[k][::2] for k in f.files}
                data['bin_momentum'][1::2]*=3
                np.savez(root/f'{e}-canonical-profile.npz',**data)
            (root/'plan.json').write_text(json.dumps(p))
            sparse=transport.extract_rn(root,4)['rnemd-exclude-0']['sponge']
            np.testing.assert_allclose(sparse['momentum'],np.arange(8))
            self.assertEqual(sparse['flux_pa'],dense['flux_pa']);self.assertEqual(sparse['duration_ps'],16.)
            self.assertEqual(sparse['kinetic_temperature_k'],125.)
            for half in ('first','second'):
                np.testing.assert_allclose(sparse[half]['momentum'],np.arange(8))
                self.assertEqual(sparse[half]['temperature_k'],120.)
            (root/'plan.json').write_text(json.dumps(dict(p,start_star=1.)))
            with self.assertRaisesRegex(ValueError,'boundaries'):transport.extract_rn(root,4)
            (root/'plan.json').write_text(json.dumps(dict(p,sample_interval=0)))
            with self.assertRaisesRegex(ValueError,'original'):transport.extract_rn(root,4)

    def spec(self):
        return dict(contract=transport.CONTRACT,duration_star=200.,start_star=40.,block_star=20.,
            gk_sample_star=.004,gk_audit_star=2.,gk_windows_star=[.5,1.,2.,4.],reference_gk_window_star=1.,
            rnemd_exchange=200,rnemd_bins=20,rnemd_exclusions=[0,1],bootstrap_draws=2000,bootstrap_seed=123)

    def test_protocol_and_paired_gk_temperature_normalization(self):
        self.assertEqual(transport.validate_spec(self.spec(),.002),dict(steps=100000,stride=2,audit_stride=1000))
        for bad in (dict(self.spec(),rnemd_exchange=150),dict(self.spec(),rnemd_bins=7),
                    dict(self.spec(),gk_windows_star=[20.]),dict(self.spec(),reference_gk_window_star=3.),
                    dict(self.spec(),bootstrap_draws=True),dict(self.spec(),start_star=190.)):
            with self.assertRaises(ValueError):transport.validate_spec(bad,.002)
        records=[]
        for i in range(4):
            engines={}
            for e,scale in [('sponge',2.),('lammps',1.)]:
                numerator=(20.+10*i)*scale;t=100.*(i+1)
                engines[e]=dict(kind='gk',numerator_mpa_s_k=numerator,temperature_k=t,temperature_scale=1.,
                    per_replica_viscosity_mpa_s=numerator/t,components_numerator_mpa_s_k=[numerator]*3,
                    first_numerator=numerator,second_numerator=numerator,first_temperature=t,second_temperature=t,
                    blocks=8,samples_per_block=100,volume_nm3=10.,actual_window_ps=1.,sample_weight_duration_ps=20.)
            records.append(dict(id=str(i),statistics={'gk-1':engines}))
        report,comparisons=transport.reduce_replicas(records,self.spec())
        self.assertAlmostEqual(report['gk-1']['lammps']['viscosity_mpa_s'],35./250.)
        self.assertNotAlmostEqual(report['gk-1']['lammps']['mean_of_separately_normalized_replicas_mpa_s'],35./250.)
        np.testing.assert_allclose(comparisons[0]['paired_ratio_95_interval'],[2.,2.],rtol=0,atol=0)

    def test_rnemd_pools_extensive_profile_and_flux_before_fit(self):
        entries=[];z=np.arange(8)+.5
        for i in range(4):
            mass=np.ones(8);mass[1:4]+=.1*i;mass[5:8]-=.1*i
            velocity=.001*(i+1)*np.where(z<4,z,8-z);momentum=mass*velocity;flux=1e6*(1+.1*i)
            half=dict(mass=mass.tolist(),momentum=momentum.tolist(),flux_pa=flux,temperature_k=120.)
            entries.append(dict(kind='rnemd',**half,first=half,second=half,length_nm=8.,exclude=0,
                temperature_scale=1.,blocks=8,intervals_per_block=20,duration_ps=10.))
        indices=np.random.default_rng(5).integers(4,size=(2000,4))
        report,draws,_=transport.estimate(entries,indices)
        mass=np.mean([x['mass'] for x in entries],axis=0);momentum=np.mean([x['momentum'] for x in entries],axis=0)
        v=momentum/mass;flux=np.mean([x['flux_pa'] for x in entries])
        slopes=[np.polyfit(z[a:b],v[a:b],1)[0]*sign for a,b,sign in [(1,4,1),(5,8,-1)]]
        expected=flux/(np.mean(slopes)*1e9)
        self.assertAlmostEqual(report['viscosity_mpa_s'],expected,places=12)
        self.assertNotAlmostEqual(report['viscosity_mpa_s'],np.mean([(1+.1*i)/(i+1) for i in range(4)]),places=3)
        wrong=[dict(x,momentum=(-np.array(x['momentum'])).tolist()) for x in entries]
        bad,draws,_=transport.estimate(wrong,indices)
        self.assertIsNone(bad['viscosity_mpa_s']);self.assertIsNone(bad['replica_bootstrap_95_interval_mpa_s'])
        self.assertTrue(np.isnan(draws).all());self.assertEqual(bad['nonpositive_branch_draws'],[2000,2000])

    def test_rnemd_total_energy_audit_is_not_exchange_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'sponge').mkdir();(root/'lammps').mkdir()
            m=dict(atoms=2,box_angstrom=10.,cutoff_angstrom=3.,c12=1000.,c6=10.,epsilon_kcal_mol=.5)
            q=np.array([[0.,0.,0.],[2.,0.,0.]])
            (root/'sponge/coordinate.txt').write_text('2\n0 0 0\n2 0 0\n')
            np.tile(q,(2,1,1)).astype(np.float32).tofile(root/'sponge/crd.dat')
            potential=1000/2**12-10/2**6;cutoff=1000/3**12-10/3**6;tail=-2*np.pi*4*10/(3*10**3*3**3)
            kinetic=.5*float(np.float32(3*paired.SP_KB))*100
            with h5py.File(root/'sponge/observables.h5','w') as f:
                f['parameters/sponge/output/status']=np.bytes_('finalized')
                f['observables/all/step']=[0,10,20]
                f['observables/all/eff_pot/value']=[potential+tail]*3
            np.savetxt(root/'lammps/energy.dat',[[t,potential-cutoff,kinetic] for t in (0,10,20)])
            plan=dict(model=m,steps=20,sample_interval=10)
            report=paired.audit_energy(root,plan,dict(kinetic_temperature=np.full(3,100.)))
            self.assertTrue(all(report['initial_energy_gate'].values()))
            self.assertEqual(report['sponge']['energy_range_per_atom_epsilon'],0.)
            # Dense profile samples between energy audits must not enter KE.
            dense_plan=dict(plan,sample_interval=5,audit_interval=10)
            dense=dict(kinetic_temperature=np.array([100.,900.,100.,700.,100.]))
            dense_report=paired.audit_energy(root,dense_plan,dense)
            self.assertEqual(dense_report['sponge'],report['sponge'])
            with self.assertRaisesRegex(ValueError,'align with profiles'):
                paired.audit_energy(root,dict(dense_plan,audit_interval=9),dense)
            with self.assertRaisesRegex(ValueError,'temperature cadence'):
                paired.audit_energy(root,dense_plan,dict(kinetic_temperature=np.full(3,100.)))
            with h5py.File(root/'sponge/observables.h5','a') as f:f['observables/all/eff_pot/value'][1]+=1.
            report=paired.audit_energy(root,plan,dict(kinetic_temperature=np.full(3,100.)))
            self.assertAlmostEqual(report['sponge']['energy_range_per_atom_epsilon'],1.)


if __name__ == '__main__':
    unittest.main()
