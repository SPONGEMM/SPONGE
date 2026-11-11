"""Published verification points, units and scope guards for reference curves."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/viscosity'))
import experimental_reference as ref

class PublishedReference(unittest.TestCase):
    def test_completed_analysis_adapter_rejects_changed_evidence(self):
        import numpy as np
        import compare_gk_reference as comparison
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);analysis=root/'analysis';analysis.mkdir()
            def save(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
            dependencies={}
            for name in ('benchmarks/viscosity/hexane_model.py','benchmarks/viscosity/water_model.py',
                         'benchmarks/viscosity/lj_model.py','scripts/analyze_green_kubo.py'):
                path=root/'analysis-sources'/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('synthetic frozen source fixture')
                dependencies[name]=comparison.digest(path)
            analyzer='benchmarks/viscosity/hexane_gk_analysis.py'
            (root/'analysis-sources'/analyzer).write_text('synthetic analyzer fixture')
            hashes={**dependencies,analyzer:comparison.digest(root/'analysis-sources'/analyzer)}
            plan=dict(contract='sponge.hexane.gk-replicas.v1',replicas=['r1','r2','r3','r4'],source_hashes=dependencies,
                analysis=dict(start_ps=[20.],decimation=[1],demean=[False],windows_ps=[1.],
                    replica_bootstrap_seed=19,replica_bootstrap_draws=1000))
            save(root/'plan.json',plan);identity=comparison.digest(root/'plan.json')
            ready=dict(plan_sha256=identity,source_hashes=hashes)
            save(root/'analysis-readiness.json',ready)
            save(root/'execution.json',dict(stage='completed_sampling_pending_analysis',completed=plan['replicas'],plan_sha256=identity))
            temperature=np.array([[290.],[300.],[310.],[320.]])
            numerator=np.array([[290.],[-300.],[620.],[960.]])
            indices=np.random.default_rng(19).integers(4,size=(1000,4))
            samples=numerator[indices].mean(axis=1)/temperature[indices].mean(axis=1)
            np.savez(analysis/'bootstrap.npz',indices=indices,viscosity_mpa_s=samples)
            replicas=[]
            for i,name in enumerate(plan['replicas']):
                save(root/'runs'/name/'summary.json',dict(fixture=True,replica=name))
                replicas.append(dict(replica=name,run_summary_sha256=comparison.digest(root/'runs'/name/'summary.json'),
                    estimates=[dict(setting='start-20-decimation-1-demean-0',window_ps=1.,
                        temperature_k=temperature[i,0],numerator_mpa_s_k=numerator[i,0])]))
            summary=dict(plan_sha256=identity,status='conditional_finite_window_diagnostics_not_transport_acceptance',
                formal_viscosity_acceptance=False,analysis_sha256=hashes[analyzer],dependencies=dependencies,
                physical_model=dict(actual_density_kg_m3=659.29),replicas=replicas,
                aggregate=[dict(setting='start-20-decimation-1-demean-0',window_ps=1.,temperature_k=305.,
                    viscosity_mpa_s=numerator.mean()/305.,
                    conditional_replica_95_interval_mpa_s=np.quantile(samples,[.025,.975]).tolist())])
            save(analysis/'summary.json',summary)
            result=comparison.run(root,analysis,root/'comparison')
            self.assertFalse(result['formal_viscosity_acceptance'])
            self.assertGreater(result['settings'][0]['nonpositive_viscosity_bootstrap_draws'],0)
            for target,altered,error in (
                (root/'execution.json',dict(stage='sampling'),'completed frozen'),
                (analysis/'summary.json',{**summary,'aggregate':[]},'all declared'),
                (root/'runs/r1/summary.json',dict(changed=True),'run summary'),
                (root/'analysis-readiness.json',{**ready,'source_hashes':{analyzer:hashes[analyzer]}},'dependency')):
                original=target.read_text();save(target,altered)
                with self.assertRaisesRegex(ValueError,error):comparison.load_campaign(root,analysis)
                target.write_text(original)
            np.savez(analysis/'bootstrap.npz',indices=indices,viscosity_mpa_s=samples+1.)
            with self.assertRaises(AssertionError):comparison.load_campaign(root,analysis)
            np.savez(analysis/'bootstrap.npz',indices=indices[::-1],viscosity_mpa_s=samples)
            with self.assertRaises(AssertionError):comparison.load_campaign(root,analysis)

    def test_joint_temperature_reference_covariance_and_signed_integrals(self):
        import numpy as np
        import compare_gk_reference as comparison
        temperature=np.array([290.,310.])
        sampled_temperature=np.tile(np.linspace(280.,320.,101)[:,None],(1,2))
        curve=np.array([ref.water_viscosity_pa_s(t,998)*1000 for t in sampled_temperature[:,0]])
        samples=curve[:,None]*np.array([2.,-3.])
        estimates=np.array([ref.water_viscosity_pa_s(t,998)*1000 for t in temperature])*[2.,-3.]
        result,_,ratios=comparison.compare('water',998,estimates,temperature,samples,sampled_temperature)
        # Exact cancellation requires preserving the pairing of viscosity and T.
        np.testing.assert_allclose(ratios,np.tile([2.,-3.],(101,1)),rtol=1e-14)
        np.testing.assert_allclose(result['conditional_ratio_95_interval'],[[2.,-3.],[2.,-3.]],rtol=1e-14)
        np.testing.assert_array_equal(result['nonpositive_draws'],[0,101])
        wrong_pairing=samples/np.array([ref.water_viscosity_pa_s(t,998)*1000 for t in sampled_temperature[::-1,0]])[:,None]
        self.assertGreater(np.ptp(wrong_pairing[:,0]),1.)
        for bad in (sampled_temperature[:1],sampled_temperature*np.nan,sampled_temperature*0):
            with self.assertRaises(ValueError):comparison.compare('water',998,estimates,temperature,samples,bad)

    def test_iapws_table4_all_noncritical_checkpoints(self):
        for t,rho,micro in [(298.15,998,889.735100),(298.15,1200,1437.649467),(373.15,1000,307.883622),(433.15,1,14.538324),(433.15,1000,217.685358),(873.15,1,32.619287),(873.15,100,35.802262),(873.15,600,77.430195),(1173.15,1,44.217245),(1173.15,100,47.640433),(1173.15,400,64.154608)]:
            with self.subTest(t=t,rho=rho):self.assertAlmostEqual(ref.water_viscosity_pa_s(t,rho)*1e6,micro,delta=.0000006)
    def test_hexane_table6_all_checkpoints(self):
        for t,rho,micro,tolerance in [(250,0,5.2584,.00006),(400,0,8.4149,.00006),(550,0,11.442,.0006),(250,700,528.20,.006),(400,600,177.62,.006),(550,500,95.002,.0006)]:
            with self.subTest(t=t,rho=rho):self.assertAlmostEqual(ref.n_hexane_viscosity_pa_s(t,rho)*1e6,micro,delta=tolerance)
    def test_invalid_and_critical_states_do_not_get_reference_label(self):
        for fn in (ref.water_viscosity_pa_s,ref.n_hexane_viscosity_pa_s):
            for t,rho in [(float('nan'),998),(300,float('inf')),(300,-1),(True,1),(300,False),(0,1),(1200,1)]:
                with self.subTest(fn=fn,t=t,rho=rho),self.assertRaises(ValueError):fn(t,rho)
        with self.assertRaisesRegex(ValueError,'critical'):ref.water_viscosity_pa_s(647.096,322.)
        with self.assertRaises(ValueError):ref.reference('argon',300,1)
        r=ref.reference('water',298.15,998)
        self.assertIsNone(r['pressure_mpa']);self.assertFalse(r['phase_and_pressure_validated']);self.assertTrue(r['not_an_md_result'])
    def test_experimental_table_has_all_points_and_per_point_uncertainty(self):
        with (ROOT/'benchmarks/viscosity/data/n_hexane_klein2019.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),11);self.assertEqual(len({r['temperature_k'] for r in rows}),11)
        self.assertEqual(rows[0]['viscosity_mpa_s'],'0.3495');self.assertEqual(rows[-1]['relative_expanded_uncertainty_percent'],'3.7')
        self.assertTrue(all(r['coverage_factor']=='2' and r['state_path']=='saturation_approximation_with_helium' for r in rows))

if __name__=='__main__':unittest.main()
