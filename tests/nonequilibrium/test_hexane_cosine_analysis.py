"""Analytical replica statistics and window accounting, not MD acceptance."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/viscosity'))
import hexane_cosine_analysis as analysis
from test_cosine_analysis import fixture


class ReplicaAnalysis(unittest.TestCase):
    def test_frozen_reference_analyzer_and_source_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=analysis.analysis_sources();hashes={}
            for name,actual in sources.items():
                dest=root/'analysis-sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(actual,dest);hashes[name]=analysis.water.digest(actual)
            plan=dict(frozen_sha256={'sources/'+n:h for n,h in hashes.items() if not n.endswith('/hexane_cosine_analysis.py')})
            (root/'plan.json').write_text(json.dumps(plan))
            ready=dict(plan_sha256=analysis.water.digest(root/'plan.json'),source_sha256=hashes)
            def save(): (root/'analysis-readiness.json').write_text(json.dumps(ready))
            save();self.assertEqual(analysis.verify_analysis_sources(root,plan),hashes)
            for name in ('benchmarks/viscosity/experimental_reference.py','benchmarks/viscosity/hexane_cosine_analysis.py'):
                target=root/'analysis-sources'/name;original=target.read_bytes();target.write_bytes(original+b'\n# changed\n')
                with self.assertRaisesRegex(ValueError,'frozen analysis source'):analysis.verify_analysis_sources(root,plan)
                target.write_bytes(original)
                old=hashes.pop(name);save()
                with self.assertRaisesRegex(ValueError,'source set'):analysis.verify_analysis_sources(root,plan)
                hashes[name]=old;save()
            plan['frozen_sha256']['sources/benchmarks/viscosity/experimental_reference.py']='incorrect'
            with self.assertRaisesRegex(ValueError,'sampling dependency'):analysis.verify_analysis_sources(root,plan)

    def test_whole_replica_pairing_and_reciprocal_of_mean(self):
        f=np.array([[1.,2.],[2.,4.],[3.,6.],[4.,8.]])
        t=np.tile(np.array([280.,290.,310.,320.])[:,None],(1,2))
        p=t-300
        result,sample=analysis.reduce(f,t,p,659.29,1000,23)
        self.assertEqual(result[0]['viscosity_mpa_s'],.4)
        self.assertNotEqual(result[0]['viscosity_mpa_s'],np.mean(1/f[:,0]))
        np.testing.assert_array_equal(sample['fluidity_per_mpa_s'][:,1],2*sample['fluidity_per_mpa_s'][:,0])
        np.testing.assert_array_equal(sample['temperature_k'],t[sample['indices']].mean(axis=1))
        np.testing.assert_array_equal(sample['pressure_mpa'],p[sample['indices']].mean(axis=1))
        for i in (0,10,999):
            expected=analysis.reference.n_hexane_viscosity_pa_s(float(sample['temperature_k'][i,0]),659.29)*1000
            self.assertEqual(sample['reference_viscosity_mpa_s'][i,0],expected)
            self.assertEqual(sample['ratio_to_reference'][i,0],1/sample['fluidity_per_mpa_s'][i,0]/expected)
        # Reversing temperatures destroys the sampled response/reference covariance.
        wrong=sample['viscosity_mpa_s']/sample['reference_viscosity_mpa_s'][::-1]
        self.assertGreater(float(np.max(abs(wrong-sample['ratio_to_reference']))),.01)

    def test_zero_and_wrong_sign_draws_are_retained(self):
        f=np.array([-1.,-1.,1.,1.])[:,None]
        report,sample=analysis.reduce(f,np.full_like(f,293.15),np.zeros_like(f),659.29,1000,24)
        self.assertIsNone(report[0]['viscosity_mpa_s'])
        self.assertIsNone(report[0]['viscosity_95_interval_mpa_s'])
        self.assertIsNone(report[0]['ratio_to_reference_95_interval'])
        self.assertGreater(report[0]['zero_draws'],0)
        self.assertGreater(report[0]['nonpositive_draws'],report[0]['zero_draws'])
        self.assertEqual(sample['fluidity_per_mpa_s'].shape,(1000,1))
        self.assertEqual(np.count_nonzero(np.isinf(sample['viscosity_mpa_s'])),report[0]['zero_draws'])
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp)/'samples.npz',**sample)
            with np.load(Path(tmp)/'samples.npz') as restored:
                for key,value in sample.items():np.testing.assert_array_equal(restored[key],value)

    def test_equal_length_edge_groups_and_half_open_windows(self):
        d=fixture(np.full(101,.25))
        for key in ('pxx','pyy','pzz'):d[key]=np.arange(101)*1e6
        plan=dict(block_ps=20.,duration_ps=100.,block_bootstrap_draws=100,block_bootstrap_seed=23)
        for start,blocks,edge in ((20.,4,40.),(40.,3,20.)):
            result=analysis.extract(d,start,plan,'fit')
            self.assertEqual(result['report']['blocks'],blocks)
            self.assertEqual(result['report']['dropped_tail_frames'],1)
            self.assertEqual(result['report']['last_time_ps'],99.)
            self.assertEqual(result['edge_duration_ps'],edge)
            self.assertAlmostEqual(result['fluidity_per_mpa_s'],1.)
            self.assertEqual(result['last_minus_first_pressure_mpa'],40.)
        self.assertEqual(analysis.edge_difference([1.,999.,3.]),2.)
        self.assertEqual(analysis.edge_difference([1.,2.,3.,4.]),2.)
        # Reversing drive and response preserves the signed fluidity.
        d['acceleration']*=-1
        for key in ('velocity_fourier','velocity_com_fourier','velocity_fit'):d[key]*=-1
        self.assertAlmostEqual(analysis.extract(d,40.,plan,'fit')['fluidity_per_mpa_s'],1.)

    def test_partial_campaign_and_duplicate_replica_shapes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'plan.json').write_text(json.dumps(dict(contract='sponge.hexane.cosine-replicas.v1')))
            (root/'execution.json').write_text(json.dumps(dict(status='production_running')))
            with self.assertRaisesRegex(ValueError,'completed frozen'):analysis.run(root,root/'analysis')
            self.assertFalse((root/'analysis').exists())
        for bad in (np.ones((3,2)),np.full((4,2),np.nan)):
            with self.assertRaises(ValueError):analysis.reduce(bad,np.ones_like(bad)*300,np.zeros_like(bad),659.29)


if __name__=='__main__':unittest.main()
