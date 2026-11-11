"""Native reference input, observation and signed-response accounting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/viscosity'))
import lj_native_cosine as native


class NativeCosine(unittest.TestCase):
    def test_block_boundaries_signs_and_temperature(self):
        p=dict(model=dict(dt_star=.1,atoms=8,mass_da=1.,box_angstrom=10.),
               start_star=1.,block_star=1.,steps=30,sample_interval=1)
        steps=np.arange(31);v=np.ones((31,3));v[10:20]=[1.,-1.,0.];v[20:30]=[3.,-1.,0.]
        v[30]=1e30  # endpoint is not another complete sample block
        report=native.block_report(steps,v,np.arange(31.),p,.01)
        self.assertEqual(report['temperature_k'],19.5)
        self.assertEqual(len(report['blocks']),2)
        self.assertEqual(report['viscosity_mpa_s'][1:],[None,None])
        self.assertGreater(report['fluidity_per_mpa_s'][0],0)
        self.assertLess(report['fluidity_per_mpa_s'][1],0)
        with self.assertRaisesRegex(ValueError,'block sample count'):
            native.block_report(steps[:-2],v[:-2],np.arange(29.),p,.01)

    def test_independent_dump_checks_cadence_and_native_dof(self):
        # Four phases with zero mean cos and exactly mean cos^2=1/2.
        n=4;box=4.;q=np.array([[0.,0.,z] for z in (0.,1.,2.,3.)]);c=np.cos(2*np.pi*q[:,2]/box)
        v=np.column_stack((.2+.3*c,np.array([.1,-.1,.1,-.1]),np.zeros(n)))
        p=dict(model=dict(atoms=n,mass_da=2.,box_angstrom=box),steps=0,sample_interval=1)
        fourier=.3;res=v.copy();res[:,0]-=fourier*c
        t_native=np.sum(res**2)*2*native.LM_TIME**2/(9*native.LM_KB)
        t_joint=.04*2*native.LM_TIME**2/(8*native.LM_KB)
        rows=np.array([[0.,t_native,.3,.3,.3,t_joint,0.,0.,0.,0.,.5]])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.dump'
            text='ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n4\nITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\nITEM: ATOMS id x y z vx vy vz\n'
            text+=''.join(str(i+1)+' '+' '.join(map(str,np.r_[q[i],v[i]]))+'\n' for i in range(n))
            path.write_text(text);self.assertEqual(native.audit_dump(path,rows,p)['frames'],1)
            wrong=rows.copy();wrong[0,1]*=9/12
            with self.assertRaisesRegex(ValueError,'observer mismatch'):native.audit_dump(path,wrong,p)
            path.write_text(text.replace('TIMESTEP\n0','TIMESTEP\n1'))
            with self.assertRaises((ValueError,IndexError)):native.audit_dump(path,rows,p)

    def test_native_input_has_no_python_integrator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'lammps').mkdir()
            p=dict(model=dict(atoms=108,mass_da=39.948),reference=dict(dt_fs=4.,temperature_k=120.,acceleration_angstrom_fs2=2e-7),
                tau_sponge_ps=2.,dt_sponge_ps=.004,sample_interval=10,steps=100000)
            (root/'plan.json').write_text(json.dumps(p))
            (root/'lammps/in.lammps').write_text('units real\nread_data model.data\nfix integrate all python/move old.Custom\nrun 100\n')
            text=native.input_text(root)
            self.assertNotIn('python',text)
            self.assertIn('fix integrate all nvt temp 120 120 2000',text)
            self.assertIn('fix_modify integrate temp flow',text)


if __name__=='__main__':unittest.main()
