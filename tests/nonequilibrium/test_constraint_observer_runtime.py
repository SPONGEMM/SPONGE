"""Real SETTLE/SHAKE trajectories with integrated HDF5 and offline C++ observers, not liquid GK."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
try:
    import h5py
    import numpy
except ImportError:
    raise SystemExit(77)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks/viscosity'))
import constraint_observer_probe as probe

class RigidRotor(unittest.TestCase):
    def test_real_rotors_recover_continuous_stress_with_second_order_error(self):
        with tempfile.TemporaryDirectory() as temp:
            result=probe.run(SPONGE,OBSERVER,Path(temp)/'campaign')
            self.assertEqual(len(result['cases']),12)
            self.assertEqual(result['status'],'passed_bounded_rotor_gates',result)
            self.assertFalse(result['formal_viscosity_acceptance'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sponge',required=True);p.add_argument('--observer',required=True)
    a,extra=p.parse_known_args();SPONGE=a.sponge;OBSERVER=a.observer
    unittest.main(argv=[__file__,*extra])
