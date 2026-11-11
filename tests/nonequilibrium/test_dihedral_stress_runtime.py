"""Dihedral stress against independent energy derivatives in the real engine."""
import argparse
from pathlib import Path
import unittest
import test_green_kubo_runtime as runtime
import numpy as np


class DihedralStress(unittest.TestCase):
    run_case = runtime.GreenKuboRuntime.run_case

    def test_energy_gradient_and_six_strain_derivatives(self):
        # All coordinates are exactly representable, including the wrapped case.
        original = np.array([[1., 2., 3.], [2.5, 2.25, 3.5],
                             [3., 3.5, 3.25], [4.25, 3., 4.5]])
        coefficients = np.array([.7, -.14, 1.57], dtype=np.float32).astype(float)

        def energy(q):
            axis = q[2]-q[1]
            axis /= np.linalg.norm(axis)
            v, w = q[0]-q[1], q[3]-q[2]
            v -= axis*(v@axis)
            w -= axis*(w@axis)
            c = (v@w)/np.linalg.norm(v)/np.linalg.norm(w)
            # cis=0, trans=pi. Chebyshev polynomials avoid atan branch choices.
            return coefficients @ [1+c, 2-2*c*c, 1+4*c**3-3*c]

        for transform, shift in [(np.eye(3), 0), (np.eye(3), 29),
                                 (np.diag([-1, 1, 1]), 0),
                                 (np.eye(3)[:, [2, 0, 1]], 0)]:
            with self.subTest(transform=transform.tolist(), shift=shift):
                # shift=29 wraps x=33.25 to 1.25 while atom 0 stays at 30:
                # the bonded chain crosses the periodic face.
                q = original@transform
                expected_force = np.zeros_like(q)
                tensor = np.zeros((3, 3))
                h = 1e-5
                for index in np.ndindex(q.shape):
                    plus, minus = q.copy(), q.copy()
                    plus[index] += h
                    minus[index] -= h
                    expected_force[index] = -(energy(plus)-energy(minus))/(2*h)
                # Differentiate all nine strains before checking symmetry.
                for a in range(3):
                    for b in range(3):
                        plus, minus = q.copy(), q.copy()
                        plus[:, a] += h*q[:, b]
                        minus[:, a] -= h*q[:, b]
                        tensor[a, b] = -(energy(plus)-energy(minus))/(2*h)
                np.testing.assert_allclose(tensor, tensor.T, atol=1e-8)
                np.testing.assert_allclose(tensor, expected_force.T@q, atol=1e-8)
                rows = [f'0 1 2 3 {i+1} {c:.17g} {np.pi if i==1 else 0:.17g}'
                        for i,c in enumerate(coefficients)]
                result = self.run_case(masses=[15.,14.,14.,15.], velocities=np.zeros((4,3)),
                    positions=(q+shift)%32,
                    options={'dt':0, 'step_limit':1, 'dihedral_in_file':'dihedral.txt'},
                    files={'dihedral.txt':'3\n'+'\n'.join(rows)+'\n'})
                force = np.frombuffer(result['frc'],np.float32).reshape(-1,4,3)[0]
                np.testing.assert_allclose(force, expected_force, rtol=3e-5, atol=3e-5)
                actual = np.array([result['values'][key][0] for key in runtime.NAMES])
                expected = np.array([tensor[a,b] for a,b in runtime.AXES])*runtime.PA/32**3
                self.assertLessEqual(np.max(abs(actual-expected)), 2+3e-5*np.max(abs(expected)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sponge',required=True)
    args,extra=parser.parse_known_args()
    runtime.SPONGE=str(Path(args.sponge).resolve())
    unittest.main(argv=[__file__,*extra])
