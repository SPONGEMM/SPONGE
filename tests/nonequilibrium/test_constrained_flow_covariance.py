"""Independent mass-whitened SVD reference for the actual C++ flow OU core."""
import argparse
import os
import subprocess
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
try:
    import numpy as np
except ImportError:
    sys.exit(77)


def check(probe, mass, q, pairs, samples=0, axis=0, box=(20., 20., 20.)):
    n = len(mass)
    dim = 3*n
    box = np.array(box)
    root_mass = np.repeat(np.sqrt(mass), 3)
    gradient = (axis+1) % 3
    cosine = np.cos(2*np.pi*q[:, gradient]/box[gradient])
    # Independent dense SVD of the whitened constraint Jacobian, never the
    # C++ component Gram/Cholesky implementation.
    jac = np.zeros((len(pairs), dim))
    for row, (a, b, length) in enumerate(pairs):
        d = q[a]-q[b]
        d -= box*np.floor(d/box+.5)
        np.testing.assert_allclose(np.linalg.norm(d), length, atol=1e-12)
        jac[row, 3*a:3*a+3] = d/length
        jac[row, 3*b:3*b+3] = -d/length
    jac /= root_mass
    if len(pairs):
        _, singular, vt = np.linalg.svd(jac, full_matrices=True)
        assert np.min(singular) > 1e-6
        tangent = vt[len(pairs):].T
        pc = tangent @ tangent.T
    else:
        pc = np.eye(dim)
    modes = np.zeros((dim, 4))
    for i in range(n):
        modes[3*i:3*i+3, :3] = np.eye(3)*np.sqrt(mass[i])
        modes[3*i+axis, 3] = np.sqrt(mass[i])*cosine[i]
    modes = pc @ modes
    u, singular, _ = np.linalg.svd(modes, full_matrices=False)
    assert min(singular) > 1e-6
    kept = u @ u.T
    ph = pc-kept
    dof = dim-len(pairs)-4
    rng = np.random.default_rng(52873)
    velocity, noise = rng.normal(size=(2, n, 3))
    damping, kbt = .63, 2.3
    lines = [f'{n} {len(pairs)} {axis} {samples} {damping} {kbt}']
    for array in (mass, q, velocity, noise, box, cosine):
        lines.append(' '.join(format(x, '.17g') for x in np.ravel(array)))
    lines.extend(f'{a} {b} {length:.17g}' for a, b, length in pairs)
    run = subprocess.run([probe, '--stream'], input='\n'.join(lines)+'\n',
                         text=True, capture_output=True, timeout=50, check=True)
    data = np.fromstring(run.stdout, sep=' ')
    assert len(data) == 2+4*dim+2*dim*dim+3*dim+3*dim*dim+2
    assert data[0] == dof
    vectors = data[2:2+4*dim].reshape(4, dim)*root_mass
    w = velocity.ravel()*root_mass
    expected = [pc@w, kept@w, ph@w,
                kept@w+damping*ph@w+np.sqrt(kbt*(1-damping**2))*ph@noise.ravel()]
    np.testing.assert_allclose(vectors, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(data[1], .5*(expected[3]@expected[3]-expected[0]@expected[0]),
                               rtol=2e-10, atol=2e-10)
    start = 2+4*dim
    matrices = data[start:start+2*dim*dim].reshape(dim, 2, dim)
    observed_pc, observed_ph = matrices[:, 0].T, matrices[:, 1].T
    for observed, reference in ((observed_pc, pc), (observed_ph, ph)):
        np.testing.assert_allclose(observed, reference, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(observed.T, observed, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(observed@observed, observed, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(jac@observed, 0, atol=2e-10)
    np.testing.assert_allclose(observed_ph@modes, 0, atol=2e-10)
    np.testing.assert_allclose(np.trace(observed_ph), dof, atol=2e-10)
    np.testing.assert_allclose(kept@expected[3], kept@expected[0], atol=2e-10)
    before = vectors[0].reshape(n, 3)/np.sqrt(mass)[:, None]
    after = vectors[3].reshape(n, 3)/np.sqrt(mass)[:, None]
    np.testing.assert_allclose(mass@after, mass@before, atol=2e-10)
    np.testing.assert_allclose((mass*cosine)@after[:, axis],
                               (mass*cosine)@before[:, axis], atol=2e-10)
    if samples:
        start += 2*dim*dim
        means = data[start:start+3*dim].reshape(3, dim)/samples
        start += 3*dim
        moments = data[start:start+3*dim*dim].reshape(3, dim, dim)/samples
        mean_fraction = covariance_fraction = 0.
        for mean, moment, cov in zip(means, moments, [kbt*ph, kbt*(1-damping**2)*ph, kbt*pc]):
            diagonal = np.maximum(np.diag(cov), 0)
            mean_bound = 6*np.sqrt(diagonal/samples)+2e-10
            # Raw second moments, with the population mean known to be zero.
            covariance_bound = 6*np.sqrt((np.outer(diagonal, diagonal)+cov**2)/samples)+2e-10
            assert np.all(np.abs(mean) <= mean_bound), 'full Gaussian mean failed'
            assert np.all(np.abs(moment-cov) <= covariance_bound), 'full covariance failed'
            mean_fraction = max(mean_fraction, float(np.max(np.abs(mean)/mean_bound)))
            covariance_fraction = max(covariance_fraction, float(np.max(np.abs(moment-cov)/covariance_bound)))
        kmean, ksecond = data[-2:]/samples
        assert abs(kmean-dof) <= 6*np.sqrt(2*dof/samples)
        # Chi-square variance estimator asymptotic SE: sqrt((mu4-var²)/N).
        assert abs(ksecond-kmean*kmean-2*dof) <= 6*np.sqrt((8*dof*dof+48*dof)/samples)
        print(f'Gaussian n={n} constraints={len(pairs)} draws={samples} dof={dof}: '
              f'max_mean_gate_fraction={mean_fraction:.8g} max_covariance_gate_fraction={covariance_fraction:.8g} '
              f'2K/kBT_mean={kmean:.8g} variance={ksecond-kmean*kmean:.8g}')
    print(f'Deterministic axis={axis} constraints={len(pairs)}: '
          f'Pc_max_error={np.max(np.abs(observed_pc-pc)):.8g} '
          f'Ph_max_error={np.max(np.abs(observed_ph-ph)):.8g}')
    return observed_pc, observed_ph


def main(probe):
    # Two disconnected, unequal-mass rigid triangles with interleaved IDs.
    mass = np.array([16., 16., 1., 1., 1., 1.])
    q = np.array([[1.,1.,1.], [6.,6.,6.], [1.6,1.8,1.],
                  [6.6,6.8,6.], [.4,1.8,1.], [5.4,6.8,6.]])
    pairs = [(0,2,1.), (0,4,1.), (2,4,1.2), (1,3,1.), (1,5,1.), (3,5,1.2)]
    pc, ph = check(probe, mass, q, pairs, 60000)
    wrapped = q.copy()
    wrapped[2] += [20., -20., 40.]
    wrapped[5] += [-40., 20., 20.]
    pc2, ph2 = check(probe, mass, wrapped, [(b,a,l) for a,b,l in reversed(pairs)])
    np.testing.assert_allclose(pc2, pc, atol=2e-10)
    np.testing.assert_allclose(ph2, ph, atol=2e-10)
    permutation = np.array([3,0,5,2,1,4])
    inverse = np.argsort(permutation)
    pc3, ph3 = check(probe, mass[permutation], q[permutation],
                     [(int(inverse[a]),int(inverse[b]),l) for a,b,l in pairs])
    indices = np.array([3*i+j for i in permutation for j in range(3)])
    np.testing.assert_allclose(pc3, pc[np.ix_(indices,indices)], atol=2e-10)
    np.testing.assert_allclose(ph3, ph[np.ix_(indices,indices)], atol=2e-10)
    check(probe, mass, q, [], 60000)
    # Oblique geometry and both other flow axes; full projector/OU identities
    # are deterministic checks, without multiplying the stochastic test count.
    direction = np.array([1.,2.,3.])/np.sqrt(14.)
    x,y,z = direction
    cross = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    rotation = np.cos(.47)*np.eye(3)+(1-np.cos(.47))*np.outer(direction,direction)+np.sin(.47)*cross
    for axis in (1,2):
        check(probe, mass, q@rotation.T, pairs, axis=axis, box=(17.,23.,29.))
    print('Independent SVD, periodic/permutation, full Gaussian covariance and OU invariance passed; 120000 draws')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', required=True)
    main(parser.parse_args().probe)
