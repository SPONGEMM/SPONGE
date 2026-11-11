"""Serial benchmark extension for LAMMPS python/move, not a native LAMMPS fix.

Uses a mass-weighted QR projection and independent NumPy PCG64 noise. Forces
remain LAMMPS's native forces. Only one run, atomic positive masses, a fixed
orthogonal periodic box and axes x/z are supported. No restart is implemented.
"""
import json
from pathlib import Path

import numpy as np


def project(mass, positions, velocity, length):
    """Project onto three translations and cos(2 pi z/L) in mass coordinates."""
    mass = np.asarray(mass, dtype=float)
    positions, velocity = np.asarray(positions), np.asarray(velocity)
    n = len(mass)
    if n < 2 or positions.shape != (n, 3) or velocity.shape != (n, 3):
        raise ValueError('invalid projection shapes')
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0) or not np.isfinite(length) or length <= 0 or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocity)):
        raise ValueError('invalid projection values')
    cosine = np.cos(2*np.pi*positions[:, 2]/length)
    mean = np.average(cosine, weights=mass)
    if np.average((cosine-mean)**2, weights=mass) <= 1e-12:
        raise ValueError('rank deficient cosine projection')
    basis = np.zeros((3*n, 4))
    for axis in range(3): basis[axis::3, axis] = 1
    basis[::3, 3] = cosine
    sqrt_mass = np.repeat(np.sqrt(mass), 3)
    orthogonal, _ = np.linalg.qr(sqrt_mass[:, None]*basis, mode='reduced')
    weighted = sqrt_mass*velocity.ravel()
    projected = (orthogonal@(orthogonal.T@weighted)/sqrt_mass).reshape(n, 3)
    return projected, orthogonal, sqrt_mass


def ou_step(mass, positions, velocity, length, damping, kt_velocity, normal):
    """kt_velocity = kBT / mvv2e; velocities and masses use LAMMPS units."""
    if not np.isfinite(damping) or not 0 <= damping <= 1 or not np.isfinite(kt_velocity) or kt_velocity < 0:
        raise ValueError('invalid OU parameters')
    normal = np.asarray(normal, dtype=float)
    if normal.shape != np.shape(velocity) or not np.all(np.isfinite(normal)):
        raise ValueError('invalid Maxwell noise')
    bias, orthogonal, sqrt_mass = project(mass, positions, velocity, length)
    noise = normal.ravel()
    thermal_noise = ((noise-orthogonal@(orthogonal.T@noise))/sqrt_mass).reshape(normal.shape)
    return bias+damping*(velocity-bias)+np.sqrt((1-damping*damping)*kt_velocity)*thermal_noise


def observe(mass, positions, velocity, box, bins, mvv2e, boltz):
    """Canonical profile from integer-time LAMMPS velocities (Angstrom/fs)."""
    bias, _, _ = project(mass, positions, velocity, box)
    mass = np.asarray(mass); total = mass.sum()
    phase = 2*np.pi*positions[:, 2]/box
    cosine, sine = np.cos(phase), np.sin(phase)
    mean = np.average(cosine, weights=mass)
    variance = np.average((cosine-mean)**2, weights=mass)
    com = np.average(velocity, axis=0, weights=mass)
    # Independent least-squares coefficient, rather than the engine formula.
    design = np.column_stack((np.ones(len(mass)), cosine))
    coef = np.linalg.lstsq(design*np.sqrt(mass[:, None]), velocity[:, 0]*np.sqrt(mass), rcond=None)[0][1]
    fourier = 2*np.sum(mass*velocity[:, 0]*cosine)/total
    record = dict(velocity_fourier=fourier*100,
        velocity_com_fourier=(fourier-2*mean*com[0])*100, velocity_fit=coef*100,
        mean_cosine=mean, variance_cosine=variance,
        thermal_temperature=np.sum(mass[:, None]*(velocity-bias)**2)*mvv2e/((3*len(mass)-4)*boltz),
        density=total*1.66053906660/box**3, length=box/10, wave_number=20*np.pi/box)
    record.update({f'com_v{axis}': com[i]*100 for i, axis in enumerate('xyz')})
    index = np.minimum((np.mod(positions[:, 2]/box, 1)*bins).astype(int), bins-1)
    record['bin_count'] = np.bincount(index, minlength=bins)
    for key, weights in dict(mass=mass, momentum=mass*velocity[:, 0]*100,
                             cosine_mass=mass*cosine, sine_mass=mass*sine).items():
        record['bin_'+key] = np.bincount(index, weights=weights, minlength=bins)
    return record


class ProjectedMiddle:
    """B/2 A/2 O_projected A/2 B/2 via native LAMMPS integration hooks."""
    def __init__(self, ptr):
        from lammps import lammps
        self.lmp = lammps(ptr=ptr)
        self.config = json.loads(Path('projected_middle.json').read_text())
        self.rng = np.random.default_rng(self.config['seed'])
        self.records = []
        self.completed = 0
        self.work = self.heat = 0.
        self.initialized = False

    def arrays(self):
        n = int(self.lmp.extract_global('nlocal'))
        if n != self.config['atoms']:
            raise ValueError('reference requires one rank, no atom creation/deletion')
        atom = self.lmp.numpy.extract_atom
        mass = atom('mass')[atom('type')[:n]]
        if np.any(mass != self.config['mass_da']):
            raise ValueError('reference mass changed')
        return mass, atom('x')[:n], atom('v')[:n], atom('f')[:n], atom('id')[:n]

    def init(self):
        if self.initialized or self.lmp.extract_global('ntimestep') != 0:
            raise ValueError('reference requires a single new run beginning at step zero')
        self.initialized = True
        if self.lmp.extract_setting('world_size') != 1:
            raise ValueError('reference supports one MPI rank only')
        self.dt = float(self.lmp.extract_global('dt'))
        self.ftm2v = float(self.lmp.extract_global('ftm2v'))
        self.mvv2e = float(self.lmp.extract_global('mvv2e'))
        self.boltz = float(self.lmp.extract_global('boltz'))
        lo, hi, xy, yz, xz, periodic, _ = self.lmp.extract_box()
        if any(value != 0 for value in (*lo, xy, yz, xz)) or list(periodic) != [1, 1, 1] or not np.allclose(hi, self.config['box_angstrom'], rtol=0, atol=1e-12):
            raise ValueError('reference requires fixed cubic zero-origin periodic geometry')
        if not np.isclose(self.dt, self.config['dt_fs'], rtol=1e-14, atol=0):
            raise ValueError('reference timestep mismatch')
        self.sample()

    def initial_integrate(self, vflag):
        mass, x, v, f, ids = self.arrays()
        v += .5*self.dt*self.ftm2v*f/mass[:, None]
        self.old_cosine = np.cos(2*np.pi*x[:, 2]/self.config['box_angstrom'])
        self.displacement = .5*self.dt*v.copy()
        x += self.displacement
        before = v.copy()
        # Stable atom-ID ordering makes random assignment independent of sorting.
        order = np.argsort(ids)
        normal = np.empty(v.shape)
        normal[order] = self.rng.standard_normal(v.shape)
        v[:] = ou_step(mass, x, v, self.config['box_angstrom'], self.config['damping'],
                       self.config['temperature_k']*self.boltz/self.mvv2e, normal)
        self.heat += .5*self.mvv2e*np.sum(mass[:, None]*(v*v-before*before))
        second = .5*self.dt*v
        x += second
        self.displacement += second
        # At the next hook atoms may have been reordered. Store by global ID.
        self.previous = {int(i): (float(c), float(dx)) for i, c, dx in zip(ids, self.old_cosine, self.displacement[:, 0])}

    def final_integrate(self):
        mass, x, v, f, ids = self.arrays()
        v += .5*self.dt*self.ftm2v*f/mass[:, None]
        old = np.array([self.previous[int(i)] for i in ids])
        cosine = np.cos(2*np.pi*x[:, 2]/self.config['box_angstrom'])
        self.work += .5*self.config['acceleration_angstrom_fs2']*self.mvv2e*np.sum(mass*(old[:, 0]+cosine)*old[:, 1])
        self.completed += 1
        if self.completed % self.config['sample_interval'] == 0: self.sample()
        if self.completed == self.config['steps']:
            arrays = {key: np.asarray([row[key] for row in self.records]) for key in self.records[0]}
            np.savez('canonical-profile.npz', **arrays)
            Path('reference-completion.json').write_text(json.dumps(dict(status='finalized', steps=self.completed,
                frames=len(self.records), work_kcal_mol=self.work, heat_kcal_mol=self.heat,
                numpy_version=np.__version__, rng='PCG64; independent of SPONGE Philox',
                implementation='custom python/move projected Middle; native LAMMPS forces'), indent=2)+'\n')

    def sample(self):
        mass, x, v, _, _ = self.arrays()
        record = observe(mass, x, v, self.config['box_angstrom'], self.config['bins'], self.mvv2e, self.boltz)
        record.update(step=self.completed, time=self.completed*self.dt/1000,
            acceleration=self.config['acceleration_nm_ps2'], coupling_work=self.work, thermostat_heat=self.heat)
        self.records.append(record)

    def reset_dt(self):
        raise ValueError('reference timestep cannot change')

    def initial_integrate_respa(self, *args):
        raise ValueError('reference does not support RESPA')

    def final_integrate_respa(self, *args):
        raise ValueError('reference does not support RESPA')
