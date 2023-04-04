import numpy as np
from scipy.special import lpmv, eval_legendre
from healpy.rotator import rotateDirection, dir2vec

from bilby.core.prior import Prior, Uniform, Sine, Cosine, Interped, PriorDict

from .para_sampler import ParameterSampler
from .prior_dict import PriorDictSampler
from ..utils import sh_normal_coeff, rotation_matrix_from_vec

from typing import Tuple
from ..typing import SHModes, ParameterVector, ParaNameList


def uniform_sky_sampler(ra_dec: bool = True) -> PriorDictSampler:
    if ra_dec:
        priors = {
            'ra': Uniform(name='ra', minimum=0, maximum=2*np.pi, boundary='periodic'),
            'dec': Cosine(name='dec')
        }
    else:
        priors = {
            'phi': Uniform(name='phi', minimum=0, maximum=2*np.pi, boundary='periodic'),
            'theta': Sine(name='theta')
        }

    priors = PriorDict(priors)
    return PriorDictSampler(priors)


class SHPhiPrior(Prior):
    def __init__(self, alms: SHModes) -> None:
        self.alms: SHModes = {}
        for (l, m), alm in alms.items():
            if m >= 0:
                self.alms[(l, m)] = alm
            else:
                al_m = -alm.conjugate()  # expect value of a_{l,-m}
                if alms.get((l, -m), al_m) != al_m:
                    raise ValueError('Wrong alm for real field')
                else:
                    self.alms[(l, -m)] = al_m
        # self.alms only contain m>0 mode
        # (for real field, corresponding -m mode is the minus conjugate)

        self._cos_theta = None
        self.cosm_coeff = {}
        self.sinm_coeff = {}

        super().__init__(minimum=0, maximum=2*np.pi)

    @property
    def cos_theta(self):
        return self._cos_theta

    @cos_theta.setter
    def cos_theta(self, value):
        self._cos_theta = value

        cosm_coeff = {}
        sinm_coeff = {}
        for (l, m), alm in self.alms.items():
            theta_item = sh_normal_coeff(l, m) * lpmv(m, l, self._cos_theta)
            # 2 for both m and -m mode
            cosm_coeff[m] = cosm_coeff.get(m, 0) + 2*alm.real*theta_item
            sinm_coeff[m] = sinm_coeff.get(m, 0) - 2*alm.imag*theta_item

        normalization = cosm_coeff[0] * 2*np.pi
        self.cosm_coeff = {
            m: value/normalization for m, value in cosm_coeff.items()}
        self.sinm_coeff = {
            m: value/normalization for m, value in sinm_coeff.items()}

    def prob(self, val):
        cos = sum([coefficient*np.cos(m*val)
                   for m, coefficient in self.cosm_coeff.items()])
        sin = sum([coefficient*np.sin(m*val)
                   for m, coefficient in self.sinm_coeff.items()])
        return cos+sin

    def rescale(self, val):
        # an upper bound of max value of PDF
        max_bound = sum(map(np.abs, self.cosm_coeff.values())) + \
            sum(map(np.abs, self.sinm_coeff.values()))
        while True:
            phi = np.random.uniform(0, 2*np.pi)
            if val <= self.prob(phi) / max_bound:
                return phi
            val = np.random.random()


class SHSkySampler(ParameterSampler):
    def __init__(self, alms: SHModes, ra_dec: bool = True):
        UNIFORM_A00 = 2 * np.pi**0.5
        a00 = alms.get((0, 0), UNIFORM_A00)
        self.alms = {lm: value/a00 * UNIFORM_A00
                     for lm, value in alms.items()}  # normalize
        self.alms[(0, 0)] = UNIFORM_A00

        self.cos_theta_prior = self.get_cos_theta_prior(self.alms)
        self.phi_prior, self.conditional_phi = self.get_phi_prior(self.alms)

        self.ra_dec = ra_dec

    def _check_positive_prior(self, prior):
        if np.any(prior < 0):
            raise ValueError(
                'Wrong sperical harmonic coefficients value, '
                'summed prior function should be larger than zero')

    def get_cos_theta_prior(self, alms: SHModes, interp_len: int = 1000) -> Prior:
        cos_theta = np.linspace(-1, 1, interp_len)
        prior = np.zeros(cos_theta.shape)
        for (l, m), alm in alms.items():
            # distribution of theta is Y(theta,phi) marginalized by phi
            if m == 0:
                prior += alm*sh_normal_coeff(l, m)*eval_legendre(l, cos_theta)
        self._check_positive_prior(prior)
        return Interped(xx=cos_theta, yy=prior, minimum=-1, maximum=1)

    def get_phi_prior(self, alms: SHModes) -> Tuple[Prior, bool]:
        if all(map(lambda lm: lm[1]==0, alms.keys())):
            # alms does not contain m!=0 item, phi is uniformly distributed.
            # faster to compute
            return Uniform(
                name='phi', minimum=0, maximum=2*np.pi, boundary='periodic'), False
        else:
            return SHPhiPrior(alms), True

    @property
    def keys(self) -> ParaNameList:
        if self.ra_dec:
            return ['ra', 'dec']
        else:
            return ['theta', 'phi']

    def sample_theta_phi(self) -> Tuple[float, float]:
        cos_theta = self.cos_theta_prior.sample()
        if self.conditional_phi:
            self.phi_prior.cos_theta = cos_theta
        phi = self.phi_prior.sample()
        return np.arccos(cos_theta), phi

    def theta_phi_to_para(self, theta, phi) -> ParameterVector:
        if self.ra_dec:
            return [phi, np.pi/2-theta]
        else:
            return [theta, phi]

    def sample(self) -> ParameterVector:
        theta, phi = self.sample_theta_phi()
        return self.theta_phi_to_para(theta, phi)


class DipoleSkySampler(SHSkySampler):
    def __init__(self, amplitude: float, dipole_theta: float, dipole_phi: float, ra_dec: bool = True) -> None:
        self.amplitude = amplitude
        self.dipole_theta = dipole_theta
        self.dipole_phi = dipole_phi

        if amplitude < 0 or amplitude > 1:
            raise ValueError('Dipole amplitude should be >=0 and <=1.')

        self.rot_mat = rotation_matrix_from_vec(
            orig_vec=np.array([0, 0, 1]),
            dest_vec=dir2vec(self.dipole_theta, self.dipole_phi))

        alms = {(1, 0): amplitude/sh_normal_coeff(1, 0)}
        super().__init__(alms=alms, ra_dec=ra_dec)

    def sample_theta_phi(self) -> Tuple[float, float]:
        theta, phi = super().sample_theta_phi()
        r_theta, r_phi = rotateDirection(self.rot_mat, theta, phi)
        return r_theta, r_phi
