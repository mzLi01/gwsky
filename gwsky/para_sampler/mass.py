import numpy as np
from bilby.core.prior import Prior, Interped

from .para_sampler import ParameterSampler

from typing import Optional
from ..typing import ParaNameList, ParameterVector


class PowerLawPeakMass(Interped):
    '''
    Power Law + Peak model in 2010.14533
    '''

    def __init__(self, m_min, m_max, delta_m, alpha, mu_m, sigma_m, lambda_peak):
        self.m_min = m_min
        self.m_max = m_max
        self.delta_m = delta_m
        self.alpha = alpha
        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.lambda_peak = lambda_peak

        m = np.linspace(m_min, m_max, 1000)

        power_law = self.smooth(m**(-self.alpha), m)
        gaussian = self.smooth(
            np.exp(-(m-self.mu_m)**2 / self.sigma_m**2 / 2), m)
        yy = (1-self.lambda_peak) * self.normalize(power_law, m) + \
            self.lambda_peak * self.normalize(gaussian, m)

        super().__init__(xx=m, yy=yy, minimum=m_min, maximum=m_max, name='mass_1')

    def smooth(self, y, m):
        smooth_i = m < self.m_min+self.delta_m
        mprime = m[smooth_i] - self.m_min
        smooth = 1 / (np.exp(self.delta_m/mprime +
                             self.delta_m/(mprime-self.delta_m)) + 1)
        y[smooth_i] *= smooth
        return y

    def normalize(self, y, x):
        return y / np.trapz(y, x)


class PowerLawPeakMassRatio(Prior):
    def __init__(self, beta_q, m_min, delta_m):
        self.beta_q = beta_q
        self.m_min = m_min
        self.delta_m = delta_m
        self._mass_1 = None
        super().__init__(minimum=0, maximum=1)

    @property
    def mass_1(self):
        return self._mass_1
    
    @mass_1.setter
    def mass_1(self, m:float):
        self._mass_1=m

    def smooth_function(self, m):
        if m < self.m_min:
            return 0
        elif m < self.m_min+self.delta_m:
            mprime = m - self.m_min
            return 1 / (np.exp(self.delta_m/mprime + self.delta_m/(mprime-self.delta_m)) + 1)
        else:
            return 1

    def rescale(self, val):
        # 采用马文淦《计算物理学》P20中介绍的第二类舍选法
        # h(x)为指数分布，g(x)为smooth函数（归一化常数被吸收到L中）
        while True:
            q_eta = val**(1/(self.beta_q+1))
            if np.random.rand() <= self.smooth_function(q_eta*self.mass_1):
                return q_eta
            val = np.random.rand()


class O3aMassSampler(ParameterSampler):
    M1_PRIOR_CONFIG = dict(
        m_min=4.59, m_max=86.22, delta_m=4.82, alpha=2.63,
        mu_m=33.07, sigma_m=5.69, lambda_peak=0.1)
    Q_PRIOR_CONFIG = dict(beta_q=1.26, m_min=4.59, delta_m=4.82)

    def __init__(self,
                 m1_prior: Optional[PowerLawPeakMass] = None,
                 q_prior: Optional[PowerLawPeakMassRatio] = None) -> None:
        if m1_prior is None:
            m1_prior = PowerLawPeakMass(**self.M1_PRIOR_CONFIG)
        if q_prior is None:
            q_prior = PowerLawPeakMassRatio(**self.Q_PRIOR_CONFIG)
        self.m1_prior = m1_prior
        self.q_prior = q_prior

    @property
    def keys(self) -> ParaNameList:
        return ['mass_1', 'mass_2']
    
    def sample(self) -> ParameterVector:
        mass_1 = self.m1_prior.sample()

        self.q_prior.mass_1 = mass_1
        mass_ratio = self.q_prior.sample()
        mass_2 = mass_1 * mass_ratio
        return [mass_1, mass_2]
