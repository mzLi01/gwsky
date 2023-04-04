
import argparse
from datetime import datetime

import numpy as np

from bilby.gw import WaveformGenerator
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector import InterferometerList, load_interferometer
from bilby.core.prior import PriorDict, Uniform, Sine
from bilby.gw.prior import UniformSourceFrame
from astropy.cosmology import Planck15

from gwsky import BilbyEvaluatorRotated, Sampler, \
    O3aMassSampler, uniform_sky_sampler, PriorDictSampler


parser = argparse.ArgumentParser()
parser.add_argument('--nsamples', type=float, help='number of samples')  # float: handle inputs like --nsamples 1e6
parser.add_argument('--nprocess', type=int, help='number of process used')
parser.add_argument('--resultpath', type=str, help='result file pathname')
parser.add_argument('--catalogpath', type=str, help='catalog file pathname')
args = parser.parse_args()

nsamples: int = int(args.nsamples)
nprocess: int = args.nprocess
result_path: str = args.resultpath
catalog_path : str = args.catalogpath
if not result_path.endswith('.csv'):
    result_path += '.csv'
if not catalog_path.endswith('.csv'):
    catalog_path += '.csv'


waveform_generator = WaveformGenerator(
    duration=4, sampling_frequency=2048,
    frequency_domain_source_model=lal_binary_black_hole,
    waveform_arguments={
        'waveform_approximant': 'TaylorF2',
        'minimum_frequency': 1e-2, 'reference_frequency': 0})

network : InterferometerList = load_interferometer('ifo/ET.ifo')
ifo_CE = load_interferometer('ifo/CE.ifo')
network.append(ifo_CE)

evaluator = BilbyEvaluatorRotated(
    network=network, waveform_generator=waveform_generator,
    fisher_parameters=['mass_1', 'mass_2', 'luminosity_distance', 'geocent_time', 'phase', 'theta_jn', 'ra', 'dec', 'psi'])

default_parameters = dict(
    phase=0, psi=np.pi/6,
    a_1=0, a_2=0, tilt_1=0, tilt_2=0, phi_12=0, phi_jl=0)

z_min, z_max = 0.1, 2
priors = {
    'luminosity_distance': UniformSourceFrame(
        name='luminosity_distance',
        minimum=Planck15.luminosity_distance(z_min).value,
        maximum=Planck15.luminosity_distance(z_max).value),
    'theta_jn': Sine(name='iota'),
    'geocent_time': Uniform(minimum=0, maximum=24*3600, name='geocent_time')
}
priors = PriorDict(priors)

para_samplers = [
    O3aMassSampler(), uniform_sky_sampler(), PriorDictSampler(priors)]

sampler = Sampler(
    para_samplers=para_samplers, default_parameters=default_parameters)
result = sampler.sample_fisher(
    nsamples=nsamples, evaluator=evaluator,
    result_path=result_path, nprocess=nprocess)

start = datetime.now()
catalog = result.generate_catalog(
    converter_from_parameter=evaluator.converter_from_parameter)
end = datetime.now()
print(f'generate catalog used time: {end-start}')

catalog.save_csv(catalog_path)
