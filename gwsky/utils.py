import numpy as np
import healpy as hp
from scipy.special import sph_harm
from functools import reduce

import quaternionic
import spherical

import matplotlib.pyplot as plt
from healpy.projaxes import HpxMollweideAxes

from typing import Tuple, Optional, List, Dict
from .typing import SHModes, Value


def ra_dec_to_theta_phi(ra: Value, dec: Value) -> Tuple[Value, Value]:
    return np.pi/2-dec, ra


def theta_phi_to_ra_dec(theta: Value, phi: Value) -> Tuple[Value, Value]:
    return phi, np.pi/2-theta


def catalog_delta_map(theta: np.ndarray, phi: np.ndarray,
                      nside: int = 64, ra_dec: bool = False) -> np.ndarray:
    if ra_dec:
        theta, phi = ra_dec_to_theta_phi(theta, phi)

    hp_map = np.zeros(hp.nside2npix(nside))
    points_ipix = hp.ang2pix(nside=nside, theta=theta, phi=phi)
    ipix, counts = np.unique(points_ipix, return_counts=True)
    hp_map[ipix] += counts

    map_mean = theta.shape[0] / hp.nside2npix(nside)
    return hp_map/map_mean - 1


def spherical_harmonic_modes(theta: np.ndarray, phi: np.ndarray, l: int, m: int,
                             weights: Optional[np.ndarray] = None, ra_dec: bool = False) -> complex:
    if ra_dec:
        theta, phi = ra_dec_to_theta_phi(theta, phi)

    if weights is None:
        weights = np.ones(theta.shape)

    normalization = theta.shape[0] / (4*np.pi)
    # sph_harm(m, n, theta, phi)
    # m,n: harmonic mode, |m|<=n
    # theta, phi: spherical coordinate, 0<theta<2*pi, 0<phi<pi
    coefficient = np.sum(sph_harm(m, l, phi, theta) * weights).conjugate() / normalization
    return coefficient


def sh_normal_coeff(l, m):
    fact_item = reduce(
        lambda x, y: x*y, range(l-np.abs(m)+1, l+np.abs(m)+1), 1)
    return ((2*l+1)/(4*np.pi) / fact_item)**0.5


def rotation_matrix_from_vec(orig_vec, dest_vec) -> np.ndarray:
    # see https://math.stackexchange.com/a/476311
    v = np.cross(orig_vec, dest_vec)
    c = np.inner(orig_vec, dest_vec)

    v_cross = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]])
    rot = np.eye(3) + v_cross + np.matmul(v_cross, v_cross)/(1+c)
    return rot


def dipole_modes(amplitude: float, theta: float, phi: float) -> SHModes:
    a10 = amplitude / sh_normal_coeff(1, 0)
    dipole_mode = spherical.Modes(
        np.array([0, 0, a10, 0], dtype=complex),
        spin_weight=0)

    rot_mat = rotation_matrix_from_vec(
        orig_vec=hp.dir2vec(theta, phi),
        dest_vec=np.array([0, 0, 1]))  # 将Y_{10}转到给定(theta, phi)方向的偶极场的旋转矩阵的逆
    rotation = quaternionic.array.from_rotation_matrix(rot_mat)
    wigner = spherical.Wigner(ell_max=2)
    # wigner.rotate返回的modes对应的是坐标旋转的逆
    # 即对于一个球谐系数为a_lm的场f，用坐标旋转RM作用之，得到的新场f'(r)=f(R^{-1} r)
    # 则f'的球谐系数为`wigner.rotate(modes=a_lm, R=1/R)`
    rot_mode_sph: spherical.Modes = wigner.rotate(modes=dipole_mode, R=rotation)
    rot_mode = {(1, m): rot_mode_sph[spherical.LM_index(1, m)] for m in range(-1, 2)}
    return rot_mode


def plot_hp_map(hp_map: np.ndarray, detectors: Optional[List[Dict]] = None,
                fig: Optional[plt.Figure] = None, label: str = '',
                grid_on: bool = True, grid_kwargs: Optional[Dict] = None,
                detector_kwargs: Optional[Dict] = None, **kwargs):
    plot_kwargs = {'flip': 'geo'}
    plot_kwargs.update(kwargs)

    hp.mollview(hp_map, fig=fig, **plot_kwargs)
    fig = plt.gcf()
    skymap_ax: HpxMollweideAxes = fig.get_axes()[0]

    det_kwargs = {'color': 'orange', 'markersize': 10}
    if detector_kwargs is not None:
        det_kwargs.update(detector_kwargs)
    for detector in detectors:
        skymap_ax.projplot(
            detector['longitude'], detector['latitude'], lonlat=True,
            marker=detector['marker'], **det_kwargs)

    if grid_on:
        grid_kwargs_real = {'dpar': 30, 'dmer': 30}
        grid_kwargs_real.update(grid_kwargs)
        skymap_ax.graticule(**grid_kwargs_real)

    cb_ax: plt.Axes = fig.get_axes()[1]
    cb_ax.set_xlabel(label)

    return fig