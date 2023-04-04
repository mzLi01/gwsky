import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .utils import spherical_harmonic_modes, catalog_delta_map, plot_hp_map

from typing import Dict, List, Optional, Tuple, Iterable
from .typing import ParameterVector, ParaNameList, SHModes, SHModeLM


class GWCatalog:
    def __init__(self, events: Optional[pd.DataFrame] = None, keys: Optional[ParaNameList] = None) -> None:
        if events is None:
            assert keys is not None
        self.events = events
        self.keys = keys
        if self.events is not None:
            self._fix_dec()

    def _fix_dec(self):
        dec = self.events.loc[:, 'dec']
        self.events.loc[dec<-np.pi/2, 'dec'] = -np.pi/2
        self.events.loc[dec>np.pi/2, 'dec'] = np.pi/2

    def append_events(self, new_events: List[ParameterVector]) -> None:
        if self.events is None:
            self.events = pd.DataFrame(new_events, columns=self.keys)
        else:
            new_df = pd.DataFrame(new_events, columns=self.events.columns)
            self.events = pd.concat([self.events, new_df], ignore_index=True)
        self._fix_dec()

    def get_parameters(self, parameter: str, indexes: Optional[pd.Index] = None) -> np.ndarray:
        if indexes is None:
            indexes = self.events.index
        return self.events.loc[indexes, parameter].to_numpy()

    def get_snr(self, indexes: Optional[pd.Index] = None) -> np.ndarray:
        if indexes is None:
            indexes = self.events.index
        return self.events.loc[indexes, 'snr'].to_numpy()

    def get_source_position(self, snr_threshold: float = 0, other_cols: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, ...]:
        snr = self.get_snr()
        index = snr > snr_threshold

        if other_cols is None:
            other_cols = []
        paras = self.get_parameters(['ra', 'dec']+list(other_cols), index)
        return tuple(paras[:, i] for i in range(paras.shape[1]))

    def clear(self):
        self.events.drop(index=self.events.index, inplace=True)

    def spherical_harmonic_modes(self, lmax: Optional[int] = None, lmin: int = 1, mmax: Optional[int] = None, mmin: int = 0,
                                 lms: Optional[Iterable[SHModeLM]] = None, weights_col: Optional[str] = None, snr_threshold: float = 0) -> SHModes:
        if lms is None:
            if lmax is None:
                raise ValueError('lmax should be passed when not passing lms')
            if mmax is None:
                mmax = lmax + 1
            lms = [(l, m) for l in range(lmin, lmax+1) for m in range(mmin, min(l, mmax)+1)]

        if weights_col is None:
            ra, dec = self.get_source_position(snr_threshold)
            weights = None
        else:
            ra, dec, weights = self.get_source_position(snr_threshold, [weights_col])

        calc_lms = {(l, abs(m)) for l, m in lms}
        modes: SHModes = {spherical_harmonic_modes(
            ra, dec, l=l, m=m, weights=weights, ra_dec=True) for l, m in calc_lms}
        for l, m in lms:
            if m < 0:
                a_l_absm = modes[(l, -m)] if (l, -m) in lms else modes.pop((l, -m))
                modes[(l, m)] = -a_l_absm.conjugate()
        return modes

    def save_csv(self, path: str, append: bool = False):
        if append:
            self.events.to_csv(path, index=False, mode='a', header=False)
        else:
            self.events.to_csv(path, index=False)

    @classmethod
    def load_csv(cls, path: str):
        events = pd.read_csv(path, index_col=False)
        return cls(events=events)

    def plot_skymap(self, snr_threshold: float = 0, detectors: Optional[List[Dict]] = None,
                    nside=64, **kwargs) -> plt.Figure:
        ra, dec = self.get_source_position(snr_threshold)
        delta_map = catalog_delta_map(ra, dec, nside, ra_dec=True)
        fig = plot_hp_map(delta_map, detectors=detectors, **kwargs)
        return fig
