# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import numpy as np
import os
import sys

from tqdm import trange
import torch
import zeus  # TODO: into setup

from .base_mcmc import MCMC
from typing import Any, Callable, Dict, Optional
from warnings import warn
import multiprocessing as mp


class ZeusSliceEnsembleMCMC(MCMC):
    def __init__(
        self,
        potential_fn: Callable,
        init_fn: Callable,
        thin: int = 10,
        warmup: int = 50,
        num_chains: int = 1,
        transforms: Optional[Dict[str, Any]] = None,
        vectorize: bool = True,
        available_cpus: int = max(mp.cpu_count() - 1, 1),
        verbose: bool = True,
        site_name: str = "theta",
        num_walkers: int = 100,
    ):
        super().__init__(
            potential_fn,
            init_fn,
            thin,
            warmup,
            num_chains,
            transforms,
            vectorize,
            available_cpus,
            verbose,
        )
        if self.vectorize is True:
            warn("vectorization not supported")

        if self.transforms is not None:
            raise NotImplementedError

        if self.available_cpus > 1:
            warn("will only use single CPU")

        self.site_name = site_name

        self.num_walkers = num_walkers

    def run(self, num_samples: int) -> Dict[str, torch.Tensor]:
        assert num_samples >= 0

        # TODO: start through init_fn
        # TODO: ndim through init_fn
        # TODO: move thin into sampler arg (remove elsewhere)
        ndim = 2
        start = 1.0 * np.random.randn(self.num_walkers, ndim)
        nsteps = num_samples * self.thin + self.warmup * self.thin

        vec = self.vectorize
        vec = 1
        sampler = zeus.sampler(self.num_walkers, ndim, self._log_prob_fn, vectorize=vec)
        samples = sampler.run_mcmc(start, nsteps)

        samples = sampler.get_chain()  # discard = warmup

        # Init chain
        # if self.x is None:
        #    self.x = self._get_init_params()
        #    self.n_dims = self.x.size  # TODO: double check

        return {self.site_name: torch.from_numpy(samples.astype(np.float32))}

    def _reset(self):
        self.rng = np.random
        self.step = 0
        self.width = None
        self.x = None

    def _log_prob_fn(self, x: np.ndarray):
        return (
            -1.0
            * self.potential_fn(
                {self.site_name: torch.from_numpy(x.astype(np.float32))}
            ).numpy()
        ).reshape(-1)

    def _get_init_params(self):
        return self.init_fn(self.site_name).numpy()
