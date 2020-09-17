# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, Optional

import os
import torch
import torch.multiprocessing as mp
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC as MCMCAPI
from .pyro_slice_kernel import Slice
from .base_mcmc import MCMC
from warnings import warn


class PyroMCMC(MCMC):
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
        kernel: str = "slice",
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

        self.track_gradients = kernel != "slice"

        kernels = dict(slice=Slice, hmc=HMC, nuts=NUTS)
        self.kernel = kernels[kernel]

        os.cpu_count = lambda: available_cpus

    def run(self, num_samples):
        with torch.set_grad_enabled(self.track_gradients):

            initial_params = self.init_fn(self.site_name)

            sampler = MCMCAPI(
                kernel=self.kernel(potential_fn=self.potential_fn),
                num_samples=(self.thin * num_samples) // self.num_chains
                + self.num_chains,
                warmup_steps=self.warmup * self.thin,
                initial_params={self.site_name: initial_params},
                num_chains=self.num_chains,
                mp_context="fork",
                disable_progbar=not self.verbose,
            )
            sampler.run()

            samples = sampler.get_samples()

            # TODO: thinning
            # TODO: detach

            # assert samples.shape[0] == num_samples

            return samples
