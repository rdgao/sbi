# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import numpy as np
import os
import sys

from tqdm import trange
import torch

from .base_mcmc import MCMC
from typing import Any, Callable, Dict, Optional
from warnings import warn
import multiprocessing as mp


class NumpySliceMCMC(MCMC):
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
        init_width: float = 0.01,
        max_width: float = float("inf"),
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

        self.init_width = init_width
        self.max_width = max_width
        self.site_name = site_name

        self.logger = open(os.devnull, "w") if not self.verbose else sys.stdout

        self._reset()

    def run(self, num_samples: int) -> Dict[str, torch.Tensor]:
        assert num_samples >= 0

        # TODO: Multiple chains
        # TODO: Parallelization?
        # TODO: Vectorisation?

        with torch.set_grad_enabled(False):

            # Init chain
            if self.x is None:
                self.x = self._get_init_params()
                self.n_dims = self.x.size  # TODO: double check

            order = list(range(self.n_dims))
            samples = np.empty([int(num_samples), int(self.n_dims)])

            if self.width is None:
                self._tune_bracket_width(self.warmup)

            if self.verbose:
                tbar = trange(int(num_samples), miniters=10)
                tbar.set_description("Generating samples")
            else:
                tbar = range(num_samples)

            for n in tbar:
                for _ in range(self.thin):
                    self.rng.shuffle(order)
                    for i in order:
                        self.x[i], _ = self._sample_from_conditional(i, self.x[i])

                samples[n] = self.x.copy()

                self.L = self._log_prob_fn(self.x)

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
        )[0]

    def _get_init_params(self):
        # NOTE: Sampler only works with single chain at a time, reshaping
        # init explicitly to -1
        return self.init_fn(self.site_name).numpy().reshape(-1)

    def _tune_bracket_width(self, num_samples):
        """
        Initial test run for tuning bracket width.
        Note that this is not correct sampling; samples are thrown away.
        :param rng: random number generator to use
        """
        self.logger.write("tuning bracket width...\n")

        order = list(range(self.n_dims))
        x = self.x.copy()
        self.width = np.full(self.n_dims, self.init_width)

        tbar = trange(num_samples, miniters=10)
        tbar.set_description("Tuning bracket width...")
        for n in tbar:
            self.rng.shuffle(order)

            for i in range(self.n_dims):
                x[i], wi = self._sample_from_conditional(i, x[i])
                self.width[i] += (wi - self.width[i]) / (n + 1)

    def _sample_from_conditional(self, i, cxi):
        """
        Samples uniformly from conditional by constructing a bracket.
        :param i: conditional to sample from
        :param cxi: current state of variable to sample
        :return: new state, final bracket width
        """

        # conditional log prob
        Li = lambda t: self._log_prob_fn(
            np.concatenate([self.x[:i], [t], self.x[i + 1 :]])
        )
        wi = self.width[i]

        # sample a slice uniformly
        logu = Li(cxi) + np.log(1.0 - self.rng.rand())

        # position the bracket randomly around the current sample
        lx = cxi - wi * self.rng.rand()
        ux = lx + wi

        # find lower bracket end
        while Li(lx) >= logu and cxi - lx < self.max_width:
            lx -= wi

        # find upper bracket end
        while Li(ux) >= logu and ux - cxi < self.max_width:
            ux += wi

        # sample uniformly from bracket
        xi = (ux - lx) * self.rng.rand() + lx

        # if outside slice, reject sample and shrink bracket
        while Li(xi) < logu:
            if xi < cxi:
                lx = xi
            else:
                ux = xi
            xi = (ux - lx) * self.rng.rand() + lx

        return xi, ux - lx
