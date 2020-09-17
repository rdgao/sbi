# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch
import torch.multiprocessing as mp


class MCMC(ABC):
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
    ):
        """Sample from `potential_fn` with MCMC

        Args:
            potential_fn: Potential function, e.g., -1 * log likelihood. Should take a
                dictionary a single input. The dictionary should contain parameter
                names as `str` keys, mapping to `torch.Tensor` values. It should return
                a 2-dimensional `torch.Tensor`, where the first dimension indexes over
                batches (if batch evaluations are unsupported, make sure to set
                `vectorize=False`).
            init_fn: Initialisation function. Should take string specifying a parameter
                name a a single input. The function should return a 1-d `torch.Tensor`
                as output returning a initial parameter. Note that initial parameter
                should be specified in unconstrained space.
            thin: Thinning to be applied.
            warmup: Length of warm-up (discarded from samples). Note that if
                `thin > 1` is used, the total number of warmup steps is going to be
                `thin * warmup`.
            num_chains: Number of chains to use to generate samples.
            transforms: Optional dictionary that specifies a transform
                for a sample site with constrained support to unconstrained space. The
                transform should be invertible, and implement `log_abs_det_jacobian`.
            vectorize: Boolean flag to indicate whether a potential function supports
                batched evaluations.
            available_cpus: Number of CPUs available for parallelization
            verbose: Show/hide additional info such as progress bars
        """
        self.potential_fn = potential_fn
        self.init_fn = init_fn
        self.thin = thin
        self.warmup = warmup
        self.num_chains = num_chains
        self.transforms = transforms
        self.vectorize = vectorize
        self.available_cpus = available_cpus
        self.verbose = verbose

    @abstractmethod
    def run(self, num_samples: int) -> Dict[str, torch.Tensor]:
        """Sample with MCMC

        Args:
            num_samples: Number of samples to generate. The total number of steps the
                MCMC sampler takes will depend on `num_warmup`, `num_chains`, and
                `thin`. The total number of likelihood evaluations will be higher
                (e.g. due to rejections and/or Gibbs updates) and depends on the
                algorithm.

        Returns:
            Dictionary containing samples in a `torch.Tensor`. Tensors have shape
            `num_samples, dim_parameter`.
        """
        pass
