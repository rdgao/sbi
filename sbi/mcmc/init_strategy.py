# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from copy import deepcopy
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch import Tensor


def prior_init(prior: Any) -> Tensor:
    """Return a sample from the prior."""
    return prior.sample((1,)).detach()


def sir_init(
    prior: Any, potential_fn: Callable, init_strategy_num_candidates: int,
) -> Tensor:
    r"""
    Return a sample obtained by sequential importance reweighing.

    This function can also do `SIR` on the conditional posterior
    $p(\theta_i|\theta_j, x)$ when a `condition` and `dims_to_sample` are passed.

    Args:
        prior: Prior distribution, candidate samples are drawn from it.
        potential_fn: Potential function that the candidate samples are weighted with.
        init_strategy_num_candidates: Number of candidate samples drawn.

    Returns:
        A single sample.
    """
    # TODO: make work with pyro potential function
    
    init_param_candidates = prior.sample((init_strategy_num_candidates,)).detach()

    log_weights = torch.cat(
        [
            potential_fn(init_param_candidates[i, :]).detach()
            for i in range(init_strategy_num_candidates)
        ]
    )
    probs = np.exp(log_weights.view(-1).numpy().astype(np.float64))
    probs[np.isnan(probs)] = 0.0
    probs[np.isinf(probs)] = 0.0
    probs /= probs.sum()
    idxs = np.random.choice(
        a=np.arange(init_strategy_num_candidates), size=1, replace=False, p=probs,
    )
    return init_param_candidates[torch.from_numpy(idxs.astype(int)), :]


class PriorWithFewerDims:
    """
    Prior which samples only from the free dimensions of the conditional.

    This is needed for the the MCMC initialization functions when conditioning.
    For the prior init, we could post-hoc select the relevant dimensions. But
    for SIR, we want to evaluate the `potential_fn` of the conditional
    posterior, which takes only a subset of the full parameter vector theta
    (only the `dims_to_sample`). This subset is provided by `.sample()` from
    this class.
    """

    def __init__(self, full_prior, dims_to_sample):
        self.full_prior = full_prior
        self.dims_to_sample = dims_to_sample

    def sample(self, *args, **kwargs):
        """
        Sample only from the relevant dimension. Other dimensions are filled in
        by the `ConditionalPotentialFunctionProvider()` during MCMC.
        """
        return self.full_prior.sample(*args, **kwargs)[:, self.dims_to_sample]

    def log_prob(self, *args, **kwargs):
        r"""
        `log_prob` is same as for the full prior, because we usually evaluate
        the $\theta$ under the full joint once we have added the condition.
        """
        return self.full_prior.log_prob(*args, **kwargs)