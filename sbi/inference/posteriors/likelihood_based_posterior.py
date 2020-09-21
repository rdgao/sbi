# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from warnings import warn

import numpy as np
import torch
from torch import Tensor, nn

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.types import Shape
from sbi.utils import del_entries
from sbi.utils.torchutils import ScalarFloat, atleast_2d_float32_tensor
from sbi.utils.torchutils import (
    ensure_theta_batched,
    ensure_x_batched,
)

class LikelihoodBasedPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNLE.<br/><br/>
    SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
    `SNLE_Posterior` class wraps the trained network such that one can directly evaluate
    the unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
    p(\theta)$ and draw samples from the posterior with MCMC.<br/><br/>
    The neural network itself can be accessed via the `.net` attribute.
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior,
        x_shape: torch.Size,
        mcmc_init: str = "prior",
        mcmc_method: str = "slice",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            method_family: One of snpe, snl, snre_a or snre_b.
            neural_net: A classifier for SNRE, a density estimator for SNPE and SNL.
            prior: Prior distribution with `.log_prob()` and `.sample()`.
            x_shape: Shape of a single simulator output.
            mcmc_init: Initialisation strategy to to for MCMC sampling.
            mcmc_method: Method used for MCMC sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(mcmc_potential_builder=PotentialFunctionProvider(), **kwargs)

        self._purpose = (
            f"It provides MCMC to .sample() from the posterior and "
            f"can evaluate the _unnormalized_ posterior density with .log_prob()."
        )

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        r"""
        Returns the log-probability of $p(x|\theta) \cdot p(\theta).$

        This corresponds to an **unnormalized** posterior log-probability.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
                back onto an `x_o` if previously provided for multi-round training, or
                to another default if set later for convenience, see `.set_default_x()`.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.

        Returns:
            `(len(θ),)`-shaped log-probability $\log(p(x|\theta) \cdot p(\theta))$.

        """
        # TODO Train exited here, entered after sampling?
        self.net.eval()

        theta, x = self._prepare_theta_and_x_for_log_prob_(theta, x)

        warn(
            "The log probability from SNL is only correct up to a normalizing constant."
        )

        with torch.set_grad_enabled(track_gradients):
            return self.net.log_prob(x, theta) + self._prior.log_prob(theta)


class PotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
    Posterior class. 

    NOTE: Why use a class?
    ----------------------
    During inference, we use deepcopy to save untrained posteriors in memory. deepcopy
    uses pickle which can't serialize nested functions
    (https://stackoverflow.com/a/12022055).

    It is important to NOT initialize attributes upon instantiation, because we need the
    most current trained posterior neural net.
    """

    def __call__(
        self, prior, net: nn.Module, x: Tensor,
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Args:
            prior: Prior distribution that can be evaluated.
            net: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$.

        Returns:
            Potential function for sampler.
        """
        self.likelihood_nn = net
        self.prior = prior
        self.x = x

        return self.pyro_potential

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return posterior log probability of parameters $p(\theta|x)$.

        Args:
            theta: Parameters $\theta$.

        Returns:
            The potential $-[\log r(x_o, \theta) + \log p(\theta)]$.
        """
        theta = next(iter(theta.values()))

        # Theta and x should have shape (batch, dim).
        theta = ensure_theta_batched(theta)
        x = ensure_x_batched(self.x)

        log_likelihood = self.likelihood_nn.log_prob(
            inputs=x, context=theta
        )

        return -(log_likelihood + self.prior.log_prob(theta))
