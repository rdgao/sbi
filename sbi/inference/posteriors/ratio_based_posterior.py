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
from sbi.utils.torchutils import (
    ScalarFloat,
    atleast_2d_float32_tensor,
    ensure_theta_batched,
    ensure_x_batched,
)


class RatioBasedPosterior(NeuralPosterior):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods, obtained with
    SNRE.<br/><br/>
    SNRE trains a neural network to approximate likelihood ratios, which in turn can be
    used obtain an unnormalized posterior $p(\theta|x) \propto p(x|\theta) \cdot
    p(\theta)$. The `SNRE_Posterior` class wraps the trained network such that one can
    directly evaluate the unnormalized posterior log-probability $p(\theta|x) \propto
    p(x|\theta) \cdot p(\theta)$ and draw samples from the posterior with
    MCMC. Note that, in the case of single-round SNRE_A / AALR, it is possible to
    evaluate the log-probability of the **normalized** posterior, but sampling still
    requires MCMC.<br/><br/>
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
            mcmc_potential_builder: Callable that builds potential function. 
            mcmc_init: Initialisation strategy to to for MCMC sampling.
            mcmc_method: Method used for MCMC sampling.
            mcmc_parameters: Dictionary overriding the default parameters for MCMC.
        """
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(mcmc_potential_builder=PotentialFunctionProvider(), **kwargs)

    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        r"""
        Returns the log-probability of $p(x|\theta) \cdot p(\theta).$

        This corresponds to an **unnormalized** posterior log-probability. Only for
        single-round SNRE_A / AALR, the returned log-probability will correspond to the
        **normalized** log-probability.

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

        self._warn_log_prob_snre()

        with torch.set_grad_enabled(track_gradients):
            log_ratio = self.net(torch.cat((theta, x), dim=1)).reshape(-1)
            return log_ratio + self._prior.log_prob(theta)

    def _warn_log_prob_snre(self) -> None:
        if self._method_family == "snre_a":
            if self._num_trained_rounds > 1:
                warn(
                    "The log-probability from AALR / SNRE-A beyond round 1 is only"
                    " correct up to a normalizing constant."
                )
        elif self._method_family == "snre_b":
            warn(
                "The log probability from SNRE_B is only correct up to a normalizing "
                "constant."
            )

    @property
    def _num_trained_rounds(self) -> int:
        return self._trained_rounds

    @_num_trained_rounds.setter
    def _num_trained_rounds(self, trained_rounds: int) -> None:
        """
        Sets the number of trained rounds and updates the purpose.

        When the number of trained rounds is 1 and the algorithm is SNRE_A, then the
        log_prob will be normalized, as specified in the purpose.

        The reason we made this a property is that the purpose gets updated
        automatically whenever the number of rounds is updated.
        """
        self._trained_rounds = trained_rounds

        normalized_or_not = (
            ""
            if (self._method_family == "snre_a" and self._trained_rounds == 1)
            else "_unnormalized_ "
        )
        self._purpose = (
            f"It provides MCMC to .sample() from the posterior and "
            f"can evaluate the {normalized_or_not}posterior density with .log_prob()."
        )


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
        self, prior, classifier: nn.Module, x: Tensor, 
    ) -> Callable:
        r"""Return potential function for posterior $p(\theta|x)$.

        Args:
            prior: Prior distribution that can be evaluated.
            net: Binary classifier approximating the likelihood up to a constant.
            x: Conditioning variable for posterior $p(\theta|x)$.

        Returns:
            Potential function for sampler.
        """

        self.classifier = net
        self.prior = prior
        self.x = x

        return self.pyro_potential

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""Return potential for Pyro sampler.

        Args:
            theta: Parameters $\theta$.

        Returns:
            Potential $-(\log r(x_o, \theta) + \log p(\theta))$.
        """
        theta = next(iter(theta.values()))

        # Theta and x should have shape (batch, dim).
        theta = ensure_theta_batched(theta)
        x = ensure_x_batched(self.x)

        log_ratio = self.classifier(torch.cat((theta, x), dim=1).reshape(1, -1))

        return -(log_ratio + self.prior.log_prob(theta))
