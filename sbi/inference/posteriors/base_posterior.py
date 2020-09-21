# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod
from copy import deepcopy
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

import numpy as np
import torch
from torch import Tensor, log
from torch import multiprocessing as mp
from torch import nn

from sbi import utils as utils
import sbi.mcmc as mcmc
from sbi.mcmc.init_strategy import prior_init, sir_init
from sbi.types import Array, Shape
from sbi.user_input.user_input_checks import process_x
from sbi.utils.torchutils import (
    ScalarFloat,
    atleast_2d_float32_tensor,
    ensure_theta_batched,
)


class NeuralPosterior(ABC):
    r"""Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods.<br/><br/>
    All inference methods in sbi train a neural network which is then used to obtain
    the posterior distribution. The `NeuralPosterior` class wraps the trained network
    such that one can directly evaluate the (unnormalized) log probability and draw
    samples from the posterior. The neural network itself can be accessed via the `.net`
    attribute.
    """

    def __init__(
        self,
        method_family: str,
        neural_net: nn.Module,
        prior,
        x_shape: torch.Size,
        mcmc_potential_builder: Callable,
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
        if method_family in ("snpe", "snle", "snre_a", "snre_b"):
            self._method_family = method_family
        else:
            raise ValueError("Method family unsupported.")

        self.net = neural_net

        self._prior = prior
        self._x = None
        self._x_shape = x_shape

        self.set_mcmc_init(mcmc_init)
        self.set_mcmc_method(mcmc_method)
        self.set_mcmc_parameters(mcmc_parameters)

        self._mcmc_init_params = None
        self._mcmc_potential_builder = mcmc_potential_builder

        self._leakage_density_correction_factor = None  # Correction factor for SNPE.
        self._num_trained_rounds = 0

    @property
    def default_x(self) -> Optional[Tensor]:
        """Return default x used by `.sample(), .log_prob` as conditioning context."""
        return self._x

    @default_x.setter
    def default_x(self, x: Tensor) -> None:
        """See `set_default_x`."""
        self.set_default_x(x)

    def set_default_x(self, x: Tensor) -> "NeuralPosterior":
        """Set new default x for `.sample(), .log_prob` to use as conditioning context.

        This is a pure convenience to avoid having to repeatedly specify `x` in calls to
        `.sample()` and `.log_prob()` - only θ needs to be passed.

        This convenience is particularly useful when the posterior is focused, i.e.
        has been trained over multiple rounds to be accurate in the vicinity of a
        particular `x=x_o` (you can check if your posterior object is focused by
        printing it).

        NOTE: this method is chainable, i.e. will return the NeuralPosterior object so
        that calls like `posterior.set_default_x(my_x).sample(mytheta)` are possible.

        Args:
            x: The default observation to set for the posterior $p(theta|x)$.

        Returns:
            `NeuralPosterior` that will use a default `x` when not explicitly passed.
        """
        processed_x = process_x(x, self._x_shape)
        self._x = processed_x

        return self

    @property
    def mcmc_init(self) -> str:
        """Returns MCMC init."""
        return self._mcmc_init

    @mcmc_init.setter
    def mcmc_init(self, init: str) -> None:
        """See `set_mcmc_init`."""
        self.set_mcmc_init(init)

    def set_mcmc_init(self, init: str) -> "NeuralPosterior":
        """Sets init strategy to for MCMC and returns `NeuralPosterior`.

        Args:
            init: Init strategy to use.

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._mcmc_init = init
        return self

    @property
    def mcmc_method(self) -> str:
        """Returns MCMC method."""
        return self._mcmc_method

    @mcmc_method.setter
    def mcmc_method(self, method: str) -> None:
        """See `set_mcmc_method`."""
        self.set_mcmc_method(method)

    def set_mcmc_method(self, method: str) -> "NeuralPosterior":
        """Sets sampling method to for MCMC and returns `NeuralPosterior`.

        Args:
            method: Method to use.

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._mcmc_method = method
        return self

    @property
    def mcmc_parameters(self) -> dict:
        """Returns MCMC parameters."""
        if self._mcmc_parameters is None:
            return {}
        else:
            return self._mcmc_parameters

    @mcmc_parameters.setter
    def mcmc_parameters(self, parameters: Dict[str, Any]) -> None:
        """See `set_mcmc_parameters`."""
        self.set_mcmc_parameters(parameters)

    def set_mcmc_parameters(self, parameters: Dict[str, Any]) -> "NeuralPosterior":
        """Sets parameters for MCMC and returns `NeuralPosterior`.

        Args:
            parameters: Dictionary overriding the default parameters for MCMC.
                The following parameters are supported: `thin` to set the thinning
                factor for the chain, `warmup_steps` to set the initial number of
                samples to discard, `num_chains` for the number of chains,
                `init_strategy` for the initialisation strategy for chains; `prior`
                will draw init locations from prior, whereas `sir` will use Sequential-
                Importance-Resampling using `init_strategy_num_candidates` to find init
                locations.

        Returns:
            `NeuralPosterior` for chainable calls.
        """
        self._mcmc_parameters = parameters
        return self

    @abstractmethod
    def log_prob(
        self, theta: Tensor, x: Optional[Tensor] = None, track_gradients: bool = False,
    ) -> Tensor:
        """See child classes for docstring."""
        pass

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        mcmc_init: Optional[str] = None,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""Return samples from posterior distribution $p(\theta|x)$ using MCMC.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).
            mcmc_init: Optional parameter to override `self.mcmc_init`.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Optional parameter to override `self.mcmc_parameters`.

        Returns:
            Samples from posterior.
        """
        sampler = self._get_sampler(
            x, sample_shape, mcmc_init, mcmc_method, mcmc_parameters
        )

        self.net.eval()
        num_samples = torch.Size(sample_shape).numel()
        samples = sampler.run(num_samples=num_samples)
        self.net.train(True)

        assert samples["theta"].shape[0] == num_samples
        return samples["theta"].reshape((*sample_shape, -1)).detach()

    def sample_conditional(
        self,
        condition: Tensor,
        dims_to_sample: List[int],
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        mcmc_init: Optional[str] = None,
        mcmc_method: Optional[str] = None,
        mcmc_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        r"""Return samples from conditional posterior $p(\theta_i|\theta_j, x)$.

        In this function, we do not sample from the full posterior, but instead only
        from a few parameter dimensions while the other parameter dimensions are kept
        fixed at values specified in `condition`.

        Samples are obtained with MCMC.

        Args:
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_sample: Which dimensions to sample from. The dimensions not
                specified in `dims_to_sample` will be fixed to values given in
                `condition`.
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).
            mcmc_init: Optional parameter to override `self.mcmc_init`.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Optional parameter to override `self.mcmc_parameters`.

        Returns:
            Samples from conditional posterior.
        """
        sampler = self._get_sampler(
            x, sample_shape, mcmc_init, mcmc_method, mcmc_parameters
        )

        self.net.eval()
        num_samples = torch.Size(sample_shape).numel()
        samples = sampler.run(num_samples=num_samples)
        self.net.train(True)

        assert samples["theta"].shape[0] == num_samples
        return samples["theta"].reshape((*sample_shape, -1)).detach()

    def _get_sampler(
        self,
        x: Tensor,
        sample_shape: Optional[Tensor],
        mcmc_init: Optional[str], 
        mcmc_method: Optional[str],
        mcmc_parameters: Optional[Dict[str, Any]],
        condition: Optional[Tensor] = None,
        dims_to_sample: Optional[List[int]] = None,
    ) -> mcmc.MCMC:
        r"""
        Return checked and (potentially default) values to sample from the posterior.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided,
                fall back onto `x_o` if previously provided for multiround training, or
                to a set default (see `set_default_x()` method).
            mcmc_init: Optional parameter to override `self.mcmc_init`.
            mcmc_method: Optional parameter to override `self.mcmc_method`.
            mcmc_parameters: Optional parameter to override `self.mcmc_parameters`.
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_sample: Which dimensions to sample from. The dimensions not
                specified in `dims_to_sample` will be fixed to values given in
                `condition`.

        Returns: Single (potentially default) $x$ with batch dimension; an integer
            number of samples; mcmc class, mcmc potential function, mcmc init function,
            mcmc parameters.
        """
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        num_samples = torch.Size(sample_shape).numel()

        mcmc_init = mcmc_init if mcmc_init is not None else self.mcmc_init
        mcmc_method = mcmc_method if mcmc_method is not None else self.mcmc_method
        mcmc_parameters = (
            mcmc_parameters if mcmc_parameters is not None else self.mcmc_parameters
        )

        if mcmc_method in ["slice", "slice_np", "np"]:
            mcmc_class = mcmc.NumpySliceMCMC
        elif mcmc_method in ["pyro"]:
            mcmc_class = mcmc.PyroMCMC
        elif mcmc_method in ["zeus"]:
            mcmc_clas = mcmc.ZeusSliceEnsembleMCMC
        else:
            raise NotImplementedError

        if condition is not None:
            raise NotImplementedError

        potential_fn = self._mcmc_potential_builder(
            prior=self._prior, net=self.net, x=x
        )

        if mcmc_init == "prior":
            init_fn = lambda parameter: prior_init(prior=self._prior)
        elif mcmc_init == "sir":
            init_fn = lambda parameter: sir_init(prior=self._prior, potential_fn=potential_fn, init_strategy_num_candidates=10_000)
        elif mcmc_init == "last_sample":
            init_fn = lambda parameter: NotImplementedError
        else:
            raise NotImplementedError

        sampler = mcmc_class(
            potential_fn=potential_fn,
            init_fn=init_fn,
            **mcmc_parameters,
        )

        return sampler

    def _prepare_theta_and_x_for_log_prob_(
        self, theta: Tensor, x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns $\theta$ and $x$ in shape that can be used by posterior.log_prob().

        Checks shapes of $\theta$ and $x$ and then repeats $x$ as often as there were
        batch elements in $\theta$.

        Args:
            theta: Parameters $\theta$.
            x: Conditioning context for posterior $p(\theta|x)$. If not provided, fall
                back onto an `x_o` if previously provided for multi-round training, or
                to another default if set later for convenience, see `.set_default_x()`.

        Returns:
            ($\theta$, $x$) with the same batch dimension, where $x$ is repeated as
            often as there were batch elements in $\theta$ orginally.
        """

        theta = ensure_theta_batched(torch.as_tensor(theta))

        # Select and check x to condition on.
        x = atleast_2d_float32_tensor(self._x_else_default_x(x))
        self._ensure_single_x(x)
        self._ensure_x_consistent_with_default_x(x)

        # Repeat `x` in case of evaluation on multiple `theta`. This is needed below in
        # when calling nflows in order to have matching shapes of theta and context x
        # at neural network evaluation time.
        x = self._match_x_with_theta_batch_shape(x, theta)

        return theta, x

    def _x_else_default_x(self, x: Optional[Array]) -> Array:
        if x is not None:
            return x
        elif self.default_x is None:
            raise ValueError(
                "Context `x` needed when a default has not been set."
                "If you'd like to have a default, use the `.set_default_x()` method."
            )
        else:
            return self.default_x

    def _ensure_x_consistent_with_default_x(self, x: Tensor) -> None:
        """Check consistency with the shape of `self.default_x` (unless it's None)."""

        # TODO: This is to check the passed x matches the NN input dimensions by
        # comparing to `default_x`, which was checked in user input checks to match the
        # simulator output. Later if we might not have `self.default_x` we might want to
        # compare to the input dimension of `self.net` here.
        if self.default_x is not None:
            assert (
                x.shape == self.default_x.shape
            ), f"""The shape of the passed `x` {x.shape} and must match the shape of `x`
            used during training, {self.default_x.shape}."""

    @staticmethod
    def _ensure_single_x(x: Tensor) -> None:
        """Raise a ValueError if multiple (a batch of) xs are passed."""

        inferred_batch_size, *_ = x.shape

        if inferred_batch_size > 1:

            raise ValueError(
                """The `x` passed to condition the posterior for evaluation or sampling
                has an inferred batch shape larger than one. This is not supported in
                sbi for reasons depending on the scenario:

                    - in case you want to evaluate or sample conditioned on several xs
                    e.g., (p(theta | [x1, x2, x3])), this is not supported yet in sbi.

                    - in case you trained with a single round to do amortized inference
                    and now you want to evaluate or sample a given theta conditioned on
                    several xs, one after the other, e.g, p(theta | x1), p(theta | x2),
                    p(theta| x3): this broadcasting across xs is not supported in sbi.
                    Instead, what you can do it to call posterior.log_prob(theta, xi)
                    multiple times with different xi.

                    - finally, if your observation is multidimensional, e.g., an image,
                    make sure to pass it with a leading batch dimension, e.g., with
                    shape (1, xdim1, xdim2). Beware that the current implementation
                    of sbi might not provide stable support for this and result in
                    shape mismatches.
                """
            )

    @staticmethod
    def _match_x_with_theta_batch_shape(x: Tensor, theta: Tensor) -> Tensor:
        """Return `x` with batch shape matched to that of `theta`.

        This is needed in nflows in order to have matching shapes of theta and context
        `x` when evaluating the neural network.
        """

        # Theta and x are ensured to have a batch dim, get the shape.
        theta_batch_size, *_ = theta.shape
        x_batch_size, *x_shape = x.shape

        assert x_batch_size == 1, "Batch size 1 should be enforced by caller."
        if theta_batch_size > x_batch_size:
            x_matched = x.expand(theta_batch_size, *x_shape)

            # Double check.
            x_matched_batch_size, *x_matched_shape = x_matched.shape
            assert x_matched_batch_size == theta_batch_size
            assert x_matched_shape == x_shape
        else:
            x_matched = x

        return x_matched

    def _get_net_name(self) -> str:
        """
        Return the name of the neural network used for inference.

        For SNRE the net is sequential because it has a standardization net. Hence,
        we only access its last entry.
        """
        try:
            # Why not `isinstance(self.net[0], StandardizeInputs)`? Because
            # `StandardizeInputs` is defined within a function in
            # neural_nets/classifier.py and can not be imported here.
            # TODO: Refactor this into the net's __str__  method.
            if self.net[0].__class__.__name__ == "StandardizeInputs":
                actual_net = self.net[-1]
            else:
                actual_net = self.net
        except TypeError:
            # If self.net is not a sequential, self.net[0] will throw an error.
            actual_net = self.net

        return actual_net.__class__.__name__.lower()

    def __repr__(self):
        desc = f"""{self.__class__.__name__}(
               method_family={self._method_family},
               net=<a {self.net.__class__.__name__}, see `.net` for details>,
               prior={self._prior!r},
               x_shape={self._x_shape!r})
               """
        return desc

    def __str__(self):
        msg = {0: "untrained", 1: "amortized"}

        focused_msg = "multi-round"

        default_x_msg = (
            f" Evaluates and samples by default at x={self.default_x.tolist()!r}."
            if self.default_x is not None
            else ""
        )

        desc = (
            f"Posterior conditional density p(θ|x) "
            f"({msg.get(self._num_trained_rounds, focused_msg)}).{default_x_msg}\n\n"
            f"This {self.__class__.__name__}-object was obtained with a "
            f"{self._method_family.upper()}-class "
            f"method using a {self._get_net_name()}.\n"
            f"{self._purpose}"
        )

        return desc


class ConditionalPotentialFunctionProvider:
    """
    Wraps the potential functions to allow for sampling from the conditional posterior.
    """

    def __init__(
        self,
        potential_fn_provider: Callable,
        condition: Tensor,
        dims_to_sample: List[int],
    ):
        """
        Args:
            potential_fn_provider: Creates potential function of unconditional
                posterior.
            condition: Parameter set that all dimensions not specified in
                `dims_to_sample` will be fixed to. Should contain dim_theta elements,
                i.e. it could e.g. be a sample from the posterior distribution.
                The entries at all `dims_to_sample` will be ignored.
            dims_to_sample: Which dimensions to sample from. The dimensions not
                specified in `dims_to_sample` will be fixed to values given in
                `condition`.
        """

        self.potential_fn_provider = potential_fn_provider
        self.condition = ensure_theta_batched(condition)
        self.dims_to_sample = dims_to_sample

    def __call__(self, prior, net: nn.Module, x: Tensor, mcmc_method: str,) -> Callable:
        """Return potential function.

        Switch on numpy or pyro potential function based on `mcmc_method`.
        """
        # Set prior, net, and x as attributes of unconditional potential_fn_provider.
        _ = self.potential_fn_provider.__call__(prior, net, x, mcmc_method)

        return self.pyro_potential

    def pyro_potential(self, theta: Dict[str, Tensor]) -> Tensor:
        r"""
        Return conditional posterior log-probability or $-\infty$ if outside prior.

        Args:
            theta: Free parameters $\theta_i$ (from pyro sampler).

        Returns:
            Conditional posterior log-probability $\log(p(\theta_i|\theta_j, x))$,
            masked outside of prior.
        """

        theta = next(iter(theta.values()))

        theta_condition = deepcopy(self.condition)
        theta_condition[:, self.dims_to_sample] = theta

        return self.potential_fn_provider.pyro_potential({"": theta_condition})
