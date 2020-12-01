from copy import deepcopy
from typing import Callable, Optional, Tuple, Union

import torch
from pyknos.nflows.nn import nets
from torch import Tensor, nn, optim, relu
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from sbi.utils.sbiutils import standardizing_net


def build_input_layer(
    batch_theta: Tensor = None,
    z_score_theta: bool = True,
    embedding_net_theta: nn.Module = nn.Identity(),
) -> nn.Module:
    r"""Builds input layer for the `RestrictionEstimator` that optionally z-scores.
    The classifier used in the `RestrictionEstimator` will receive batches of $\theta$s.
    Args:
        batch_theta: Batch of $\theta$s, used to infer dimensionality and (optional)
            z-scoring.
        z_score_theta: Whether to z-score $\theta$s passing into the network.
        embedding_net_theta: Optional embedding network for $\theta$s.
    Returns:
        Input layer that optionally z-scores.
    """
    if z_score_theta:
        input_layer = nn.Sequential(standardizing_net(batch_theta), embedding_net_theta)
    else:
        input_layer = embedding_net_theta

    return input_layer


class Sensitivity:
    def __init__(self, posterior):
        self._posterior = posterior
        self._regression_net = None
        self._theta = None
        self._emergent_property = None

    def add_property(
        self,
        theta: Tensor,
        emergent_property: Tensor,
        model: Union[str, Callable] = "resnet",
        hidden_features: int = 100,
        num_blocks: int = 2,
        dropout_probability: float = 0.5,
        z_score: bool = True,
        embedding_net: nn.Module = nn.Identity(),
    ):
        self._theta = theta
        self._emergent_property = emergent_property

        if isinstance(model, str):
            if model == "resnet":

                def build_nn(theta):
                    classifier = nets.ResidualNet(
                        in_features=theta.shape[1],
                        out_features=2,
                        hidden_features=hidden_features,
                        context_features=None,
                        num_blocks=num_blocks,
                        activation=relu,
                        dropout_probability=dropout_probability,
                        use_batch_norm=True,
                    )
                    input_layer = build_input_layer(theta, z_score, embedding_net)
                    classifier = nn.Sequential(input_layer, classifier)
                    return classifier

            elif model == "mlp":

                def build_nn(theta):
                    classifier = nn.Sequential(
                        nn.Linear(theta.shape[1], hidden_features),
                        nn.BatchNorm1d(hidden_features),
                        nn.ReLU(),
                        nn.Linear(hidden_features, hidden_features),
                        nn.BatchNorm1d(hidden_features),
                        nn.ReLU(),
                        nn.Linear(hidden_features, 2),
                    )
                    input_layer = build_input_layer(theta, z_score, embedding_net)
                    classifier = nn.Sequential(input_layer, classifier)
                    return classifier

            else:
                raise NameError
        else:
            build_nn = model

        self._build_nn = build_nn

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: Optional[int] = None,
        clip_max_norm: Optional[float] = 5.0,
    ):

        # Get indices for permutation of the data.
        num_examples = len(self._theta)
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(self._theta, self._emergent_property)

        # Create neural_net and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=training_batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_examples - num_training_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices),
        )

        if self._classifier is None:
            self._classifier = self._build_nn(self._theta[train_indices])

        optimizer = optim.Adam(list(self._classifier.parameters()), lr=learning_rate,)
        max_num_epochs = 2 ** 31 - 1 if max_num_epochs is None else max_num_epochs

        # criterion / loss
        criterion = MSELoss()

        epoch, self._val_log_prob = 0, float("-Inf")
        while epoch <= max_num_epochs and not self._converged(epoch, stop_after_epochs):
            self._classifier.train()
            for parameters, observations in train_loader:
                optimizer.zero_grad()
                outputs = self._classifier(parameters)
                loss = criterion(outputs, observations)
                loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._classifier.parameters(), max_norm=clip_max_norm,
                    )
                optimizer.step()

            epoch += 1

            # calculate validation performance
            self._classifier.eval()

            val_loss = 0.0
            with torch.no_grad():
                for parameters, observations in val_loader:
                    outputs = self._classifier(parameters)
                    loss = criterion(outputs, observations)
                    val_loss += loss.item()
            self._val_log_prob = -val_loss / num_validation_examples

            print("Training neural network. Epochs trained: ", epoch, end="\r")

        return deepcopy(self._classifier)

    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        r"""
        Return whether the training converged yet and save best model state so far.
        Checks for improvement in validation performance over previous epochs.
        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.
        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        posterior_nn = self._classifier

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(posterior_nn.state_dict())
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            posterior_nn.load_state_dict(self._best_model_state_dict)
            converged = True

        return converged

    def find_active(
        self,
        posterior_log_prob_as_property: bool = False,
        num_monte_carlo_samples: int = 1000,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return eigenvectors and values corresponding to directions of sensitivity.

        The directions of sensitivity are the directions along which a specific
        property changes in the fastest way.

        Args:
            posterior_log_prob_as_property: Whether to use the posterior
                log-probability the key property whose sensitivity is analysed. If
                `False`, one must have specified an emergent property and trained a
                regression network using `.add_property().train()`. If `True`,
                any previously specified property is ignored.
            num_monte_carlo_samples: Number of Monte Carlo samples that the average is
                based on. A larger value will make the results more accurate while
                requiring more compute time.

        Returns: Eigenvectors and corresponding eigenvalues. They are sorted from
            largest to smallest eigenvalue. If multiple emergent properties were
            specified, this will return a list where the n-th entry corresponds to the
            eigenvectors and values of the n-th property.
        """

        if self._emergent_property is None and not posterior_log_prob_as_property:
            raise ValueError(
                "You have not yet passed an emergent property whose "
                "sensitivity you would like to analyse. Please use "
                "`.add_emergent_property().train()` to do so. If you want "
                "to use all features that had also been used to infer the "
                "posterior distribution (i.e. you want to analyse the "
                "sensitivity of the posterior probability), use:"
                "`.find_active(posterior_log_prob_as_property=True)`."
            )

        thetas = self._posterior.sample((num_monte_carlo_samples,))

        thetas.requires_grad = True
        predictions = self._regression_net.forward(thetas)
        loss = predictions.mean(dim=1)
        loss.backward()
        gradient_input = torch.squeeze(thetas.grad)
        outer_products = torch.einsum("bi,bj->bij", (gradient_input, gradient_input))
        average_outer_product = outer_products.mean(dim=0)

        eigen_values, eigen_vectors = torch.symeig(
            average_outer_product, eigenvectors=True
        )

        return eigen_values, eigen_vectors
