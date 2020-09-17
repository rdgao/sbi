import pyro.distributions as dist

from sbi.mcmc import NumpySliceMCMC, PyroMCMC, ZeusSliceEnsembleMCMC
from typing import Dict
import torch


num_dim = 2


def init_fn(parameter: str):
    return torch.zeros((num_dim,))


x_o = torch.zeros((num_dim,))


def potential_fn(parameters: Dict):
    return -1.0 * dist.Normal(loc=x_o, scale=1.0).to_event(1).log_prob(
        parameters["theta"]
    )


sampler = ZeusSliceEnsembleMCMC(potential_fn=potential_fn, init_fn=init_fn)
samples = sampler.run(10)

xxx
