import numpy as np
import torch
import functorch


class HyperNetwork(torch.nn.Module):
    def __init__(self, hyper: torch.nn.Module, base: torch.nn.Module):
        super().__init__()
        self._hyper = hyper
        params_lookup = [[None, None, 0, 0]]
        for name, param in base.named_parameters():
            s = params_lookup[-1][-1]
            params_lookup.append([name, param.shape, s, s+np.prod(param.shape)])
        self._params_lookup = params_lookup[1:]
        buffers = {}
        self._call = lambda params, data: torch.func.functional_call(base, (params, buffers), (data,))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        params = self._hyper(x)
        params_dict = {}
        for i in self._params_lookup:
            params_dict[i[0]] = params[:, i[2]:i[3]].reshape(-1, *i[1])
        return torch.vmap(self._call, in_dims=(0, 0))(params_dict, z)