from typing import Type
import torch.nn as nn


def freeze_module(module: Type[nn.Module]):
    """
    Freezes the parameters of a module so gradient will not be computed for them.

    Parameters
    ----------
    module : torch.nn.Module
        Any subclass of torch.nn.Module

    Returns
    -------

    """
    for param in module.parameters():
        param.requires_grad = False


if __name__ == '__main__':
    import torch.nn as nn

    m = nn.Linear(20, 50)
    freeze_module(m)

    all_params = set([p.requires_grad for p in m.parameters()])

    if len(all_params) != 1:
        print(f"Test failed: expected 'all_params' to contain only False values but contains {all_params}")
    else:
        print("Test passed!")