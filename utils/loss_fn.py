"""
File with implementation of maskNLLLoss
"""
from typing import Tuple

import torch


def mask_nll_loss(out: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
                  ) -> Tuple[torch.Tensor, int]:
    """
    Function implementing maskNLLLoss.
    It calculates the average negative log likelihood of the
    elements that correspond to a 1 in mask tensor.

    Parameters
    ----------
    out: torch.Tensor
        Tensor of outputs calculated by decoder.
    target: torch.Tensor
        Tensor of target values.
    mask: torch.Tensor
        Binary tensor describing the padding od the target tensor.
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple which contains calculated loss value and sum of ones in mask tensor.
    """
    total: int = mask.sum()
    cross_entropy: torch.Tensor = -torch.log(
        torch.gather(out, 1, target.view(-1, 1)).squeeze(1)
    )
    loss: torch.Tensor = cross_entropy.masked_select(mask).mean()
    return loss, total.item()
