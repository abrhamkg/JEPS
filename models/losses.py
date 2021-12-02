import torch
import torch.nn as nn


class AEJEPSLoss(nn.Module):
    """
    A loss function that implements a custom loss whose equation is shown below.

    .. math::
        L = (\hat{I} - I)^2 + \sum_{i=1}^{m} {(\hat{W_i} - W_i)^2} + \sum_{i=1}^{n} {(\hat{M_i} - M_i)^2} + R(h)

    where \(\hat{I}\), \(\hat{M_i}\) and \(\hat{W_i}\) are the reconstructed image, motor command at time step i and
    reconstructed word at time step i respectively, I, \(M_i\) and \(W_i\) are the target outputs at time step i respectively,
    h is the hidden representation and R(h) is a regularization function that ensures the network learns useful representation.

    Parameters
    ----------
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the weighted mean of the output is taken, 'sum': the output will be summed.
        Default: 'mean'
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, goal_img_out, text_out, cmd_out, goal_img, text_target, cmd_target):
        """

        Parameters
        ----------
        goal_img_out : torch.Tensor
            The goal image output from the model
        text_out : torch.Tensor
            The text output from the model
        cmd_out : torch.Tensor
            The motor command output from the model
        goal_img : torch.Tensor
            The reference goal image
        text_target : torch.Tensor
            The reference text sequence
        cmd_target : torch.Tensor
            The reference motor command sequence

        Returns
        -------
        The computed loss
        """
        L_img = nn.functional.mse_loss(goal_img_out, goal_img, reduction=self.reduction)
        L_text = nn.functional.mse_loss(text_out, text_target, reduction=self.reduction)
        L_cmd = nn.functional.mse_loss(cmd_out.float(), cmd_target.float(), reduction=self.reduction)
        return L_img, L_text, L_cmd



LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "mse": nn.MSELoss,
    'smooth_l1': nn.SmoothL1Loss,
    "aejeps_loss": AEJEPSLoss,
}


def get_loss_func(loss_name):
    """
    Returns a class for a loss function that can be instantiated.
    Parameters
    ----------
    loss_name : str
        The name of the loss function to use can be one cross_entropy | bce | mse | smooth_l1 | aejeps_loss

    Returns
    -------
    A torch.nn.Module subclass implementing the loss specified.
    """
    if loss_name not in LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return LOSSES[loss_name]

