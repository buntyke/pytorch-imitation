import torch.nn.functional as F


# weighted l1+l2 loss
def mujoco_loss(output, target, weights):
    loss = weights[0]*F.l1_loss(output, target) + weights[1]*F.mse_loss(output, target)
    return loss
