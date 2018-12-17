import torch
import torch.nn.functional as F


def l1_dist(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        loss = F.l1_loss(output, target)
    return loss

def l2_dist(output, target):
    with torch.no_grad():
        assert output.shape[0] == len(target)
        loss = F.mse_loss(output, target)
    return loss
