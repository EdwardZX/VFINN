import torch
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

def warpForward(tensorInput, tensorFlow):
    tensorFlow = torch.cat([
        tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
    ], 1)
    # to [-1, 1], note that when first point such as [1] -> [255], the v calculated 254
    return torch.nn.functional.grid_sample(
        input=tensorInput,
        grid= tensorFlow.permute(0, 2, 3, 1), # B, [vx,vy], H, W -> B , H, W, [vx, vy]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)


def multiscaleUnsupervisorError(tensorFlowForward,
                                tensorFirst,
                                tensorSecond):
    lambda_s = 3.0
    # lambda_c = 0.2
    loss = (warpLoss(tensorFirst, tensorSecond, tensorFlowForward) 
            + lambda_s * secondSmoothnessLoss(tensorFlowForward))
    return loss


def charbonnierLoss(x, alpha=0.45, beta=1.0, epsilon=1e-3):
    """Compute the generalized charbonnier loss for x
    Args:
        x(tesnor): [batch, channels, height, width]
    Returns:
        loss
    """
    batch, channels, height, width = x.shape
    normalization = torch.tensor(batch * height * width * channels,
                                 requires_grad=False)

    error = torch.pow(
        (x * torch.tensor(beta)).pow(2) + torch.tensor(epsilon).pow(2), alpha)

    return torch.sum(error) / normalization


# photometric difference
def warpLoss(tensorFirst, tensorSecond, tensorFlow):
    """Differentiable Charbonnier penalty function"""
    tensorDifference = tensorSecond - warpForward(tensorInput=tensorFirst,
                                              tensorFlow=tensorFlow)
    return charbonnierLoss(tensorDifference, beta=255.0)


# 2nd order smoothness loss
def _secondOrderDeltas(tensorFlow):
    """2nd order smoothness, compute smoothness loss components"""
    out_channels = 4
    in_channels = 1
    kh, kw = 3, 3

    filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
    filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
    filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
    filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
    weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
    weight[0, 0, :, :] = torch.FloatTensor(filter_x)
    weight[1, 0, :, :] = torch.FloatTensor(filter_y)
    weight[2, 0, :, :] = torch.FloatTensor(filter_diag1)
    weight[3, 0, :, :] = torch.FloatTensor(filter_diag2)

    uFlow, vFlow = torch.split(tensorFlow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(uFlow, weight.to(device))
    delta_v = F.conv2d(vFlow, weight.to(device))
    return delta_u, delta_v


def secondSmoothnessLoss(tensorFlow):
    """Compute 2nd order smoothness loss"""
    delta_u, delta_v = _secondOrderDeltas(tensorFlow)
    return charbonnierLoss(delta_u) + charbonnierLoss(delta_v)

