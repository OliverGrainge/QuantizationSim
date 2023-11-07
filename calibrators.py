

import torch 
from torch.optim import Adam
import torch.nn.functional as F
from scipy import optimize
import numpy as np
from weight_quantization import get_qrange, quantize_filter, quantize_channel, quantize_tensor


####################################### MinMax Calibrator ################################

def minmax_scaler(tensor: torch.tensor, precision: str = "int8", granularity="tensor"):
    if precision == "fp32" or precision == "fp16":
        return None
    if granularity == "tensor" or tensor.ndim <= 2:
        qmin, qmax = get_qrange(precision=precision)
        min_val, max_val = tensor.min(), tensor.max()
        max_abs = max(abs(min_val), abs(max_val))
        scale = max_abs / max(qmax, -qmin)
        return torch.tensor([scale])

    elif granularity == "channel":
        scales = []
        reshaped_tensor = (
            tensor.permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)
        )
        for i in range(reshaped_tensor.size(0)):
            channel = reshaped_tensor[i]
            qmin, qmax = get_qrange(precision=precision)
            min_val, max_val = channel.min(), channel.max()
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / max(qmax, -qmin)
            scales.append(scale)
        return torch.tensor(scales).clone().detach().requires_grad_(True)

    elif granularity == "filter":
        scales = []
        for filter in tensor:
            qmin, qmax = get_qrange(precision=precision)
            min_val, max_val = filter.min(), filter.max()
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / max(qmax, -qmin)
            scales.append(scale)
        scales = torch.tensor(scales).clone().detach().requires_grad_(True)
        return scales
    



####################################### Entropy Calibrator ################################

def kl_divergence_objective(scale, fp_tensor, granularity, precision):
    scale = torch.tensor(scale)
    qmin, qmax = get_qrange(precision=precision)

    if granularity == "tensor" or fp_tensor.ndim <= 2:
        q_tensor = quantize_tensor(fp_tensor, scale, qmin, qmax)
    elif granularity == "filter":
        q_tensor = quantize_tensor(fp_tensor, scale, qmin, qmax)
    elif granularity == "channel":
        q_tensor = quantize_channel(fp_tensor, scale, qmin, qmax)
    else:
        raise Exception("Granularity not Implemented")

    hist, bins = torch.histogram(fp_tensor, bins=100)
    qhist, bins = torch.histogram(q_tensor.float(), bins.float())
    hist = hist.detach().float().numpy()
    qhist = qhist.detach().float().numpy()
    # replace zeros to avoid division by zero 
    hist = np.where(hist == 0, np.finfo(float).eps, hist)
    qhist = np.where(qhist == 0, np.finfo(float).eps, qhist)

    kl_div = np.sum(hist * np.log(hist / qhist))
    return kl_div


def entropy_scaler(tensor: torch.tensor, precision: str = "int8", granularity="tensor"):
    if precision == "fp32" or precision == "fp16":
        return None
    
    initial_scale = minmax_scaler(tensor, precision=precision, granularity=granularity).detach().numpy()
    bounds = [(b - 0.01, b + 0.01) for b in initial_scale]
    result = optimize.minimize(
        kl_divergence_objective,
        initial_scale,
        args=(tensor, granularity, precision),
        method="L-BFGS-B",
        bounds=bounds
    ) 
    return torch.tensor(result.x)



####################################### Perceintile Calibrator ########################

def percentile_scaler(tensor: torch.tensor, precision: str="int8", granularity="tensor", percentile=99):
    if precision == "fp32" or precision == "fp16":
        return None
    
    qmin, qmax = get_qrange(precision=precision)
    if granularity == "tensor" or tensor.ndim <= 2:
        _, bins = torch.histogram(tensor, bins=100)
        max_abs = max(abs(bins[100-percentile].item()),abs(bins[percentile].item()))
        scale = max_abs / max(qmax, -qmin)
        return scale
    
    elif granularity == "filter":
        scales = []
        for filter in tensor:
            _, bins = torch.histogram(filter, bins=100)
            max_abs = max(abs(bins[100-percentile].item()), abs(bins[percentile].item()))
            scales.append(max_abs / max(qmax, -qmin))
        return torch.tensor(scales)

    elif granularity == "channel":
        scales = []
        reshaped_tensor = (
            tensor.permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)
        )

        for i in range(reshaped_tensor.size(0)):
            channel = reshaped_tensor[i]
            _, bins = torch.histogram(channel, bins=100)
            max_abs = max(abs(bins[100-percentile].item()), abs(bins[percentile].item()))
            scales.append(max_abs/max(qmax, -qmin))
        return torch.tensor(scales)


####################################### MSE Calibrator ################################


def mse_scaler(tensor: torch.tensor, precision: str = "int8", granularity="tensor"):
    if precision == "fp32" or precision == "fp16":
        return None

    tensor.requires_grad_(True)
    # Calculate the histogram of the activations
    # The number of bins could be set to the range of the precision
    qmin, qmax = get_qrange(precision=precision)
    # Initialize the scale with some reasonable values
    if granularity == "tensor" or tensor.ndim <= 2:
        max_abs = max(abs(tensor.min()), abs(tensor.max()))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)
    elif granularity == "filter":
        max_abs = torch.amax(tensor.abs(), dim=(1, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)
    elif granularity == "channel":
        max_abs = torch.amax(tensor.abs(), dim=(0, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)

    scale = scale.detach()

    optimizer = Adam([scale], lr=0.0001)
    for round in range(20):  # Run for a number of iterations
        optimizer.zero_grad()

        # Compute the quantized tensor
        if granularity == "tensor" or tensor.ndim <= 2:
            quantized_tensor = quantize_tensor(tensor, scale, qmin, qmax)

        elif granularity == "filter":
            quantized_tensor = quantize_filter(tensor, scale, qmin, qmax)

        elif granularity == "channel":
            quantized_tensor = quantize_channel(tensor, scale, qmin, qmax)

        # Ensure that scale is always positive and zero_point is within range
        loss = F.mse_loss(tensor, quantized_tensor)
        

        # Perform the optimization step
        loss.backward(retain_graph=(round < 19))
        optimizer.step()
        scale.data.clamp_(min=1e-8)
    return scale.detach()




################################## Calibration Function ##########################################

def calibration(tensor: torch.tensor, precision: str = "int8", granularity="tensor", calibration_type: str="minmax"):
    if calibration_type == "minmax":
        scale = minmax_scaler(
            tensor, precision=precision, granularity=granularity
        )
        return scale
    elif calibration_type == "entropy":
        scale = entropy_scaler(
            tensor, precision=precision, granularity=granularity
        )
        return scale
    elif calibration_type == "mse":
        scale = mse_scaler(
            tensor, precision=precision, granularity=granularity
                    )
        return scale
    elif calibration_type == "percentile":
        scale = percentile_scaler(
            tensor, precision=precision, granularity=granularity, percentile=1
        )
        return scale
    else: 
        raise Exception("Calibrator Not Implemented")

