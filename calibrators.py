

import torch 
from torch.optim import Adam
import torch.nn.functional as F
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
        return scale.clone().detach().requires_grad_(True)

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

def kl_divergence(p, q):
    # KL divergence between two distributions
    return (p * (p / q).log()).sum()


def entropy_scaler(tensor: torch.tensor, precision: str = "int8", granularity="tensor"):
    if precision == "fp32" or precision == "fp16":
        return None

    tensor.requires_grad_(True)
    # Calculate the histogram of the activations
    # The number of bins could be set to the range of the precision
    qmin, qmax = get_qrange(precision=precision)
    num_bins = int(qmax - qmin + 1)

    if granularity == "tensor" or tensor.ndim <= 2:
        hist = torch.histc(
            tensor, bins=num_bins, min=tensor.min().item(), max=tensor.max().item()
        )
        prob_dist = hist / hist.sum()
    elif granularity == "filter":
        hist = torch.stack(
            [
                torch.histc(
                    filt, bins=num_bins, min=filt.min().item(), max=filt.max().item()
                )
                for filt in tensor
            ]
        )
    elif granularity == "channel":
        tensor_reshaped = (
            tensor.permute(1, 0, 2, 3).contiguous().view(tensor.size(1), -1)
        )
        hist = torch.stack(
            [
                torch.histc(
                    chan, bins=num_bins, min=chan.min().item(), max=chan.max().item()
                )
                for chan in tensor_reshaped
            ]
        )

    # Initialize the scale with some reasonable values
    if granularity == "tensor" or tensor.ndim <= 2:
        max_abs = max(abs(tensor.min()), abs(tensor.max()))
        scale = torch.tensor(max_abs / max(qmax, -qmin), requires_grad=True)
        scale = torch.nn.Parameter(scale)
    elif granularity == "filter":
        max_abs = torch.amax(tensor.abs(), dim=(1, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)
    elif granularity == "channel":
        max_abs = torch.amax(tensor.abs(), dim=(0, 2, 3))
        scale = max_abs / max(qmax, -qmin)
        scale = torch.nn.Parameter(scale)

    optimizer = Adam([scale], lr=0.01)

    for _ in range(1000):  # Run for a number of iterations
        optimizer.zero_grad()

        # Compute the quantized tensor
        if granularity == "tensor" or tensor.ndim <= 2:
            quantized_tensor = quantize_tensor(tensor, scale, qmin, qmax)

            # Compute the histogram of the quantized tensor
            q_hist = torch.histc(
                quantized_tensor,
                bins=num_bins,
                min=tensor.min().item(),
                max=tensor.max().item(),
            )
            q_prob_dist = q_hist / q_hist.sum()

            # Compute the KL divergence
            loss = kl_divergence(prob_dist, q_prob_dist)

            # Perform the optimization step
            loss.backward()
            optimizer.step()

        elif granularity == "filter":
            quantized_tensor = quantize_filter(tensor, scale, qmin, qmax)

            # Compute the histogram of the quantized tensor
            q_hist = torch.stack(
                [
                    torch.histc(
                        filt, bins=num_bins, min=tensor[i].min(), max=tensor[i].max()
                    )
                    for i, filt in enumerate(quantized_tensor)
                ]
            )
            q_prob_dist = q_hist / q_hist.sum(1, keepdims=True)

            # Compute the KL divergence
            loss = torch.stack(
                [
                    kl_divergence(prob_dist[i], q_prob_dist[i])
                    for i in range(q_hist.size(0))
                ]
            ).mean()

            # Perform the optimization step
            loss.backward()
            optimizer.step()

        elif granularity == "channel":
            quantized_tensor = quantize_channel(tensor, scale, qmin, qmax)

            # Compute the histogram of the quantized tensor
            reshaped_qtensor = (
                quantized_tensor.permute(1, 0, 2, 3)
                .contiguous()
                .view(quantized_tensor.size(1), -1)
            )
            q_hist = torch.stack(
                [
                    torch.histc(
                        chan,
                        bins=num_bins,
                        min=tensor_reshaped[i].min(),
                        max=tensor_reshaped[i].max(),
                    )
                    for i, chan in enumerate(reshaped_qtensor)
                ]
            )
            q_prob_dist = q_hist / q_hist.sum(1, keepdims=True)

            # Compute the KL divergence
            loss = torch.stack(
                [
                    kl_divergence(prob_dist[i], q_prob_dist[i])
                    for i in range(q_hist.size(0))
                ]
            ).mean()

            # Perform the optimization step
            loss.backward()
            optimizer.step()

        # Ensure that scale is always positive and zero_point is within range
        scale.data.clamp_(min=1e-8)
    print("THEERE")
    return scale.item()


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

