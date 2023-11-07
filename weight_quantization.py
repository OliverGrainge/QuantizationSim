
import torch 


def quantize_tensor(
    tensor: torch.tensor, scale: torch.tensor, qmin: float, qmax: float
):
    q_tensor = (tensor / scale).clamp(qmin, qmax).round()
    q_tensor = q_tensor * scale
    return q_tensor


def quantize_channel(
    tensor: torch.tensor, scale: torch.tensor, qmin: float, qmax: float
):
    q_channels = []
    reshaped_tensor = tensor.permute(1, 0, 2, 3)
    for i in range(reshaped_tensor.size(0)):
        channel_weights = reshaped_tensor[i]
        q_channel = (channel_weights / scale[i]).clamp(qmin, qmax).round()
        q_channel = q_channel * scale[i]
        q_channels.append(q_channel)
    q_channels = torch.stack(q_channels).permute(1, 0, 2, 3)
    return q_channels


def quantize_filter(
    tensor: torch.tensor, scale: torch.tensor, qmin: float, qmax: float
):
    q_tensors = []
    for i, channel in enumerate(tensor):
        q_tensor = (channel / scale[i]).clamp(qmin, qmax).round()
        q_tensor = q_tensor * scale[i]
        q_tensors.append(q_tensor)
    q_tensor = torch.stack(q_tensors)
    return q_tensor


def get_qrange(precision: str = "int8"):
    if precision == "fp32":
        return None, None
    elif precision == "fp16":
        return None, None
    elif precision == "int32":
        qmin = -(2.0 ** (32 - 1))  # Minimum quantization value
        qmax = 2.0 ** (32 - 1) - 1  # Maximum quantization value
    elif precision == "int8":
        qmin = -(2.0 ** (8 - 1))
        qmax = 2.0 ** (8 - 1) - 1
    elif precision == "int4":
        qmin = -(2.0 ** (4 - 1))
        qmax = 2.0 ** (4 - 1) - 1
    else:
        raise Exception("Can only quantize to int4, int8, int32, fp16, fp32 precisions")
    return torch.tensor(qmin, requires_grad=True), torch.tensor(
        qmax, requires_grad=True
    )


def quantize_weights(
    tensor: torch.tensor,
    scale: torch.tensor,
    precision: str = "int8",
    granularity="tensor",
):
    # Floating Point quantization
    if precision == "fp32":
        return tensor
    elif precision == "fp16":
        return tensor.half().float()

    qmin, qmax = get_qrange(precision=precision)
    # Integer Quantization
    if granularity == "tensor" or tensor.ndim <= 2 and precision:
        tensor = quantize_tensor(tensor, scale, qmin, qmax)
    elif granularity == "channel":
        tensor = quantize_channel(tensor, scale, qmin, qmax)
    elif granularity == "filter":
        tensor = quantize_filter(tensor, scale, qmin, qmax)
    return tensor
