

import torch
import torch.nn as nn
from weight_quantization import get_qrange


########################################## Activation Collector ################################

class ActivationCollector:
    def __init__(self, activation_layer_types):
        self.activation_layer_types = activation_layer_types
        self.activations = []

    def hook_fn(self, module, input, output):
        self.activations.append(output)

    def register_hooks(self, model):
        hooks = []
        for layer in model.children():
            if isinstance(layer, self.activation_layer_types):
                hooks.append(layer.register_forward_hook(self.hook_fn))
            self.register_hooks(layer)  # Recursive call for nested layers
        return hooks

    def __call__(self, model, input):
        self.activations = []  # Reset previous activations
        hooks = self.register_hooks(model)
        model(input)
        for hook in hooks:
            hook.remove()
        return self.activations
    

########################################## Activation Qauntization ##############################################

class QintTensorLayer(nn.Module):
    def __init__(self, scale, precision):
        super().__init__()
        
        self.scale = torch.nn.Parameter(scale)
        self.qmin, self.qmax = get_qrange(precision=precision)

    def forward(self, x):
        q_tensor = (x / self.scale).clamp(self.qmin, self.qmax).round()
        q_tensor = q_tensor * self.scale
        return q_tensor

class Qfloat16TensorLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.half().float()
    