

import torch
import torch.nn as nn
from weight_quantization import get_qrange


########################################## Activation Collector ################################

class ActivationCollector:
    def __init__(self, activation_layer_types):
        self.activation_layer_types = activation_layer_types
        self.activations = {}

    def hook_fn(self, module, input, output):
        layer_id = id(module)  # or any other unique identifier of the layer
        self.activations[str(layer_id)] = output

    def register_hooks(self, model):
        hooks = []
        
        def recursive_register_hooks(module):
            for layer in module.children():
                # Check if this is a leaf module (does not have children of its own)
                if list(layer.children()):
                    # If it's not a leaf module, we do a recursive call
                    recursive_register_hooks(layer)
                elif isinstance(layer, self.activation_layer_types):
                    # Only register a hook if it's a leaf module of the correct type
                    hooks.append(layer.register_forward_hook(self.hook_fn))

        recursive_register_hooks(model)
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
    