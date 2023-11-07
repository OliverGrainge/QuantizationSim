import torch 
import torch.nn as nn
from activation_quantization import ActivationCollector, Qfloat16TensorLayer, QintTensorLayer
from calibrators import calibration
from weight_quantization import quantize_weights

def quantize_layer_recursive(
    model,
    module,
    module_name,
    state_dict,
    configuration,
    granularity,
    calibration_type="minmax",
    new_state_dict=None,
    idx=None,
    layer_types=(nn.Conv2d, nn.BatchNorm2d, nn.Linear)
):
    if new_state_dict is None:
        new_state_dict = {}
    if idx is None:
        idx = 0

    is_leaf_module = len(list(module.children())) == 0
    if is_leaf_module and isinstance(module, layer_types):
        # Apply the function to the module's weights and bias if they exist
        for name, param in module.named_parameters():
            if name == "weight":
                param_key = f"{module_name}.{name}" if module_name else name
                weight = param.data.clone().detach()
                scale = calibration(weight, precision=configuration[idx], granularity=granularity, calibration_type=calibration_type)
                qtensor = quantize_weights(weight, scale, precision=configuration[idx], granularity=granularity)
                new_state_dict[param_key] = qtensor
        idx += 1

    else:
        # Call this function recursively for each submodule
        for name, submodule in module.named_children():
            if name:  # avoid self-recursion on the module itself
                full_name = f"{module_name}.{name}" if module_name else name
                model, new_state_dict, idx = quantize_layer_recursive(
                    model,
                    submodule,
                    full_name,
                    state_dict,
                    configuration,
                    granularity,
                    calibration_type=calibration_type,
                    new_state_dict=new_state_dict,
                    layer_types=layer_types,
                    idx=idx,
                )
    return model, new_state_dict, idx



class Quantizer:
    def __init__(self, model, 
                 layer_precision="int8",
                 activation_precision="int32",
                 layer_configuration=None,
                 activation_configuration=None,
                 layer_granularity="channel",
                 activation_granularity="tensor", 
                 calibration_type="mse", 
                 calibration_loader=None): 
        # the pytorch model to quantize
        self.model = model
        self.state_dict = model.state_dict()
        # the per layer/activation precision configurations
        self.layer_configuration = layer_configuration
        self.activation_configuration = activation_configuration
        # layer/activation precision in the absence of a configuration
        self.layer_precision = layer_precision
        self.activation_precision = activation_precision
        # the granularity of quantization preicision
        self.layer_granularity = layer_granularity 
        self.activation_granularity = activation_granularity 
        # calibration type
        self.calibration_type = calibration_type
        self.calibration_loader = calibration_loader


    def collect_activations(self):
        if self.calibration_loader is None:
            raise Exception("Must Pass a Calibration loader if quantizing activations")
        else: 
            activations = []
            for batch in self.calibration_loader:
                collector = ActivationCollector()
                collector.register_hooks(model, layers=(nn.ReLU6, nn.ReLU, nn.BatchNorm2d))
                model(batch)
                activations.append(collector.activations)
        
        min_length = min(len(sublist) for sublist in activations)
        # Now, stack (or concatenate) the tensors for the first and second elements across all sublists
        stacked_activations = []
        for i in range(min_length):
            # Using torch.stack to create a new dimension for stacking
            stacked = torch.stack([sublist[i] for sublist in activations])
            stacked_activations.append(stacked)
        self.activation_data = stacked_activations


    def quantize_layers(self):
        if self.layer_configuration is None:
            self.layer_configuration = [self.layer_precision for _ in range(1000)]

        state_dict = self.model.state_dict()

        self.model, new_state_dict, _ = quantize_layer_recursive(
            self.model, self.model, None, state_dict,
            self.layer_configuration,
            self.layer_granularity,
            calibration_type=self.calibration_type,
            layer_types=(nn.Conv2d, nn.Linear) 
        )

        for key in state_dict.keys():
            if key not in list(new_state_dict.keys()):
                new_state_dict[key] = state_dict[key]

        self.model.load_state_dict(new_state_dict)
        return self.model

    def quantize_activations(self):
        if self.activation_configuration == None:
            self.activation_configuration = [self.activation_precision for _ in range(1000)]
            
        def replace_relu_with_sequential_identity(model, idx=None):
            if idx == None:
                idx = 0
            
            for name, module in model.named_children():
                if isinstance(module, (nn.ReLU6, nn.ReLU, nn.BatchNorm2d)):
                    # Replace the ReLU with Sequential containing ReLU and Identity
                    if self.activation_configuration[0] == "fp32":
                        setattr(model, name, nn.Sequential(module, nn.Identity()))
                    elif self.activation_configuration[idx] == "fp16":
                        setattr(model, name, nn.Sequential(module, Qfloat16TensorLayer()))
                    else: 
                        scale = calibration(self.activation_data[idx], 
                                            precision=self.activation_configuration[idx], 
                                            granularity=self.activation_granularity, 
                                            calibration_type=self.calibration_type)
                        
                        setattr(model, name, nn.Sequential(module, QintTensorLayer(scale=scale, precision=self.activation_configuration[idx])))
                    idx += 1

                elif len(list(module.children())) > 0:  # If module has children, recursively apply the function
                    idx = replace_relu_with_sequential_identity(module, idx)
                    
            
            return idx

        replace_relu_with_sequential_identity(self.model)
        return self.model
