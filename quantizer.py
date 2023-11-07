import torch 
import torch.nn as nn
from activation_quantization import ActivationCollector, Qfloat16TensorLayer, QintTensorLayer
from calibrators import calibration
from weight_quantization import quantize_weights
from tqdm import tqdm




class Quantizer:
    def __init__(self, model, 
                 layer_precision="int8",
                 activation_precision="int32",
                 layer_configuration=None,
                 activation_configuration=None,
                 layer_granularity="channel",
                 activation_granularity="tensor", 
                 calibration_type="mse", 
                 calibration_loader=None, 
                 activation_layers=(nn.ReLU, nn.ReLU6, nn.BatchNorm2d),
                 weight_layers=(nn.Linear, nn.Conv2d, nn.BatchNorm2d)): 
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
        # which are the activation layers 
        self.activation_layers = activation_layers
        self.weight_layers = weight_layers

        # Progress bar 
        self.pbar = tqdm(total=self.num_activations() + self.num_layers(), desc="Quantizing Model")

    def collect_activations(self):
        if self.calibration_loader is None:
            raise Exception("Must Pass a Calibration loader if quantizing activations")
        else: 
            activations = []
            for batch in self.calibration_loader:
                collector = ActivationCollector(self.activation_layers)
                collector.register_hooks(self.model)
                self.model(batch)
                activations.append(collector.activations)

        total_activations = {}
        for key in activations[0].keys():
            total_activations[key] = []

        for activation_dict in activations:
            for key, value in activation_dict.items():
                total_activations[key].append(value)

        for key in total_activations.keys():
            total_activations[key] = torch.stack(total_activations[key])
        
        self.activation_data = list(total_activations.values())
        return self.activation_data


    def quantize_layers(self):
        if self.layer_configuration is None:
            self.layer_configuration = [self.layer_precision for _ in range(1000)]

        state_dict = self.model.state_dict()


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
                self.pbar.update(1)
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

        self.model, new_state_dict, idx = quantize_layer_recursive(
            self.model, self.model, None, state_dict,
            self.layer_configuration,
            self.layer_granularity,
            calibration_type=self.calibration_type,
            layer_types=self.weight_layers
        )
        self.layers_quantized = idx
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
                if isinstance(module, self.activation_layers):
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
                    self.pbar.update(1)
                    idx += 1
                    

                elif len(list(module.children())) > 0:  # If module has children, recursively apply the function
                    idx = replace_relu_with_sequential_identity(module, idx)
                    
            return idx

        idx = replace_relu_with_sequential_identity(self.model)
        self.activations_quantized = idx
        return self.model
    
    def num_activations(self):
        if self.calibration_loader == None:
            raise Exception("Must pass a calibration loader to quantizer")
        for batch in self.calibration_loader:
            collector = ActivationCollector(self.activation_layers)
            collector.register_hooks(self.model)
            self.model(batch)
            return len(collector.activations)

    def num_layers(self):

        def count_layer(
            model,
            module,
            idx=None,
            layer_types=(nn.Conv2d, nn.BatchNorm2d, nn.Linear)
        ):
            if idx is None:
                idx = 0

            is_leaf_module = len(list(module.children())) == 0
            if is_leaf_module and isinstance(module, layer_types):
                idx += 1
            else:
                # Call this function recursively for each submodule
                for name, submodule in module.named_children():
                    if name:  # avoid self-recursion on the module itself
                        model, idx = count_layer(
                            model,
                            submodule,
                            layer_types=layer_types,
                            idx=idx,
                        )
            return model, idx
        
        _, layer_count = count_layer(self.model, self.model, layer_types=self.weight_layers)
        return layer_count

    def fit(self):
        self.collect_activations()
        self.model = self.quantize_layers()
        self.quantize_activations()
        self.pbar.close()
        return self.model

