import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from quantizer import Quantizer
import torch.nn.functional as F

from activation_quantization import ActivationCollector

qmodel = models.MobileNetV2().eval()

print(qmodel)
class CalibrationDataset(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images
    
    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx]

cal_ds = CalibrationDataset(torch.randn(1, 3, 224, 224))
cal_dl = DataLoader(cal_ds, batch_size=1)

quantizer = Quantizer(qmodel,
    layer_precision="int8",
    activation_precision="fp16",
    activation_granularity="tensor",
    layer_granularity="tensor",
    calibration_type="mse", 
    calibration_loader=cal_dl,
    activation_layers = (nn.ReLU, nn.ReLU6, nn.BatchNorm2d))

x = torch.randn(1, 3, 480, 640)

qmodel = quantizer.fit()
print("Layer Quantized ", quantizer.layers_quantized)
print("Layer Count", quantizer.num_layers())
print("Activation Count", quantizer.num_activations())
print("Activations Quantized", quantizer.activations_quantized)
print("Number Activation Collected:", len(quantizer.activation_data))

