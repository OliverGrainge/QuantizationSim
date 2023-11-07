import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from quantizer import Quantizer


model = models.resnet18(pretrained=True)
qmodel = models.resnet18(pretrained=True)


class CalibrationDataset(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images
    
    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        return self.images[idx]

cal_ds = CalibrationDataset(torch.randn(20, 3, 480, 640))
cal_dl = DataLoader(cal_ds, batch_size=10)

quantizer = Quantizer(qmodel,
    layer_precision="int8",
    activation_precision="fp16",
    activation_granularity="tensor",
    layer_granularity="channel",
    calibration_type="minmax", 
    calibration_loader=cal_dl)

x = torch.randn(1, 3, 480, 640)

quantizer.collect_activations()
qmodel = quantizer.quantize_activations()
qmodel = quantizer.quantize_layers()

print(qmodel)