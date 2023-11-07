import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from quantizer import Quantizer

# Define a simple 3-layer CNN with input shape 3x480x640
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

cal_ds = CalibrationDataset(torch.randn(20, 3, 224, 224))
cal_dl = DataLoader(cal_ds, batch_size=10)

quantizer = Quantizer(qmodel,
    layer_precision="int8",
    activation_precision="fp16",
    activation_granularity="tensor",
    layer_granularity="channel",
    calibration_type="entropy", 
    calibration_loader=cal_dl,
    activation_layers = (nn.ReLU, nn.MaxPool2d, nn.Conv2d))

x = torch.randn(1, 3, 480, 640)

qmodel = quantizer.fit()

out = model(x).flatten().detach().numpy()
out1 = qmodel(x).flatten().detach().numpy()

#for i in range(len(out)):
#    print(out[i], out1[i])

