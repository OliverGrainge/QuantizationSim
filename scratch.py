import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from quantizer import Quantizer

# Define a simple 3-layer CNN with input shape 3x480x640
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Layer 1 - Convolution Layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 2 - Convolution Layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Since the input image is larger (480x640), we need to add one more convolution and pooling layer
        # Layer 3 - Convolution Layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # After 3 layers of pooling, the feature map size is 60x80, so we add a fully connected layer.
        # We will flatten the 128x60x80 feature map to connect it to the fully connected layer.
        self.fc = nn.Linear(in_features=128 * 60 * 80, out_features=10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x

# Create the model instance with the input shape of 3x480x640
model = SimpleCNN()
qmodel = SimpleCNN()


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
    calibration_type="percentile", 
    calibration_loader=cal_dl,
    activation_layers = (nn.ReLU, nn.MaxPool2d, nn.Conv2d))

x = torch.randn(1, 3, 480, 640)

qmodel = quantizer.fit()