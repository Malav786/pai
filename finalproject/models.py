import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super().__init__()
        c = 32
        self.conv1 = nn.Conv2d(in_channels, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c*2, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Sequential(nn.Flatten(), 
                                nn.Linear((c*2)*(int(62*0.5)**2), 256), # adjust size to image dims nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(256, num_classes)
                               ) 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.fc(x)
        return x 