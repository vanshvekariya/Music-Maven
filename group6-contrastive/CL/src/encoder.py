import torch.nn.functional as F
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.projection_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x, return_features=False):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        if return_features:
            return F.normalize(x, dim=1)  

        x = self.projection_head(x)
        x = F.normalize(x, dim=1)
        return x