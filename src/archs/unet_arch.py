import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# conv blocks at one level
class convBlock(nn.Module):
  def __init__(self, in_channels, num_features):
    super(convBlock, self).__init__()

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=num_features, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=num_features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=num_features),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.conv(x)

# Unet
class UNET(nn.Module):
  def __init__(
      self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
  ):
    super(UNET, self).__init__()

    self.contract = nn.ModuleList()
    self.expand = nn.ModuleList()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Contracting Path
    for feature in features:
      self.contract.append(convBlock(in_channels, feature))
      in_channels = feature

    # Expanding Path
    for feature in reversed(features):
      self.expand.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
      self.expand.append(convBlock(feature*2, feature))

    # Bottleneck layer
    self.bottleneck = convBlock(features[-1], features[-1]*2)
    self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

  def forward(self, x):
    skip_connections = []

    for layer in self.contract:
      x = layer(x)
      skip_connections.append(x)
      x = self.pool(x)

    x = self.bottleneck(x)
    skip_connections = skip_connections[::-1]

    for i in range(0, len(self.expand), 2):
      x = self.expand[i](x)
      skip = skip_connections[i//2]

      if x.shape!=skip.shape:
        x = TF.resize(x, size=skip.shape[2:])

      concat_skip = torch.cat((skip, x), dim=1)
      x = self.expand[i+1](concat_skip)

    return self.final_conv(x)