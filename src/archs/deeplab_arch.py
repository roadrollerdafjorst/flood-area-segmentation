import torch
import torchvision

def DeepLabV3(in_channels=256, out_channels=1):
    deeplab = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    deeplab.classifier[4] = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
    
    return deeplab