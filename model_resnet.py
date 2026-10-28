import torch
import torch.nn as nn
import torchvision.models as models


class QuickDrawResNet(nn.Module):

  def __init__(self, num_classes=345):
    super(QuickDrawResNet, self).__init__()

    # Load pretrained ResNet-18 weights from torchvision
    # (Note: In newer torchvision versions, 'weights=models.ResNet18_Weights.DEFAULT' is standard)
    self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 1. Modify the first convolutional layer to accept 1-channel grayscale instead of 3 RGB channels
    # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.resnet.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # 2. Modify the final fully connected layer to output 345 classes instead of 1000
    in_features = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(in_features, num_classes)

  def forward(self, x):
    # Optional safety check: if input is 28x28, ResNet downsamples too quickly in early layers.
    # We can use functional interpolation to upscale 28x28 -> 64x64 for better feature extraction.
    if x.shape[2] == 28 and x.shape[3] == 28:
      x = nn.functional.interpolate(
          x, size=(64, 64), mode='bilinear', align_corners=False
      )

    return self.resnet(x)


if __name__ == '__main__':
  # Test the architecture with a dummy batch
  model = QuickDrawResNet(num_classes=345)
  dummy_input = torch.randn(4, 1, 28, 28)
  output = model(dummy_input)
  print('ResNet-18 Adapted Output Shape:', output.shape)  # Expected: torch.Size([4, 345])