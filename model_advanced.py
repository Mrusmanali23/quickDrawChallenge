import torch
import torch.nn as nn
import torchvision.models as models


class QuickDrawAdvancedModel(nn.Module):

  def __init__(self, num_classes=345, model_name='efficientnet_b0'):
    super(QuickDrawAdvancedModel, self).__init__()

    if model_name == 'efficientnet_b0':
      # Load pretrained EfficientNet-B0
      self.backbone = models.efficientnet_b0(
          weights=models.EfficientNet_B0_Weights.DEFAULT
      )
      # Modify first conv layer to accept 1 channel (grayscale)
      # EfficientNet's first conv is at backbone.features[0][0]
      original_conv = self.backbone.features[0][0]
      self.backbone.features[0][0] = nn.Conv2d(
          1,
          original_conv.out_channels,
          kernel_size=original_conv.kernel_size,
          stride=original_conv.stride,
          padding=original_conv.padding,
          bias=False,
      )
      # Modify classifier head
      in_features = self.backbone.classifier[1].in_features
      self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'resnet50':
      self.backbone = models.resnet50(
          weights=models.ResNet50_Weights.DEFAULT
      )
      self.backbone.conv1 = nn.Conv2d(
          1, 64, kernel_size=7, stride=2, padding=3, bias=False
      )
      in_features = self.backbone.fc.in_features
      self.backbone.fc = nn.Linear(in_features, num_classes)

    self.model_name = model_name

  def forward(self, x):
    # Upscale 28x28 to 128x128 so advanced backbones can extract rich stroke features
    if x.shape[2] == 28 and x.shape[3] == 28:
      x = nn.functional.interpolate(
          x, size=(128, 128), mode='bicubic', align_corners=False
      )
    return self.backbone(x)


if __name__ == '__main__':
  model = QuickDrawAdvancedModel(num_classes=345, model_name='efficientnet_b0')
  dummy = torch.randn(2, 1, 28, 28)
  print(f'Output shape: {model(dummy).shape}')  # Expected: torch.Size([2, 345])