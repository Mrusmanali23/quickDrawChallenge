import torch
import torch.nn as nn


class QuickDrawCNN(nn.Module):

  def __init__(self, num_classes=345):
    super(QuickDrawCNN, self).__init__()

    # Feature Extractor: Scans the 28x28 image for lines and curves
    self.feature_extractor = nn.Sequential(
        # Input shape: (Batch, 1, 28, 28)
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Shrinks to (32, 14, 14)
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Shrinks to (64, 7, 7)
    )

    # Classifier: Converts extracted features into class scores
    self.classifier = nn.Sequential(
        nn.Flatten(),  # Flattens 64 * 7 * 7 into a 3136-element vector
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.3),  # Prevents overfitting by dropping 30% of random weights
        nn.Linear(128, num_classes),  # Final output layer for 345 classes
    )

  def forward(self, x):
    # Pass input through feature extractor, then classifier
    x = self.feature_extractor(x)
    x = self.classifier(x)
    return x


if __name__ == '__main__':
  # Test the model structure with dummy data
  model = QuickDrawCNN(num_classes=345)
  dummy_input = torch.randn(16, 1, 28, 28)  # Batch of 16 random images
  output = model(dummy_input)
  print('Test Model Output Shape:', output.shape)  # Expected: torch.Size([16, 345])