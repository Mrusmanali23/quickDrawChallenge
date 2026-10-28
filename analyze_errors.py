import numpy as np
import torch
from dataset import QuickDrawDataset
from model import QuickDrawCNN
from torch.utils.data import DataLoader


def analyze_per_class_performance():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')

  # Load validation dataset
  val_dataset = QuickDrawDataset('data/val_images.npy', 'labels/val_labels.npy')
  val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

  # Load best saved weights
  model = QuickDrawCNN(num_classes=345).to(device)
  model.load_state_dict(torch.load('quickdraw_cnn.pth'))
  model.eval()

  # Track correct and total samples per class (345 classes)
  class_correct = {i: 0 for i in range(345)}
  class_total = {i: 0 for i in range(345)}

  print('Analyzing per-class accuracy on validation set...')
  with torch.no_grad():
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)

      for label, prediction in zip(labels, predicted):
        cls = label.item()
        class_total[cls] += 1
        if label == prediction:
          class_correct[cls] += 1

  # Compute accuracy per class
  class_accuracies = {}
  for cls in range(345):
    if class_total[cls] > 0:
      class_accuracies[cls] = 100 * class_correct[cls] / class_total[cls]
    else:
      class_accuracies[cls] = 0.0

  # Sort classes by accuracy
  sorted_classes = sorted(class_accuracies.items(), key=lambda item: item[1])

  print('\n--- 10 Worst Performing Classes ---')
  for cls, acc in sorted_classes[:10]:
    print(f'Class ID {cls}: {acc:.2f}% accuracy ({class_correct[cls]}/{class_total[cls]} correct)')

  print('\n--- 10 Best Performing Classes ---')
  for cls, acc in sorted_classes[-10:]:
    print(f'Class ID {cls}: {acc:.2f}% accuracy ({class_correct[cls]}/{class_total[cls]} correct)')


if __name__ == '__main__':
  analyze_per_class_performance()