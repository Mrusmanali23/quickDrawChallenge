import numpy as np
import torch
from dataset import QuickDrawDataset  # Reusing our dataset class
from model import QuickDrawCNN
from torch.utils.data import DataLoader


def evaluate_model():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')

  # 1. Load Validation Dataset
  print('Loading validation data...')
  val_dataset = QuickDrawDataset('data/val_images.npy', 'labels/val_labels.npy')
  val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

  # 2. Load Trained Model Weights
  model = QuickDrawCNN(num_classes=345).to(device)
  model.load_state_dict(torch.load('quickdraw_cnn.pth'))
  model.eval()  # Put model in evaluation mode (turns off dropout)

  correct_top1 = 0
  correct_top5 = 0
  total = 0

  print('Evaluating model on validation set...')
  with torch.no_grad():  # No need to compute gradients during evaluation (saves VRAM & speed)
    for images, labels in val_loader:
      images, labels = images.to(device), labels.to(device)

      outputs = model(images)

      # Top-1 Accuracy
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct_top1 += (predicted == labels).sum().item()

      # Top-5 Accuracy
      _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
      labels_expanded = labels.view(-1, 1).expand_as(top5_pred)
      correct_top5 += (top5_pred == labels_expanded).sum().item()

  top1_acc = 100 * correct_top1 / total
  top5_acc = 100 * correct_top5 / total

  print(f'\n--- Evaluation Results ---')
  print(f'Total Validation Samples: {total}')
  print(f'Top-1 Accuracy: {top1_acc:.2f}%')
  print(f'Top-5 Accuracy: {top5_acc:.2f}%')


if __name__ == '__main__':
  evaluate_model()