import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import QuickDrawDataset
from model_advanced import QuickDrawAdvancedModel
from torch.utils.data import DataLoader


def train_advanced_model():
  # 1. Device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')

  # 2. Load Datasets & Dataloaders
  print('Loading datasets into memory...')
  train_dataset = QuickDrawDataset(
      'data/train_images.npy', 'labels/train_labels.npy', is_train=True
  )
  val_dataset = QuickDrawDataset(
      'data/val_images.npy', 'labels/val_labels.npy', is_train=False
  )

  # Using batch size 64 to safely fit EfficientNet-B0 with 128x128 inputs into 4GB VRAM
  train_loader = DataLoader(
      train_dataset, batch_size=64, shuffle=True, num_workers=0
  )
  val_loader = DataLoader(
      val_dataset, batch_size=128, shuffle=False, num_workers=0
  )

  # 3. Initialize Advanced Model (EfficientNet-B0), Loss, Optimizer, and Scheduler
  model = QuickDrawAdvancedModel(
      num_classes=345, model_name='efficientnet_b0'
  ).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=2
  )

  num_epochs = 15
  best_val_acc = 0.0

  print(
      f'Starting Advanced EfficientNet-B0 training for {num_epochs} epochs...'
  )

  for epoch in range(num_epochs):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if (batch_idx + 1) % 1500 == 0:
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Batch'
            f' [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}'
        )

    epoch_time = time.time() - start_time
    train_loss = running_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    correct_top1 = 0
    total = 0

    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct_top1 += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct_top1 / total

    scheduler.step(val_loss)

    print(
        f'--> Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s |'
        f' Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val'
        f' Top-1 Acc: {val_acc:.2f}%'
    )

    # --- MODEL CHECKPOINTING ---
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), 'quickdraw_efficientnet.pth')
      print(
          f'>>> Saving new best EfficientNet checkpoint with Validation'
          f' Accuracy: {val_acc:.2f}%\n'
      )
    else:
      print(f'    (Accuracy did not improve from best: {best_val_acc:.2f}%)\n')

  print(
      f'Training complete! Best EfficientNet model saved as'
      f' quickdraw_efficientnet.pth with {best_val_acc:.2f}% accuracy.'
  )


if __name__ == '__main__':
  train_advanced_model()