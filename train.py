import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import QuickDrawDataset
from model import QuickDrawCNN
from torch.utils.data import DataLoader


def train_model():
  # 1. Device configuration: Automatically detects your RTX 3050 Ti GPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')

  # 2. Load Dataset & Dataloader (uses your dataset.py)
  print('Loading datasets into memory...')
  train_dataset = QuickDrawDataset('data/train_images.npy', 'labels/train_labels.npy')
  val_dataset = QuickDrawDataset('data/val_images.npy', 'labels/val_labels.npy')

  batch_size = 128  # Number of images processed simultaneously in VRAM
  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
  )
  val_loader = DataLoader(
      val_dataset, batch_size=256, shuffle=False, num_workers=0
  )
  # 3. Initialize Model, Loss Function, and Optimizer
  model = QuickDrawCNN(num_classes=345).to(device)
  criterion = (
      nn.CrossEntropyLoss()
  )  # Measures error for multi-class classification
  optimizer = optim.Adam(
      model.parameters(), lr=0.001
  )  # Adjusts weights efficiently

# WHY ReduceLROnPlateau? If validation loss stops improving, it drops the learning rate (e.g., by multiplying by 0.5)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=2
  )

  # 4. Training Loop Execution
  num_epochs = 20  # Start with 3 epochs to test training speed on your laptop
  best_val_acc = 0.0
  print(
      f'Starting training for {num_epochs} epochs on {len(train_dataset)}'
      ' samples...'
  )

  for epoch in range(num_epochs):
    model.train()  # Put model in training mode
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
      # Move batch tensors to GPU memory
      images, labels = images.to(device), labels.to(device)

      # Step A: Clear old gradients from previous step
      optimizer.zero_grad()

      # Step B: Forward pass (Get predictions)
      outputs = model(images)

      # Step C: Compute error (Loss)
      loss = criterion(outputs, labels)

      # Step D: Backward pass (Compute gradients)
      loss.backward()

      # Step E: Update weights
      optimizer.step()

      running_loss += loss.item()

      # Print progress every 1000 batches
    #   if (batch_idx + 1) % 1000 == 0:
    #     print(
    #         f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}'
    #     )

    epoch_time = time.time() - start_time    
    train_loss = running_loss / len(train_loader)
    print(
        f'--> Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds'
        f' | Average Loss: {train_loss:.4f}\n'
    )

# --- VALIDATION PHASE (To check accuracy and save the best model) ---
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

    # Step the learning rate scheduler based on validation loss
    scheduler.step(val_loss)

    print(
        f'--> Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s |'
        f' Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val'
        f' Top-1 Acc: {val_acc:.2f}%'
    )

# --- MODEL CHECKPOINTING ---
    # WHY: Save weights ONLY if this epoch achieved a higher validation accuracy than ever before
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), 'quickdraw_cnn.pth')
      print(
          f'>>> Saving new best model checkpoint with Validation Accuracy:'
          f' {val_acc:.2f}%\n'
      )
    else:
      print(f'    (Accuracy did not improve from best: {best_val_acc:.2f}%)\n')

  print(
      f'Training complete! Best model saved as quickdraw_cnn.pth with'
      f' {best_val_acc:.2f}% accuracy.'
  )

if __name__ == '__main__':
  train_model()