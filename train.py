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
  train_dataset = QuickDrawDataset('train_images.npy', 'train_labels.npy')

  batch_size = 128  # Number of images processed simultaneously in VRAM
  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  # 3. Initialize Model, Loss Function, and Optimizer
  model = QuickDrawCNN(num_classes=345).to(device)
  criterion = (
      nn.CrossEntropyLoss()
  )  # Measures error for multi-class classification
  optimizer = optim.Adam(
      model.parameters(), lr=0.001
  )  # Adjusts weights efficiently

  # 4. Training Loop Execution
  num_epochs = 3  # Start with 3 epochs to test training speed on your laptop
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
      if (batch_idx + 1) % 1000 == 0:
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}'
        )

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    print(
        f'--> Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds'
        f' | Average Loss: {avg_loss:.4f}\n'
    )

  # Save the learned weights to a file
  torch.save(model.state_dict(), 'quickdraw_cnn.pth')
  print('Training complete! Model weights saved as quickdraw_cnn.pth')


if __name__ == '__main__':
  train_model()