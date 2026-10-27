import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# 1. Create a Custom PyTorch Dataset Class
# WHY: PyTorch needs a structured way to fetch one image-label pair at a time.
class QuickDrawDataset(Dataset):

  def __init__(self, images_path, labels_path):
    # Load data from .npy files into memory
    print(f'Loading data from {images_path}...')
    self.images = np.load(images_path)
    self.labels = np.load(labels_path)

  def __len__(self):
    # Returns the total number of samples in the dataset
    return len(self.images)

  def __getitem__(self, idx):
    # HOW: This method fetches a single sample by its index `idx`
    image = self.images[idx]
    label = self.labels[idx]

    # Preprocessing:
    # 1. Reshape image from (28, 28) to (1, 28, 28) -> PyTorch CNNs expect a channel dimension (Channels, Height, Width)
    # 2. Normalize pixel values from [0, 255] to [0.0, 1.0] by dividing by 255.0
    image = (
        torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
    )  # unsqueeze(0) adds the channel dim

    # Convert label to a torch integer tensor
    label = torch.tensor(label, dtype=torch.long)

    return image, label


# Test the dataset if this file is run directly
if __name__ == '__main__':
  # Instantiate the dataset
  data='data/'
  labels='labels/'
  train_dataset = QuickDrawDataset(data+'train_images.npy', labels+'train_labels.npy')

  # Create a DataLoader
  # WHY: DataLoader automatically handles batching, shuffling, and multi-threaded loading.
  batch_size = 128
  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
  )

  print(f'Total training samples: {len(train_dataset)}')
  print(f'Total batches per epoch: {len(train_loader)}')

  # Grab one batch to inspect shapes
  sample_images, sample_labels = next(iter(train_loader))
  print('Batch Images Shape (Batch, Channel, Height, Width):', sample_images.shape)
  print('Batch Labels Shape:', sample_labels.shape)