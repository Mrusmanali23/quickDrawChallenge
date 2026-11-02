import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# 1. Create a Custom PyTorch Dataset Class
# WHY: PyTorch needs a structured way to fetch one image-label pair at a time.
class QuickDrawDataset(Dataset):

  def __init__(self, images_path, labels_path, is_train=False):
    print(f'Loading data from {images_path}...')
    self.images = np.load(images_path)
    self.labels = np.load(labels_path)
    self.is_train = is_train

    # Define professional augmentation pipeline for training data
    # Quick, Draw! doodles benefit heavily from slight rotation, translation, and scaling.
    self.train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
    ])

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]
    label = self.labels[idx]

    # Reshape flat 784 arrays back to 28x28 if necessary
    if image.shape == (784,):
      image = image.reshape(28, 28)

    # Convert to float32 tensor and add channel dim (1, 28, 28)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

    # Apply data augmentation only during training
    if self.is_train:
      # To apply torchvision transforms, we convert temporarily to PIL or use tensor transforms
      # Here we let our train_transform handle it cleanly:
      image_pil = transforms.ToPILImage()(image_tensor)
      image_tensor = self.train_transform(image_pil)

    label_tensor = torch.tensor(label, dtype=torch.long)

    return image_tensor, label_tensor

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