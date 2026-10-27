import numpy as np
import torch

# 1. Verify GPU Access for your RTX 3050 Ti
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
  print("GPU Device:", torch.cuda.get_device_name(0))

# 2. Load the .npy files
print('\nLoading datasets...')
train_images = np.load('data/train_images.npy')
train_labels = np.load('labels/train_labels.npy')
classes = np.load('classes.npy', allow_pickle=True)

# 3. Print shapes and info
print('Train Images Shape:', train_images.shape)  # Expected: (N, 28, 28)
print('Train Labels Shape:', train_labels.shape)
print('Number of Classes:', len(classes))
print('Sample Classes:', classes[:5])