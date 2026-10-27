Main Project Report 
------------------------------------------------------------------------
Tools :
Python : 3.11
Anaconda 
VsCode
CUDA Available: True
GPU Device: NVIDIA GeForce RTX 3050 Ti Laptop GPU

Loading datasets...
Train Images Shape: (586500, 28, 28)
Train Labels Shape: (586500,)
Number of Classes: 345
Sample Classes: ['The Eiffel Tower' 'The Great Wall of China' 'The Mona Lisa'
 'aircraft carrier' 'airplane']
---------------------------------------------------------------------------
Second Step : Data Preprocessing 
Loading data from data/train_images.npy...
Total training samples: 586500
Total batches per epoch: 4583
Batch Images Shape (Batch, Channel, Height, Width): torch.Size([128, 1, 28, 28])
Batch Labels Shape: torch.Size([128])

---------------------------------------------------------------------------
Model Class
Convolution Small filters using (CNN)
Activation Function: ReLU  For non-linearity 
MaxPooling2d: shrinks image 
Fully Connected Layers: 345 output values 
Feature Extractor: Scans the 28x28 image
First convolution= in-> 1 out-> 32 krenel_size=3, padding =1
Max Pool= kernel size = 2 
stride = 2
Second Convolution in-> 32 , out -> 64 krenel_size=3, padding =1
Max Pool= kernel_size=2, stride=2
Flattens 64 * 7 * 7 into a 3136-element vector
----------------------------------------------------------------------------
Train Class
Forward Pass
Loss Calculation= Cross Entropy Loss(Compares the model's prediction against the true label to see how wrong it was.)
Backward Pass= Calculates gradients
Optimizer= Updates the model weights to make fewer mistakes next time.

First Evaluation completed:103500 total samples 
Top-1 Accuracy=53.86%
Top-5 Accuracy=73.48%


