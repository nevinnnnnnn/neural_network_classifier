# neural_network_classifier
This project demonstrates how to build a simple neural network from scratch using NumPy to classify images of the letters A, B, and C. The goal is to understand the fundamental concepts of neural networks—forward and backward propagation, activation functions, and gradient descent—without relying on high-level deep learning libraries like TensorFlow or PyTorch.

Approach
1. Creating the Dataset
We start by defining the base patterns for the letters A, B, and C as 5×6 binary grids (where 1 represents a black pixel and 0 represents white). To make the problem more realistic, we generate training and test datasets by adding random noise (flipping some pixels) to these base patterns. This helps the model generalize better rather than just memorizing exact shapes.

Training Data: 300 samples (100 per letter) with slight noise.

Test Data: 30 samples (10 per letter) with different noisy variations.

Each label is one-hot encoded (e.g., A = [1, 0, 0], B = [0, 1, 0], C = [0, 0, 1]).

2. Building the Neural Network
The neural network consists of:

Input Layer: 30 neurons (flattened 5×6 image).

Hidden Layer: 10 neurons with sigmoid activation to introduce non-linearity.

Output Layer: 3 neurons with softmax activation, producing probabilities for each class (A, B, or C).

Weights are initialized with small random values, and biases start at zero.

3. Training the Model
The model learns by iteratively adjusting its weights and biases using backpropagation:

Forward Pass: The input passes through the hidden layer (sigmoid) and output layer (softmax) to compute predictions.

Loss Calculation: Cross-entropy loss measures how far predictions are from true labels.

Backward Pass: Gradients of the loss with respect to weights and biases are computed.

Gradient Descent: Weights and biases are updated to minimize loss (learning rate = 0.1).

Training runs for 1000 epochs, with loss and accuracy tracked at each step.

4. Evaluating Performance
After training, the model is tested on unseen noisy data to check its accuracy. We also visualize:

Training Progress: Graphs of loss and accuracy over epochs.

Base Patterns: The original noise-free shapes of A, B, and C.

Test Predictions: A sample test image is shown alongside its predicted and true labels.

Why This Matters
This project provides hands-on insight into how neural networks work under the hood. By implementing key components—matrix operations, activation functions, gradient descent, and backpropagation—from scratch, we gain a deeper understanding of deep learning fundamentals.

The techniques used here form the basis of more complex models, making this a valuable learning exercise for anyone entering machine learning.

Tools Used: NumPy, Matplotlib
Concepts Covered: Neural Networks, Backpropagation, Activation Functions, Gradient Descent, Image Classification
