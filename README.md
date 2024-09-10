# Multi-Layer Perceptron (MLP) - Machine Learning Project
This project implements a basic Multi-Layer Perceptron (MLP) for binary classification using a feedforward neural network with backpropagation. The network consists of input nodes, hidden nodes, and output nodes, and is trained using gradient descent.

## Project Overview
Training the Network: Using training data to adjust the weights of the network to minimize the error. <br />
Testing the Network: Evaluating the performance of the trained network on test data. <br />
Graphing Results: Generating a graph to visualize the training progress.

## How It Works
### 1. Network Setup

The network consists of multiple layers:

Input Layer: Contains InputNode instances. <br />
Hidden Layer: Contains HiddenNode instances. <br />
Output Layer: Contains OutputNode instances. <br />
Bias Nodes: Added to each layer to provide additional degrees of freedom to the network.
### 2. Training

Forward Propagation: Calculates the output of each node by passing inputs through the network. <br />
Backward Propagation: Adjusts weights using gradient descent to minimize the error. <br />
Error Calculation: Computes the squared error to evaluate performance and guide the weight adjustments.
### 3. Testing

After training, the network is tested with new data to evaluate its performance. The results are printed to the console.

### 4. Graphing

The training error for each epoch is saved to a file and used to generate a graph, which visualizes how the network's error decreases over time.
 
## How To Run
Open and run the .sln file in Visual Studio.
