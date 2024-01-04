#import math
import random
import numpy as np
from micrograd.gradient_ops import Value
from micrograd.neural_network import MLP
from micrograd.neural_network import Neuron
import micrograd.plotter  as plotter


def main():
    
    # a very simple example
    x = Value(1.0)
    y = (x * 2 + 1).relu()
    y.backward()
    plotter.draw_dot(y)

    # a simple 2D neuron
    random.seed(1337)
    n = Neuron(2)
    x = [Value(1.0), Value(-2.0)]
    y = n(x)
    y.backward()

    plotter.draw_dot(y)
    
    # Define the input data (xs) and the corresponding target outputs (ys)
    xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets

    # Initialize a Multi-Layer Perceptron (MLP) with 3 input nodes, two hidden layers with 4 nodes each, and 1 output node
    n = MLP(3, [4, 4, 1], activation='tanh') 
    print(n)

    print(f"# of parameters= {len(n.parameters())}")

    for k in range(20):
       # Forward pass: Compute predictions for each input
        ypred = [n(x) for x in xs]
        
        # Compute the loss as the sum of squared differences between predictions and targets
        #loss = sum((Value(yout) - Value(ygt))**2 for ygt, yout in zip(ys, ypred))
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
        # Draw the computational graph for the first iteration
        plotter.draw_dot(loss) if k == 0 else None
      
        # Backward pass: Reset gradients to zero before backpropagation
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # Print gradients for debugging
        #for i, p in enumerate(n.parameters()):
        #    print(f"Param {i}, data {p.data}, Gradient: {p.grad}")

        # update
        for p in n.parameters():
            #print(f"data {p.data}, Gradient: {p.grad}")
            p.data += -0.1 * p.grad
            #print(f"data {p.data}, Gradient: {p.grad}")

        print(f"Iteration {k}, after change: {loss.data}")
        
    print(f"target {ys}")
    ypred_data = [v.data for v in ypred]
    print(f"result {ypred_data}")

  
if __name__ == "__main__":
    main()

