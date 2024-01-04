import random
from micrograd.gradient_ops import Value


class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  def parameters(self):
        return []

class Neuron(Module):
  
  def __init__(self, nin, nonlin=True, activation='relu'):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
    self.nonlin = nonlin
    self.activation = activation
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    if self.activation == 'relu':
      return act.relu() if self.nonlin else act
    elif self.activation == 'tanh':
      return act.tanh() if self.nonlin else act
  
  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    #return f"Neuron(nin={len(self.w)})"
    return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    #return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
  def __init__(self, nin, nout, nonlin=True, activation='relu'):
    self.neurons = [Neuron(nin, nonlin, activation) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

  def __repr__(self):
      return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
      
class MLP(Module):
  def __init__(self, nin, nouts, activation='relu'):
    sz = [nin] + nouts
    #self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1,  activation=activation) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  
  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"