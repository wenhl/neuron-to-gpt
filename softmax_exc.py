#import math
from micrograd.gradient_ops import Value


import torch

def verify_gradients_with_torch(custom_gradients, logits):

    # Convert to PyTorch tensors and enable gradient tracking
    logits_torch = torch.tensor(logits, dtype=torch.float32, requires_grad=True)

    # Softmax in PyTorch
    probs_torch = torch.softmax(logits_torch, dim=0)

    # Negative log likelihood loss for a specified class (e.g., class 3 here)
    loss_torch = -probs_torch[3].log()

    # Perform backpropagation
    loss_torch.backward()
    print(f"--- verify with torch libraray {loss_torch.data}")
    # Print and compare gradients
    #print(f"PyTorch Gradients: {logits_torch.grad}")
    #print(f"Custom Gradients: {custom_gradients}")


    # Verifying the gradients
    for dim, (custom_grad, torch_grad) in enumerate(zip(custom_gradients, logits_torch.grad)):
        ok = 'OK' if abs(custom_grad - torch_grad.item()) < 1e-5 else 'WRONG!'
        print(f"{ok} for dim {dim}: expected {custom_grad}, yours returns {torch_grad.item()}")


def main():
    def softmax(logits):
        counts = [logit.exp() for logit in logits]
        denominator = sum(counts)
        out = [c/denominator for c in counts]
        return out

    # this is the negative log likelihood loss function, pervasive in classification
    a, b, c, d  = 0.0, 3.0, -2.0, 1.0
    logits = [Value(a), Value(b), Value(c), Value(d)]
    probs = softmax(logits)
    loss = -probs[3].log() # dim 3 acts as the label for this input example
    loss.backward()
    print(f"--- verify with cusotm implementation {loss.data}")

    ans = [0.041772570515350445, 0.8390245074625319, 0.005653302662216329, -0.8864503806400986]
    for dim in range(4):
        ok = 'OK' if abs(logits[dim].grad - ans[dim]) < 1e-5 else 'WRONG!'
        print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {logits[dim].grad}")

    logits_t = [a, b, c, d]
    verify_gradients_with_torch(ans, logits_t)

if __name__ == "__main__":
    main()


