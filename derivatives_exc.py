import math
from math import sin, cos


def main():

    # Define the function f(a, b, c)
    def f(a, b, c):
        return -a**3 + sin(3*b) - 1.0/c + b**2.5 - a**0.5

    # Compute the gradient of f analytically
    def gradf(a, b, c):
        # Partial derivative with respect to a
        df_da = -3*a**2 - 0.5/math.sqrt(a)
        # Partial derivative with respect to b
        df_db = 3 * math.cos(3 * b) + 2.5 * b**1.5
        # Partial derivative with respect to c
        df_dc = 1 / c**2
        return df_da, df_db, df_dc

     # Compute the gradient of f numerically (forward difference)
    def numerical_grad(a, b, c):
        h=0.00001
        grad_a = (f(a + h, b, c) - f(a, b, c)) / h
        grad_b = (f(a, b + h, c) - f(a, b, c)) / h
        grad_c = (f(a, b, c + h) - f(a, b, c)) / h
        return [grad_a, grad_b, grad_c]

    # Pre-computed expected answers for gradient
    ans = [-12.353553390593273, 10.25699027111255, 0.0625]
    a, b, c = 2, 3, 4
    
    # Compare analytical gradient with expected answers
    yours = gradf(a, b, c)
    for dim in range(3):
        ok = 'OK' if abs(yours[dim] - ans[dim]) < 1e-5 else 'WRONG!'
        print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {yours[dim]}")

    # Compare numerical gradient (forward difference) with expected answers
    grad = numerical_grad(a,b,c)

    for dim in range(3):
        ok = 'OK' if abs(grad[dim] - ans[dim]) < 1e-5 else 'WRONG!'
        print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {grad[dim]}")


    # Compute the gradient of f numerically (central difference)
    def numerical_grad2(a, b, c, h=0.0001):
        grad_a = (f(a + h, b, c) - f(a - h, b, c)) / (2 * h)
        grad_b = (f(a, b + h, c) - f(a, b - h, c)) / (2 * h)
        grad_c = (f(a, b, c + h) - f(a, b, c - h)) / (2 * h)
        return [grad_a, grad_b, grad_c]

    # Compare numerical gradient (central difference) with expected answers
    grad2 = numerical_grad2(a, b, c)

    for dim in range(3):
        ok = 'OK' if abs(grad2[dim] - ans[dim]) < 1e-5 else 'WRONG!'
        print(f"{ok} for dim {dim}: expected {ans[dim]}, yours returns {grad2[dim]}")

if __name__ == "__main__":
    main()

