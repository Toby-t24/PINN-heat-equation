import numpy as np

# Tanh(x) is used here as the activation function for the hidden layers

def tanh(x):
    
    return np.tanh(x)


def tanh_deriv(x):
    
    return 1 - np.tanh(x)**2



def init_params():
    
    """ initial parameters"""

    W1 = np.random.randn(64, 2)   * np.sqrt(2/2)
    b1 = np.zeros((64, 1))
    
    W2 = np.random.randn(64, 64)  * np.sqrt(2/64)
    b2 = np.zeros((64, 1))
    
    W3 = np.random.randn(1, 64)   * np.sqrt(2/64)
    b3 = np.zeros((1, 1))

    return W1, b1, W2, b2, W3, b3


def forward_propagation(W1, b1, W2, b2, W3, b3, X):

    """using given weights and biases to propagate through the basic network"""
    
    Z1 = W1.dot(X) + b1
    A1 = tanh(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = tanh(Z2)
    
    Z3 = W3.dot(A2) + b3
    A3 = Z3
    
    return Z1, A1, Z2, A2, Z3, A3


W1, b1, W2, b2, W3, b3 = init_params()



def back_propagation(W1, b1, W2, b2, W3, b3, X, Y):
    
    """
    back propagation through the basic network, 
    determining how much each weight and bias affects the loss function
    
    W1, W2, W3: weight matrices
    b1, b2, b3: bias vectors
    X: input data
    Y: target data
    """

    m = Y.shape[1]
    
    Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
    
    CA3 = A3 - Y
    
    dW3 = 1 / m * CA3.dot(A2.T)
    db3 = 1 / m * np.sum(CA3, axis=1, keepdims=True)
    
    CA2 = W3.T.dot(CA3) * tanh_deriv(Z2)
    
    dW2 = 1 / m * CA2.dot(A1.T)
    db2 = 1 / m * np.sum(CA2, axis=1, keepdims=True)
    
    CA1 = W2.T.dot(CA2) * tanh_deriv(Z1)
    
    dW1 = 1 / m * CA1.dot(X.T)
    db1 = 1 / m * np.sum(CA1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3


def update_param(W1, b1, W2, b2, W3, b3,
                 dW1, db1, dW2, db2, dW3, db3, rate):
    
    """update the weights and biases based on back propagation output"""

    W1 = W1 - rate * dW1
    b1 = b1 - rate * db1
    
    W2 = W2 - rate * dW2
    b2 = b2 - rate * db2
    
    W3 = W3 - rate * dW3
    b3 = b3 - rate * db3
    
    return W1, b1, W2, b2, W3, b3


def gradient_descent(X, Y, lr_0, iterations):
    
    """
    minimising the loss function by repeatedly forward and backward propagating,
    updating parameters so that Loss decreases
     """

    W1, b1, W2, b2, W3, b3 = init_params()
    
    for i in range(iterations):
        
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(W1, b1, W2, b2, W3, b3, X)
        
        dW1, db1, dW2, db2, dW3, db3 = back_propagation(W1, b1, W2, b2, W3, b3, X, Y)
        
        lr = lr_0 / (1 + i * 0.001) # decreases learning rate over iterations
        
        W1, b1, W2, b2, W3, b3 = update_param(
            W1, b1, W2, b2, W3, b3,
            dW1, db1, dW2, db2, dW3, db3, lr
        )
        
        if i % 100 == 0:
            
            loss = np.mean((A3 - Y)**2) # calculates the loss every 100 iterations
            
            print(f"Iteration {i}, Loss: {loss:.6f}")
    
    return W1, b1, W2, b2, W3, b3
