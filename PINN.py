import numpy as np



def tanh(x):
  return np.tanh(x)

def tanh_deriv1(x):
  return 1 - np.tanh(x)**2

def tanh_deriv2(x):
  return -2 * np.tanh(x) * (1 - np.tanh(x)**2)



def init_params():

    W1 = np.random.randn(128, 2) * np.sqrt(1/2)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(128, 128) * np.sqrt(1/128)
    b2 = np.zeros((128, 1))
    W3 = np.random.randn(1, 128) * np.sqrt(1/128)
    b3 = np.zeros((1, 1))

    return W1, b1, W2, b2, W3, b3


def propagate_layer(Z, A, W, b, A_t, A_x, A_xx):

  """
  For the PINN a propagation function is needed to keep track of the partial
  derivatives through the layers"""

  Z_new = W @ A + b
  A_new = tanh(Z_new)

  Z_new_t = W @ A_t
  Z_new_x = W @ A_x
  Z_new_xx = W @ A_xx

  A_new_t = tanh_deriv1(Z_new) * Z_new_t
  A_new_x = tanh_deriv1(Z_new) * Z_new_x
  A_new_xx = tanh_deriv2(Z_new) * (Z_new_x)**2 + tanh_deriv1(Z_new) * Z_new_xx

  return Z_new, A_new, Z_new_t, Z_new_x, Z_new_xx, A_new_t, A_new_x, A_new_xx


def forward_propagation(W1, b1, W2, b2, W3, b3, X):

  """
  Takes in the initial partial derivatives and calls on propagate_layer to
  propagate through the layers, keeping track of the new parameters
  """

  N = X.shape[1]

  A_t_initial = np.vstack([np.zeros((1, N)), np.ones((1, N))])
  A_x_initial = np.vstack([np.ones((1, N)), np.zeros((1, N))])
  A_xx_initial = np.vstack([np.zeros((1, N)), np.zeros((1, N))])

  Z1, A1, Z1_t, Z1_x, Z1_xx, A1_t, A1_x, A1_xx = propagate_layer(0, X, W1, b1, A_t_initial, A_x_initial, A_xx_initial)
  
  Z2, A2, Z2_t, Z2_x, Z2_xx, A2_t, A2_x, A2_xx = propagate_layer(Z1, A1, W2, b2, A1_t, A1_x, A1_xx)

  Z3,_, Z3_t, Z3_x, Z3_xx,_,_,_ = propagate_layer(Z2, A2, W3, b3, A2_t, A2_x, A2_xx) ; A3, A3_t, A3_x, A3_xx = Z3, Z3_t, Z3_x, Z3_xx

  return Z1, A1, Z1_t, Z1_x, Z1_xx, A1_t, A1_x, A1_xx, Z2, A2, Z2_t, Z2_x, Z2_xx, A2_t, A2_x, A2_xx, Z3, Z3_t, Z3_x, Z3_xx, A3, A3_t, A3_x, A3_xx, A_t_initial, A_x_initial, A_xx_initial

W1, b1, W2, b2, W3, b3 = init_params()


def back_propagation_data(W1, b1, W2, b2, W3, b3, X, Y):

  """
  Back propagation function that only deals with the raw data, essentially the
  same as the function for the basic neural network
  """

  m = Y.shape[1]
  
  Z1, A1, Z1_t, Z1_x, Z1_xx, A1_t, A1_x, A1_xx, Z2, A2, Z2_t, Z2_x, Z2_xx, A2_t, A2_x, A2_xx, Z3, Z3_t, Z3_x, Z3_xx, A3, A3_t, A3_x, A3_xx,_,_,_ = forward_propagation(W1, b1, W2, b2, W3, b3, X)

  CA3 = A3 - Y

  dW3 = 1 / m * (CA3 @ A2.T)
  db3 = 1 / m * (np.sum(CA3,axis=1,keepdims=True))
  
  CA2 = W3.T @ CA3 * tanh_deriv1(Z2)

  dW2 = 1 / m * (CA2 @ A1.T)
  db2 = 1 / m * (np.sum(CA2 , axis=1, keepdims=True))
  
  CA1 = W2.T @ CA2 * tanh_deriv1(Z1)

  dW1 = 1 / m * (CA1 @ X.T)
  db1 = 1 / m * (np.sum(CA1, axis=1, keepdims=True))

  return dW1, db1, dW2, db2, dW3, db3

def back_propagation_physics(W1, b1, W2, b2, W3, b3, X, Y):

  """
  The back propagation function with a loss function derived from the 
  heat equation, minimising this loss means the output from the PINN network
  will be closer to the true mathmatical solution 
  """

  m = Y.shape[1]
  
  Z1, A1, Z1_t, Z1_x, Z1_xx, A1_t, A1_x, A1_xx, Z2, A2, Z2_t, Z2_x, Z2_xx, A2_t, A2_x, A2_xx, Z3, Z3_t, Z3_x, Z3_xx, A3, A3_t, A3_x, A3_xx, A_t_initial, A_x_initial, A_xx_initial = forward_propagation(W1, b1, W2, b2, W3, b3, X)
  
  residual = (1/T) * A3_t - (alpha/L**2) * A3_xx
  dA3_t = (1/T)*residual
  dA3_xx = -(alpha/L**2) * residual

  dW3 = 1 / m * (dA3_t @ A2_t.T + dA3_xx @ A2_xx.T)
  db3 = 1 / m * (np.sum(dA3_t,axis=1,keepdims=True) + np.sum(dA3_xx,axis=1,keepdims=True))

  dA2_t = W3.T @ dA3_t * tanh_deriv1(Z2)
  dA2_xx = W3.T @ dA3_xx * tanh_deriv1(Z2)

  dW2 = 1 / m * (dA2_t @ A1_t.T + dA2_xx @ A1_xx.T)
  db2 = 1 / m * (np.sum(dA2_t, axis=1, keepdims=True) + np.sum(dA2_xx, axis=1, keepdims=True))

  dA1_t = W2.T @ dA2_t * tanh_deriv1(Z1)
  dA1_xx = W2.T @ dA2_xx * tanh_deriv1(Z1)

  dW1 = 1 / m * (dA1_t @ A_t_initial.T + dA1_xx @ A_xx_initial.T)
  db1 = 1 / m * (np.sum(dA1_t, axis=1, keepdims=True) + np.sum(dA1_xx, axis=1, keepdims=True))

  return dW1, db1, dW2, db2, dW3, db3

def back_propagation_combine(W1, b1, W2, b2, W3, b3, X_data, Y_data, X_col, Y_col, lambda_pde, lambda_initial, lambda_boundry):

  """
  This function calls on the basic back propagation function to deal with data: 
  
  Raw data
  Initial Conditions
  Boundary Conditions
  
  It calls on the physics back propagation function to deal with the heat equation:
  
  Initial Conditions
  
  lambda_pde, lambda_initial, lambda_boundry are constants that are used to
  alter the affect that each loss function has on the final output

  """

  dW1_initial, db1_initial, dW2_initial, db2_initial, dW3_initial, db3_initial = back_propagation_data(W1, b1, W2, b2, W3, b3, initial_data, u_initial)
  
  dW1_bc, db1_bc, dW2_bc, db2_bc, dW3_bc, db3_bc = back_propagation_data(W1, b1, W2, b2, W3, b3, bc_data, u_bc)

  dW1_data, db1_data, dW2_data, db2_data, dW3_data, db3_data = back_propagation_data(W1, b1, W2, b2, W3, b3, X_data, Y_data)

  dW1_col, db1_col, dW2_col, db2_col, dW3_col, db3_col = back_propagation_physics(W1, b1, W2, b2, W3, b3, X_col, Y_col)

  dW1 = dW1_data + lambda_pde * dW1_col + (lambda_initial ) * dW1_initial + (lambda_boundry ) * dW1_bc
  db1 = db1_data + lambda_pde * db1_col + (lambda_initial ) * db1_initial + (lambda_boundry ) * db1_bc
  dW2 = dW2_data + lambda_pde * dW2_col + (lambda_initial ) * dW2_initial + (lambda_boundry ) * dW2_bc
  db2 = db2_data + lambda_pde * db2_col + (lambda_initial ) * db2_initial + (lambda_boundry ) * db2_bc
  dW3 = dW3_data + lambda_pde * dW3_col + (lambda_initial ) * dW3_initial + (lambda_boundry ) * dW3_bc
  db3 = db3_data + lambda_pde * db3_col + (lambda_initial ) * db3_initial + (lambda_boundry ) * db3_bc

  return dW1, db1, dW2, db2, dW3, db3


def update_param(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, rate):
  
  W1 = W1 - rate * dW1
  b1 = b1 - rate * db1
  W2 = W2 - rate * dW2
  b2 = b2 - rate * db2
  W3 = W3 - rate * dW3
  b3 = b3 - rate * db3
  
  return W1, b1, W2, b2, W3, b3


def gradient_descent(X, Y, X_col, Y_col, rate_0, iterations):

  """
  Minimising the loss function by repeatedly forward and backward propagating,
  updating parameters so that Loss decreases
  """
  
  W1, b1, W2, b2, W3, b3 = init_params()

  lambda_pde = 1
  
  lambda_initial = 10
  
  lambda_boundry = 2

  
  for i in range(iterations):

    Z1, A1, Z1_t, Z1_x, Z1_xx, A1_t, A1_x, A1_xx, Z2, A2, Z2_t, Z2_x, Z2_xx, A2_t, A2_x, A2_xx, Z3, Z3_t, Z3_x, Z3_xx, A3, A3_t, A3_x, A3_xx,_,_,_ = forward_propagation(W1, b1, W2, b2, W3, b3, X)
   
    dW1, db1, dW2, db2, dW3, db3 = back_propagation_combine(W1, b1, W2, b2, W3, b3, X, Y, X_col, Y_col, lambda_pde , lambda_initial, lambda_boundry)
   
    rate = rate_0 / (1 + i * 0.001) # Decreases the learing rate throughout the iterations
   
    W1, b1, W2, b2, W3, b3 = update_param(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, rate)
    
    if i % 100 == 0:
      
      # prints loss information every 100 iterations

      data_loss = np.mean((A3 - Y)**2)
      
      pde_loss = np.mean((A3_t - alpha * A3_xx)**2)
      
      print(f"Iteration {i}, Data Loss: {data_loss:.6f}, PDE Loss: {pde_loss:.6f}")
  
  return W1, b1, W2, b2, W3, b3
