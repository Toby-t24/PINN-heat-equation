# Modelling the heat equation using a bar of length 2 being heated over time
# period 1.


L = 2
T = 1

alpha = 0.5


def u(x, t):
    
    """the true solution to the heat equation"""

    k = 2 * np.pi / L
    
    return np.cos(k * x) * np.exp(-alpha * (k**2) * t)


# Creating the data for the basic neural network (and the foundation of the
# PINN)

N = 2000

x_data = np.random.rand(N) * L
t_data = np.random.rand(N) * T


x_norm = x_data / L
t_norm = t_data / T


train_data = np.vstack([x_norm, t_norm])


u_data_train  = u(x_data, t_data)

U_max = np.max(np.abs(u_data_train))

u_data_train = u_data_train / U_max
u_data_train = u_data_train.reshape(1, -1)


# The PINN needs additional collocation, initial, and boundary data points to 
# use in the additional loss functions


# Collocation Points

N_col = 1000

x_col = np.random.rand(N_col) * L
t_col = np.random.rand(N_col) * T
u_col = u(x_col, t_col)

#Normalise

x_col_norm = x_col / L
t_col_norm = t_col / T

col_data = np.vstack([x_col_norm, t_col_norm])

u_data_col = u_col / U_max
u_data_col = u_data_col.reshape(1,-1)



# Initial Points

N_initial = 2000

x_initial = np.random.rand(N_initial) * L
t_initial = np.zeros(N_initial)

u_initial = np.cos((2 * np.pi / L) * x_initial)
u_initial = u_initial / U_max
u_initial = u_initial.reshape(1,-1)

# Normalise

x_initial_norm = x_initial / L
t_initial_norm = t_initial / T

initial_data = np.vstack([x_initial_norm, t_initial_norm])



# Boundary Conditions

N_bc = 1000
t_bc = np.random.rand(N_bc) * T


x0 = np.zeros(N_bc)
u0 = np.zeros(N_bc)


xL = L * np.ones(N_bc)
uL = np.zeros(N_bc)


x_bc = np.concatenate([x0, xL]) / L
t_bc = np.concatenate([t_bc, t_bc]) / T
u_bc = np.concatenate([u0, uL]).reshape(1, -1)

bc_data = np.vstack([x_bc, t_bc])
