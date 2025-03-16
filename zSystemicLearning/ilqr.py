import numpy as np
 
# System dynamics (nonlinear)
def system_dynamics(x, u):
    return x + np.sin(u)
 
# Cost function for a single step
def cost_function(x, u):
    return 0.5 * (x**2 + u**2)
 
# Derivative of the cost function w.r.t. control input u (l_u)
def cost_u(u):
    return u
 
# Derivative of the cost function w.r.t. state x (l_x)
def cost_x(x):
    return x
 
# Second derivative of the cost function w.r.t. control input u (l_uu)
def cost_uu():
    return 1
 
# Second derivative of the cost function w.r.t. state x (l_xx)
def cost_xx():
    return 1
 
 
# Function to calculate the initial state trajectory based on control sequence
def compute_initial_trajectory(x0, u):
    x = np.zeros(N+1)
    x[0] = x0
    for k in range(N):
        x[k+1] = system_dynamics(x[k], u[k])
    return x
 
# iLQR algorithm with different stopping conditions
def ilqr_with_conditions(x, u, iterations, epsilon_u, epsilon_J, epsilon_x):
    prev_cost = np.inf
    for i in range(iterations):
        # Backward pass
        V_x = np.zeros(N+1)
        V_xx = np.zeros(N+1)
        V_x[-1] = x[-1]  # Terminal value for V_x
        V_xx[-1] = 1  # Terminal value for V_xx (quadratic cost on terminal state)
 
        du = np.zeros(N)  # Control updates
 
        # Backward pass: compute Q function and control update
        for k in range(N-1, -1, -1):
            # Compute Q-function terms
            f_u = np.cos(u[k])  # Derivative of system dynamics w.r.t. u
            Q_u = cost_u(u[k]) + f_u * V_x[k+1]  # Q_u = l_u + f_u^T * V_x(k+1)
            Q_uu = cost_uu() + f_u**2 * V_xx[k+1]  # Q_uu = l_uu + f_u^T * V_xx(k+1) * f_u
            Q_x = cost_x(x[k]) + V_x[k+1]  # Q_x = l_x + f_x^T * V_x(k+1)
            Q_xx = cost_xx() + V_xx[k+1]  # Q_xx = l_xx + f_x^T * V_xx(k+1) * f_x
 
            # Update control input
            du[k] = -Q_u / Q_uu  # Control update
            V_x[k] = Q_x + Q_uu * du[k]  # Update value function gradient
            V_xx[k] = Q_xx  # Update value function Hessian (V_xx)
 
        # Forward pass: update trajectory using the new control inputs
        x_new = np.zeros(N+1)
        u_new = np.zeros(N)
        x_new[0] = x0
 
        for k in range(N):
            u_new[k] = u[k] + du[k]  # Update control
            x_new[k+1] = system_dynamics(x_new[k], u_new[k])  # Update state
 
        
        # Compute the total cost for the current trajectory
        current_cost = np.sum([cost_function(x_new[k], u_new[k]) for k in range(N)])
 
        # 1. Stop based on control input change
        if np.max(np.abs(du)) < epsilon_u:
            print(f"Stopped due to control input convergence at iteration {i}")
            break
        
        # Update for next iteration
        x = x_new
        u = u_new
        prev_cost = current_cost
 
    return x, u, i
 
if __name__ == "__main__":
 
    # iLQR parameters
    N = 3  # Number of time steps
    x0 = 1  # Initial state
    iterations = 50  # Maximum number of iterations
    epsilon_u = 1e-3  # Tolerance for control input changes
    epsilon_J = 1e-4  # Tolerance for cost function change
    epsilon_x = 1e-4  # Tolerance for state trajectory change
 
    # Initialize control sequence and state trajectory
    u = np.zeros(N)  # Initial control sequence
    x = np.zeros(N+1)  # State trajectory
    x[0] = x0
 
 
    # Compute initial trajectory
    x_initial = compute_initial_trajectory(x0, u)
 
    # Run iLQR with stopping conditions
    x_final, u_final, num_iterations = ilqr_with_conditions(x_initial, u, iterations, epsilon_u, epsilon_J, epsilon_x)
 
    # Output the final results and number of iterations
    print(x_final, u_final, num_iterations)