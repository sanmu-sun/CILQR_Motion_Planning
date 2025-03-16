# https://blog.csdn.net/a8598671/article/details/141957635
# cost J = 0.5*x^2 + 0.5*u^2;
# x_k+1 = x_k + u_k

import numpy as np
 
 
# System dynamics (linearized)
def system_dynamics_lqr(x, u):
    return x + u  # Linearized system: x_{k+1} = x_k + u_k
 
# Cost function for a single step
def cost_function_lqr(x, u):
    return 0.5 * (x**2 + u**2)
 
# Backward pass: solve Riccati equation to find P matrices and feedback gains K
def solve_lqr(A, B, Q, R, N):
    P = np.zeros(N+1)
    K = np.zeros(N)
 
    # Terminal cost (at time N)
    P[N] = Q  # At final time step, P_N = Q
 
    # Backward pass to compute P matrices and feedback gains K
    for k in range(N-1, -1, -1):
        P[k] = Q + A**2 * P[k+1] - A * P[k+1] * B * (R + B**2 * P[k+1])**(-1) * B * P[k+1] * A
        K[k] = (R + B**2 * P[k+1])**(-1) * B * P[k+1] * A  # Feedback gain
 
    return P, K
 
# Forward pass: apply feedback gains K to compute the optimal trajectory
def apply_lqr_control(x0, K, N):
    x = np.zeros(N+1)
    u = np.zeros(N)
    x[0] = x0
 
    for k in range(N):
        u[k] = -K[k] * x[k]
        x[k+1] = system_dynamics_lqr(x[k], u[k])
 
    return x, u
 
if __name__ == "__main__":
    
    # LQR parameters
    N = 5  # Number of time steps
    x0 = 1  # Initial state
    Q = 5  # State cost
    R = 1  # Control cost
 
    # Matrices for the linearized system
    A = 1  # State transition matrix
    B = 1  # Control matrix
 
    # Initialize control sequence and state trajectory
    u = np.zeros(N)  # Initial control sequence
    x = np.zeros(N+1)  # State trajectory
    x[0] = x0
    
    # Solve for P and K matrices using LQR
    P, K = solve_lqr(A, B, Q, R, N)
 
    # Apply LQR control to get the optimal trajectory
    x_final, u_final = apply_lqr_control(x0, K, N)
 
    # Output the final results
    print(x_final, u_final)