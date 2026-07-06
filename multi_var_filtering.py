"""
This is an example of a multivariate generalized filter for perception. 
The agent models Hooke's Law, which describes the oscilation of a linear spring. 

Please note that according to the Gemini, the math is actually not Hooke's Law. 
It has been modified to show a more interesting, oscillatory behavior.

Please forgive the following rather opaque explanation: 

####################################################
# Derivation of the true Hooke's Law state update: #
####################################################

Hooke's Law:
F = -kx
   
Newton's Second Law:
F = ma

ma = -kx
a  = (-k/m) * x

accleration as the derivative of velocity x_1:
dx_1 = (-k/m) * x_0

velocity as the derivative of position x_0:
dx_0 = x_1

Therefore the state update is:
position: x_0 + x_1 * t
velocity: x_1 + (-k/m) * x_0 * t]

The example from the book uses the following velocity, acceleration, and state update:

velocity as the derivative of position:
v_0 + x_1

acceleration as the derivative of velocity:
(k / m) * v_1 - x_0

position: x_0 + (v_1 + x_0) * t
velocity: x_1 + ((k/m) * v_1 - x_0) * t 

This creates a more interesting oscillatory behavior than the true Hooke's Law.
"""


import matplotlib.pyplot as plt
import numpy as np

T_STEP = 0.01

# Used for the loop
LOOP_T = np.arange(0, 10, T_STEP)

# Used for the graphs
# Add one extra time step for the initial values of x, y, e_x, e_y, and f
T = np.append(LOOP_T, LOOP_T[-1] + T_STEP) 

# random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

# Generative Process
def generate_state(x_star, theta_star_x):
    k = theta_star_x[0]
    m = theta_star_x[1]
    v = theta_star_x[2]

    # Velocity of external state
    x_star_dot = [v[1] + x_star[1] + rng.normal(0.0, 8.0), (k / m) * v[0] - x_star[0] + rng.normal(0.0, 8.0)] 

    # x* update
    new_x_star = [x_star[0] + x_star_dot[0] * T_STEP, x_star[1] + x_star_dot[1] * T_STEP]
    return new_x_star

def generate_observation(x_star, theta_y):
    new_y = [x_star[0] - theta_y + rng.normal(0.0, 0.5), x_star[1] - theta_y + rng.normal(0.0, 0.5)]
    return new_y

# Generative Model
def state_transition_function(theta_x, u_x):
    k = theta_x[0]
    m = theta_x[1]
    v = theta_x[2]

    x_n = [v[1] + u_x[1], (k / m) * v[0] - u_x[0]]
    return x_n

def observation_generating_function(u_x, theta_y):
    u_y = [u_x[0] - theta_y, u_x[1] - theta_y]
    return u_y

def update_hidden_state(u_x, lambda_y, e_y, dg_du, lambda_x, e_x, df_du, k):
    # Free Energy Gradient
    # In this case there is no place to plug in u_x as the slope is the same for all possible values

    # Notice in the print statement below that the error arrays e_y and e_x are printed [e_1, e_2] instead of [[e_1], [e_2]].
    # This is because numpy automatically converted the python list to a 1D array. Also, attempting to transpose a 1D numpy
    # array does nothing. Luckily, numpy automatically transposes it when matrix multiplication requires it. Terefore, the .T 
    # of the errors in the gradient are not necessary but are there so no one gets confused. 
    
    # print(f"lambda_y: \n{lambda_y}, \ne_y: \n{np.array(e_y)}, \ndg_du: \n{dg_du}, \nlambda_x: \n{lambda_x}, \ne_x: \n{np.array(e_x).T}, \ndf_du: \n{df_du}")
    
    gradient = dg_du.T @ lambda_y @ np.array(e_y).T + df_du.T @ lambda_x @ np.array(e_x).T
    new_u_x = [u_x[0] + T_STEP * k * gradient[0], u_x[1] + T_STEP * k * gradient[1]]
    return new_u_x

def recalculate_free_energy(lambda_y_arr, e_y, lambda_x_arr, e_x):  
    f_x = 0
    for i, lam in enumerate(lambda_x_arr): 
        f_x += lam * e_x[i] ** 2 + np.log(lam ** -1)
    f_y = 0
    for i, lam in enumerate(lambda_y_arr): 
        f_y += lam * e_y[i] ** 2 + np.log(lam ** -1)
    new_f = 0.5 * (f_x + f_y)
    return new_f

def recalculate_prediction_error(u_x, u_x_prev, theta_x, y, theta_y):
    x_n = state_transition_function(theta_x, u_x_prev)
    next_e_x = [u_x[0] - x_n[0], u_x[1] - x_n[1]]

    u_y = observation_generating_function(u_x_prev, theta_y)
    next_e_y = [y[0] - u_y[0], y[1] - u_y[1]]
    return next_e_x, next_e_y, u_y

def graph_results(x_star, y, u_x, u_y, e_x, e_y, f):
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    
    x_star_arr = np.array(x_star)
    y_arr = np.array(y)
    u_x_arr = np.array(u_x)
    u_y_arr = np.array(u_y)
    e_x_arr = np.array(e_x)
    e_y_arr = np.array(e_y)

    # --- GRAPH 1: 4 Values ---
    axs[0].plot(T, x_star_arr[:, 0], label="x* 0")
    axs[0].plot(T, x_star_arr[:, 1], label="x* 1")
    axs[0].legend(loc='lower center')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 2: 2 Values ---
    axs[1].plot(T, y_arr[:, 0], label="y 0")
    axs[1].plot(T, y_arr[:, 1], label="y 1")
    axs[1].legend(loc='lower center')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 3: 2 Values ---
    axs[2].plot(T, x_star_arr[:, 0], label="x* 0")
    axs[2].plot(T, x_star_arr[:, 1], label="x* 1")
    axs[2].plot(T, u_x_arr[:, 0], label="u_x 0")
    axs[2].plot(T, u_x_arr[:, 1], label="u_x 1")
    axs[2].legend(loc='lower center')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 4: 2 Values ---
    axs[3].plot(T, y_arr[:, 0], label="y 0")
    axs[3].plot(T, y_arr[:, 1], label="y 1")
    axs[3].plot(T, u_y_arr[:, 0], label="u_y 0")
    axs[3].plot(T, u_y_arr[:, 1], label="u_y 1")
    axs[3].legend(loc='lower center')
    axs[3].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 5: 1 Value ---
    axs[4].plot(T, e_x_arr[:, 0], label="state error 0")
    axs[4].plot(T, e_x_arr[:, 1], label="state error 1")
    axs[4].plot(T, e_y_arr[:, 0], label="observation error 0")
    axs[4].plot(T, e_y_arr[:, 1], label="observation error 1")
    axs[4].legend(loc='lower center')
    axs[4].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 6: 1 Value ---
    axs[5].plot(T, f, color='purple', linewidth=2, label='Free Energy')
    axs[5].set_ylabel('Energy / Error')
    axs[5].set_xlabel('Time')
    axs[5].legend(loc='upper right')
    axs[5].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()  # Automatically adjusts margins so titles/labels don't overlap
    plt.savefig('results/multi_var_filtering_results.png')

def main():
    # Generative process vars
    k = 4
    m = 3
    v = [5, 0]
    theta_star_x = [k, m, v]
    x_star = [[0, 0]]

    theta_star_y = 3
    initial_observation = [x_star[-1][0] - theta_star_y, x_star[-1][1] - theta_star_y]
    y = [initial_observation]

    # Generative model vars
    k = 1

    u_x = [[8, 8]]

    lambda_x_mat = np.linalg.inv(2 * np.identity(2))
    lambda_y_mat = np.linalg.inv(0.1 * np.identity(2))
    lambda_x_arr = np.diag(lambda_x_mat)
    lambda_y_arr = np.diag(lambda_y_mat)

    theta_x = [k, m, v]
    theta_y = 3

    df_du = np.array([[0, 1], 
                      [-1, 0]])
    dg_du = np.array([[1, 0], 
                      [0, 1]])

    # Initial state prediction error
    x_n = state_transition_function(theta_x, u_x[-1])
    e_x = [[u_x[-1][0] - x_n[0], u_x[-1][1] - x_n[1]]]

    # Initial observation prediction error
    u_y = [observation_generating_function(u_x[-1], theta_y)]
    e_y = [[initial_observation[0] - u_y[-1][0], initial_observation[1] - u_y[-1][1]]]
    
    # Initial Free Energy
    f_x = 0
    for i, lam in enumerate(lambda_x_arr): 
        f_x += lam * e_x[0][i] ** 2 + np.log(lam ** -1)
    f_y = 0
    for i, lam in enumerate(lambda_y_arr): 
        f_y += lam * e_y[0][i] ** 2 + np.log(lam ** -1)
    f = [0.5 * (f_x + f_y)]

    # AIF
    for _ in LOOP_T:
        ####### Generative Process #######
        
        # Generate new external state and observation 
        x_star.append(generate_state(x_star[-1], theta_star_x))    
        y.append(generate_observation(x_star[-1], theta_star_y))
        # print(f"External state: {x_star}")
        # print(f"Observation: {y}")
        
        ####### Generative Model #######

        # Update hidden state using observation and generative model
        u_x.append(update_hidden_state(u_x[-1], lambda_y_mat, e_y[-1], dg_du, lambda_x_mat, e_x[-1], df_du, k))
        
        # Update free energy calculation
        f.append(recalculate_free_energy(lambda_y_arr, e_y[-1], lambda_x_arr, e_x[-1]))
        
        # Update prediction errors using new observation and hidden state estimate 
        next_e_x, next_e_y, next_u_y = recalculate_prediction_error(u_x[-1], u_x[-2], theta_x, y[-1], theta_y)
        e_x.append(next_e_x)
        e_y.append(next_e_y)
        u_y.append(next_u_y)

    graph_results(x_star, y, u_x, u_y, e_x, e_y, f)

main()