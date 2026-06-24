"""
This is the multivariate generalized filter for perception. In this the agent models Hooke's Law, which describes the oscilation of a linear spring. 

The code below is copy pasted from geralisezed_filtering.py and is a work in progress. 
"""

import matplotlib as mpl
from matplotlib.pylab import rand
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 

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

    # Velocity of external state
    x_star_dot = theta_star_x - x_star + rng.normal(0.0, 8.0)

    # x* update
    new_x_star = x_star + x_star_dot * T_STEP
    return new_x_star

def generate_observation(x_star, theta_y):
    new_y = x_star - theta_y + rng.normal(0.0, 0.5)
    return new_y

# Generative Model
def state_transition_function(theta_x, u_x):
    x_n = theta_x - u_x
    return x_n

def observation_gnerating_function(u_x, theta_y):
    u_y = u_x - theta_y
    return u_y

def update_hidden_state(u_x, lambda_y, e_y, dg_du, lambda_x, e_x, df_du, k):
    # Free Energy Gradient
    # In this case there is no place to plug in u_x as the slope is the same for all possible values
    gradient = lambda_y * e_y * dg_du + lambda_x * e_x * df_du
    new_u_x = u_x + T_STEP * k * gradient
    return new_u_x

def recalculate_free_energy(lambda_y, e_y, lambda_x, e_x):  # should this be ln ???
    new_f = lambda_y * (e_y ** 2) + lambda_x * (e_x ** 2) + np.log((1 / lambda_y) * (1 / lambda_x))
    return new_f

def recalculate_prediction_error(u_x, theta_x, y, theta_y):
    x_n = state_transition_function(theta_x, u_x)
    next_e_x = u_x - x_n

    u_y = observation_gnerating_function(u_x, theta_y)
    next_e_y = y - u_y
    return next_e_x, next_e_y, u_y

def graph_results(x_star, y, u_x, u_y, e_x, e_y, f):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- GRAPH 1: 4 Values ---
    axs[0].plot(T, x_star, label="x*")
    axs[0].plot(T, y, label="y")
    axs[0].plot(T, u_x, label="u_x")
    axs[0].plot(T, u_y, label="u_y")
    axs[0].legend(loc='lower center')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 2: 2 Values ---
    axs[1].plot(T, e_x, label="state error")  # 'k--' makes a black dashed line
    axs[1].plot(T, e_y, label="observation error")
    axs[1].legend(loc='lower center')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 3: 1 Value ---
    axs[2].plot(T, f, color='purple', linewidth=2, label='Free Energy')
    axs[2].set_ylabel('Energy / Error')
    axs[2].set_xlabel('Time')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    # 3. Clean up the spacing and save the file
    plt.tight_layout()  # Automatically adjusts margins so titles/labels don't overlap
    plt.savefig('generalized_filtering_results.png')

def main():
    # Generative process vars
    theta_star_x = 10
    x_star = [5]

    theta_star_y = 3
    initial_observation = x_star[-1] - theta_star_y
    y = [initial_observation]

    # Generative model vars
    k = 0.1

    u_x = [15]

    lambda_y = 50
    theta_y = 3

    lambda_x = 0.2
    theta_x = 10

    # Initial state prediction error
    x_n = state_transition_function(theta_x, u_x[-1])
    e_x = [u_x[-1] - x_n]

    # Initial observation prediction error
    u_y = [observation_gnerating_function(u_x[-1], theta_y)]
    e_y = [initial_observation - u_y[-1]]

    # Initial Free Energy
    f = [lambda_y * (e_y[-1] ** 2) + lambda_x * (e_x[-1] ** 2) + np.log((1 / lambda_y) * (1 / lambda_x))] 

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
        dg_du = 1
        df_du = -1
        u_x.append(update_hidden_state(u_x[-1], lambda_y, e_y[-1], dg_du, lambda_x, e_x[-1], df_du, k))
        
        # Update free energy calculation
        f.append(recalculate_free_energy(lambda_y, e_y[-1], lambda_x, e_x[-1]))

        # Update prediction errors using new observation and hidden state estimate 
        next_e_x, next_e_y, next_u_y = recalculate_prediction_error(u_x[-1], theta_x, y[-1], theta_y)
        e_x.append(next_e_x)
        e_y.append(next_e_y)
        u_y.append(next_u_y)

    graph_results(x_star, y, u_x, u_y, e_x, e_y, f)

main()