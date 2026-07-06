"""
This script is based off of chapter 6.1 of The Fundamentals of Active Inference by Sanjeev Namjoshi. 

It is the simplest example of generalized filtering for perception given by the book. In it, an agent 
observes a hidden state that changes with each time step t. At each time step the agent receives a new 
observation that it passes to the generative model to estimate the external state. In this case the update 
is done via gradient descent. We find the gradient of the agent's free energy (the slope at any point) 
with respect to the estimate u_x, and either add or subtract to the state estimate based on the slope of 
the gradient. The updated free energy of the state estimation can be calculated using this new estimation. 
The prediction errors are also graphed to show whether the agent trusts its priors or observations more. 

The accuracy of the hidden state update via Euller's method depends on the accuracy of the generative model. 
The generative model is encoded into the free energy gradient, which we decend to model the external state 
based on the new observation at each time step.

This explanation is very high level and I highly recomend reading the textbook and doing the derivations, 
as this will show you why it works the way it does. There are a quite few clever algebra and calculus tricks 
that make this work. Only the results are shown here. 
"""

import matplotlib.pyplot as plt
import numpy as np

T_STEP = 0.01

# Used for the loop
LOOP_T = np.arange(0, 10, T_STEP)

# Used for the graphs
# Add one extra time step for the initial values of x, y, e_x, e_y, and f so that the time arrary and the value arrays are the same length
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
    new_y = x_star - theta_y + rng.normal(0.0, 0.1)
    return new_y

# Generative Model
def state_transition_function(theta_x, u_x):
    # This is the agent's expectation of the velocity of the external state
    # It is used to calculate the state prediction error 
    x_n = theta_x - u_x
    return x_n

def observation_generating_function(u_x, theta_y):
    u_y = u_x - theta_y
    return u_y

def update_hidden_state(u_x, lambda_y, e_y, dg_du, lambda_x, e_x, df_du, k):
    # Free Energy Gradient
    # In this case there is no place to plug in u_x as the slope is the same for all possible values
    gradient = lambda_y * e_y * dg_du + lambda_x * e_x * df_du
    new_u_x = u_x + T_STEP * k * gradient
    return new_u_x

def recalculate_free_energy(lambda_y, e_y, lambda_x, e_x):  
    new_f = lambda_y * (e_y ** 2) + lambda_x * (e_x ** 2) + np.log((1 / lambda_y) * (1 / lambda_x))
    return new_f

def recalculate_prediction_error(u_x, u_x_prev, theta_x, y, theta_y):
    # The agnet expects the external state to move at a velocity of x_n
    x_n = state_transition_function(theta_x, u_x_prev)

    # u_x is the agent's estimatate of the external state position at time t
    # x_n is the agent's prediction of the velocity at time t-1, 
    # meaning subtracting it from u_x gives you the error of the velocity prediction.  
    next_e_x = u_x - x_n

    u_y = observation_generating_function(u_x, theta_y)
    next_e_y = y - u_y
    return next_e_x, next_e_y, u_y

def graph_results(x_star, y, u_x, u_y, e_x, e_y, f):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- GRAPH 1: States and Observations ---
    axs[0].plot(T, x_star, label="x*")
    axs[0].plot(T, y, label="y")
    axs[0].plot(T, u_x, label="u_x")
    axs[0].plot(T, u_y, label="u_y")
    axs[0].legend(loc='lower center')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 2 Prediction Errors ---
    axs[1].plot(T, e_x, label="state error")  # 'k--' makes a black dashed line
    axs[1].plot(T, e_y, label="observation error")
    axs[1].legend(loc='lower center')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # --- GRAPH 3 Free Energy ---
    axs[2].plot(T, f, color='purple', linewidth=2, label='Free Energy')
    axs[2].set_ylabel('Energy / Error')
    axs[2].set_xlabel('Time')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    # Clean up the spacing and save the file
    plt.tight_layout()  # Automatically adjusts margins so titles/labels don't overlap
    plt.savefig('results/generalized_filtering_results.png')

def main():
    ####### Generative process vars #######

    theta_star_x = 10 
    theta_star_y = 3
    
    # A list containing the external state of x for each time step. The initial external state is 5. 
    x_star = [5]

    # A list containing the agent's observation for each time step. The initial observation is calculated here using the observation generating function. 
    initial_observation = observation_generating_function(x_star[-1], theta_star_y)
    y = [initial_observation]

    ####### Generative model vars #######

    # The learning rate kappa of the gradient descent step
    k = 0.1

    # The list of hidden state estimations at each time step. The initial guess is 15 but can be set to anything. 
    u_x = [15]

    # Theta_x is used in the agents state transition function theta_x - u_x to generate it's hypothesis regarding the new value of u_x.  
    theta_x = 10
    
    # Theta_y is the agent's approximation of theta_star_y from the environment and is used to predict the next observation. 
    theta_y = 3

    # The precisions (inverse variances) of hidden states x and observations y 
    lambda_x = 0.2
    lambda_y = 50

    # The derivatives of the observation generating function g(m) and the state transition function f(m) with respect to u_x. 
    # These are used to derive the Free Energy gradient used for the gradient descent u_x update. 
    dg_du = 1
    df_du = -1

    # Initial state prediction error
    # Typically the error is calculated using the observed state u_x and the predicted state, 
    # but there is no previous state on the first step so we use the u_x - f(u_x, theta_x) 
    x_n = state_transition_function(theta_x, u_x[-1])  
    e_x = [u_x[-1] - x_n]

    # Initial observation prediction error
    u_y = [observation_generating_function(u_x[-1], theta_y)]
    e_y = [initial_observation - u_y[-1]]

    # Initial Free Energy
    f = [lambda_y * (e_y[-1] ** 2) + lambda_x * (e_x[-1] ** 2) + np.log((1 / lambda_y) * (1 / lambda_x))] 

    # AIF
    for _ in LOOP_T:
        ####### Generative Process #######

        # Generate new external state and observation 
        x_star.append(generate_state(x_star[-1], theta_star_x))    
        y.append(generate_observation(x_star[-1], theta_y))
        # print(f"External state: {x_star}")
        # print(f"Observation: {y}")
        
        ####### Generative Model #######

        # Update hidden state using observation and generative model
        u_x.append(update_hidden_state(u_x[-1], lambda_y, e_y[-1], dg_du, lambda_x, e_x[-1], df_du, k))
        
        # Update free energy calculation
        f.append(recalculate_free_energy(lambda_y, e_y[-1], lambda_x, e_x[-1]))

        # Update prediction errors using new observation and hidden state estimate 
        next_e_x, next_e_y, next_u_y = recalculate_prediction_error(u_x[-1], u_x[-2], theta_x, y[-1], theta_y)
        e_x.append(next_e_x)
        e_y.append(next_e_y)
        u_y.append(next_u_y)

    graph_results(x_star, y, u_x, u_y, e_x, e_y, f)

main()