"""
Implementation of example 6.6 in Fundamentals of Active Inference: Generalized Coordinates for External State Trajectory Prediction  

The earlier examples have shown how an agent percieves an external state. How can an agent predict the future trajectory of the     
state to allow planning? One method is using generalized coordinates of motion. 

Generalized coordinates (GC) are a vector of position and its derivatives (i.e. velocity, acceleration, jerk, snap, crackle, pop).   
These coordinates can be plugged into a Taylor Series Approximation to predict the motion of the external states as many times 
steps into the future as you like. As this is an approximation, it gets less accurate over time. To find the GC, the agent 
initializes an estimation of the external state, observes the external state, uses an internal model to predict the full trajectory  
at the current time t, then calculates the error between its estimate and prediction and uses this error to update its estimation 
of the external state at time t+1. This loops until the free energy threshold is met or the time step limnit of the loop is reached. 
I'll go into a bit more detail about how this works below:

The first step is to initialize a belief about the external state called the hidden state: u_x. This initialization can either be   
random or based off of a prior belief. It is a vector containing the estimated position, velocity, acceleration, and jerk of the 
external state. Next, at each time step t the agent makes an observation of the external state y. The agent can never know the true 
external state of the environment, as it only sees an encoding of it (i.e. light intensity). Y is this encoding, and is refered to 
as the Generalized Measurement (GM). 

This initial u_x is plugged into the State Transition Function to generate what the agent expects the velocity, acceleration, and   
jerk to be at the current moment: u_x_prediction. We want to calculate the error between u_xprediction and u_x, so we shift the u_x 
vector to get a new vector Du_x that only includes the higher order derivatives. If u_x = [x, v, a, j], then Du_x = [v, a, j, snap]. 
The last element of Du_x is assumed to be 0. This allows you to calculate the prediction error by subtracting: Du_x - u_x_prediction. 
We can also use the initial u_x and the agent's Generalized Observation Generation Function to generate a predition of what the 
agent predicts to be observed at the current time t: y_prediction. The prediction error of the observation is calculated by 
subtracting: y - y_prediction.                       

Once you have these errors you can calculate the free energy gradient and use it to estimate the hidden state u_x at time t+1. This 
is the hidden state update step. After the update, the agent can use the new u_x and the next observation of the external state to 
run the next iteration of the loop. 

Both the estimate and prediction of the GC are a full trajectory prediction of the external state and can be used predict the 
trajectory as far into the future as you want. The accuracy goes down as t gets larger. If you consider modeling the behavior of a  
complex organism, one GC vector would struggle to predict every movement the animal will ever make, but each update of the 
trajectory predicts what it is likely to do in the immediate future. 

Anyway, this was a lot to take in and this explanation skips over the derivations of the graident in terms of prediction error, as 
well as other details. It has been worth it to piece it all together and it would have taken way, way longer without the book and 
Gemini Flash.  
"""

"""
TO DO

move vars to main loop
put noise vars into initialization function in main before loop 
maybe write up a separate noise kernel set for the generative coordinates instead of using the x noise. 
maybe use a new seed for each noise type 
"""

import matplotlib.pyplot as plt
import numpy as np

T_STEP = 0.01

# Used for the loop
LOOP_T = np.arange(0, 10, T_STEP)

# Used for the graphs
# Add two extra time step for the initial values of x, y, e_x, e_y, and f so that the time arrary and the value arrays are the same length
T = np.append(LOOP_T, LOOP_T[-1] + T_STEP) 

# random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(seed=42)

def build_noise_kernels():
    sigma_w = 8.0          # Matching your rng.normal(0.0, 8.0) external state variance
    sigma_y = 0.1          # Observation variance 
    gamma_val = 1.0      # Smoothness parameter gamma
    y_gamma_val = 1.0      # y smoothness
    window_radius = 30     # How far back in time to blend

    # Build the h_omega weights and it's derivatives relative to a time-lag axis. 
    # It is much harder to understand this than it is to derive it and code it up. I wrote it all down first.   
    t_axis = np.arange(-window_radius, window_radius + 1) * T_STEP
    scaling_factor_0 = np.sqrt(T_STEP / (sigma_w * np.sqrt(np.pi)))
    scaling_factor_1 = (-gamma_val / 2) * t_axis
    scaling_factor_2 = (-gamma_val / 2)
    noise_kernel_0 = scaling_factor_0 * np.exp(-(t_axis**2 / 4) * gamma_val)
    noise_kernel_0 /= np.sum(noise_kernel_0) # Normalize to preserve total variance
    noise_kernel_1 = noise_kernel_0 * scaling_factor_1
    noise_kernel_2 = noise_kernel_1 * scaling_factor_1 + noise_kernel_0 * scaling_factor_2

    # The observations need there own noise kernels because it has a smaller variance than the states
    scaling_factor_y_0 = np.sqrt(T_STEP / (sigma_y * np.sqrt(np.pi)))
    scaling_factor_y_1 = (-y_gamma_val / 2) * t_axis
    scaling_factor_y_2 = (-y_gamma_val / 2)
    noise_kernel_y_0 = scaling_factor_y_0 * np.exp(-(t_axis**2 / 4) * y_gamma_val)  #
    noise_kernel_y_0 /= np.sum(noise_kernel_y_0)                                    # Normalize to preserve total variance
    noise_kernel_y_1 = noise_kernel_y_0 * scaling_factor_y_1
    noise_kernel_y_2 = noise_kernel_y_1 * scaling_factor_y_1 + noise_kernel_y_0 * scaling_factor_y_2
    noise_kernel_y_3 = noise_kernel_y_2 * scaling_factor_y_1 - y_gamma_val * scaling_factor_y_1

    # Sliding memory buffers for white noise, filled with zeros to start. There is one for each noise type. 
    ext_x_raw_noise_buffer = np.zeros(len(noise_kernel_0))       # External state x white noise
    y_raw_noise_buffer = np.zeros(len(noise_kernel_0))           # Observation and generalized measuremnts white noise
    gen_coords_raw_noise_buffer = np.zeros(len(noise_kernel_0))  # Generative coordinates noise
    gen_measurements_raw_noise_buffer = np.zeros(len(noise_kernel_0))  # Generative measurements noise

    noise_dict = {
        "noise_kernel_0": noise_kernel_0,
        "noise_kernel_1": noise_kernel_1,
        "noise_kernel_2": noise_kernel_2,
        "noise_kernel_y_0": noise_kernel_y_0,
        "noise_kernel_y_1": noise_kernel_y_1,
        "noise_kernel_y_2": noise_kernel_y_2,
        "noise_kernel_y_3": noise_kernel_y_3,
        "ext_x_raw_noise_buffer": ext_x_raw_noise_buffer,
        "y_raw_noise_buffer": y_raw_noise_buffer,
        "gen_coords_raw_noise_buffer": gen_coords_raw_noise_buffer,
        "gen_measurements_raw_noise_buffer": gen_measurements_raw_noise_buffer
    }

    return noise_dict

# Smooth Noise depending on noise type i.e. x, y, or gen_coords noise. 
def smooth_noise(noise_type, white_noise, noise_dict):
    # external state x noise
    if noise_type == 0:
        # 1. Slide the memory buffer (drop oldest, add newest)
        noise_dict["ext_x_raw_noise_buffer"] = np.append(noise_dict["ext_x_raw_noise_buffer"][1:], white_noise)
        # print(f"\next_x_raw_noise_buffer: {ext_x_raw_noise_buffer}")
        # print(f"noise_kernel_0: {noise_kernel_0}")

        # 2. Convolve! Dot product blends the history with the h_\omega weights
        smooth_w = [np.dot(noise_dict["ext_x_raw_noise_buffer"], noise_dict["noise_kernel_0"]),
                    np.dot(noise_dict["ext_x_raw_noise_buffer"], noise_dict["noise_kernel_1"]),
                    np.dot(noise_dict["ext_x_raw_noise_buffer"], noise_dict["noise_kernel_2"])]

    # y and generalized measurements noise buffer 
    if noise_type == 1:
        # 1. Slide the memory buffer (drop oldest, add newest)
        noise_dict["y_raw_noise_buffer"] = np.append(noise_dict["y_raw_noise_buffer"][1:], white_noise)

        # 2. Convolve! Dot product blends the history with the h_omega weights
        smooth_w = [np.dot(noise_dict["y_raw_noise_buffer"], noise_dict["noise_kernel_y_0"]), 
                    np.dot(noise_dict["y_raw_noise_buffer"], noise_dict["noise_kernel_y_1"]), 
                    np.dot(noise_dict["y_raw_noise_buffer"], noise_dict["noise_kernel_y_2"]), 
                    np.dot(noise_dict["y_raw_noise_buffer"], noise_dict["noise_kernel_y_3"])]

    # Generalized Coordinates noise buffer
    if noise_type == 2:
        # 1. Slide the memory buffer (drop oldest, add newest)
        noise_dict["gen_coords_raw_noise_buffer"] = np.append(noise_dict["gen_coords_raw_noise_buffer"][1:], white_noise)

        # 2. Convolve! Dot product blends the history with the h_\omega weights
        smooth_w = [np.dot(noise_dict["gen_coords_raw_noise_buffer"], noise_dict["noise_kernel_0"]), 
                    np.dot(noise_dict["gen_coords_raw_noise_buffer"], noise_dict["noise_kernel_1"]), 
                    np.dot(noise_dict["gen_coords_raw_noise_buffer"], noise_dict["noise_kernel_2"])]
    
    # # Generalized Measurements noise buffer
    if noise_type == 3:
        # 1. Slide the memory buffer (drop oldest, add newest)
        noise_dict["gen_measurements_raw_noise_buffer"] = np.append(noise_dict["gen_measurements_raw_noise_buffer"][1:], white_noise)

        # 2. Convolve! Dot product blends the history with the h_\omega weights
        smooth_w = [np.dot(noise_dict["gen_measurements_raw_noise_buffer"], noise_dict["noise_kernel_y_0"]), 
                    np.dot(noise_dict["gen_measurements_raw_noise_buffer"], noise_dict["noise_kernel_y_1"]), 
                    np.dot(noise_dict["gen_measurements_raw_noise_buffer"], noise_dict["noise_kernel_y_2"]), 
                    np.dot(noise_dict["gen_measurements_raw_noise_buffer"], noise_dict["noise_kernel_y_3"])]

    return smooth_w

# Generative Process
def generate_state(x_star, theta_star_x, noise_dict):
    white_noise_sample = rng.normal(0.0, 2.0)

    smooth_w = smooth_noise(0, white_noise_sample, noise_dict)

    # Velocity of external state
    velocity = theta_star_x - x_star[0] + smooth_w[0]
    acceleration = - x_star[1] + smooth_w[1]
    jerk = - x_star[2] + smooth_w[2]

    # x* update
    new_position = x_star[0] + T_STEP * velocity 

    new_x_star = [new_position, velocity, acceleration, jerk]
    print(f"\nx_noise: {smooth_w}")
    print(f"next_x_tilde: {new_x_star}")
    return new_x_star

# Generate an observation of the external state position 
def generate_observation(x_star, theta_y, noise_dict):
    white_noise_sample = rng.normal(0.0, 0.1)

    smooth_w = smooth_noise(1, white_noise_sample, noise_dict)

    new_y = x_star[0] - theta_y + smooth_w[0]
    new_y_velocity = x_star[1] + smooth_w[1]
    new_y_acceleration = x_star[2] + smooth_w[2]
    new_y_jerk = x_star[3] + smooth_w[3]

    new_y_tilde = np.array([new_y, new_y_velocity, new_y_acceleration, new_y_jerk])
    print(f"\ny_noise: {smooth_w}")
    print(f"next_y_tilde: {new_y_tilde}")
    return new_y_tilde

# Generative Model
def state_transition_function(theta_x, u_x, noise_dict):
    # This setup should be fine for generating smooth noise because we are using the same buffer every time in smooth_noise()
    white_noise_sample = rng.normal(0.0, 8.0)
    smooth_w = smooth_noise(2, white_noise_sample, noise_dict)

    # These are the derivatives of the Taylor series approximation given the agent's model of external state velocity
    velocity = theta_x - u_x[0] + smooth_w[0] 
    acceleration = - u_x[1] + smooth_w[1]       
    jerk = - u_x[2] + smooth_w[2]                

    # Generalized Coordinates of Motion
    gen_coords = np.array([velocity, acceleration, jerk, 0])
    print(f"\ngc_noise: {smooth_w}")
    print(f"gc: {gen_coords}")
    return gen_coords

# Generate a y trajectory prediction 
def observation_generating_function(u_x, theta_y, noise_dict):
    # Generate the agent's internal observation prediciton noise
    white_noise_sample = rng.normal(0.0, 0.1)
    smooth_w = smooth_noise(3, white_noise_sample, noise_dict)
    
    # These are the derivatives of the Taylor series approximation given the agent's observation generator 
    y = u_x[0] - theta_y + smooth_w[0]
    velocity = u_x[1] + smooth_w[1]
    acceleration = u_x[2] + smooth_w[2]
    jerk = u_x[3] + smooth_w[3] 

    # Generalized Measurement of Motion
    gen_measuremnt = np.array([y, velocity, acceleration, jerk])
    print(f"\ngm_noise: {smooth_w}")
    print(f"gm: {gen_measuremnt}")
    return gen_measuremnt

def update_hidden_state(u_x, lambda_y_matrix, e_y, lambda_x_matrix, e_x, k, embeddings, state_elements):
    # Free Energy Gradient
    # In this case there is no place to plug in u_x as the slope is the same for all possible values

    # Build the shift operator matrix D = the Kronecker product of S and I of size D=[mc x mc]
    # where S is a 0 matrix of size [m x m] with ones in the superdiagonal 
    # and I is the identity matrix of size [c x c] 
    s = np.eye(embeddings, k=1)      # Create a square identity matrix of size embeddings with ones in the superdiagonal 1 instead of the main diagonal 0  
    i = np.identity(state_elements)  # Another square identity matrix
    d = s * i 

    print(f"d: {d}")

    # Generalized Coordinates Jacobian
    df_du = np.array([[-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, -1]])
    
    # Generalized Measurements Jacobian
    dg_du = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    
    # print(f"\nd: {d}")    
    # print(f"df_du: {df_du}")
    # print(f"lambda_x_matrix: {lambda_x_matrix}")
    # print(f"e_x: {e_x}")
    # print(f"dg_du: {dg_du}")
    # print(f"lambda_y_matrix: {lambda_y_matrix}")
    # print(f"e_y: {e_y}")

    gradient = (d - df_du).T @ lambda_x_matrix @ e_x - (dg_du).T @ lambda_y_matrix @ e_y
    Du_x = np.append(u_x[1:], 0)
    u_x_velocity = Du_x - k * gradient
    new_u_x = u_x + T_STEP * u_x_velocity
    return new_u_x

def recalculate_free_energy(lambda_y_base, lambda_y_vector, e_y, lambda_x_base, lambda_x_vector, e_x):  
    new_f = lambda_y_base * np.sum(e_y[-1] ** 2) + lambda_x_base * np.sum(e_x[-1] ** 2) + np.sum(np.log(lambda_y_vector)) + np.sum(np.log(lambda_x_vector))
    return new_f

def recalculate_prediction_error(u_x, u_x_prev, theta_x, y, theta_y, noise_dict):
    motion_of_expectation = state_transition_function(theta_x, u_x_prev, noise_dict)
    
    Du_x = np.append(u_x[1:], 0) 
    next_e_x = Du_x - motion_of_expectation

    y_prediction = observation_generating_function(u_x, theta_y, noise_dict)
    next_e_y = y - y_prediction
    return next_e_x, next_e_y, y_prediction, motion_of_expectation 

def graph_results(x_star, y, u_x, u_x_predictions, y_predictions, e_x, e_y, f):
    # print(e_x)
    # print(e_y)
    # print(u_x)
    
    # Convert lists of vectors to proper 2D NumPy arrays for correct column slicing [:, i]
    x_star_arr = np.array(x_star)
    y_arr = np.array(y)
    u_x_arr = np.array(u_x[1:])
    u_x_pred_arr = np.array(u_x_predictions)
    y_pred_arr = np.array(y_predictions)
    e_x_arr = np.array(e_x)
    e_y_arr = np.array(e_y)

    # --- GRAPH 1: External state, state estimation prediction, and state estimation ---
    fig, axs = plt.subplots(1, 1, figsize=(12, 8)) 
    axs.plot(T, x_star_arr[:, 0], linewidth=3, label="x* true position")
    axs.plot(T, x_star_arr[:, 1], linewidth=3, label="x* true velocity")
    axs.plot(T, x_star_arr[:, 2], linewidth=3, label="x* true acceleration")
    axs.plot(T, x_star_arr[:, 3], linewidth=3, label="x* true jerk")
    
    axs.plot(T, u_x_arr[:, 0], linewidth=3, linestyle='dashed', label="u_x position", )
    axs.plot(T, u_x_arr[:, 1], linewidth=3, linestyle='dashed', label="u_x velocity")
    axs.plot(T, u_x_arr[:, 2], linewidth=3, linestyle='dashed', label="u_x acceleration")
    axs.plot(T, u_x_arr[:, 3], linewidth=3, linestyle='dashed', label="u_x jerk")
    
    axs.legend(loc="upper right") 
    axs.grid(True, linestyle='--', alpha=0.5)
    axs.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('results/gen_coords/states.png')

    # --- GRAPH 2: Observations of the encoded external state and predictions of the observations of the encoded external state ---
    fig, axs = plt.subplots(1, 1, figsize=(12, 8)) 
    axs.plot(T, y_arr[:, 0], linewidth=3, label="y[0] observation of encoded position")
    axs.plot(T, y_arr[:, 1], linewidth=3, label="y[1] observation of encoded velocity")
    axs.plot(T, y_arr[:, 2], linewidth=3, label="y[2] observation of encoded acceleration")
    axs.plot(T, y_arr[:, 3], linewidth=3, label="y[3] observation of encoded jerk")
    
    axs.plot(T, y_pred_arr[:, 0], linewidth=3,linestyle='dashed', label="y[0] prediction of observation of encoded position")
    axs.plot(T, y_pred_arr[:, 1], linewidth=3, linestyle='dashed', label="y[1] prediction of observation of encoded velocity")
    axs.plot(T, y_pred_arr[:, 2], linewidth=3, linestyle='dashed', label="y[2] prediction of observation of encoded acceleration")
    axs.plot(T, y_pred_arr[:, 3], linewidth=3, linestyle='dashed', label="y[3] prediction of observation of encoded jerk")
    
    axs.legend(loc="upper right") 
    axs.grid(True, linestyle='--', alpha=0.5)
    axs.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('results/gen_coords/observations.png')

    # --- GRAPH 3: State and observation prediction errors ---
    fig, axs = plt.subplots(1, 1, figsize=(12, 8)) 
    axs.plot(T, e_x_arr[:, 0], linewidth=3, label="e_x[0] state prediction error of position")
    axs.plot(T, e_x_arr[:, 1], linewidth=3, label="e_x[1] state prediction error of velocity")
    axs.plot(T, e_x_arr[:, 2], linewidth=3, label="e_x[2] state prediction error of acceleration")

    axs.plot(T, e_y_arr[:, 0], linewidth=3, linestyle='dashed', label="e_y[0] observation prediction error of encoded position")
    axs.plot(T, e_y_arr[:, 1], linewidth=3, linestyle='dashed', label="e_y[1] observation prediction error of encoded velocity")
    axs.plot(T, e_y_arr[:, 2], linewidth=3, linestyle='dashed', label="e_y[2] observation prediction error of encoded acceleration")
    axs.plot(T, e_y_arr[:, 3], linewidth=3, linestyle='dashed', label="e_y[3] observation prediction error of encoded jerk")

    axs.legend(loc="upper right") 
    axs.grid(True, linestyle='--', alpha=0.5)
    axs.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('results/gen_coords/errors.png')

    # --- GRAPH 4: u_x and u_x predictions ---
    fig, axs = plt.subplots(1, 1, figsize=(12, 8)) 
    axs.plot(T, u_x_arr[:, 0], linewidth=3, label="u_x[0] estimation of position")
    axs.plot(T, u_x_arr[:, 1], linewidth=3, label="u_x[1] estimation of encoded velocity")
    axs.plot(T, u_x_arr[:, 2], linewidth=3, label="u_x[2] estimation of encoded acceleration")
    axs.plot(T, u_x_arr[:, 3], linewidth=3, label="u_x[3] estimation of encoded jerk")

    axs.plot(T, u_x_pred_arr[:, 0], linewidth=3, linestyle='dashed', label="u_x_pred[0] prediction of estimation of position")
    axs.plot(T, u_x_pred_arr[:, 1], linewidth=3, linestyle='dashed', label="u_x_pred[1] prediction of estimation of velocity")
    axs.plot(T, u_x_pred_arr[:, 2], linewidth=3, linestyle='dashed', label="u_x_pred[2] prediction of estimation of acceleration")
    axs.plot(T, u_x_pred_arr[:, 3], linewidth=3, linestyle='dashed', label="u_x_pred[3] prediction of estimation of jerk")

    axs.legend(loc="upper right") 
    axs.grid(True, linestyle='--', alpha=0.5)
    axs.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('results/gen_coords/u_x.png')
    
    fig, axs = plt.subplots(1, 1, figsize=(12, 8)) 
    axs.plot(T, f, label="free energy")
    axs.legend(loc="upper right") # Moved legend slightly out of the way
    axs.grid(True, linestyle='--', alpha=0.5)
    axs.set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('results/gen_coords/free_energy.png')

def main():
    noise_dict = build_noise_kernels()

    ####### Generative process vars #######

    theta_star_x = 10 
    theta_star_y = 3
    
    x_star = [[5, 0, 0, 0]]                                            # A list containing the external state of x for each time step. The initial external state is 5. 
    y_observations = [generate_observation(x_star[-1], theta_star_y, noise_dict)]  # A list containing the agent's observation for each time step. The initial observation is calculated here using the observation generating function. 

    ####### Generative model vars #######
    
    embeddings = 4         # Generalized Coordinates Embedding Depth 
    state_elements = 1     # The number of states per observation 
    k = 0.1                # The learning rate kappa of the gradient descent step
    u_x = [[15, 7, 4, 1]]  # The list of hidden state estimations at each time step. The initial guess is 15, 7, 4, 1 but can be set to anything. 
    u_x_predictions = []   # The agent's prediction of the next estimate of higher order derivatives of position given the last estimate 
    y_predictions = []     # The agent's expectation of y given u_x (the expectation of x). This is calcualted in observation_generating_function()
    theta_x = 10           # Theta_x is used in the agents state transition function theta_x - u_x to generate it's hypothesis regarding the new value of u_x.      
    theta_y = 3            # Theta_y is the agent's approximation of theta_star_y from the environment and is used to predict the next observation. 

    # The precisions (inverse variances) of hidden states x and observations y 
    lambda_x_base = 0.2
    lambda_y_base = 50
    lambda_x_matrix = lambda_x_base * np.identity(embeddings)
    lambda_y_matrix = lambda_y_base * np.identity(embeddings)
    lambda_x_vector = np.diag(lambda_x_matrix)
    lambda_y_vector = np.diag(lambda_y_matrix)

    u_x_predictions.append(state_transition_function(theta_x, u_x[-1], noise_dict))  # Initial expectation of motion i.e. the derivatives of the Taylor Series approximation  
    Du_x = np.append(u_x[-1][1:], [0], axis=0)                           # Initial motion of expectation i.e. the shifted expectation of motion
    e_x = [Du_x - u_x_predictions[-1]]                                   # Initial state prediction error vector    

    y_predictions.append(observation_generating_function(u_x[-1], theta_y, noise_dict))  # Initial expectation of motion i.e. the derivatives of the Taylor Series approximation 
    e_y = [y_observations[-1] - y_predictions[-1]]                           # Initial state prediction error vector    

    # Initial Free Energy
    f = [lambda_y_base * np.sum(e_y[-1] ** 2) + lambda_x_base * np.sum(e_x[-1] ** 2) + np.sum(np.log(lambda_y_vector)) + np.sum(np.log(lambda_x_vector))] 

    # Initial Update 
    u_x.append(update_hidden_state(u_x[-1], lambda_y_matrix, e_y[-1], lambda_x_matrix, e_x[-1], k, embeddings, state_elements))

    # AIF
    for _ in LOOP_T:
        print("\n")
        ####### Generative Process #######
        x_star.append(generate_state(x_star[-1], theta_star_x, noise_dict))           # Generate new external state  
        y_observations.append(generate_observation(x_star[-1], theta_y, noise_dict))  # Generate new observation
        
        ####### Generative Model #######
        # Update hidden state using observation and generative model
        u_x.append(update_hidden_state(u_x[-1], lambda_y_matrix, e_y[-1], lambda_x_matrix, e_x[-1], k, embeddings, state_elements))
        
        # Update free energy calculation
        f.append(recalculate_free_energy(lambda_y_base, lambda_y_vector, e_y[-1], lambda_x_base, lambda_x_vector, e_x[-1]))

        # Update prediction errors using new observation and hidden state prediction 
        next_e_x, next_e_y, y_prediction, u_x_prediction = recalculate_prediction_error(u_x[-1], u_x[-2], theta_x, y_observations[-1], theta_y, noise_dict)
        e_x.append(next_e_x)
        e_y.append(next_e_y)
        y_predictions.append(y_prediction)
        u_x_predictions.append(u_x_prediction)

    graph_results(x_star, y_observations, u_x, u_x_predictions, y_predictions, e_x, e_y, f)

main()