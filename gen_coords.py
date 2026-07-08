"""
Implementation of example 6.6 in Fundamentals of Active Inference

Generalized Coordinates for External State Trajectory Prediction

The earlier examples have shown how an agent percieves an external state. How can an agent predict the future 
trajectory of the state to allow planning? One method is using generalized coordinates of motion. 

Generalized coordinates (GC) are a vector of the values of higher order derivatives of position (i.e. velocity, acceleration, jerk, 
snap, crackle, pop etc.). 

(rewrite this part)

It is worth noting that the agent never truly measures the external state but that the VFE gradient update of the GC uses the observed data 
to provide an indirect measurement.  

Also, it is worth noting that every "measurement" and prediction of the GC is a full trajectory prediction of the external state. This means 
that you can take the measurement or prediction at any time t and predict the trajectory as far into the future as you want. The accuracy goes 
down as t gets larger. If you consider modeling the behavior of a complex organism, one GC vector would struggle to predict every movement the 
animal will ever make, but each update of the trajectory predicts what it is likely to do in the immediate future. 

Anyway, this was a lot to take in and this explanation skips over the details. It has been worth it to piece it all together and it would
have taken way, way longer without the book and my model gf (Gemini Flash) (thanks babe). I'm not going to explain the derivations to you. 
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

# Colored noise kernel per Gemini (thanks babe)
# The following is used to turn white noise to colored noise because white noise is undifferentiable,
# and we need to differentiate the noise when building the trajectories.
# In the future the predictability of the colored noise will allow the model to learn the patterns and correct itself. 
# White noise is random and therefore impossible to learn. 
# --- Appendix C.94 Smooth Noise Configuration ---
sigma_w = 8.0          # Matching your rng.normal(0.0, 8.0) external state variance
sigma_y = 0.1          # Observation variance 
gamma_val = 1.0        # Smoothness parameter gamma
y_gamma_val = 1.0      # y smoothness
window_radius = 40     # How far back in time to blend

# Build the h_omega weights and it's derivatives relative to a time-lag axis. 
# It is much harder to understand this than it is to derive it and code it up. I wrote it all down first.   
t_axis = np.arange(-window_radius, window_radius + 1) * T_STEP
scaling_factor_0 = np.sqrt(T_STEP / (sigma_w * np.sqrt(np.pi)))
scaling_factor_1 = (-gamma_val / 2) * t_axis
scaling_factor_2 = (-gamma_val / 2)

noise_kernel_0 = scaling_factor_0 * np.exp(-(t_axis**2 / 4) * gamma_val)
noise_kernel_0 /= np.sum(noise_kernel_0) # Normalize to preserve total variance

# The following are the derivatives that are currently used for both the external state and the generative coordinates. 
# I could have used a differrent variance but I'm sticking with the external state's variance of 8 
# so I don't have to rewrite all of this again like for the obervation kernel below. 
# If this prooves to be too much noise for the GCs I will create a sepparate set of kernels with a lower variance. 
noise_kernel_1 = noise_kernel_0 * scaling_factor_1
noise_kernel_1 /= np.sum(noise_kernel_1) # Normalize to preserve total variance

noise_kernel_2 = noise_kernel_1 * scaling_factor_1 + noise_kernel_0 * scaling_factor_2
noise_kernel_2 /= np.sum(noise_kernel_2) # Normalize to preserve total variance

# The observations need there own noise kernels because it has variance sigma_y
scaling_factor_y_0 = np.sqrt(T_STEP / (sigma_y * np.sqrt(np.pi)))
scaling_factor_y_1 = (-y_gamma_val / 2) * t_axis
scaling_factor_y_2 = (-y_gamma_val / 2)

noise_kernel_y_0 = scaling_factor_y_0 * np.exp(-(t_axis**2 / 4) * y_gamma_val)
noise_kernel_y_0 /= np.sum(noise_kernel_0) # Normalize to preserve total variance

# The following are the derivatives that are used for the generative measurements. 
noise_kernel_y_1 = noise_kernel_y_0 * scaling_factor_y_1
noise_kernel_y_1 /= np.sum(noise_kernel_y_1) # Normalize to preserve total variance

noise_kernel_y_2 = noise_kernel_y_1 * scaling_factor_y_1 + noise_kernel_y_0 * scaling_factor_y_2
noise_kernel_y_2 /= np.sum(noise_kernel_y_2) # Normalize to preserve total variance

noise_kernel_y_3 = noise_kernel_y_2 * scaling_factor_y_1 - y_gamma_val * scaling_factor_y_1
noise_kernel_y_3 /= np.sum(noise_kernel_y_3) # Normalize to preserve total variance

# Sliding memory buffers for white noise, filled with zeros to start. There is one for each noise type. 
ext_x_raw_noise_buffer = np.zeros(len(noise_kernel_0))       # External state x white noise
y_raw_noise_buffer = np.zeros(len(noise_kernel_0))           # Observation and generalized measuremnts white noise
gen_coords_raw_noise_buffer = np.zeros(len(noise_kernel_0))  # Generative coordinates noise

# Smooth Noise depending on noise type i.e. x, y, or gen_coords noise. 
def smooth_noise(noise_type, white_noise):
    # Use the global keyword to change the stored values. 
    global ext_x_raw_noise_buffer, y_raw_noise_buffer, gen_coords_raw_noise_buffer

    # external state x noise
    if noise_type == 0:
        # 1. Slide the memory buffer (drop oldest, add newest)
        ext_x_raw_noise_buffer = np.append(ext_x_raw_noise_buffer[1:], white_noise)

        # 2. Convolve! Dot product blends the history with the h_\omega weights
        smooth_w = [np.dot(ext_x_raw_noise_buffer, noise_kernel_0),
                    np.dot(ext_x_raw_noise_buffer, noise_kernel_1),
                    np.dot(ext_x_raw_noise_buffer, noise_kernel_2)]

    # y and generalized measurements noise buffer 
    if noise_type == 1:
        # 1. Slide the memory buffer (drop oldest, add newest)
        y_raw_noise_buffer = np.append(y_raw_noise_buffer[1:], white_noise)

        # 2. Convolve! Dot product blends the history with the h_omega weights
        smooth_w = [np.dot(y_raw_noise_buffer, noise_kernel_y_0), 
                    np.dot(y_raw_noise_buffer, noise_kernel_y_1), 
                    np.dot(y_raw_noise_buffer, noise_kernel_y_2), 
                    np.dot(y_raw_noise_buffer, noise_kernel_y_3)]

    # Generalized Coordinates noise buffer
    if noise_type == 2:
        # 1. Slide the memory buffer (drop oldest, add newest)
        gen_coords_raw_noise_buffer = np.append(gen_coords_raw_noise_buffer[1:], white_noise)

        # 2. Convolve! Dot product blends the history with the h_\omega weights
        # I could have put this in a for loop but it is more readable and arguably less complicated this way  
        smooth_w = [np.dot(gen_coords_raw_noise_buffer, noise_kernel_0), 
                    np.dot(gen_coords_raw_noise_buffer, noise_kernel_1), 
                    np.dot(gen_coords_raw_noise_buffer, noise_kernel_2)]

    return smooth_w

# Generative Process
def generate_state(x_star, theta_star_x):
    white_noise_sample = rng.normal(0.0, 8.0)

    smooth_w = smooth_noise(0, white_noise_sample)

    # Velocity of external state
    velocity = theta_star_x - x_star[0] + smooth_w[0]
    acceleration = - x_star[0] + smooth_w[1]
    jerk = smooth_w[2]

    # x* update
    new_position = x_star[0] + T_STEP * velocity 
    new_velocity = x_star[1] + T_STEP * acceleration
    new_aceleration = x_star[2] + T_STEP * jerk
    new_jerk = smooth_w[2] + T_STEP * 0

    new_x_star = [new_position, new_velocity, new_aceleration, new_jerk]
    return new_x_star

# Generate an observation of the external state position 
def generate_observation(x_star, theta_y):
    white_noise_sample = rng.normal(0.0, 0.1)

    smooth_w = smooth_noise(1, white_noise_sample)

    new_y = x_star[0] - theta_y + smooth_w[0]
    new_y_velocity = x_star[1] + smooth_w[1]
    new_y_acceleration = x_star[2] + smooth_w[2]
    new_y_jerk = x_star[3] + smooth_w[3]

    new_y_tilde = np.array([new_y, new_y_velocity, new_y_acceleration, new_y_jerk])
    return new_y_tilde

# Generative Model
def state_transition_function(theta_x, u_x):
    # This setup should be fine for generating smooth noise because we are using the same buffer every time in smooth_noise()
    white_noise_sample = rng.normal(0.0, 8.0)
    smooth_w = smooth_noise(2, white_noise_sample)

    # These are the derivatives of the Taylor series approximation given the agent's model of external state velocity
    velocity = theta_x - u_x[0] + smooth_w[0]  
    acceleration = -1 * u_x[1] + smooth_w[1]       
    jerk = -1 * u_x[2] + smooth_w[2]                

    # Generalized Coordinates of Motion
    gen_coords = np.array([velocity, acceleration, jerk, 0])

    return gen_coords

# Generate a y trajectory prediction 
def observation_generating_function(u_x, theta_y):
    # Generate the agent's internal observation prediciton noise
    white_noise_sample = rng.normal(0.0, 8.0)
    smooth_w = smooth_noise(1, white_noise_sample)
    
    # These are the derivatives of the Taylor series approximation given the agent's observation generator 
    y = u_x[0] - theta_y + smooth_w[0]
    velocity = u_x[1] + smooth_w[1]
    acceleration = u_x[2] + smooth_w[2]
    jerk = u_x[3] + smooth_w[3] 

    # Generalized Measurement of Motion
    gen_measuremnt = np.array([y, velocity, acceleration, jerk])
    print(f"GM: {gen_measuremnt}")

    return gen_measuremnt

def update_hidden_state(motion_of_expectation, lambda_y_matrix, e_y, lambda_x_matrix, e_x, k, embeddings, state_elements):
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
    
    # Those are calculated by hand. Maybe in the future I can use some auto differentiation given the velocity function. 
    # print(f"d - df: {d- df_du}")
    # print(f"lambda_x_matrix: {lambda_x_matrix}")
    # print(f"e_x: {e_x}")
    gradient = (d - df_du).T @ lambda_x_matrix @ e_x + (dg_du).T @ lambda_y_matrix @ e_y
    Du_x = np.append(motion_of_expectation[1:], 0)
    u_x_velocity = Du_x - k * gradient
    new_u_x = motion_of_expectation + T_STEP * u_x_velocity
    return new_u_x

def recalculate_free_energy(lambda_y_base, lambda_y_vector, e_y, lambda_x_base, lambda_x_vector, e_x):  
    new_f = lambda_y_base * (e_y[-1] ** 2) + lambda_x_base * (e_x[-1] ** 2) + np.sum(np.log(lambda_y_vector)) + np.sum(np.log(lambda_x_vector))
    return new_f

def recalculate_prediction_error(u_x, u_x_prev, theta_x, y, theta_y):
    # The agent expects the external state to move at a velocity of x_n
    motion_of_expectation = state_transition_function(theta_x, u_x_prev)
    
    Du_x = np.append(u_x[1:], 0) 
    next_e_x = Du_x - motion_of_expectation

    y_prediction = observation_generating_function(u_x, theta_y)
    next_e_y = y - y_prediction
    return next_e_x, next_e_y, y_prediction, motion_of_expectation 

def graph_results(x_star, y, u_x, u_x_predictions, y_predictions, e_x, e_y, f):
    # Changed to 1, 1. axs is now a single object, not an array!
    fig, axs = plt.subplots(1, 1, figsize=(12, 8)) 

    # Convert lists of vectors to proper 2D NumPy arrays for correct column slicing [:, i]
    x_star_arr = np.array(x_star)
    y_arr = np.array(y)
    u_x_arr = np.array(u_x[1:])
    u_x_pred_arr = np.array(u_x_predictions)
    y_pred_arr = np.array(y_predictions)

    # --- GRAPH 1: States and Observations ---
    # Called directly on axs (no [0] subscript)
    axs.plot(T, x_star_arr[:, 0], label="x* true position")
    axs.plot(T, x_star_arr[:, 1], label="x* true velocity")
    axs.plot(T, x_star_arr[:, 2], label="x* true acceleration")
    axs.plot(T, x_star_arr[:, 3], label="x* true jerk")
 
    axs.plot(T, y_arr[:, 0], label="y[0] observation of encoded position")
    axs.plot(T, y_arr[:, 1], label="y[1] observation of encoded velocity")
    axs.plot(T, y_arr[:, 2], label="y[2] observation of encoded acceleration")
    axs.plot(T, y_arr[:, 3], label="y[3] observation of encoded jerk")

    axs.plot(T, u_x_arr[:, 0], label="u_x[0] estimated position")
    axs.plot(T, u_x_arr[:, 1], label="u_x[1] estimated velocity")
    axs.plot(T, u_x_arr[:, 2], label="u_x[2] estimated acceleration")
    axs.plot(T, u_x_arr[:, 3], label="u_x[3] estimated jerk")

    axs.plot(T, u_x_pred_arr[:, 0], label="f(u_x[0]) predicted estimate of velocity")
    axs.plot(T, u_x_pred_arr[:, 1], label="f(u_x[1]) predicted estimate of acceleration")
    axs.plot(T, u_x_pred_arr[:, 2], label="f(u_x[2]) predicted estimate of jerk")
    axs.plot(T, u_x_pred_arr[:, 3], label="f(u_x[3]) predicted estimate of snap")

    axs.plot(T, y_pred_arr[:, 0], label="g(u_x)[0] prediction of encoded position")
    axs.plot(T, y_pred_arr[:, 1], label="g(u_x)[1] prediction of encoded velocity")
    axs.plot(T, y_pred_arr[:, 2], label="g(u_x)[2] prediction of encoded acceleration")
    axs.plot(T, y_pred_arr[:, 3], label="g(u_x)[3] prediction of encoded jerk")

    axs.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0)) # Moved legend slightly out of the way
    axs.grid(True, linestyle='--', alpha=0.5)
    axs.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('results/gen_coords_results.png')

def main():
    ####### Generative process vars #######

    theta_star_x = 10 
    theta_star_y = 3
    
    x_star = [[5, 0, 0, 0]]  # A list containing the external state of x for each time step. The initial external state is 5. 
    # A list containing the agent's observation for each time step. The initial observation is calculated here using the observation generating function. 
    y_observations = [generate_observation(x_star[-1], theta_star_y)]

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

    # Initial expectation of motion i.e. the derivatives of the Taylor Series approximation 
    motion_of_expectation = state_transition_function(theta_x, u_x[-1]) 
    u_x_predictions.append(motion_of_expectation)

    # Initial motion of expectation i.e. the shifted expectation of motion
    # I haven't implemented this as the matrix multiplication but it can be implemented as such. (see book p144)
    Du_x = np.append(motion_of_expectation[1:], 0)

    # Initial state prediction error vector    
    e_x = [Du_x - motion_of_expectation]
    print(e_x)

    # Initial expectation of motion i.e. the derivatives of the Taylor Series approximation 
    y_predictions.append(observation_generating_function(u_x[-1], theta_y))

    # Initial state prediction error vector    
    e_y = [y_observations[-1] - y_predictions[-1]]

    # Initial Free Energy
    ############################## THERE MAY BE A MATHEMATICAL ERROR HERE ##############################
    f = [lambda_y_base * (e_y[-1] ** 2) + lambda_x_base * (e_x[-1] ** 2) + np.sum(np.log(lambda_y_vector)) + np.sum(np.log(lambda_x_vector))] 

    # Initial Update 
    u_x.append(update_hidden_state(motion_of_expectation, lambda_y_matrix, e_y[-1], lambda_x_matrix, e_x[-1], k, embeddings, state_elements))

    # AIF
    for _ in LOOP_T:
        ####### Generative Process #######

        # Generate new external state and observation 
        x_star.append(generate_state(x_star[-1], theta_star_x))    
        y_observations.append(generate_observation(x_star[-1], theta_y))
        
        # print(f"External state: {x_star}")
        # print(f"Observation: {y}")
        
        ####### Generative Model #######
        # Update hidden state using observation and generative model
        u_x.append(update_hidden_state(u_x[-1], lambda_y_matrix, e_y[-1], lambda_x_matrix, e_x[-1], k, embeddings, state_elements))
        
        # Update free energy calculation
        f.append(recalculate_free_energy(lambda_y_base, lambda_y_vector, e_y[-1], lambda_x_base, lambda_x_vector, e_x[-1]))

        # Update prediction errors using new observation and hidden state prediction 
        next_e_x, next_e_y, y_prediction, u_x_prediction = recalculate_prediction_error(u_x[-1], u_x[-2], theta_x, y_observations[-1], theta_y)
        e_x.append(next_e_x)
        e_y.append(next_e_y)
        y_predictions.append(y_prediction)
        u_x_predictions.append(u_x_prediction)

    graph_results(x_star, y_observations, u_x, u_x_predictions, y_predictions, e_x, e_y, f)

main()