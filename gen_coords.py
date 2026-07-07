"""
Implementation of example 6.6 in Fundamentals of Active Inference

Generalized Coordinates for External State Trajectory Prediction

The earlier examples have shown how an agent percieves an external state. How can an agent predict the future 
trajectory of the state to allow planning? One method is using generalized coordinates of motion. 

Generalized coordinates (GC) are a vector of the values of higher order derivatives of position (i.e. velocity, acceleration, jerk, 
snap, crackle, pop etc.). For an agent to percive and predict an external state trajectory, it makes an observation y and an initial guess 
of the GC. To make this initial guess of the GC at position x and time t, the agent uses its internal physics model to find the velocity, 
and a Taylor Series approximation (TSA) to extrapolate the values of all other higher order derivatives. This means the prediction is only 
dependent on the position and the velocity function. If you measure the true GC at time t you can subtract the prediction from the 
measurement to calculate the error between them. The error is then used to construct a Variational Free Energy (VFE) gradient (the slope of
VFE for each GC element given any GC vector), which is used to update the generalized coordinates. The problem is often that directly 
measuring the GC is not possible. Instead, during the first iteration of the loop you assume that the true GC can be found by shifting the 
TSA GC_prediction one index to the left such that if the TSA GC_prediction = [1, 2, 3] the true GC must be D_GC = [2, 3, 0]. This way the 
GC update is GC@t+1 = GC[i]@t + delta_t * D_GC[i]@t. In english, each element of the shifted prediction D_GC is the slope of each element of 
the unshifted TSA GC_prediction. An example is new_position = position + delta_t * velocity. Once this first error is calculated using the 
shifted TSA GC_prediction and the gradient updates the measurement with GC[t+1], the loop starts over with observation two. Observation two 
is used to find the TSA GC_prediction and the second error value can be then be calcutated with GC[t+1] and TSA GC_prediction[t+1]. This is 
done by again shifting the GC measurement to D_GC and subtracting so that error 2 = D_GC[t+1] - TSA GC_prediction[t+1]. The unshifted GC 
include the position in index 1. The error compares only the higher order derivatives of position. 

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

Correct the loop: 
generate full external x_tilde and pass it to the agent to calcualte true y
calculate the agents y prediction using the initial u_x_tilde belief
fix y error calcualtion by using true y - y prediction

move vars to main loop
put noise vars into initialization function in main before loop 
add a "light intensity" simile to explain why the observations of the trajectory predictions are different from the trajectory predictions themselves
maybe write up a separate noise kernel set for the generative coordinates instead of using the x noise. 
maybe use a new seed for each noise type 
add noise to TSA trajectory embeddings 
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
    velocity = theta_star_x - x_star + smooth_w[0]
    acceleration = - x_star + smooth_w[1]
    jerk = smooth_w[2]

    # x* update
    new_x_star = x_star * T_STEP
    return new_x_star

# Generate an observation of the external state position 
def generate_observation(x_star, theta_y):
    white_noise_sample = rng.normal(0.0, 0.1)

    smooth_w = smooth_noise(1, white_noise_sample)

    new_y = x_star - theta_y + smooth_w[0]
    return new_y, smooth_w

# Generative Model
def state_transition_function(theta_x, u_x):
    # This setup should be fine for generating smooth noise because we are using the same buffer every time in smooth_noise()
    white_noise_sample = rng.normal(0.0, 8.0)
    smooth_w = smooth_noise(2, white_noise_sample)

    # These are the derivatives of the Taylor series approximation given the agent's model of external state velocity
    velocity = theta_x - u_x + smooth_w[0]  # smooth_w[1] is the first derivative 
    acceleration = -1 + smooth_w[1]       # smooth_w[2] is the second derivative
    jerk = 0 + smooth_w[2]                # smooth_w[3] is the third derivative

    # Generalized Coordinates of Motion
    gen_coords = np.array([velocity, acceleration, jerk])

    return gen_coords

# Generate a y trajectory prediction 
# i.e. what the "light intensity" observation of the predicted external state velocity, acceleration, jerk, etc. will be  
# The y noise vector is generated using the white noise sample from the observation in the generative process step in generate_observation(). 
def observation_generating_function(u_x, theta_y, y_noise):
    # These are the derivatives of the Taylor series approximation given the agent's model of observation velocity
    velocity = u_x - theta_y + y_noise[1]
    acceleration = 1 + y_noise[2]
    jerk = 0 + y_noise[3] 

    # Generalized Measurement of Motion
    gen_measuremnt = np.array([velocity, acceleration, jerk])

    return gen_measuremnt

def update_hidden_state(u_x_tilde, lambda_y_matrix, e_y, dg_du, lambda_x_matrix, e_x, df_du, k, embeddings, state_elements):
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
    df_du = np.array([[0, 0, 0]
                      [0, 0, 0]
                      [0, 0, 0]])
    
    # Generalized Measurements Jacobian
    dg_du = np.array([[0, 0, 0]
                      [0, 0, 0]
                      [0, 0, 0]])
    
    # Those are calculated by hand. Maybe in the future I can use some auto differentiation given the velocity function. 

    gradient = (d - df_du).T * lambda_x_matrix * e_x + (dg_du).T * lambda_y_matrix * e_y
    Du_x = np.append(u_x_tilde[1:], 0)
    u_x_velocity = Du_x - k * gradient
    new_u_x = u_x_tilde + T_STEP * u_x_velocity
    return new_u_x

def recalculate_free_energy(lambda_y_base, lambda_y_vector, e_y, lambda_x_base, lambda_x_vector, e_x):  
    new_f = lambda_y_base * (e_y[-1] ** 2) + lambda_x_base * (e_x[-1] ** 2) + np.sum(np.log(lambda_y_vector)) + np.sum(np.log(lambda_x_vector))
    return new_f

def recalculate_prediction_error(u_x, u_x_prev, theta_x, y, theta_y, y_noise):
    # The agnet expects the external state to move at a velocity of x_n
    x_n = state_transition_function(theta_x, u_x_prev)
    
    # u_x is the agent's estimatate of the external state position at time t
    # x_n is the agent's prediction of the velocity at time t-1, 
    # meaning subtracting it from u_x gives you the error of the velocity prediction.
    next_e_x = u_x - x_n

    u_y = observation_generating_function(u_x, theta_y, y_noise)
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
    plt.savefig('results/gen_coords_results.png')

def main():
    ####### Generative process vars #######

    theta_star_x = 10 
    theta_star_y = 3
    
    # A list containing the external state of x for each time step. The initial external state is 5. 
    x_star = [[5, 0, 0, 0]]

    # A list containing the agent's observation for each time step. The initial observation is calculated here using the observation generating function. 
    initial_observation, initial_y_noise = generate_observation(x_star[-1], theta_star_y)
    y = [[initial_observation]]

    ####### Generative model vars #######
    
    # Generalized Coordinates Embedding Depth
    embeddings = 3 

    # The number of states per observation 
    state_elements = 1

    # The learning rate kappa of the gradient descent step
    k = 0.1

    # The list of hidden state estimations at each time step. The initial guess is 15 but can be set to anything. 
    u_x = np.array([[15]])

    # Theta_x is used in the agents state transition function theta_x - u_x to generate it's hypothesis regarding the new value of u_x.  
    theta_x = 10
    
    # Theta_y is the agent's approximation of theta_star_y from the environment and is used to predict the next observation. 
    theta_y = 3

    # The precisions (inverse variances) of hidden states x and observations y 
    lambda_x_base = 0.2
    lambda_y_base = 50
    lambda_x_matrix = lambda_x_base * np.identity(embeddings)
    lambda_y_matrix = lambda_y_base * np.identity(embeddings)
    lambda_x_vector = np.diag(lambda_x_matrix)
    lambda_y_vector = np.diag(lambda_y_matrix)

    # Initial state prediction error vector
    # Typically the error is calculated using the observed generalized coordinates u_x and the predicted state, 
    # but there is no observation on the first step so we use the shifted prediction Du_x to get the loop off the ground
    
    # Initial expectation of motion i.e. the derivatives of the Taylor Series approximation 
    u_x_tilde = state_transition_function(theta_x, u_x[-1][0]) 
    u_x = np.append(u_x, [u_x_tilde], axis=1)  # Now u_x = [[position, vel, acc, jk]]

    # Initial motion of expectation i.e. the shifted expectation of motion
    # I haven't implemented this as the matrix multiplication but it can be implemented as such. (see book p144)
    Du_x = np.append(u_x_tilde[1:], 0)

    # Initial state prediction error vector    
    e_x = Du_x - u_x_tilde

    # Initial observation prediction error

    # Initial expectation of motion i.e. the derivatives of the Taylor Series approximation 
    y_tilde = observation_generating_function(u_x[-1][0], theta_y, initial_y_noise)
    y = np.append(y, [y_tilde], axis=1)  # Now y = [[position, vel, acc, jk]]

    # Initial motion of expectation i.e. the shifted expectation of motion
    # I haven't implemented this as the matrix multiplication but it can be implemented as such. (see book p144)
    Du_y = np.append(y_tilde[1:], 0)

    # Initial state prediction error vector    
    e_y = Du_y - y_tilde

    # Initial Free Energy
    ############################## THERE MAY BE A MATHEMATICAL ERROR HERE ##############################
    f = [lambda_y_base * (e_y[-1] ** 2) + lambda_x_base * (e_x[-1] ** 2) + np.sum(np.log(lambda_y_vector)) + np.sum(np.log(lambda_x_vector))] 

    # AIF
    for _ in LOOP_T:
        ####### Generative Process #######

        # Generate new external state and observation 
        x_star.append(generate_state(x_star[-1], theta_star_x))    
        observation, y_noise = generate_observation(x_star[-1], theta_y)
        y.append(observation)
        # print(f"External state: {x_star}")
        # print(f"Observation: {y}")
        
        ####### Generative Model #######

        # Update hidden state using observation and generative model
        u_x.append(update_hidden_state(u_x[-1], lambda_y_matrix, e_y[-1], lambda_x_matrix, e_x[-1], k, embeddings, state_elements))
        
        # Update free energy calculation
        f.append(recalculate_free_energy(lambda_y_base, lambda_y_vector, e_y[-1], lambda_x_base, lambda_x_vector, e_x[-1]))

        # Update prediction errors using new observation and hidden state prediction 
        next_e_x, next_e_y, next_u_y = recalculate_prediction_error(u_x[-1], u_x[-2], theta_x, y[-1], theta_y, y_noise)
        e_x.append(next_e_x)
        e_y.append(next_e_y)
        u_y.append(next_u_y)

    graph_results(x_star, y, u_x, u_y, e_x, e_y, f)

main()