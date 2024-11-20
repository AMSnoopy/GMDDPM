import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import train_gmddpm
from tqdm import tqdm
import os
import GMDDPM

def generate_onehot_from_random(num_classes):
    # Randomly generate a 22-dimensional vector with each value being 0 or 1
    random_classes = torch.randint(0, 2, (num_classes,))

    # Convert the 22-dimensional vector to one-hot encoding with 44 dimensions
    onehot = torch.zeros(num_classes * 2)
    onehot[::2] = random_classes  # Assign class values to even indices
    onehot[1::2] = 1 - random_classes  # Assign 1 - class values to odd indices
  
    return onehot

def reverse_diffusion(model, gaussian_process, multinomial_process):
    # Start from standard normal noise for continuous features
    x_t_num = torch.randn(continuous_dim)  # Gaussian noise (continuous part)
    x_t_cat = generate_onehot_from_random(22)  # Random discrete noise (44 dimensions, discrete part)

    # Perform reverse diffusion starting from the maximum timestep down to 0
    for t in range(gaussian_process.timesteps - 1, -1, -1):
        t = torch.tensor(t, dtype=torch.long)  # Current timestep

        # Combine discrete and continuous parts
        x_t = torch.cat([x_t_cat, x_t_num])

        # Predict x_0^ (discrete and continuous) using the model
        predicted_cat_num = model(x_t, t)
        predicted_cat = predicted_cat_num[:discrete_dim]  # Predicted discrete part
        predicted_num = predicted_cat_num[discrete_dim:]  # Predicted continuous part
     
        # Compute alpha_t and related values
        alpha_t = torch.tensor(gaussian_process.alphas[t], dtype=torch.float32)
        alpha_bar_t = torch.tensor(gaussian_process.alphas_cumprod[t], dtype=torch.float32)
        alpha_bar_t_prev = torch.tensor(gaussian_process.alphas_cumprod[t - 1], dtype=torch.float32)

        # Reverse update for continuous data using Gaussian reverse diffusion
        x_t_num = (1 / torch.sqrt(alpha_t)) * (
            x_t_num - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_num)
        
        # Add noise at intermediate steps
        if t > 0:
            z = torch.randn_like(x_t_num)  # Generate random noise
            sigma_t = torch.sqrt((1 - alpha_bar_t) / (1 - alpha_bar_t_prev) * (1 - alpha_t))
            x_t_num += sigma_t * z
       
        # Reverse update for discrete data using multinomial reverse diffusion
        theta_post = GMDDPM.compute_theta_post(x_t_cat, predicted_cat, alpha_t, alpha_bar_t)
        theta_post = theta_post.squeeze(0)

        # Sampling from theta_post and convert to one-hot encoding
        num_classes = theta_post.shape[0] // 2
        samples = torch.multinomial(theta_post.view(-1, 2), num_samples=1).view(num_classes)
        xt_onehot = torch.zeros(num_classes * 2)
        xt_onehot.scatter_(0, samples + torch.arange(num_classes) * 2, 1)
        x_t_cat = xt_onehot

        if t == 0:  # At the final step, use softmax for the discrete part
            theta_post_reshaped = predicted_cat.view(22, 2)
            theta_post_softmax = F.softmax(theta_post_reshaped, dim=-1)
            x_t_cat = torch.argmax(theta_post_softmax, dim=-1)
            x_t_num = (x_t_num + 1) / 2  # Normalize continuous data to [0, 1]

    return x_t_num, x_t_cat

# Get dimensions for discrete and continuous features
discrete_dim = train_gmddpm.discrete_dim
continuous_dim = train_gmddpm.continuous_dim

# Load the pre-trained model
model = train_gmddpm.model
model = train_gmddpm.load_model(model, 'models//gmddpm_ct_model.pth')

# Initialize lists to store generated data
continuous_data_list = []
categorical_data_list = []

# Define a function to check if there are too many small values
def has_too_many_small_values(data, threshold=1.0e-04, max_count=1):
    return np.sum(data < threshold) > max_count

# Generate data through reverse diffusion
for i in range(1):
    categorical_data_list.clear()
    continuous_data_list.clear()
    for _ in tqdm(range(1000)):
        # Generate one row of continuous and discrete data
        generated_num, generated_cat = reverse_diffusion(model, train_gmddpm.gaussian_process, train_gmddpm.multinomial_process)
        
        # Convert generated data to numpy arrays
        continuous_data = generated_num.detach().numpy().flatten()
        categorical_data = generated_cat.detach().numpy().flatten()
        
        # Append generated data to the respective lists
        continuous_data_list.append(continuous_data)
        categorical_data_list.append(categorical_data)

    # Define column names for the data
    x_ls = ['Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 
            'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
            'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 
            'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
    x_lianxu = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 
                'Rate', 'Srate', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
                'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 
                'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']
    X_columns = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'ack_count',
        'syn_count', 'fin_count', 'urg_count', 'rst_count', 
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
        'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
        'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
        'Radius', 'Covariance', 'Variance', 'Weight'
    ]

    # Convert the data lists to DataFrames
    df_num = pd.DataFrame(continuous_data_list, columns=x_lianxu)
    df_cat = pd.DataFrame(categorical_data_list, columns=x_ls)

    # Combine continuous and discrete data into one DataFrame
    df_combined = pd.concat([df_num, df_cat], axis=1)

    # Rearrange columns to match the expected order
    df_combined = df_combined[X_columns]

    # Add a label column
    df_combined['label'] = train_gmddpm.label

    # Save the generated data to a CSV file
    file_exists = os.path.isfile(f'generated_{train_gmddpm.label}.csv')
    df_combined.to_csv('generated_1.csv', mode='a', header=not file_exists, index=False)

    print(f"CSV file saved {i} times, containing {1000 * (i + 1)} rows of data.")
