
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

def process_data(path, label):
    # Load the dataset
    train_df = pd.read_csv(path)

    # Extract rows where the 'label' column matches the given label
    df = train_df[train_df['label'] == label]
    df = df.drop(df.index[-1])  # Drop the last row

    # Define discrete and continuous feature columns
    x_ls = ['Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 
            'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
            'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 
            'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']

    x_lianxu = ['flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 
                'Rate', 'Srate', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
                'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 
                'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']

    # Extract discrete features
    categorical_features = df[x_ls].values

    # Apply one-hot encoding: Convert 0 to [1, 0] and 1 to [0, 1]
    # Use np.eye(2) to create a (2, 2) identity matrix, then use indexing for one-hot encoding
    one_hot_encoded = np.eye(2)[categorical_features.astype(int)]

    # Reshape the encoded features from (batch_size, 22, 2) to (batch_size, 44)
    one_hot_encoded = one_hot_encoded.reshape(categorical_features.shape[0], -1)
    categorical_features = one_hot_encoded

    # Check the transformed results
    print(categorical_features[1, :])
    print(categorical_features.shape)  # Should be (batch_size, 44)

    # Extract continuous features
    numerical_features = df[x_lianxu].values

    # Extract labels
    labels = df['label'].values  # Assumes a column named 'label' represents the labels

    mean = 0.5
    std = 0.5

    # Normalize the continuous features to the range [-1, 1]
    # Scale values using Normalize method: map [0, 1] data to [-1, 1] range
    numerical_features = (numerical_features - mean) / std

    # Convert features to PyTorch tensors
    categorical_features = torch.tensor(categorical_features, dtype=torch.float32)
    numerical_features = torch.tensor(numerical_features, dtype=torch.float32)

    # Get dimensions for discrete and continuous features
    discrete_dim = categorical_features.shape[1]
    continuous_dim = numerical_features.shape[1]

    # Return the processed features and their dimensions
    return categorical_features, numerical_features, discrete_dim, continuous_dim
