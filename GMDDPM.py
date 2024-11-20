import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusionProcess:
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def sample_q(self, x0, t):
        # Convert alpha_bar_t to a tensor and compute noise
        alpha_bar_t = torch.tensor(self.alphas_cumprod[t], dtype=torch.float32)
        epsilon = torch.randn_like(x0)  # Generate random noise with the same shape as x0
        # Compute xt by diffusing x0 with noise epsilon
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        return xt, epsilon

    def forward(self, x0, t):
        # Apply forward diffusion
        return self.sample_q(x0, t)


class MultinomialDiffusionProcess:
    def __init__(self, timesteps, beta_start=0.0001, beta_end=0.02, K=2):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.K = K  # Number of classes

    def sample_q(self, x0, t):
        # Convert alpha_bar_t to a tensor
        alpha_bar_t = torch.tensor(self.alphas_cumprod[t], dtype=torch.float32)
        # Compute logits
        logits = alpha_bar_t * x0 + (1 - alpha_bar_t) / self.K
        batch_size = logits.shape[0]
        num_classes = logits.shape[1] // 2

        # Sample and convert to one-hot encoding
        samples = torch.multinomial(logits.view(-1, 2), num_samples=1)
        samples = samples.view(batch_size, num_classes)
        xt_onehot = torch.zeros(batch_size, num_classes * 2)
        xt_onehot.scatter_(1, samples + torch.arange(num_classes) * 2, 1)
        return xt_onehot

    def forward(self, x0, t):
        # Apply forward diffusion
        xt = self.sample_q(x0, t)
        return xt


class MLPDiffusion(nn.Module):
    def __init__(self, discrete_dim, continuous_dim, n_steps, num_units=128, dropout_rate=0.01):
        super(MLPDiffusion, self).__init__()
        
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim
        self.num_units = num_units
        self.layers = nn.ModuleList(
            [
                nn.Linear(discrete_dim + continuous_dim, num_units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        )
        
        self.norm_layers = nn.ModuleList([nn.LayerNorm(num_units) for _ in range(3)])

        self.residual_layers = nn.ModuleList(
            [
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units)
            ]
        )

        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.LayerNorm(num_units),
                nn.Embedding(n_steps, num_units),
                nn.LayerNorm(num_units),
                nn.Embedding(n_steps, num_units),
                nn.LayerNorm(num_units),
            ]
        )
        
        self.output_layer = nn.Linear(num_units, discrete_dim + continuous_dim)

    def forward(self, x, t):
        # Remove the last dimension from t
        t = t.squeeze(-1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            t_embedding = self.step_embeddings[2 * (i // 3)](t)  # Obtain time embedding
            t_embedding = self.step_embeddings[2 * (i // 3) + 1](t_embedding)  # Normalize time embedding
            x = x + t_embedding
            # Apply LayerNorm every 3 layers
            if i % 3 == 2:
                x = self.norm_layers[i // 3](x)

        # Add residual connection
        residual = self.residual_layers[0](x)  # Compute residual part
        residual = self.residual_layers[1](residual)  # Activation
        residual = self.residual_layers[2](residual)  # Linear transformation
        x = x + residual  # Add residual connection

        # Final output layer
        x = self.output_layer(x)
        
        return x
    
# Compute posterior distribution
def compute_theta_post(x_t, x_0, alpha_t, alpha_bar_t_1):
    # Convert alpha_t and alpha_bar_t_1 to tensors if necessary
    if isinstance(alpha_t, np.ndarray):
        alpha_t = torch.tensor(alpha_t, dtype=torch.float32)
    if isinstance(alpha_bar_t_1, np.ndarray):
        alpha_bar_t_1 = torch.tensor(alpha_bar_t_1, dtype=torch.float32)
    # Compute \tilde{\theta}
    theta_post = (alpha_t * x_t + (1 - alpha_t) / 2) * (alpha_bar_t_1 * x_0 + (1 - alpha_bar_t_1) / 2)
    theta_tilde = theta_post.view(-1, 22, 2)

    # Apply softmax across every two dimensions (i.e., each class's two values)
    theta_post = F.softmax(theta_tilde, dim=-1)

    # Reshape back to [batch_size, 44]
    theta_post = theta_post.view(-1, 44)
   
    return theta_post

def compute_kl_loss(real_post, predicted_post):
    """
    Compute KL divergence loss
    real_post: Ground truth posterior q(x_{t-1}|x_t, x_0)
    predicted_post: Model-predicted posterior p(x_{t-1}|x_t)
    """
    predicted_post = predicted_post.view(-1, 22, 2)
    real_post = real_post.view(-1, 22, 2)

    # Compute KL divergence; F.kl_div defaults to KL(p || q), but we want KL(q || p)
    kl_loss = F.kl_div(torch.log(predicted_post), real_post, reduction='batchmean')
   
    return kl_loss
