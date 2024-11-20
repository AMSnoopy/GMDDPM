import torch
import torch.nn.functional as F
import csv
import pandas as pd
import matplotlib.pyplot as plt
import Process
import GMDDPM 

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

# Training function
def train(model, gaussian_process, multinomial_process, numerical_features, categorical_features, num_epochs, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_history = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_gaussian_loss = 0.0  # To record Gaussian loss for each epoch
        total_kl_loss = 0.0  # To record KL loss for each epoch
        for i in range(0, len(numerical_features), batch_size):
            num_batch = numerical_features[i:i+batch_size]
            cat_batch = categorical_features[i:i+batch_size]
            current_batch_size = num_batch.size(0)  # Dynamically adjust batch size for the last batch

            # Randomly select timesteps and ensure t is of Long type
            t = torch.randint(0, gaussian_process.timesteps, (current_batch_size // 2,), dtype=torch.long)
            t = torch.cat([t, gaussian_process.timesteps - 1 - t], dim=0)  # Symmetrically concatenate t
            t = t.unsqueeze(-1)  # Add one dimension for consistency

            # Forward diffusion for Gaussian process
            xt_num, epsilon = gaussian_process.forward(num_batch, t)
            
            # Forward diffusion for Multinomial process
            xt_cat = multinomial_process.forward(cat_batch, t)
            xt = torch.cat([xt_cat, xt_num], dim=-1)

            # Model prediction
            predicted_cat_num = model(xt.float(), t)  # Pass t directly without repeating
            predicted_cat = predicted_cat_num[:, :discrete_dim]
            predicted_num = predicted_cat_num[:, discrete_dim:]

            # Compute losses
            gaussian_loss = F.mse_loss(predicted_num, epsilon)
            real_post = GMDDPM.compute_theta_post(xt_cat, cat_batch, gaussian_process.alphas[t], gaussian_process.alphas_cumprod[t])
            predicted_post = GMDDPM.compute_theta_post(xt_cat, predicted_cat, gaussian_process.alphas[t], gaussian_process.alphas_cumprod[t])
            kl_loss = GMDDPM.compute_kl_loss(real_post, predicted_post)*15
            # log_likelihood_loss = F.cross_entropy(predicted_cat, cat_batch)
            loss = gaussian_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record losses
            total_loss += loss.item()
            total_gaussian_loss += gaussian_loss.item()
            total_kl_loss += kl_loss.item()
        
        # Print epoch-wise losses
        avg_gaussian_loss = total_gaussian_loss / len(numerical_features)
        avg_kl_loss = total_kl_loss / len(numerical_features)
        avg_total_loss = total_loss / len(numerical_features)
        loss_history.append((avg_gaussian_loss, avg_kl_loss, avg_total_loss))
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Gaussian Loss: {avg_gaussian_loss:.6f}, KL Loss: {avg_kl_loss:.6f}, Total Loss: {avg_total_loss:.6f}')

    # Save loss history to CSV
    with open('logs//loss_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Gaussian Loss', 'KL Loss', 'Total Loss'])
        writer.writerows(loss_history)

    # Plot loss curves
    loss_data = pd.read_csv('logs//loss_history.csv')
    plt.figure(figsize=(10, 6))
    plt.plot(loss_data.index, loss_data['Gaussian Loss'], label='Gaussian Loss')
    plt.plot(loss_data.index, loss_data['KL Loss'], label='KL Loss')
    plt.plot(loss_data.index, loss_data['Total Loss'], label='Total Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    model_path = 'models//gmddpm_ct_model.pth'
    save_model(model, model_path)
label=1
num_epochs = 200
batch_size = 128
hidden_dim = 128
timesteps = 1000
categorical_features, numerical_features, discrete_dim, continuous_dim = Process.process_data(path='data/train_data.csv', label=label)
model = GMDDPM.MLPDiffusion(discrete_dim, continuous_dim, timesteps, hidden_dim)
gaussian_process = GMDDPM.GaussianDiffusionProcess(timesteps=timesteps)
multinomial_process = GMDDPM.MultinomialDiffusionProcess(timesteps=timesteps, K=2)
if __name__ == "__main__":
    # 仅在脚本直接运行时执行的代码

    train(model, gaussian_process, multinomial_process, numerical_features, categorical_features, num_epochs, batch_size)

    # Load and test the model
    trained_model = GMDDPM.MLPDiffusion(discrete_dim, continuous_dim, timesteps, hidden_dim)
    load_model(trained_model, 'models//gmddpm_ct_model.pth')
