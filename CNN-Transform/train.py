import torch
from dataprocess import preprocess_data
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Transformer import onedCNN_Transformer


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Data preprocessing
train_loader, test_loader = preprocess_data()

# Initialize the model and optimizer
hidden_dim = 256
output_dim = 8  # Assuming it's a seven-class classification task
dropout_prob = 0.01  # For larger tasks, set dropout rate between 0.2-0.5 to avoid overfitting
model = onedCNN_Transformer(output_size=output_dim, hidden_size=hidden_dim, dropout_prob=dropout_prob).to(device)

# Load pre-trained model for further training
# model.load_state_dict(torch.load("models/best_model.pth"))

# Define loss function
criterion = nn.CrossEntropyLoss().to(device)

# Define optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Initial learning rate (if needed)
optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Fine-tuned learning rate

# Training parameters
num_epochs = 20
best_accuracy = 0  # To record the best test accuracy

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0

    # Display training progress using tqdm
    for batch_idx, (X_batch, y_batch) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
        optimizer.zero_grad()

        # Move data to the current device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)

        # Compute loss
        loss = criterion(outputs, y_batch)
        loss = loss.to(device)

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate loss and correct predictions
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == y_batch).sum().item()

    # Calculate average loss and training accuracy
    avg_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / len(train_loader.dataset)

    # Evaluate the model at the end of each epoch on the test set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_test, y_test in test_loader:
            # Move test data to the current device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Forward pass on test data
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)

            # Count correct predictions
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

        # Compute test accuracy
        test_accuracy = correct / total

    # Print epoch metrics
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Save the model if the current test accuracy is the best so far
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f'Saved best model with test accuracy: {best_accuracy:.4f}')