from dataprocess import preprocess_data
from tqdm import tqdm
from Transformer import onedCNN_Transformer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check whether to use GPU
train_loader, test_loader = preprocess_data()  # Data preprocessing

# Initialize the model and optimizer
input_dim = 163
cnn_out_channels = 64
hidden_dim = 256
output_dim = 8  # Assume it's an 8-class classification task
dropout_prob = 0.2
model = onedCNN_Transformer(output_size=output_dim, hidden_size=hidden_dim, dropout_prob=dropout_prob).to(device)

# Load the pre-trained model from train.py
model.load_state_dict(torch.load('models/best_model.pth'))

# Set the model to evaluation mode. In evaluation mode, the model disables features such as Dropout
# and uses the trained parameters for prediction.
model.eval()

# Add the following code to the model evaluation section
classes = ['Benign', 'BruteForce', 'DDoS', 'Dos', 'Mirai', 'Recon', 'Spoofing', 'Web']
all_preds = []
all_labels = []

# Disable gradient calculation during prediction to save memory
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model
        outputs = model.forward(inputs)  # Model prediction
        _, preds = torch.max(outputs, 1)  # Get predicted class, `_` ignores the maximum value itself

        all_preds.extend(preds.cpu().numpy())  # Collect predicted results
        all_labels.extend(labels.cpu().numpy())  # Collect true labels

# Convert to numpy arrays for easier computation
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

from collections import Counter
def print_count_numbers_separately(arr1, arr2):
    # Use Counter to calculate the frequency of each element in both arrays
    counts1 = Counter(arr1)
    counts2 = Counter(arr2)

    # Print counts for array 1
    print("Array 1:")
    for num, freq in counts1.items():
        print(f"Number {num}: {freq} times")

    # Print counts for array 2
    print("\nArray 2:")
    for num, freq in counts2.items():
        print(f"Number {num}: {freq} times")


# Calculate overall accuracy
overall_accuracy = accuracy_score(all_labels, all_preds)
print(f"Overall Accuracy: {overall_accuracy}")

# Calculate precision, recall, F1-score, and support for each class
precision, recall, f1_score, support = precision_recall_fscore_support(all_labels, all_preds, average=None)

# Output class-wise metrics
print("Class-wise Metrics:")
for i in range(8):  # Assume it's an 8-class classification problem
    print(f"class {classes[i]}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1 Score={f1_score[i]:.4f}, "
          f"Support={support[i]}")

# Treat non-zero values as the positive class (other values as the negative class)
binary_preds = (all_preds != 0).astype(int)  # Non-zero predictions are converted to 1 (True)
binary_labels = (all_labels != 0).astype(int)

# Calculate binary classification accuracy
binary_accuracy = accuracy_score(binary_labels, binary_preds)
print(f"Binary Accuracy: {binary_accuracy}")

# Calculate binary classification metrics
report = classification_report(binary_labels, binary_preds, zero_division=0)
print("Binary Classification Report:\n", report)

# Print the counts of each class in the labels and predictions
print_count_numbers_separately(all_labels, all_preds)











Cha