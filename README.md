# GMDDPM-CT: A Diffusion-Based Model for Network Intrusion Detection

## Overview

**GMDDPM-CT** is a research project designed to explore the use of Gaussian and Multinomial Diffusion Processes in network intrusion detection. This repository includes implementations for synthetic data generation and classification using CNN-Transformer models.

------

## Project Structure

```
bashCopy codeGMDDPM-CT/
├── data/                     # Directory for datasets
├── logs/                     # Training logs and results
├── models/                   # Directory for saving trained models
├── CNN-Transform/            # CNN-Transformer-related scripts and models
│   ├── dataprocess.py        # Data preprocessing for classification
│   ├── test.py               # Testing script for CNN-Transformer
│   ├── train.py              # Training script for CNN-Transformer
│   ├── Transformer.py        # CNN-Transformer model implementation
├── Generate_data.py          # Script for synthetic data generation using reverse diffusion
├── GMDDPM.py                 # Diffusion processes and model implementation
├── Process.py                # Data preprocessing for GMDDPM
├── train_gmddpm.py           # Script to train the GMDDPM model
└── README.md                 # Project documentation
```

------

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- scikit-learn
- tqdm
- Matplotlib

Install all dependencies with:

```
pip install -r requirements.txt
```

------

## How to Use

### Step 1: Data Preparation

Place your dataset in the `data/` directory. Ensure the dataset is formatted as expected by the preprocessing scripts.

### Step 2: Train the GMDDPM Model

Run the training script to train the GMDDPM model:

```
python train_gmddpm.py
```

This will save the trained model in the `models/` directory.

### Step 3: Generate Synthetic Data

Use the `Generate_data.py` script to generate synthetic network traffic data:

```
python Generate_data.py
```

The generated data will be saved in CSV format.

### Step 4: Train the CNN-Transformer Classifier

Navigate to the `CNN-Transform` directory and train the CNN-Transformer model:

```
python train.py
```

### Step 5: Test the CNN-Transformer Model

Evaluate the trained classifier on the test dataset:

```
python test.py
```
