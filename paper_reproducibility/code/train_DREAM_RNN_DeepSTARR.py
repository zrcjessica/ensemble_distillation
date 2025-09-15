#!/usr/bin/env python3
"""
Train DREAM-RNN models using the official @DREAM_paper implementation

This script implements the exact DREAM-RNN architecture and training methodology
from the @DREAM_paper tutorial notebooks for DeepSTARR data.
"""

import argparse
import os
# Suppress TensorFlow warnings since we're using PyTorch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import h5py
import yaml
from pathlib import Path
import sys
import os
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F

# Note: This script is PyTorch-only and does not require the TensorFlow-based
# paper_reproducibility/code/utils.py (which depends on evoaug_tf). Avoid importing
# it to prevent optional dependency warnings.

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, continuing without logging")


class BHIFirstLayersBlock(nn.Module):
    """
    First layer block from BHI team's DREAM-RNN implementation
    Based on @DREAM_paper: "First layer block: Same as DREAM-CNN"
    """
    def __init__(self, in_channels=6, out_channels=320, seqsize=249, 
                 kernel_sizes=[9, 15], pool_size=1, dropout=0.2):
        super(BHIFirstLayersBlock, self).__init__()
        self.out_channels = out_channels
        
        # Two Conv1D layers with different kernel sizes
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels//2, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_channels//2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])
    
    def forward(self, x):
        # Apply both conv blocks and concatenate
        conv_outputs = [conv(x) for conv in self.conv_blocks]
        return torch.cat(conv_outputs, dim=1)  # Concatenate along channel dimension
    
    def infer_outseqsize(self):
        return 249  # For DeepSTARR


class BHICoreBlock(nn.Module):
    """
    Core layer block from BHI team's DREAM-RNN implementation
    Based on @DREAM_paper: "Bi-LSTM with 320 hidden dimensions each (640 total) + CNN block"
    """
    def __init__(self, in_channels=320, out_channels=320, seqsize=249,
                 lstm_hidden_channels=320, kernel_sizes=[9, 15], pool_size=1,
                 dropout1=0.2, dropout2=0.5):
        super(BHICoreBlock, self).__init__()
        self.out_channels = out_channels
        
        # Bi-LSTM layer
        self.lstm = nn.LSTM(in_channels, lstm_hidden_channels, 
                           bidirectional=True, batch_first=True)
        
        # CNN blocks after LSTM (similar to first layer block)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(lstm_hidden_channels * 2, out_channels//2, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_channels//2),
                nn.ReLU(),
                nn.Dropout(dropout1)
            ) for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout2)
    
    def forward(self, x):
        # LSTM expects (batch, seq, features) but input is (batch, features, seq)
        # Transpose for LSTM: (batch, features, seq) -> (batch, seq, features)
        x_lstm = x.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x_lstm)
        
        # Transpose back for CNN: (batch, seq, features) -> (batch, features, seq)
        lstm_out = lstm_out.transpose(1, 2)
        
        # Apply CNN blocks and concatenate
        conv_outputs = [conv(lstm_out) for conv in self.conv_blocks]
        conv_concat = torch.cat(conv_outputs, dim=1)
        
        # Apply final dropout
        return self.dropout(conv_concat)
    
    def infer_outseqsize(self):
        return 249  # For DeepSTARR


class AutosomeFinalLayersBlock(nn.Module):
    """
    Final layer block from Autosome.org team's DREAM-RNN implementation
    Based on @DREAM_paper: "Final layer block: Same as DREAM-CNN and DREAM-RNN"
    """
    def __init__(self, in_channels=320):
        super(AutosomeFinalLayersBlock, self).__init__()
        
        # Point-wise convolution followed by global average pooling
        self.final_conv = nn.Conv1d(in_channels, 256, kernel_size=1)
        self.final_dense = nn.Linear(256, 2)  # 2 outputs for DeepSTARR: Dev and Hk
    
    def forward(self, x):
        # Apply final convolution
        x = self.final_conv(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # Final dense layer
        x = self.final_dense(x)
        return x


class DREAM_RNN_Official(nn.Module):
    """
    Complete DREAM-RNN model based on @DREAM_paper architecture
    Combines BHI first/core blocks with Autosome final block
    Fixed to match saved model architecture: 4 input channels, pointwise_conv
    """
    def __init__(self, seqsize=249, in_channels=4):
        super(DREAM_RNN_Official, self).__init__()
        
        # Use 4 input channels for DeepSTARR (no additional channels)
        # This matches the saved model architecture
        
        # First layer block
        self.first_block = BHIFirstLayersBlock(
            in_channels=in_channels,  # 4 channels for DeepSTARR
            out_channels=320,
            seqsize=seqsize,
            kernel_sizes=[9, 15],
            dropout=0.2
        )
        
        # Core layer block
        self.core_block = BHICoreBlock(
            in_channels=320,
            out_channels=320,
            seqsize=seqsize,
            lstm_hidden_channels=320,
            kernel_sizes=[9, 15],
            dropout1=0.2,
            dropout2=0.5
        )
        
        # Final layer block - use pointwise_conv to match saved models
        self.final_block = nn.ModuleDict({
            'pointwise_conv': nn.Conv1d(320, 256, kernel_size=1),
            'final_dense': nn.Linear(256, 2)  # 2 outputs: Dev and Hk
        })
    
    def forward(self, x):
        # Forward pass through blocks - no additional channels needed
        x = self.first_block(x)
        x = self.core_block(x)
        
        # Final layer block - use pointwise_conv
        x = self.final_block['pointwise_conv'](x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.final_block['final_dense'](x)
        
        return x


# Using PyTorch's built-in OneCycleLR instead of custom implementation


def process_DeepSTARR_data_DREAM_official(X_train, y_train, X_test, y_test, X_val, y_val):
    """
    Process DeepSTARR data according to @DREAM_paper methodology
    - 6-channel input: 4 OHE + singleton + reverse complement
    - 249bp sequences
    """
    print("Processing DeepSTARR data for DREAM-RNN official implementation...")
    
    # Handle sequence length (249bp for DeepSTARR)
    def handle_sequence_length(X, target_length=249):
        current_length = X.shape[2]
        if current_length == target_length:
            return X
        elif current_length > target_length:
            return X[:, :, :target_length]
        else:
            # Extend to target length
            extension_length = target_length - current_length
            extension = np.zeros((X.shape[0], X.shape[1], extension_length), dtype=np.float32)
            X_extended = np.concatenate([extension, X], axis=2)
            return X_extended
    
    # Process each dataset
    X_train = handle_sequence_length(X_train, target_length=249)
    X_test = handle_sequence_length(X_test, target_length=249)
    X_val = handle_sequence_length(X_val, target_length=249)
    
    # Create additional channels
    def create_singleton_channel(X):
        # For DeepSTARR, we don't have singleton information, so use zeros
        return np.zeros((X.shape[0], 1, X.shape[2]), dtype=np.float32)
    
    def create_reverse_complement_channel(X):
        # For DeepSTARR, we don't have reverse complement info, so use zeros
        return np.zeros((X.shape[0], 1, X.shape[2]), dtype=np.float32)
    
    # Add singleton and reverse complement channels
    X_train_singleton = create_singleton_channel(X_train)
    X_train_rev = create_reverse_complement_channel(X_train)
    X_train_processed = np.concatenate([X_train, X_train_singleton, X_train_rev], axis=1)
    
    X_test_singleton = create_singleton_channel(X_test)
    X_test_rev = create_reverse_complement_channel(X_test)
    X_test_processed = np.concatenate([X_test, X_test_singleton, X_test_rev], axis=1)
    
    X_val_singleton = create_singleton_channel(X_val)
    X_val_rev = create_reverse_complement_channel(X_val)
    X_val_processed = np.concatenate([X_val, X_val_singleton, X_val_rev], axis=1)
    
    # Transpose to (batch, channels, seq_len) format for PyTorch
    X_train_processed = X_train_processed.transpose(0, 2, 1)
    X_test_processed = X_test_processed.transpose(0, 2, 1)
    X_val_processed = X_val_processed.transpose(0, 2, 1)
    
    print(f"Input shape after processing: {X_train_processed.shape}")
    print(f"Expected shape: (batch, 6, 249) where 6 = 4 OHE + 1 singleton + 1 reverse")
    print(f"Sequences adjusted to 249 bp for DeepSTARR dataset")
    
    return X_train_processed, y_train, X_test_processed, y_test, X_val_processed, y_val


def load_data(args):
    """Load DeepSTARR data from unified dataset"""
    print(f"Loading data from {args.data}")
    
    # Load from unified DeepSTARR dataset
    with h5py.File(args.data, 'r') as f:
        # Load training data
        X_train = f['Train/X'][:]  # (N, 249, 4)
        y_train = f['Train/y'][:]  # (N, 2) - Dev and Hk
        
        # Load validation data
        X_val = f['Val/X'][:]  # (N, 249, 4)
        y_val = f['Val/y'][:]  # (N, 2) - Dev and Hk
        
        # Load test data
        X_test = f['Test/X'][:]  # (N, 249, 4)
        y_test = f['Test/y'][:]  # (N, 2) - Dev and Hk
    
    # Convert to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    print(f"Original data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Process data according to @DREAM_paper methodology
    # The data is already 249bp, so no sequence length adjustment needed
    print(f"Sequence length is already 249bp, no adjustment needed")
    
    # Transpose to (batch, channels, seq_len) format for PyTorch
    # Input is (N, 249, 4), we want (N, 4, 249)
    # The transpose (0, 2, 1) means: keep batch dim 0, move channels from dim 2 to dim 1, move seq_len from dim 1 to dim 2
    X_train = np.transpose(X_train, (0, 2, 1))  # (N, 249, 4) -> (N, 4, 249)
    X_val = np.transpose(X_val, (0, 2, 1))  # (N, 249, 4) -> (N, 4, 249)
    X_test = np.transpose(X_test, (0, 2, 1))  # (N, 249, 4) -> (N, 4, 249)
    
    # Optional additional downsampling for quick smoke tests
    if hasattr(args, 'downsample') and args.downsample < 1.0:
        print(f"Downsampling training data to {args.downsample:.1%} with fixed seed for reproducibility")
        rng = np.random.default_rng(1234)
        n_samples = max(1, int(len(X_train) * args.downsample))
        indices = rng.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"  Training data downsampled from {int(len(X_train)/args.downsample)} to {len(X_train)} samples")

    print(f"Processed data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"Expected shape: (batch, 4, 249) where 4 = A, C, G, T channels")
    
    # Note: Using pre-downsampled dataset, no on-the-fly downsampling needed
    print(f"Using pre-downsampled dataset with ratio: {args.downsample}")
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def train_model(model, train_loader, val_loader, device, args, config):
    """Train the model using @DREAM_paper methodology"""
    
    # AdamW optimizer with weight decay as specified in @DREAM_paper
    optimizer = optim.AdamW(model.parameters(), lr=config['optim_lr'], weight_decay=0.01)
    
    # MSE loss for regression
    criterion = nn.MSELoss()
    
    # One-cycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=config['optim_lr'], steps_per_epoch=steps_per_epoch, epochs=config['epochs'])
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Calculate Pearson correlation for model selection
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # Calculate Pearson for each output (Dev, Hk)
        pearson_dev = pearsonr(val_predictions[:, 0], val_targets[:, 0])[0]
        pearson_hk = pearsonr(val_predictions[:, 1], val_targets[:, 1])[0]
        avg_pearson = (pearson_dev + pearson_hk) / 2
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Pearson: {avg_pearson:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())
    
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    # Calculate metrics for each output
    metrics = {}
    for i, output_name in enumerate(['Dev', 'Hk']):
        pred = test_predictions[:, i]
        target = test_targets[:, i]
        
        pearson = pearsonr(pred, target)[0]
        spearman = spearmanr(pred, target)[0]
        mse = np.mean((pred - target) ** 2)
        
        metrics[f'{output_name}_pearson'] = pearson
        metrics[f'{output_name}_spearman'] = spearman
        metrics[f'{output_name}_mse'] = mse
    
    # Average metrics
    metrics['avg_pearson'] = (metrics['Dev_pearson'] + metrics['Hk_pearson']) / 2
    metrics['avg_spearman'] = (metrics['Dev_spearman'] + metrics['Hk_spearman']) / 2
    metrics['avg_mse'] = (metrics['Dev_mse'] + metrics['Hk_mse']) / 2
    
    return metrics, test_predictions, test_targets


def main(args):
    print("=== DEBUG: Starting main function ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Set random seeds - use model index to ensure different initializations
    seed = 42 + args.ix  # Different seed for each model
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Disable cuDNN to avoid version conflicts
    torch.backends.cudnn.enabled = False

    # Set CUDA launch blocking for better error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Set device compatibility flags
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=== DEBUG: Loading config ===")
    # Load config
    with open(args.config, 'r') as f:
        config_raw = yaml.safe_load(f)
    
    config = {}
    for key, item in config_raw.items():
        if isinstance(item, dict) and 'value' in item:
            config[key] = item['value']
        else:
            config[key] = item
    
    print("=== DEBUG: Loading data ===")
    # Load data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(args)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # Use different worker seeds for different shuffling patterns
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                             worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print("=== DEBUG: Creating model ===")
    # Create model
    model = DREAM_RNN_Official(in_channels=4, seqsize=249)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("=== DEBUG: Starting training ===")
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, device, args, config)
    
    # Evaluate model
    print("Evaluating model...")
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nTest Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_dir / f"{args.ix}_model.pth")
    
    # Save predictions
    np.save(output_dir / f"{args.ix}_predictions.npy", predictions)
    np.save(output_dir / f"{args.ix}_targets.npy", targets)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / f"{args.ix}_performance.csv", index=False)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DREAM-RNN using official @DREAM_paper implementation")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--data", type=str, required=True, help="Input data file")
    parser.add_argument("--config", type=str, default="paper_reproducibility/config/DREAM_RNN_DeepSTARR.yaml", help="Config file")
    parser.add_argument("--dataset", type=str, default="DeepSTARR", help="Dataset name")
    parser.add_argument("--celltype", type=str, default="Dev", help="Cell type")
    parser.add_argument("--downsample", type=float, default=1.0, help="Downsample ratio")
    parser.add_argument("--ix", type=int, default=0, help="Model index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    args = parser.parse_args()
    main(args)
