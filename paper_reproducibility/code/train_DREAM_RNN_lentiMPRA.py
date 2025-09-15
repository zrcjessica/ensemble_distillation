#!/usr/bin/env python3
"""
Train DREAM-RNN models on lentiMPRA datasets using @DREAM_paper tutorial protocols

This script implements the exact DREAM-RNN architecture and training methodology
from the @DREAM_paper tutorial notebooks for lentiMPRA data.
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
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

# Add the code directory to path for potential future imports
sys.path.append('/home/jessica/ensemble_distillation/code')
# import utils  # Not needed for core functionality

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
    def __init__(self, in_channels=4, out_channels=320, seqsize=230, 
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
        return 230  # For lentiMPRA


class BHICoreBlock(nn.Module):
    """
    Core layer block from BHI team's DREAM-RNN implementation
    Based on @DREAM_paper: "Bi-LSTM with 320 hidden dimensions each (640 total) + CNN block"
    """
    def __init__(self, in_channels=320, out_channels=320, seqsize=230,
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
    
    def infer_outseqsize(self):
        return 230  # For lentiMPRA
    
    def forward(self, x):
        # LSTM expects (batch, seq, features) but we have (batch, features, seq)
        x = x.transpose(1, 2)  # (batch, seq, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        lstm_out = lstm_out.transpose(1, 2)  # (batch, hidden*2, seq)
        
        # Apply CNN blocks and concatenate
        conv_outputs = [conv(lstm_out) for conv in self.conv_blocks]
        output = torch.cat(conv_outputs, dim=1)
        return self.dropout(output)


class AutosomeFinalLayersBlock(nn.Module):
    """
    Final layer block from @DREAM_paper tutorial
    Based on @DREAM_paper: "Point-wise convolution + global average pooling + SoftMax"
    """
    def __init__(self, in_channels=320, seqsize=230):
        super(AutosomeFinalLayersBlock, self).__init__()
        self.pointwise_conv = nn.Conv1d(in_channels, 256, kernel_size=1, padding='same')
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.final_dense = nn.Linear(256, 2)  # 2 outputs for activity and aleatoric uncertainty
        self.activation = nn.Identity()  # Linear activation for regression
    
    def forward(self, x):
        x = self.pointwise_conv(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.final_dense(x)
        return self.activation(x)


class DREAM_RNN_LentiMPRA(nn.Module):
    """
    DREAM-RNN model for lentiMPRA datasets following @DREAM_paper tutorial
    """
    def __init__(self, in_channels=4, seqsize=230):
        super(DREAM_RNN_LentiMPRA, self).__init__()
        self.first_block = BHIFirstLayersBlock(in_channels=in_channels, out_channels=320, seqsize=seqsize, kernel_sizes=[9, 15], pool_size=1, dropout=0.2)
        self.core_block = BHICoreBlock(in_channels=self.first_block.out_channels, out_channels=320, seqsize=self.first_block.infer_outseqsize(), lstm_hidden_channels=320, kernel_sizes=[9, 15], pool_size=1, dropout1=0.2, dropout2=0.5)
        self.final_block = AutosomeFinalLayersBlock(in_channels=self.core_block.out_channels, seqsize=self.core_block.infer_outseqsize())
    
    def forward(self, x):
        x = self.first_block(x)
        x = self.core_block(x)
        x = self.final_block(x)
        return x


def load_lentiMPRA_data(args):
    """Load lentiMPRA data following @DREAM_paper tutorial protocols"""
    print(f"Loading lentiMPRA data from {args.data}")
    
    # Load from lentiMPRA dataset
    with h5py.File(args.data, 'r') as f:
        # Load training data
        X_train = f['Train/X'][:]  # (N, 230, 4)
        y_train = f['Train/y'][:]  # (N, 2) - activity and aleatoric uncertainty
        
        # Load validation data
        X_val = f['Val/X'][:]  # (N, 230, 4)
        y_val = f['Val/y'][:]  # (N, 2) - activity and aleatoric uncertainty
        
        # Load test data
        X_test = f['Test/X'][:]  # (N, 230, 4)
        y_test = f['Test/y'][:]  # (N, 2) - activity and aleatoric uncertainty
    
    # Convert to float32 and extract only first 2 columns (activity and aleatoric) from targets
    X_train = X_train.astype(np.float32)
    y_train = y_train[:, :2].astype(np.float32)  # Only first 2 columns: activity and aleatoric
    X_val = X_val.astype(np.float32)
    y_val = y_val[:, :2].astype(np.float32)  # Only first 2 columns: activity and aleatoric
    X_test = X_test.astype(np.float32)
    y_test = y_test[:, :2].astype(np.float32)  # Only first 2 columns: activity and aleatoric
    
    # Apply downsampling if specified (with fixed seed for reproducibility)
    if hasattr(args, 'downsample') and args.downsample < 1.0:
        print(f"Downsampling training data to {args.downsample:.1%} with fixed seed for reproducibility")
        rng = np.random.default_rng(1234)  # Fixed seed for consistency across runs
        n_samples = int(len(X_train) * args.downsample)
        indices = rng.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"  Training data downsampled from {len(X_train)/args.downsample:.0f} to {len(X_train)} samples")
    
    print(f"Original data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape} (using first 2 columns: activity and aleatoric)")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape} (using first 2 columns: activity and aleatoric)")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape} (using first 2 columns: activity and aleatoric)")
    
    # For lentiMPRA, we don't have reverse complement info, so keep 4 channels
    # Transpose to (batch, channels, seq_len) format for PyTorch
    # Input is (N, 230, 4), we want (N, 4, 230)
    X_train = np.transpose(X_train, (0, 2, 1))  # (N, 230, 4) -> (N, 4, 230)
    X_val = np.transpose(X_val, (0, 2, 1))  # (N, 230, 4) -> (N, 4, 230)
    X_test = np.transpose(X_test, (0, 2, 1))  # (N, 230, 4) -> (N, 4, 230)
    
    print(f"Processed data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"Expected shape: (batch, 4, 230) where 4 = A, C, G, T channels")
    print(f"Using lentiMPRA dataset with 230bp sequences")
    
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
        
        # Calculate Pearson for each output (activity, aleatoric)
        pearson_activity = pearsonr(val_predictions[:, 0], val_targets[:, 0])[0]
        pearson_aleatoric = pearsonr(val_predictions[:, 1], val_targets[:, 1])[0]
        avg_pearson = (pearson_activity + pearson_aleatoric) / 2
        
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
    
    # Calculate metrics for each output (only first 2 columns: activity and aleatoric)
    metrics = {}
    for i, output_name in enumerate(['activity', 'aleatoric']):
        pred = test_predictions[:, i]
        target = test_targets[:, i]
        
        pearson = pearsonr(pred, target)[0]
        spearman = spearmanr(pred, target)[0]
        mse = np.mean((pred - target) ** 2)
        
        metrics[f'{output_name}_pearson'] = pearson
        metrics[f'{output_name}_spearman'] = spearman
        metrics[f'{output_name}_mse'] = mse
    
    # Average metrics
    metrics['avg_pearson'] = (metrics['activity_pearson'] + metrics['aleatoric_pearson']) / 2
    metrics['avg_spearman'] = (metrics['activity_spearman'] + metrics['aleatoric_spearman']) / 2
    metrics['avg_mse'] = (metrics['activity_mse'] + metrics['aleatoric_mse']) / 2
    
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
    X_train, y_train, X_test, y_test, X_val, y_val = load_lentiMPRA_data(args)
    
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
    model = DREAM_RNN_LentiMPRA(in_channels=4, seqsize=230)
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
    parser = argparse.ArgumentParser(description="Train DREAM-RNN on lentiMPRA using @DREAM_paper tutorial protocols")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--data", type=str, required=True, help="Input data file")
    parser.add_argument("--config", type=str, default="paper_reproducibility/config/DREAM_RNN_lentiMPRA.yaml", help="Config file")
    parser.add_argument("--dataset", type=str, default="lentiMPRA", help="Dataset name")
    parser.add_argument("--celltype", type=str, default="K562", help="Cell type")
    parser.add_argument("--downsample", type=float, default=1.0, help="Downsample ratio")
    parser.add_argument("--ix", type=int, default=0, help="Model index")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    args = parser.parse_args()
    main(args)

