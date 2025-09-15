#!/usr/bin/env python3
"""
DREAM-RNN Ensemble Evaluation and Distillation Data Generation

This script follows the DEGU methodology to:
1. Load all models from a trained ensemble
2. Generate predictions on train/val/test data
3. Calculate ensemble mean and standard deviation (epistemic uncertainty)
4. Save distillation training data
5. Evaluate ensemble performance

Usage:
    python evaluate_ensemble_and_generate_distillation_data.py \
        --ensemble_dir dream_rnn_official_0.1 \
        --dataset DeepSTARR \
        --celltype Dev \
        --downsample_ratio 0.1 \
        --data_path /path/to/data.h5 \
        --output_dir distillation_data_0.1
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Temporarily disable cuDNN to avoid version incompatibility
torch.backends.cudnn.enabled = False
from pathlib import Path
import h5py
import gc
# Avoid importing keras to suppress TF initialization warnings; provide a tiny shim
class _KerasShim:
    class backend:
        @staticmethod
        def clear_session():
            pass
keras = _KerasShim()
import pandas as _pd
from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr

# Local minimal helpers to avoid importing utils (which may require evoaug_tf)
def _downsample(X, Y, rng, frac):
    n = int(len(X) * float(frac))
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], Y[idx]

def _summarise_deepstarr(perf_pred, y_true):
    # Compute Pearson/Spearman/MSE per head and return a small DataFrame
    cols = []
    for i, name in enumerate(['Dev','Hk'][:perf_pred.shape[1]]):
        p = _pearsonr(perf_pred[:,i], y_true[:,i])[0]
        s = _spearmanr(perf_pred[:,i], y_true[:,i])[0]
        mse = ((perf_pred[:,i] - y_true[:,i])**2).mean()
        cols.append({
            'output': name,
            'pearson_r': p,
            'spearman_rho': s,
            'mse': mse
        })
    return _pd.DataFrame(cols)

def _summarise_lentimpra(perf_pred, y_true, celltype, aleatoric=False, epistemic=False):
    # Handle shape mismatch: squeeze y_true if needed
    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)
    # Build names to match number of prediction heads
    if perf_pred.ndim == 1:
        perf_pred = perf_pred[:, None]
    n_heads = perf_pred.shape[1]
    # Align y_true heads to n_heads if necessary
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_true.shape[1] < n_heads:
        # pad y_true with zeros for missing heads to avoid index errors
        pad = np.zeros((y_true.shape[0], n_heads - y_true.shape[1]), dtype=y_true.dtype)
        y_true = np.concatenate([y_true, pad], axis=1)
    names = []
    if n_heads >= 1:
        names.append('activity')
    if n_heads >= 2 and aleatoric:
        names.append(f'{celltype}-aleatoric')
    out = []
    for i, name in enumerate(names):
        p = _pearsonr(perf_pred[:,i], y_true[:,i])[0]
        s = _spearmanr(perf_pred[:,i], y_true[:,i])[0]
        mse = ((perf_pred[:,i] - y_true[:,i])**2).mean()
        out.append({'output': name, 'pearson_r': p, 'spearman_rho': s, 'mse': mse})
    return _pd.DataFrame(out)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

## Removed Prix Fixe dependency; use the configured PyTorch model below as canonical

def create_deepstarr_model_fallback(seqsize=249, in_channels=4):
    """
    Fallback DeepSTARR model definition that matches the trained models exactly
    Used when Prix Fixe framework is not available
    """
    class DeepSTARRModel(nn.Module):
        def __init__(self, seqsize=249, in_channels=4):
            super(DeepSTARRModel, self).__init__()
            
            # First Layer Block: BHIFirstLayersBlock structure
            # Matches: first_block.conv_blocks.0.0.weight, first_block.conv_blocks.1.0.weight
            self.first_block = nn.ModuleDict({
                'conv_blocks': nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(in_channels, 160, kernel_size=9, padding=4),
                        nn.BatchNorm1d(160),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(in_channels, 160, kernel_size=15, padding=7),
                        nn.BatchNorm1d(160),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    )
                ])
            })
            
            # Core Layer Block: BHICoreBlock structure
            # Matches: core_block.lstm.weight_ih_l0, core_block.conv_blocks.0.0.weight
            self.core_block = nn.ModuleDict({
                'lstm': nn.LSTM(320, 320, batch_first=True, bidirectional=True),
                'conv_blocks': nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(640, 160, kernel_size=9, padding=4),  # 640 from bidirectional LSTM
                        nn.BatchNorm1d(160),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(640, 160, kernel_size=15, padding=7),  # 640 from bidirectional LSTM
                        nn.BatchNorm1d(160),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    )
                ])
            })
            
            # Final Layer Block: AutosomeFinalLayersBlock structure
            # Matches: final_block.pointwise_conv.weight, final_block.final_dense.weight
            self.final_block = nn.ModuleDict({
                'pointwise_conv': nn.Conv1d(320, 256, kernel_size=1),  # 320 from concatenated conv outputs
                'final_dense': nn.Linear(256, 2)  # 2 outputs: Dev and Hk
            })
        
        def forward(self, x):
            # First layer block
            conv_outputs = []
            for conv_block in self.first_block['conv_blocks']:
                conv_outputs.append(conv_block(x))
            x = torch.cat(conv_outputs, dim=1)  # 320 channels
            
            # Core block
            x = x.transpose(1, 2)  # (batch, seq, channels)
            x, _ = self.core_block['lstm'](x)
            x = x.transpose(1, 2)  # (batch, channels, seq)
            
            # Apply core conv blocks
            conv_outputs = []
            for conv_block in self.core_block['conv_blocks']:
                conv_outputs.append(conv_block(x))
            x = torch.cat(conv_outputs, dim=1)  # 320 channels
            
            # Final layer block
            x = self.final_block['pointwise_conv'](x)
            x = nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
            x = self.final_block['final_dense'](x)
            
            return x
    
    return DeepSTARRModel(seqsize=seqsize, in_channels=in_channels)

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load a trained DeepSTARR DREAM-RNN checkpoint into the configured PyTorch model."""
    model = create_deepstarr_model_fallback().to(device).float()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

class EnsembleEvaluator:
    """Evaluates DREAM-RNN ensembles and generates distillation training data"""
    
    def __init__(self, ensemble_dir: str, dataset: str, celltype: str, downsample_ratio: float, data_path: str, save_both_celltypes: bool = False):
        self.ensemble_dir = ensemble_dir
        self.dataset = dataset
        self.celltype = celltype
        self.downsample_ratio = downsample_ratio
        self.data_path = data_path
        self.save_both_celltypes = save_both_celltypes
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set dataset-specific parameters
        if dataset == "DeepSTARR":
            self.seqsize = 249
            self.n_outputs = 2  # Dev, Hk
        elif dataset == "lentiMPRA":
            self.seqsize = 230
            self.n_outputs = 2  # activity, aleatoric uncertainty
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Load data
        self.load_data()
        
        # Find model files
        self.model_files = self.find_model_files()
        print(f"Found {len(self.model_files)} model files")
        
        # Load models
        self.models = self.load_models()
        print(f"Successfully loaded {len(self.models)} models")

    def load_data(self):
        """Load the dataset"""
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # Load data from Train/Val/Test groups
            if self.dataset == "DeepSTARR":
                # DeepSTARR data structure: Train/Val/Test groups with X and y
                self.X_train = torch.tensor(f['Train']['X'][:], dtype=torch.float32)
                self.y_train = torch.tensor(f['Train']['y'][:], dtype=torch.float32)
                self.X_val = torch.tensor(f['Val']['X'][:], dtype=torch.float32)
                self.y_val = torch.tensor(f['Val']['y'][:], dtype=torch.float32)
                self.X_test = torch.tensor(f['Test']['X'][:], dtype=torch.float32)
                self.y_test = torch.tensor(f['Test']['y'][:], dtype=torch.float32)
                
                # Transpose to (batch, channels, seq_len) format for PyTorch
                self.X_train = self.X_train.transpose(1, 2)
                self.X_val = self.X_val.transpose(1, 2)
                self.X_test = self.X_test.transpose(1, 2)
                
                print(f"DeepSTARR data loaded:")
                print(f"  Train: {self.X_train.shape}, {self.y_train.shape}")
                print(f"  Val:   {self.X_val.shape}, {self.y_val.shape}")
                print(f"  Test:  {self.X_test.shape}, {self.y_test.shape}")
                
                # Check if ensemble_mean and std are available for distillation
                if 'ensemble_mean' in f['Train'] and 'std' in f['Train']:
                    self.ensemble_mean_train = torch.tensor(f['Train']['ensemble_mean'][:], dtype=torch.float32)
                    self.std_train = torch.tensor(f['Train']['std'][:], dtype=torch.float32)
                    print(f"  Ensemble mean train: {self.ensemble_mean_train.shape}")
                    print(f"  Std train: {self.std_train.shape}")
                else:
                    self.ensemble_mean_train = None
                    self.std_train = None
                    print("  No pre-computed ensemble data found")
                
            elif self.dataset == "lentiMPRA":
                # lentiMPRA data structure: Train/Val/Test groups with X and y
                X_train_full = f['Train']['X'][:]
                y_train_full = f['Train']['y'][:]
                X_val_full = f['Val']['X'][:]
                y_val_full = f['Val']['y'][:]
                X_test_full = f['Test']['X'][:]
                y_test_full = f['Test']['y'][:]
                
                # Apply downsampling if specified (with fixed seed for reproducibility)
                # This matches the exact same downsampling logic used in train_DREAM_RNN_lentiMPRA.py
                if self.downsample_ratio < 1.0:
                    print(f"Downsampling training data to {self.downsample_ratio:.1%} with fixed seed for reproducibility")
                    rng = np.random.default_rng(1234)  # Fixed seed for consistency across runs
                    n_samples = int(len(X_train_full) * self.downsample_ratio)
                    indices = rng.choice(len(X_train_full), n_samples, replace=False)
                    X_train_full = X_train_full[indices]
                    y_train_full = y_train_full[indices]
                    print(f"  Training data downsampled from {len(X_train_full)/self.downsample_ratio:.0f} to {len(X_train_full)} samples")
                
                # Convert to tensors and transpose to (batch, channels, seq_len) format for PyTorch
                self.X_train = torch.tensor(X_train_full, dtype=torch.float32).transpose(1, 2)
                self.y_train = torch.tensor(y_train_full, dtype=torch.float32)
                self.X_val = torch.tensor(X_val_full, dtype=torch.float32).transpose(1, 2)
                self.y_val = torch.tensor(y_val_full, dtype=torch.float32)
                self.X_test = torch.tensor(X_test_full, dtype=torch.float32).transpose(1, 2)
                self.y_test = torch.tensor(y_test_full, dtype=torch.float32)
                
                print(f"lentiMPRA data loaded:")
                print(f"  Train: {self.X_train.shape}, {self.y_train.shape}")
                print(f"  Val:   {self.X_val.shape}, {self.y_val.shape}")
                print(f"  Test:  {self.X_test.shape}, {self.y_test.shape}")
            else:
                raise ValueError(f"Unknown dataset: {self.dataset}")

    def find_model_files(self) -> List[str]:
        """Find all model checkpoint files in the ensemble directory"""
        model_files = []
        ensemble_path = Path(self.ensemble_dir)
        
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found: {self.ensemble_dir}")
        
        # Look for .pth files
        for model_file in ensemble_path.glob("*.pth"):
            model_files.append(str(model_file))
        
        # Look for .pt files
        for model_file in ensemble_path.glob("*.pt"):
            model_files.append(str(model_file))
        
        # Look for model_best files
        for model_file in ensemble_path.glob("*model_best*"):
            model_files.append(str(model_file))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.ensemble_dir}")
        
        return sorted(model_files)

    def load_models(self) -> List[nn.Module]:
        """Load all models in the ensemble"""
        models = []
        
        for model_file in self.model_files:
            try:
                # Load the model using the new load_model function
                model = load_model(model_file, self.device)
                models.append(model)
                print(f"Successfully loaded: {Path(model_file).name}")
                
            except Exception as e:
                print(f"Failed to load {model_file}: {e}")
                continue
        
        if not models:
            raise ValueError("No models could be loaded from ensemble directory")
        
        return models
    
    def generate_ensemble_predictions(self, models: List[nn.Module], 
                                    X_data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensemble predictions and calculate mean/std"""
        print(f"Generating ensemble predictions on {X_data.shape[0]} sequences")
        
        all_predictions = []
        
        with torch.no_grad():
            for i, model in enumerate(models):
                print(f"  Model {i+1}/{len(models)}")
                
                # Generate predictions in batches
                batch_size = 1024
                predictions = []
                
                for j in range(0, len(X_data), batch_size):
                    batch_end = min(j + batch_size, len(X_data))
                    X_batch = X_data[j:batch_end].to(self.device)
                    
                    # Forward pass
                    y_pred = model(X_batch)
                    predictions.append(y_pred.cpu().numpy())
                
                # Concatenate all batches
                model_predictions = np.concatenate(predictions, axis=0)
                all_predictions.append(model_predictions)
                
                print(f"    Predictions shape: {model_predictions.shape}")
        
        # Stack all model predictions
        all_predictions = np.stack(all_predictions, axis=0)
        print(f"Stacked predictions shape: {all_predictions.shape}")
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(all_predictions, axis=0)
        ensemble_std = np.std(all_predictions, axis=0)
        
        print(f"Ensemble mean shape: {ensemble_mean.shape}")
        print(f"Ensemble std shape: {ensemble_std.shape}")
        
        return ensemble_mean, ensemble_std
    
    def evaluate_ensemble_performance(self, ensemble_mean: np.ndarray, 
                                    y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance using mean predictions"""
        print("Evaluating ensemble performance")
        
        # Calculate Pearson correlation for each output
        correlations = []
        for i in range(ensemble_mean.shape[1]):
            if self.dataset == "DeepSTARR":
                output_name = ["Dev", "Hk"][i]
            else:
                output_name = ["activity", "aleatoric"][i]
            
            # Calculate Pearson correlation
            corr_matrix = np.corrcoef(ensemble_mean[:, i], y_true[:, i])
            pearson_r = corr_matrix[0, 1]
            
            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            spearman_rho, _ = spearmanr(ensemble_mean[:, i], y_true[:, i])
            
            # Calculate MSE
            mse = np.mean((ensemble_mean[:, i] - y_true[:, i]) ** 2)
            
            correlations.append({
                'output': output_name,
                'pearson_r': pearson_r,
                'spearman_rho': spearman_rho,
                'mse': mse
            })
            
            print(f"  {output_name}: Pearson r = {pearson_r:.4f}, Spearman ρ = {spearman_rho:.4f}, MSE = {mse:.6f}")
        
        return correlations
    
    def save_distillation_data(self, ensemble_mean_train: np.ndarray, 
                              ensemble_std_train: np.ndarray,
                              ensemble_mean_val: np.ndarray,
                              ensemble_std_val: np.ndarray,
                              ensemble_mean_test: np.ndarray,
                              ensemble_std_test: np.ndarray,
                              performance_metrics: list):
        """Save distillation training data and performance metrics"""
        print("Saving distillation training data")
        
        # Save ensemble predictions for training data (used for distillation)
        distillation_data = {
            'train_mean': ensemble_mean_train,
            'train_std': ensemble_std_train,
            'val_mean': ensemble_mean_val,
            'val_std': ensemble_std_val,
            'test_mean': ensemble_mean_test,
            'test_std': ensemble_std_test
        }
        
        # Save ensemble outputs in the ensemble directory structure
        ensemble_outputs_dir = Path(self.ensemble_dir) / "ensemble_outputs"
        ensemble_outputs_dir.mkdir(exist_ok=True)
        
        distillation_file = ensemble_outputs_dir / f"distillation_data_{self.dataset}_{self.celltype}_{self.downsample_ratio}.npz"
        np.savez_compressed(distillation_file, **distillation_data)
        print(f"  Saved distillation data to: {distillation_file}")
        
        # If save_both_celltypes is True and this is DeepSTARR, save distillation data for both celltypes
        if self.save_both_celltypes and self.dataset == "DeepSTARR":
            # Save Dev distillation data if current celltype is not Dev
            if self.celltype != "Dev":
                dev_distillation_file = ensemble_outputs_dir / f"distillation_data_{self.dataset}_Dev_{self.downsample_ratio}.npz"
                np.savez_compressed(dev_distillation_file, **distillation_data)
                print(f"  Saved Dev distillation data to: {dev_distillation_file}")
            
            # Save Hk distillation data if current celltype is not Hk
            if self.celltype != "Hk":
                hk_distillation_file = ensemble_outputs_dir / f"distillation_data_{self.dataset}_Hk_{self.downsample_ratio}.npz"
                np.savez_compressed(hk_distillation_file, **distillation_data)
                print(f"  Saved Hk distillation data to: {hk_distillation_file}")
        
        # Save ensemble performance metrics
        performance_file = ensemble_outputs_dir / f"ensemble_performance_{self.dataset}_{self.celltype}_{self.downsample_ratio}.csv"
        
        # Create performance summary with actual metrics
        performance_summary = []
        for i, metrics in enumerate(performance_metrics):
            if self.dataset == "DeepSTARR":
                output_name = ["Dev", "Hk"][i]
            else:
                output_name = ["activity", "aleatoric"][i]
            
            performance_summary.append({
                'dataset': self.dataset,
                'celltype': self.celltype,
                'downsample_ratio': self.downsample_ratio,
                'output': output_name,
                'ensemble_size': len(self.models),
                'pearson_r': metrics['pearson_r'],
                'spearman_rho': metrics['spearman_rho'],
                'mse': metrics['mse'],
                'data_shape': f"{ensemble_mean_train.shape[0]}x{ensemble_mean_train.shape[1]}"
            })
        
        performance_df = pd.DataFrame(performance_summary)
        performance_df.to_csv(performance_file, index=False)
        print(f"  Saved performance metrics to: {performance_file}")
        
        # Also save a simplified performance file for the plotting script
        # This will contain the main performance metric (Pearson correlation) for the celltype of interest
        if self.dataset == "DeepSTARR":
            # For DeepSTARR, we need to find which output corresponds to our celltype
            if self.celltype == "Dev":
                output_idx = 0
            elif self.celltype == "Hk":
                output_idx = 1
            else:
                output_idx = 0
            
            # Get the performance for the celltype of interest
            celltype_performance = performance_metrics[output_idx]
            
            # Save a simplified performance file for plotting
            plot_performance_file = ensemble_outputs_dir / f"plot_performance_{self.dataset}_{self.celltype}_{self.downsample_ratio}.csv"
            plot_performance_df = pd.DataFrame([{
                'dataset': self.dataset,
                'celltype': self.celltype,
                'downsample_ratio': self.downsample_ratio,
                'method': 'Ensemble',
                'pearson_r': celltype_performance['pearson_r'],
                'n_models': len(self.models)
            }])
            plot_performance_df.to_csv(plot_performance_file, index=False)
            print(f"  Saved plotting performance data to: {plot_performance_file}")
            
            # If save_both_celltypes is True, save plot performance files for both Dev and Hk
            if self.save_both_celltypes:
                # Save Dev performance if current celltype is not Dev
                if self.celltype != "Dev":
                    dev_performance = performance_metrics[0]  # Dev is at index 0
                    dev_plot_file = ensemble_outputs_dir / f"plot_performance_{self.dataset}_Dev_{self.downsample_ratio}.csv"
                    dev_plot_df = pd.DataFrame([{
                        'dataset': self.dataset,
                        'celltype': 'Dev',
                        'downsample_ratio': self.downsample_ratio,
                        'method': 'Ensemble',
                        'pearson_r': dev_performance['pearson_r'],
                        'n_models': len(self.models)
                    }])
                    dev_plot_df.to_csv(dev_plot_file, index=False)
                    print(f"  Saved Dev plotting performance data to: {dev_plot_file}")
                
                # Save Hk performance if current celltype is not Hk
                if self.celltype != "Hk":
                    hk_performance = performance_metrics[1]  # Hk is at index 1
                    hk_plot_file = ensemble_outputs_dir / f"plot_performance_{self.dataset}_Hk_{self.downsample_ratio}.csv"
                    hk_plot_df = pd.DataFrame([{
                        'dataset': self.dataset,
                        'celltype': 'Hk',
                        'downsample_ratio': self.downsample_ratio,
                        'method': 'Ensemble',
                        'pearson_r': hk_performance['pearson_r'],
                        'n_models': len(self.models)
                    }])
                    hk_plot_df.to_csv(hk_plot_file, index=False)
                    print(f"  Saved Hk plotting performance data to: {hk_plot_file}")
        
        # Save detailed ensemble statistics
        stats_file = ensemble_outputs_dir / f"ensemble_statistics_{self.dataset}_{self.celltype}_{self.downsample_ratio}.txt"
        with open(stats_file, 'w') as f:
            f.write(f"DREAM-RNN Ensemble Statistics\n")
            f.write(f"============================\n\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Cell Type: {self.celltype}\n")
            f.write(f"Downsample Ratio: {self.downsample_ratio}\n")
            f.write(f"Ensemble Size: {len(self.models)}\n\n")
            
            f.write(f"Performance Metrics:\n")
            for i, metrics in enumerate(performance_metrics):
                if self.dataset == "DeepSTARR":
                    output_name = ["Dev", "Hk"][i]
                else:
                    output_name = ["activity", "aleatoric"][i]
                f.write(f"  {output_name}:\n")
                f.write(f"    Pearson r: {metrics['pearson_r']:.6f}\n")
                f.write(f"    Spearman ρ: {metrics['spearman_rho']:.6f}\n")
                f.write(f"    MSE: {metrics['mse']:.6f}\n\n")
            
            f.write(f"Training Data:\n")
            f.write(f"  Mean shape: {ensemble_mean_train.shape}\n")
            f.write(f"  Std shape: {ensemble_std_train.shape}\n")
            f.write(f"  Mean range: [{ensemble_mean_train.min():.6f}, {ensemble_mean_train.max():.6f}]\n")
            f.write(f"  Std range: [{ensemble_mean_train.min():.6f}, {ensemble_mean_train.max():.6f}]\n\n")
            
            f.write(f"Validation Data:\n")
            f.write(f"  Mean shape: {ensemble_mean_val.shape}\n")
            f.write(f"  Std shape: {ensemble_std_val.shape}\n\n")
            
            f.write(f"Test Data:\n")
            f.write(f"  Mean shape: {ensemble_mean_test.shape}\n")
            f.write(f"  Std shape: {ensemble_std_test.shape}\n")
        
        print(f"  Saved detailed statistics to: {stats_file}")
    
    def evaluate_ensemble(self):
        """Main method to evaluate the ensemble and generate distillation data"""
        print(f"\n{'='*60}")
        print(f"Evaluating {self.dataset} ensemble with {len(self.models)} models")
        print(f"Data path: {self.data_path}")
        print(f"Ensemble directory: {self.ensemble_dir}")
        print(f"{'='*60}")
        
        try:
            # 1. Load data - already done in __init__
            print("Data already loaded in initialization")
            
            # 2. Models already loaded in __init__
            print(f"Models already loaded: {len(self.models)}")
            
            # 3. Generate ensemble predictions
            print("\nGenerating ensemble predictions...")
            ensemble_mean_train, ensemble_std_train = self.generate_ensemble_predictions(self.models, self.X_train)
            ensemble_mean_val, ensemble_std_val = self.generate_ensemble_predictions(self.models, self.X_val)
            ensemble_mean_test, ensemble_std_test = self.generate_ensemble_predictions(self.models, self.X_test)
            
            # 4. Evaluate ensemble performance
            print("\nEvaluating ensemble performance...")
            performance = self.evaluate_ensemble_performance(ensemble_mean_test, self.y_test.numpy())
            
            # 5. Save distillation data
            print("\nSaving distillation data...")
            self.save_distillation_data(
                ensemble_mean_train, ensemble_std_train,
                ensemble_mean_val, ensemble_std_val,
                ensemble_mean_test, ensemble_std_test,
                performance
            )
            
            # 6. Print summary
            print(f"\n{'='*60}")
            print("ENSEMBLE EVALUATION COMPLETE")
            print(f"{'='*60}")
            print(f"Dataset: {self.dataset}")
            print(f"Cell type: {self.celltype}")
            print(f"Downsample ratio: {self.downsample_ratio}")
            print(f"Number of models: {len(self.models)}")
            print(f"Test performance: {performance}")
            print(f"Distillation data saved to: {self.ensemble_dir}/distillation_data.h5")
            
            return True
            
        except Exception as e:
            print(f"Error during ensemble evaluation: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DREAM-RNN ensemble and generate distillation data"
    )
    
    # Unified CLI supports both legacy (ensemble_dir, data_path, downsample_ratio)
    # and repo-style (model_dir, n_mods, data, out, eval, distill)
    parser.add_argument("--ensemble_dir", type=str, help="Directory containing trained ensemble models")
    parser.add_argument("--model_dir", type=str, help="Directory containing trained models (repo style)")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["DeepSTARR", "lentiMPRA"],
        help="Dataset type (DeepSTARR or lentiMPRA)"
    )
    
    parser.add_argument(
        "--celltype", 
        type=str, 
        required=True,
        help="Cell type (Dev, Hk for DeepSTARR; K562, HepG2 for lentiMPRA)"
    )
    
    parser.add_argument(
        "--save_both_celltypes",
        action="store_true",
        help="For DeepSTARR, save plot performance files for both Dev and Hk celltypes"
    )
    
    parser.add_argument("--downsample_ratio", type=float, help="Downsample ratio (legacy)")
    parser.add_argument("--downsample", type=float, help="Downsample ratio (repo style)")
    
    parser.add_argument("--data_path", type=str, help="Path to HDF5 (legacy)")
    parser.add_argument("--data", type=str, help="Path to HDF5 (repo style)")
    parser.add_argument("--n_mods", type=int, help="Number of models in ensemble (repo style)")
    parser.add_argument("--out", type=str, default=None, help="Output directory (repo style)")
    parser.add_argument("--eval", action='store_true', help="Evaluate ensemble average on test set")
    parser.add_argument("--distill", action='store_true', help="Write ensemble average on train set")
    parser.add_argument("--aleatoric", action='store_true', help="lentiMPRA: use aleatoric head in metrics")
    

    
    args = parser.parse_args()
    
    # Determine which CLI mode to use
    repo_mode = args.model_dir is not None and args.data is not None and args.n_mods is not None

    if repo_mode:
        # Repo-style flow: predict like ensemble_predict_*.py but for DREAM-RNN (PyTorch)
        model_dir = args.model_dir
        n_mods = args.n_mods
        data_path = args.data
        outdir = args.out if args.out else model_dir
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # Load data
        with h5py.File(data_path, 'r') as f:
            X_train = f['Train']['X'][:]
            y_train = f['Train']['y'][:]
            X_val = f['Val']['X'][:]
            y_val = f['Val']['y'][:]
            X_test = f['Test']['X'][:]
            y_test = f['Test']['y'][:]
        if args.distill and args.downsample:
            rng = np.random.default_rng(1234)
            X_train, y_train = _downsample(X_train, y_train, rng, args.downsample)
            print(f'number of training samples after downsampling: {X_train.shape[0]}')
        # Transpose for PyTorch
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

        # Find checkpoints
        ensemble_path = Path(model_dir)
        model_files = []
        for i in range(n_mods):
            p = ensemble_path / f"{i}_model.pth"
            if p.exists():
                model_files.append(str(p))
        if len(model_files) < n_mods:
            extras = sorted([str(p) for p in ensemble_path.glob("*.pth") if str(p) not in model_files])
            model_files += extras[:max(0, n_mods - len(model_files))]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        print(f"predicting with {len(model_files)}/{n_mods} models")

        # Choose builder
        if args.dataset == "DeepSTARR":
            def build_model(device):
                return create_deepstarr_model_fallback().to(device).eval()
        else:
            # decide outputs: use flag (aleatoric -> 2 heads), else 1
            n_outputs = 2 if args.aleatoric else (y_test.shape[1] if y_test.ndim == 2 else 1)
            def build_model(device):
                # Reuse DeepSTARR fallback but change final head size dynamically
                m = create_deepstarr_model_fallback().to(device)
                m.final_block['final_dense'] = torch.nn.Linear(256, n_outputs).to(device)
                m.eval()
                return m

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Predict
        avg_train_pred, avg_test_pred = None, None
        std_train_pred, std_test_pred = None, None
        if args.distill:
            train_preds = []
            for idx, mf in enumerate(model_files):
                print(f'predicting with model {idx+1}/{len(model_files)}')
                keras.backend.clear_session(); gc.collect()
                model = build_model(device)
                state = torch.load(mf, map_location=device)
                # Drop incompatible final head if present
                for k in [
                    'final_block.final_dense.weight',
                    'final_block.final_dense.bias'
                ]:
                    if k in state:
                        try:
                            w = state[k]
                            if hasattr(model, 'final_block'):
                                # infer expected out_features
                                pass
                        finally:
                            state.pop(k)
                model.load_state_dict(state, strict=False)
                out_batches = []
                with torch.no_grad():
                    for j in range(0, X_train.shape[0], 1024):
                        xb = torch.from_numpy(X_train[j:j+1024]).to(device).float()
                        out_batches.append(model(xb).cpu().numpy())
                train_preds.append(np.concatenate(out_batches, axis=0))
            train_stack = np.stack(train_preds, axis=0)
            avg_train_pred = train_stack.mean(axis=0)
            std_train_pred = train_stack.std(axis=0)
            np.save(os.path.join(outdir, "ensemble_avg_y_train.npy"), avg_train_pred)

        if args.eval:
            test_preds = []
            for idx, mf in enumerate(model_files):
                print(f'predicting with model {idx+1}/{len(model_files)}')
                keras.backend.clear_session(); gc.collect()
                model = build_model(device)
                state = torch.load(mf, map_location=device)
                for k in [
                    'final_block.final_dense.weight',
                    'final_block.final_dense.bias'
                ]:
                    if k in state:
                        state.pop(k)
                model.load_state_dict(state, strict=False)
                out_batches = []
                with torch.no_grad():
                    for j in range(0, X_test.shape[0], 1024):
                        xb = torch.from_numpy(X_test[j:j+1024]).to(device).float()
                        out_batches.append(model(xb).cpu().numpy())
                test_preds.append(np.concatenate(out_batches, axis=0))
            test_stack = np.stack(test_preds, axis=0)
            avg_test_pred = test_stack.mean(axis=0)
            std_test_pred = test_stack.std(axis=0)
            
            # Process validation data for distillation
            val_preds = []
            for idx, mf in enumerate(model_files):
                model = build_model(device)
                state = torch.load(mf, map_location=device)
                # Drop incompatible final head if present
                for k in [
                    'final_block.final_dense.weight',
                    'final_block.final_dense.bias'
                ]:
                    if k in state:
                        state.pop(k)
                model.load_state_dict(state, strict=False)
                out_batches = []
                with torch.no_grad():
                    for j in range(0, X_val.shape[0], 1024):
                        xb = torch.from_numpy(X_val[j:j+1024]).to(device).float()
                        out_batches.append(model(xb).cpu().numpy())
                val_preds.append(np.concatenate(out_batches, axis=0))
            val_stack = np.stack(val_preds, axis=0)
            avg_val_pred = val_stack.mean(axis=0)
            std_val_pred = val_stack.std(axis=0)

            # Write performance like existing scripts
            if args.dataset == "DeepSTARR":
                perf = _summarise_deepstarr(avg_test_pred, y_test)
            else:
                if avg_test_pred.shape != y_test.shape:
                    perf = _summarise_lentimpra(avg_test_pred, np.expand_dims(y_test, axis=-1), args.celltype, aleatoric=args.aleatoric, epistemic=False)
                else:
                    perf = _summarise_lentimpra(avg_test_pred, y_test, args.celltype, aleatoric=args.aleatoric, epistemic=False)
            perf.to_csv(os.path.join(outdir, "ensemble_performance_avg.csv"), index=False)

        # Save unified distillation NPZ if we have any predictions
        if args.distill or args.eval:
            npz_payload = {}
            if avg_train_pred is not None:
                npz_payload['train_mean'] = avg_train_pred
            if std_train_pred is not None:
                npz_payload['train_std'] = std_train_pred
            if avg_val_pred is not None:
                npz_payload['val_mean'] = avg_val_pred
            if std_val_pred is not None:
                npz_payload['val_std'] = std_val_pred
            if avg_test_pred is not None:
                npz_payload['test_mean'] = avg_test_pred
            if std_test_pred is not None:
                npz_payload['test_std'] = std_test_pred
            if npz_payload:
                tag = f"{args.dataset}_{(args.celltype or 'NA')}_{args.downsample or 'full'}"
                np.savez_compressed(os.path.join(outdir, f"distillation_data_{tag}.npz"), **npz_payload)

        return

    # Legacy mode: keep original behavior
    # Validate arguments
    if args.dataset == "DeepSTARR" and args.celltype not in ["Dev", "Hk"]:
        raise ValueError("For DeepSTARR, celltype must be 'Dev' or 'Hk'")
    if args.dataset == "lentiMPRA" and args.celltype not in ["K562", "HepG2"]:
        raise ValueError("For lentiMPRA, celltype must be 'K562' or 'HepG2'")
    if args.downsample_ratio not in [0.1, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("Downsample ratio must be one of: 0.1, 0.25, 0.5, 0.75, 1.0")

    evaluator = EnsembleEvaluator(
        ensemble_dir=args.ensemble_dir,
        dataset=args.dataset,
        celltype=args.celltype,
        downsample_ratio=args.downsample_ratio,
        data_path=args.data_path,
        save_both_celltypes=args.save_both_celltypes
    )
    evaluator.evaluate_ensemble()

if __name__ == "__main__":
    main()
