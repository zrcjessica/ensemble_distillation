#!/usr/bin/env python3
"""
Unified DeepSTARR DEGU Distillation Training Script

This script trains DEGU distilled models that learn to predict both Dev and Hk outputs
from DeepSTARR ensemble predictions, following the @DEGU_paper methodology.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
import h5py

# Add the code directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))

# Import the model classes
from train_DREAM_RNN_DeepSTARR import DREAM_RNN_Official

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedDeepSTARRDEGUTrainer:
    """Trainer for unified DeepSTARR DEGU distilled models (both Dev and Hk outputs)"""
    
    def __init__(self, distillation_data_path, output_dir, model_index=0, 
                 batch_size=1024, learning_rate=0.005, epochs=80, device='cuda'):
        self.distillation_data_path = Path(distillation_data_path)
        self.output_dir = Path(output_dir)
        self.model_index = model_index
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + model_index)
        np.random.seed(42 + model_index)
        logger.info(f"Set random seed to {42 + model_index} for model index {model_index}")
        
        # Initialize data and model
        print("=== DEBUG: Starting data loading ===")
        sys.stdout.flush()
        self._load_distillation_data()
        print("=== DEBUG: Distillation data loaded ===")
        sys.stdout.flush()
        self._load_actual_sequences()
        print("=== DEBUG: Actual sequences loaded ===")
        sys.stdout.flush()
        self._create_data_loaders()
        print("=== DEBUG: Data loaders created ===")
        sys.stdout.flush()
        self._initialize_model()
        print("=== DEBUG: Model initialized ===")
        sys.stdout.flush()
        self._setup_optimizer()
        print("=== DEBUG: Optimizer setup complete ===")
        sys.stdout.flush()
        
        logger.info(f"Initialized unified DeepSTARR DEGU distillation trainer for model {model_index}")
        logger.info(f"Device: {self.device}")
    
    def _load_distillation_data(self):
        """Load ensemble distillation data (mean and std predictions)"""
        print("=== DEBUG: Starting distillation data loading ===")
        sys.stdout.flush()
        logger.info(f"Loading distillation data from: {self.distillation_data_path}")
        
        # Check if file is HDF5 or NPZ format
        if str(self.distillation_data_path).endswith('.h5'):
            print("=== DEBUG: Loading HDF5 file ===")
            sys.stdout.flush()
            # Load HDF5 file
            with h5py.File(self.distillation_data_path, 'r') as f:
                print("=== DEBUG: HDF5 file opened, loading tensors ===")
                sys.stdout.flush()
                # Load ensemble predictions (mean and std for both Dev and Hk)
                self.train_mean = torch.FloatTensor(f['train_mean'][:])  # (N, 2): [Dev_mean, Hk_mean]
                print("=== DEBUG: train_mean loaded ===")
                sys.stdout.flush()
                self.train_std = torch.FloatTensor(f['train_std'][:])    # (N, 2): [Dev_std, Hk_std]
                print("=== DEBUG: train_std loaded ===")
                sys.stdout.flush()
                self.val_mean = torch.FloatTensor(f['val_mean'][:])
                print("=== DEBUG: val_mean loaded ===")
                sys.stdout.flush()
                self.val_std = torch.FloatTensor(f['val_std'][:])
                print("=== DEBUG: val_std loaded ===")
                sys.stdout.flush()
                self.test_mean = torch.FloatTensor(f['test_mean'][:])
                print("=== DEBUG: test_mean loaded ===")
                sys.stdout.flush()
                self.test_std = torch.FloatTensor(f['test_std'][:])
                print("=== DEBUG: test_std loaded ===")
                sys.stdout.flush()
        else:
            print("=== DEBUG: Loading NPZ file ===")
            sys.stdout.flush()
            # Load NPZ file
            data = np.load(self.distillation_data_path)
            print("=== DEBUG: NPZ file loaded, creating tensors ===")
            sys.stdout.flush()
            # Load ensemble predictions (mean and std for both Dev and Hk)
            self.train_mean = torch.FloatTensor(data['train_mean'])  # (N, 2): [Dev_mean, Hk_mean]
            print("=== DEBUG: train_mean tensor created ===")
            sys.stdout.flush()
            self.train_std = torch.FloatTensor(data['train_std'])    # (N, 2): [Dev_std, Hk_std]
            print("=== DEBUG: train_std tensor created ===")
            sys.stdout.flush()
            self.val_mean = torch.FloatTensor(data['val_mean'])
            print("=== DEBUG: val_mean tensor created ===")
            sys.stdout.flush()
            self.val_std = torch.FloatTensor(data['val_std'])
            print("=== DEBUG: val_std tensor created ===")
            sys.stdout.flush()
            self.test_mean = torch.FloatTensor(data['test_mean'])
            print("=== DEBUG: test_mean tensor created ===")
            sys.stdout.flush()
            self.test_std = torch.FloatTensor(data['test_std'])
            print("=== DEBUG: test_std tensor created ===")
            sys.stdout.flush()
        
        print("=== DEBUG: All distillation data loaded successfully ===")
        sys.stdout.flush()
        logger.info(f"Loaded distillation data:")
        logger.info(f"  Training: {self.train_mean.shape}")
        logger.info(f"  Validation: {self.val_mean.shape}")
        logger.info(f"  Test: {self.test_mean.shape}")
    
    def _load_actual_sequences(self):
        """Load actual DNA sequences from the original dataset"""
        # Extract downsample ratio from distillation data path
        path_str = str(self.distillation_data_path)
        if "0.005" in path_str:
            downsample_ratio = 0.005
        elif "0.01" in path_str:
            downsample_ratio = 0.01
        elif "0.1" in path_str:
            downsample_ratio = 0.1
        elif "0.25" in path_str:
            downsample_ratio = 0.25
        elif "0.5" in path_str:
            downsample_ratio = 0.5
        elif "0.75" in path_str:
            downsample_ratio = 0.75
        elif "1.0" in path_str:
            downsample_ratio = 1.0
        else:
            raise ValueError("Could not determine downsample ratio from path")
        
        # Use the full dataset and downsample on-the-fly like the standard training script
        data_file = "/home/jessica/ensemble_distillation/zenodo/data/DeepSTARR_distillation_data.h5"
        logger.info(f"Loading actual sequences from: {data_file}")
        
        with h5py.File(data_file, 'r') as f:
            train_sequences = f['Train/X'][:]
            val_sequences = f['Val/X'][:]
            test_sequences = f['Test/X'][:]
        
        # Apply on-the-fly downsampling to training data if needed (same as standard training script)
        if downsample_ratio < 1.0:
            logger.info(f"Downsampling training sequences to {downsample_ratio:.1%} with fixed seed for reproducibility")
            rng = np.random.default_rng(1234)  # Same fixed seed as standard training script
            n_samples = max(1, int(len(train_sequences) * downsample_ratio))
            indices = rng.choice(len(train_sequences), n_samples, replace=False)
            train_sequences = train_sequences[indices]
            logger.info(f"  Training sequences downsampled from {int(len(train_sequences)/downsample_ratio)} to {len(train_sequences)} samples")
        
        # Ensure sequence counts match distillation data
        if train_sequences.shape[0] != self.train_mean.shape[0]:
            logger.warning(f"Training sequence count mismatch: {train_sequences.shape[0]} vs {self.train_mean.shape[0]}")
            min_size = min(train_sequences.shape[0], self.train_mean.shape[0])
            train_sequences = train_sequences[:min_size]
            self.train_mean = self.train_mean[:min_size]
            self.train_std = self.train_std[:min_size]
            logger.info(f"Truncated training to {min_size} samples")
        
        if val_sequences.shape[0] != self.val_mean.shape[0]:
            logger.warning(f"Validation sequence count mismatch: {val_sequences.shape[0]} vs {self.val_mean.shape[0]}")
            min_size = min(val_sequences.shape[0], self.val_mean.shape[0])
            val_sequences = val_sequences[:min_size]
            self.val_mean = self.val_mean[:min_size]
            self.val_std = self.val_std[:min_size]
            logger.info(f"Truncated validation to {min_size} samples")
        
        if test_sequences.shape[0] != self.test_mean.shape[0]:
            logger.warning(f"Test sequence count mismatch: {test_sequences.shape[0]} vs {self.test_mean.shape[0]}")
            min_size = min(test_sequences.shape[0], self.test_mean.shape[0])
            test_sequences = test_sequences[:min_size]
            self.test_mean = self.test_mean[:min_size]
            self.test_std = self.test_std[:min_size]
            logger.info(f"Truncated test to {min_size} samples")
        
        # Convert to PyTorch tensors and transpose to (batch, channels, length) format
        self.train_sequences = torch.FloatTensor(train_sequences).transpose(1, 2)  # (N, 4, 249)
        self.val_sequences = torch.FloatTensor(val_sequences).transpose(1, 2)      # (N, 4, 249)
        self.test_sequences = torch.FloatTensor(test_sequences).transpose(1, 2)    # (N, 4, 249)
        
        logger.info(f"Loaded actual DNA sequences:")
        logger.info(f"  Training: {self.train_sequences.shape}")
        logger.info(f"  Validation: {self.val_sequences.shape}")
        logger.info(f"  Test: {self.test_sequences.shape}")
    
    def _create_data_loaders(self):
        """Create PyTorch data loaders for training and validation with actual sequences"""
        # Training data: (sequences, mean, std) triplets
        train_dataset = TensorDataset(self.train_sequences, self.train_mean, self.train_std)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Validation data: (sequences, mean, std) triplets
        val_dataset = TensorDataset(self.val_sequences, self.val_mean, self.val_std)
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        logger.info(f"Created optimized data loaders with batch size {self.batch_size}")
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def _initialize_model(self):
        """Initialize the DREAM-RNN model with 4 outputs for unified DEGU distillation"""
        print("=== DEBUG: Starting model initialization ===")
        sys.stdout.flush()
        logger.info("Initializing unified DeepSTARR DREAM-RNN model...")
        
        print("=== DEBUG: Creating DREAM_RNN_Official model ===")
        sys.stdout.flush()
        # Create model with 4 outputs: [Dev_mean, Hk_mean, Dev_std, Hk_std]
        self.model = DREAM_RNN_Official(seqsize=249, in_channels=4)
        print("=== DEBUG: DREAM_RNN_Official model created successfully ===")
        sys.stdout.flush()
        
        print("=== DEBUG: Modifying final layer ===")
        sys.stdout.flush()
        # Modify final layer to have 4 outputs for unified DEGU distillation
        self.model.final_block['final_dense'] = nn.Linear(256, 4)
        print("=== DEBUG: Final layer modified successfully ===")
        sys.stdout.flush()
        
        print("=== DEBUG: About to move model to device ===")
        print(f"=== DEBUG: Device: {self.device} ===")
        print(f"=== DEBUG: Model device before: {next(self.model.parameters()).device} ===")
        print(f"=== DEBUG: Available GPU memory before: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)} bytes ===")
        sys.stdout.flush()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        print("=== DEBUG: Model successfully moved to device ===")
        print(f"=== DEBUG: Model device after: {next(self.model.parameters()).device} ===")
        print(f"=== DEBUG: Available GPU memory after: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)} bytes ===")
        sys.stdout.flush()
        logger.info(f"Initialized model with 4 outputs: [Dev_mean, Hk_mean, Dev_std, Hk_std]")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler following @DREAM_paper methodology"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Use AdamW optimizer with weight decay to mirror teacher (DREAM-RNN standard)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        # Use OneCycleLR scheduler with default parameters (matching @DREAM_paper)
        total_steps = len(self.train_loader) * self.epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps
        )
        
        logger.info(f"Optimizer: Adam with lr={self.learning_rate}")
        logger.info(f"Scheduler: OneCycleLR with {total_steps} total steps")
    
    def _compute_loss(self, outputs, mean_targets, std_targets):
        """Compute DEGU distillation loss: L = L_mean + λ * L_uncertainty for each task"""
        # Split outputs: [Dev_mean, Hk_mean, Dev_std, Hk_std]
        dev_mean_pred = outputs[:, 0:1]    # Dev mean prediction
        hk_mean_pred = outputs[:, 1:2]     # Hk mean prediction
        dev_std_pred = outputs[:, 2:3]     # Dev std prediction
        hk_std_pred = outputs[:, 3:4]      # Hk std prediction
        
        # Split targets: mean_targets=[Dev_mean, Hk_mean], std_targets=[Dev_std, Hk_std]
        dev_mean_target = mean_targets[:, 0:1]  # Dev mean target
        hk_mean_target = mean_targets[:, 1:2]   # Hk mean target
        dev_std_target = std_targets[:, 0:1]    # Dev std target
        hk_std_target = std_targets[:, 1:2]     # Hk std target
        
        # Compute DEGU loss for each task separately (following DEGU paper methodology)
        # L = L_mean + λ * L_uncertainty 
        # For Figure 2 replication, prioritize mean prediction (activity performance)
        # Since Figure 2 only evaluates activity predictions, try λ=0 to focus purely on mean predictions
        lambda_uncertainty = 0.0  # Focus purely on mean predictions for Figure 2 replication
        
        # Dev task loss
        dev_mean_loss = nn.functional.mse_loss(dev_mean_pred, dev_mean_target)
        dev_uncertainty_loss = nn.functional.mse_loss(dev_std_pred, dev_std_target)
        dev_total_loss = dev_mean_loss + lambda_uncertainty * dev_uncertainty_loss
        
        # Hk task loss
        hk_mean_loss = nn.functional.mse_loss(hk_mean_pred, hk_mean_target)
        hk_uncertainty_loss = nn.functional.mse_loss(hk_std_pred, hk_std_target)
        hk_total_loss = hk_mean_loss + lambda_uncertainty * hk_uncertainty_loss
        
        # Total loss: sum of both tasks
        total_loss = dev_total_loss + hk_total_loss
        
        return total_loss, {
            'dev_mean_loss': dev_mean_loss.item(),
            'dev_uncertainty_loss': dev_uncertainty_loss.item(),
            'dev_total_loss': dev_total_loss.item(),
            'hk_mean_loss': hk_mean_loss.item(),
            'hk_uncertainty_loss': hk_uncertainty_loss.item(),
            'hk_total_loss': hk_total_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {
            'dev_mean_loss': 0.0, 'dev_uncertainty_loss': 0.0, 'dev_total_loss': 0.0,
            'hk_mean_loss': 0.0, 'hk_uncertainty_loss': 0.0, 'hk_total_loss': 0.0,
            'total_loss': 0.0
        }
        
        for batch_idx, (sequences, mean_targets, std_targets) in enumerate(self.train_loader):
            sequences = sequences.to(self.device)
            mean_targets = mean_targets.to(self.device)
            std_targets = std_targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss, components = self._compute_loss(outputs, mean_targets, std_targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            for key, value in components.items():
                loss_components[key] += value
        
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {key: value / len(self.train_loader) for key, value in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, epoch):
        """Validate against ensemble predictions (consistent with training targets)"""
        self.model.eval()
        total_loss = 0.0
        loss_components = {
            'dev_activity_loss': 0.0, 'hk_activity_loss': 0.0, 'total_loss': 0.0
        }
        
        with torch.no_grad():
            # Use ensemble predictions for validation (consistent with training)
            val_dataset = TensorDataset(self.val_sequences, self.val_mean, self.val_std)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            for sequences, mean_targets, std_targets in val_loader:
                sequences = sequences.to(self.device)
                mean_targets = mean_targets.to(self.device)
                std_targets = std_targets.to(self.device)
                
                outputs = self.model(sequences)
                
                # For validation, use ensemble predictions as targets (consistent with training)
                dev_pred = outputs[:, 0:1]  # Dev activity prediction
                hk_pred = outputs[:, 1:2]   # Hk activity prediction
                dev_target = mean_targets[:, 0:1]  # Dev ensemble mean target
                hk_target = mean_targets[:, 1:2]   # Hk ensemble mean target
                
                # Use MSE loss for validation (standard practice)
                dev_loss = nn.MSELoss()(dev_pred, dev_target)
                hk_loss = nn.MSELoss()(hk_pred, hk_target)
                total_batch_loss = dev_loss + hk_loss
                
                total_loss += total_batch_loss.item()
                loss_components['dev_activity_loss'] += dev_loss.item()
                loss_components['hk_activity_loss'] += hk_loss.item()
        
        # Average losses
        avg_loss = total_loss / len(val_loader)
        avg_components = {key: value / len(val_loader) for key, value in loss_components.items()}
        
        return avg_loss, avg_components
    

    def train(self):
        """Train the unified DeepSTARR DEGU distilled model for full epochs (no early stopping)"""
        start_epoch = 1
        best_val_loss = float('inf')
        
        logger.info("Starting unified DeepSTARR DEGU distillation training")
        
        logger.info(f"Training for {self.epochs} epochs (no early stopping)")
        
        training_history = []
        
        for epoch in range(start_epoch, self.epochs + 1):
            # Train
            train_loss, train_components = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_components = self.validate(epoch)
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Log progress
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Dev Activity Loss: {val_components['dev_activity_loss']:.4f}, Val Hk Activity Loss: {val_components['hk_activity_loss']:.4f}")
            
            # Store history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dev_activity_loss': val_components['dev_activity_loss'],
                'val_hk_activity_loss': val_components['hk_activity_loss'],
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
        
        # Save final model
        self._save_checkpoint(self.epochs, is_best=True, is_final=True)
        
        # Save training history
        history_df = pd.DataFrame(training_history)
        history_path = self.output_dir / f"{self.model_index}_training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to: {history_path}")
        
        # Evaluate the final model and save performance results
        self._evaluate_and_save_performance()
        
        return training_history
    
    def _save_checkpoint(self, epoch, is_best=False, is_final=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if is_final:
            checkpoint_path = self.output_dir / f"{self.model_index}_degu_model_final.pth"
        else:
            checkpoint_path = self.output_dir / f"{self.model_index}_degu_model_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best and not is_final:
            logger.info(f"Saved best model checkpoint to: {checkpoint_path}")
        elif is_final:
            logger.info(f"Saved final model to: {checkpoint_path}")
    
    def _evaluate_and_save_performance(self):
        """Evaluate the trained model on original experimental test data and save performance results"""
        print("Evaluating unified model performance on original experimental test data...")
        
        # Load the best model
        best_model_path = self.output_dir / f"{self.model_index}_degu_model_final.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from: {best_model_path}")
        
        self.model.eval()
        
        # Load original experimental test data (not ensemble predictions)
        # Always use the full dataset for evaluation
        data_file = "/home/jessica/ensemble_distillation/zenodo/data/DeepSTARR_distillation_data.h5"
        
        with h5py.File(data_file, 'r') as f:
            test_sequences = torch.FloatTensor(f['Test/X'][:]).transpose(1, 2)
            test_y = torch.FloatTensor(f['Test/y'][:])  # Original experimental values
        
        # Evaluate on original experimental test data
        dev_predictions = []
        hk_predictions = []
        dev_targets = []
        hk_targets = []
        
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(test_sequences), batch_size):
                batch_sequences = test_sequences[i:i+batch_size].to(self.device)
                batch_targets = test_y[i:i+batch_size]  # Original experimental values
                
                outputs = self.model(batch_sequences)
                
                # Handle 4 outputs: Dev mean, Hk mean, Dev std, Hk std
                dev_pred = outputs[:, 0:1]  # Dev activity (mean)
                hk_pred = outputs[:, 1:2]   # Hk activity (mean)
                
                # Store predictions and targets for both Dev and Hk
                dev_predictions.append(dev_pred.cpu())
                hk_predictions.append(hk_pred.cpu())
                dev_targets.append(batch_targets[:, 0:1].cpu())  # Dev experimental target
                hk_targets.append(batch_targets[:, 1:2].cpu())   # Hk experimental target
        
        # Concatenate all predictions and targets
        dev_predictions = torch.cat(dev_predictions, dim=0).numpy()
        hk_predictions = torch.cat(hk_predictions, dim=0).numpy()
        dev_targets = torch.cat(dev_targets, dim=0).numpy()
        hk_targets = torch.cat(hk_targets, dim=0).numpy()
        
        # Calculate performance metrics for both Dev and Hk
        from scipy.stats import pearsonr, spearmanr
        
        dev_pred_flat = dev_predictions.flatten()
        dev_target_flat = dev_targets.flatten()
        hk_pred_flat = hk_predictions.flatten()
        hk_target_flat = hk_targets.flatten()
        
        # Dev performance
        dev_pearson_r, _ = pearsonr(dev_pred_flat, dev_target_flat)
        dev_spearman_rho, _ = spearmanr(dev_pred_flat, dev_target_flat)
        dev_mse = np.mean((dev_pred_flat - dev_target_flat) ** 2)
        
        # Hk performance
        hk_pearson_r, _ = pearsonr(hk_pred_flat, hk_target_flat)
        hk_spearman_rho, _ = spearmanr(hk_pred_flat, hk_target_flat)
        hk_mse = np.mean((hk_pred_flat - hk_target_flat) ** 2)
        
        print(f"Test Performance on Original Experimental Data:")
        print(f"  Dev - Pearson r: {dev_pearson_r:.4f}, Spearman ρ: {dev_spearman_rho:.4f}, MSE: {dev_mse:.6f}")
        print(f"  Hk  - Pearson r: {hk_pearson_r:.4f}, Spearman ρ: {hk_spearman_rho:.4f}, MSE: {hk_mse:.6f}")
        
        # Save performance results
        performance_data = {
            'Dev_pearson': dev_pearson_r,
            'Hk_pearson': hk_pearson_r,
            'Dev_spearman': dev_spearman_rho,
            'Hk_spearman': hk_spearman_rho,
            'Dev_mse': dev_mse,
            'Hk_mse': hk_mse,
            'n_test_samples': len(dev_targets)
        }
        
        # Save as CSV
        performance_df = pd.DataFrame([performance_data])
        performance_path = self.output_dir / f"{self.model_index}_performance.csv"
        performance_df.to_csv(performance_path, index=False)
        print(f"Saved performance results to: {performance_path}")

def main():
    parser = argparse.ArgumentParser(description='Train unified DeepSTARR DEGU distilled model')
    parser.add_argument('--distillation_data', required=True, help='Path to distillation data NPZ file')
    parser.add_argument('--output_dir', required=True, help='Output directory for trained model')
    parser.add_argument('--model_index', type=int, default=0, help='Model index for reproducibility')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Disable cuDNN to avoid version conflicts
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to avoid version conflicts")
    
    # Create trainer and train
    trainer = UnifiedDeepSTARRDEGUTrainer(
        distillation_data_path=args.distillation_data,
        output_dir=args.output_dir,
        model_index=args.model_index,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )
    
    # Train the model
    trainer.train()
    print("Unified DeepSTARR DEGU distillation training completed!")

if __name__ == "__main__":
    main()
