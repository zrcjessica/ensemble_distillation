#!/usr/bin/env python3
"""
GPU-accelerated evaluation script for DeepSTARR models
"""

import os
import sys
import torch
import torch.nn as nn
import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Disable cuDNN to avoid version compatibility issues
torch.backends.cudnn.enabled = False

# Add the dream-challenge-2022 directory to the path
sys.path.insert(0, 'zenodo/dream-challenge-2022')

# Import the model class
from train_DREAM_RNN_official import DREAM_RNN_Official

def evaluate_model(model_path, downsample_ratio, output_path, device='cuda:0'):
    """Evaluate a DeepSTARR model on GPU"""
    print(f"Evaluating model: {model_path}")
    print(f"Using device: {device}")
    
    # Load the model
    model = DREAM_RNN_Official(
        seqsize=249,
        in_channels=4
    )
    
    # Modify final layer to have 4 outputs for unified DEGU distillation
    model.final_block['final_dense'] = nn.Linear(256, 4)
    
    # Load the state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Get the correct data file path
    if downsample_ratio == "1.0":
        data_file = "zenodo/data/DeepSTARR_distillation_data.h5"
    else:
        data_file = f"zenodo/data/DeepSTARR_downsample{downsample_ratio}_distillation_data.h5"
    
    print(f"Loading test data from: {data_file}")
    
    # Load test data
    with h5py.File(data_file, 'r') as f:
        test_sequences = torch.FloatTensor(f['Test/X'][:]).transpose(1, 2)
        test_y = torch.FloatTensor(f['Test/y'][:])
    
    print(f"Loaded {len(test_sequences)} test sequences")
    
    # Use larger batch size for GPU processing
    batch_size = 4096
    n_batches = (len(test_sequences) + batch_size - 1) // batch_size
    
    print(f"Processing {n_batches} batches with batch size {batch_size}")
    
    # Evaluate with progress bar
    dev_predictions = []
    hk_predictions = []
    dev_targets = []
    hk_targets = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_sequences), batch_size), desc="Evaluating"):
            batch_sequences = test_sequences[i:i+batch_size].to(device)
            batch_targets = test_y[i:i+batch_size]
            
            outputs = model(batch_sequences)
            
            if outputs.shape[1] == 4:
                dev_pred = outputs[:, 0].cpu().numpy()
                hk_pred = outputs[:, 1].cpu().numpy()
            else:
                dev_pred = outputs[:, 0].cpu().numpy()
                hk_pred = outputs[:, 1].cpu().numpy()
            
            dev_target = batch_targets[:, 0].numpy()
            hk_target = batch_targets[:, 1].numpy()
            
            dev_predictions.extend(dev_pred)
            hk_predictions.extend(hk_pred)
            dev_targets.extend(dev_target)
            hk_targets.extend(hk_target)
    
    print("Calculating metrics...")
    
    # Convert to numpy arrays
    dev_predictions = np.array(dev_predictions)
    hk_predictions = np.array(hk_predictions)
    dev_targets = np.array(dev_targets)
    hk_targets = np.array(hk_targets)
    
    # Calculate metrics
    dev_pearson = pearsonr(dev_predictions, dev_targets)[0]
    hk_pearson = pearsonr(hk_predictions, hk_targets)[0]
    activity_pearson = pearsonr(np.concatenate([dev_predictions, hk_predictions]), 
                               np.concatenate([dev_targets, hk_targets]))[0]
    
    dev_spearman = spearmanr(dev_predictions, dev_targets)[0]
    hk_spearman = spearmanr(hk_predictions, hk_targets)[0]
    activity_spearman = spearmanr(np.concatenate([dev_predictions, hk_predictions]), 
                                 np.concatenate([dev_targets, hk_targets]))[0]
    
    dev_mse = mean_squared_error(dev_targets, dev_predictions)
    hk_mse = mean_squared_error(hk_targets, hk_predictions)
    activity_mse = mean_squared_error(np.concatenate([dev_targets, hk_targets]), 
                                     np.concatenate([dev_predictions, hk_predictions]))
    
    # Create results
    results = {
        'Dev_pearson': dev_pearson,
        'Hk_pearson': hk_pearson,
        'activity_pearson': activity_pearson,
        'Dev_spearman': dev_spearman,
        'Hk_spearman': hk_spearman,
        'activity_spearman': activity_spearman,
        'Dev_mse': dev_mse,
        'Hk_mse': hk_mse,
        'activity_mse': activity_mse,
        'n_test_samples': len(test_sequences)
    }
    
    # Save results
    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to: {output_path}")
    print(f"Dev Pearson: {results['Dev_pearson']:.4f}")
    print(f"Hk Pearson: {results['Hk_pearson']:.4f}")
    
    return results

def main():
    # Use GPU 0 (free GPU)
    device = 'cuda:0'
    
    # Evaluate the models
    models = [
        {
            'model_path': '/tmp/3_degu_model_final.pth',
            'downsample_ratio': '1.0',
            'output_path': 'deepstarr_1.0_model3_performance.csv'
        },
        {
            'model_path': '/tmp/4_degu_model_final.pth',
            'downsample_ratio': '1.0',
            'output_path': 'deepstarr_1.0_model4_performance.csv'
        }
    ]
    
    for model_info in models:
        try:
            if os.path.exists(model_info['model_path']):
                evaluate_model(
                    model_info['model_path'],
                    model_info['downsample_ratio'],
                    model_info['output_path'],
                    device
                )
            else:
                print(f"Model file not found: {model_info['model_path']}")
        except Exception as e:
            print(f"Error evaluating {model_info['model_path']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
