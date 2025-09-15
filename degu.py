import tensorflow as tf
from tensorflow import keras 
import numpy as np
from scipy import stats
from sklearn import metrics


######################################################################
# DEGU class
######################################################################

class DEGU():
    """
    Deep Ensemble with Gaussian Uncertainty (DEGU) implementation.
    
    This class implements an ensemble-based uncertainty quantification method that:
    1. Trains multiple models (ensemble) on the same data with different weight initializations
    2. Uses the ensemble to generate predictions with uncertainty estimates
    3. Distills the ensemble knowledge into a single student model for efficiency
    
    The key idea is that ensemble disagreement provides a measure of epistemic uncertainty.
    """
    
    def __init__(self, model, num_ensemble, uncertainty_fun=None):
        """
        Initialize DEGU ensemble framework.
        
        Args:
            model: Base neural network model to be used as template for ensemble members
            num_ensemble: Number of models in the ensemble (typically 5-10)
            uncertainty_fun: Function to compute uncertainty from ensemble predictions
                           (default: log variance of predictions across ensemble)
        """
        if uncertainty_fun is None:
            uncertainty_fun = uncertainty_logvar
        self.teacher_base_model = model              # Template model for ensemble members
        self.num_ensemble = num_ensemble             # Size of ensemble
        self.uncertainty_fun = uncertainty_fun       # Uncertainty quantification function
        self.teacher_ensemble_paths = []             # Paths to saved ensemble model weights
        self.ensemble_history = []                   # Training histories for each ensemble member

    def train_ensemble(self, x_train, y_train, validation_data, optimizer, loss, train_fun, save_prefix):
        """
        Train an ensemble of models with different random weight initializations.
        
        Each model in the ensemble is trained on the same data but starts with different
        random weights, leading to different local minima and thus ensemble diversity.
        
        Args:
            x_train: Training input features
            y_train: Training target labels
            validation_data: Validation dataset (x_val, y_val) for early stopping
            optimizer: Keras optimizer instance for model compilation
            loss: Loss function for model compilation
            train_fun: Training function that handles fitting with callbacks
            save_prefix: Prefix for saved model weight files
        """
        self.ensemble_paths = []      # Reset paths for new ensemble
        self.ensemble_history = []    # Reset training histories
        
        # Train each ensemble member
        for model_idx in range(self.num_ensemble):
            print('Training model %d'%(model_idx + 1))

            # Reinitialize model weights randomly to ensure diversity
            self._reinitialize_model_weights(self.teacher_base_model)
            fresh_optimizer = optimizer.__class__(**optimizer.get_config())
            self.teacher_base_model.compile(optimizer=fresh_optimizer, loss=loss)

            # Train the model using provided training function
            history = train_fun(self.teacher_base_model, x_train, y_train, validation_data, save_prefix)
            self.ensemble_history.append(history)

            # Save trained model weights with unique identifier
            ensemble_name = save_prefix+'_'+str(model_idx)+'.weights.h5'
            self.teacher_base_model.save_weights(ensemble_name)
            self.ensemble_paths.append(ensemble_name)


    def pred_ensemble(self, x, batch_size=512):
        """
        Generate ensemble predictions with uncertainty estimates.
        
        This method:
        1. Loads each trained ensemble member
        2. Gets predictions from each member
        3. Computes ensemble mean (aleatoric uncertainty reduction)
        4. Computes ensemble disagreement (epistemic uncertainty estimate)
        
        Args:
            x: Input data for prediction
            batch_size: Batch size for prediction (memory management)
            
        Returns:
            ensemble_mean: Average prediction across all ensemble members
            ensemble_uncertainty: Uncertainty measure based on ensemble disagreement
        """
        preds = []
        
        # Get predictions from each ensemble member
        for model_idx in range(self.num_ensemble):
            # Load weights for current ensemble member
            self.teacher_base_model.load_weights(self.ensemble_paths[model_idx])

            # Generate predictions for this ensemble member
            preds.append(self.teacher_base_model.predict(x, batch_size=batch_size, verbose=False))

        # Convert to numpy array for easier manipulation: shape (num_ensemble, num_samples, num_outputs)
        preds = np.array(preds)
        
        # Ensemble mean: average across ensemble members (axis=0)
        # This reduces aleatoric (data) uncertainty through averaging
        ensemble_mean = np.mean(preds, axis=0)
        
        # Ensemble uncertainty: measure of disagreement between ensemble members
        # This captures epistemic (model) uncertainty
        ensemble_uncertainty = self.uncertainty_fun(preds, axis=0)
        
        return ensemble_mean, ensemble_uncertainty

    def eval_ensemble(self, x, y, eval_fun, batch_size=512):
        """
        Comprehensive evaluation of the ensemble on test data.
        
        Evaluates both individual ensemble members and the combined ensemble,
        providing detailed performance analysis.
        
        Args:
            x: Test input features
            y: True test labels
            eval_fun: Evaluation function (returns metrics like MSE, correlation, etc.)
            batch_size: Batch size for prediction
            
        Returns:
            ensemble_results: Performance metrics for ensemble predictions
            preds_ensemble: Combined ensemble predictions and uncertainties
            standard_results: List of individual model performance metrics
            all_preds: Raw predictions from all ensemble members
        """
        all_preds = []          # Store all individual predictions
        standard_results = []   # Store individual model evaluation results
        
        # Evaluate each ensemble member individually
        for model_idx in range(self.num_ensemble):
            print('Model %d'%(model_idx))

            # Load weights for current ensemble member
            self.teacher_base_model.load_weights(self.ensemble_paths[model_idx])

            # Get predictions from current model
            preds = self.teacher_base_model.predict(x, batch_size=batch_size, verbose=False)
            all_preds.append(preds)

            # Evaluate individual model performance
            results = eval_fun(preds, y)
            standard_results.append(results)

        # Convert to array for ensemble calculations
        all_preds = np.array(all_preds)
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(all_preds, axis=0)
        ensemble_uncertainty = self.uncertainty_fun(all_preds, axis=0)
        
        # Combine predictions and uncertainties for comprehensive output
        preds_ensemble = np.concatenate([ensemble_mean, ensemble_uncertainty], axis=1)

        # Evaluate ensemble performance (typically better than individual models)
        print('Ensemble')
        ensemble_results = eval_fun(ensemble_mean, y)

        return ensemble_results, preds_ensemble, standard_results, all_preds

    def distill_student(self, student_model, x_train, y_train, train_fun,
                        save_prefix, validation_data, batch_size=512):
        """
        Knowledge distillation: Train a single student model to mimic ensemble behavior.
        
        The ensemble (teacher) is computationally expensive for inference. This method
        trains a single student model to predict both the ensemble mean and uncertainty,
        providing efficient uncertainty-aware predictions.
        
        Args:
            student_model: Single model to be trained (more efficient than ensemble)
            x_train: Training input features
            y_train: Original training labels
            train_fun: Training function for student model
            save_prefix: Prefix for saving student model weights
            validation_data: Original validation data
            batch_size: Batch size for ensemble prediction generation
            
        Returns:
            history: Training history of student model
        """
        # Generate "soft targets" from ensemble for training data
        # These include both predictions and uncertainty estimates
        print('Generate ensemble labels')
        train_mean, train_unc = self.pred_ensemble(x_train, batch_size=batch_size)
        y_train_ensemble = np.concatenate([train_mean, train_unc], axis=1)

        # Generate ensemble predictions for validation data
        # Use original labels for validation but add uncertainty estimates
        x_valid, y_valid = validation_data
        valid_mean, valid_unc = self.pred_ensemble(x_valid, batch_size=batch_size)
        y_valid_ensemble = np.concatenate([y_valid, valid_unc], axis=1)
        validation_data = [x_valid, y_valid_ensemble]

        # Train student model to predict both ensemble mean and uncertainty
        # Student learns to mimic ensemble behavior in a single forward pass
        print('Train distilled model')
        history = train_fun(student_model, x_train, y_train_ensemble, validation_data, save_prefix)

        # Save the trained student model
        student_model.save_weights(save_prefix+'.weights.h5')

        return history

    def eval_student(self, student_model, x, y, eval_fun, batch_size=512):
        """
        Evaluate the distilled student model against ensemble performance.
        
        Compares student model predictions with ensemble ground truth to assess
        how well knowledge distillation preserved ensemble behavior.
        
        Args:
            student_model: Trained student model
            x: Test input features
            y: True test labels
            eval_fun: Evaluation function
            batch_size: Batch size for predictions
            
        Returns:
            results: Performance metrics comparing student to ensemble
            pred: Student model predictions
            y_ensemble: Ensemble predictions and uncertainties (ground truth)
        """
        # Generate ensemble predictions as ground truth
        test_mean, test_unc = self.pred_ensemble(x, batch_size=512)

        # Construct evaluation targets: true labels + ensemble uncertainties
        y_ensemble = np.concatenate([y, test_unc], axis=1)

        # Get student model predictions
        pred = student_model.predict(x, batch_size=batch_size)
        
        # Evaluate how well student matches ensemble behavior
        results = eval_fun(pred, y_ensemble)
        
        # Also provide ensemble mean + uncertainty for comparison
        y_ensemble = np.concatenate([test_mean, test_unc], axis=1)

        return results, pred, y_ensemble

    def _reinitialize_model_weights(self, model):
        """
        Reinitialize all trainable weights in the model to random values.
        
        This ensures each ensemble member starts from a different point in weight space,
        promoting diversity in the ensemble which is crucial for uncertainty estimation.
        
        Args:
            model: Neural network model to reinitialize
        """
        # Iterate through all layers in the model
        for layer in model.layers:
            # Check if layer has trainable weights (Dense, Conv, etc.)
            if hasattr(layer, 'kernel_initializer'):
                # Reinitialize kernel weights (main weight matrix)
                if hasattr(layer, 'kernel'):
                    kernel_initializer = layer.kernel_initializer
                    layer.kernel.assign(kernel_initializer(shape=layer.kernel.shape))
                    
                # Reinitialize bias weights if they exist
                if hasattr(layer, 'bias') and layer.bias is not None:
                    bias_initializer = layer.bias_initializer
                    layer.bias.assign(bias_initializer(shape=layer.bias.shape))


######################################################################
# Utility functions
######################################################################

def uncertainty_logvar(x, axis=0):
    """
    Calculate uncertainty as log variance across ensemble predictions.
    
    Log variance is used instead of raw variance for numerical stability
    and because it provides a better uncertainty measure that scales well.
    
    Args:
        x: Predictions array with shape (num_ensemble, num_samples, num_outputs)
        axis: Axis along which to compute variance (0 = across ensemble members)
        
    Returns:
        Log variance of predictions across ensemble members
    """
    return np.log(np.var(x, axis=axis))

def uncertainty_std(x, axis=0):
    """
    Calculate uncertainty as standard deviation across ensemble predictions.
    
    Standard deviation provides an intuitive uncertainty measure in the same
    units as the predictions themselves.
    
    Args:
        x: Predictions array with shape (num_ensemble, num_samples, num_outputs)
        axis: Axis along which to compute standard deviation
        
    Returns:
        Standard deviation of predictions across ensemble members
    """
    return np.std(x, axis=axis)


######################################################################
# Evaluation functions
######################################################################

def eval_regression(pred, y, verbose=1):
    """
    Comprehensive evaluation metrics for regression tasks.
    
    Computes multiple regression metrics to assess model performance
    from different perspectives: error magnitude, linear correlation,
    and rank correlation.
    
    Args:
        pred: Model predictions, shape (num_samples, num_tasks)
        y: True values, shape (num_samples, num_tasks)
        verbose: If 1, print metrics for each task
        
    Returns:
        list: Performance metrics [MSE, Pearson, Spearman] per task
    """
    num_tasks = y.shape[1]  # Support multi-task regression
    results = []
    
    for i in range(num_tasks):
        # Mean Squared Error: measures prediction accuracy
        mse = metrics.mean_squared_error(y[:,i], pred[:,i])
        
        # Pearson correlation: measures linear relationship strength
        pearsonr = stats.pearsonr(y[:,i], pred[:,i])[0]
        
        # Spearman correlation: measures monotonic relationship (rank correlation)
        spearmanr = stats.spearmanr(y[:,i], pred[:,i])[0]
        
        results.append([mse, pearsonr, spearmanr])
        
        if verbose:
            print('Task %d  MSE      = %.4f'%(i, mse))
            print('Task %d  Pearson  = %.4f'%(i, pearsonr))
            print('Task %d  Spearman = %.4f'%(i, spearmanr))
            
    return results

def eval_classification(pred, y, verbose=1):
    """
    Comprehensive evaluation metrics for binary classification tasks.
    
    Provides multiple classification metrics that capture different aspects
    of model performance: ranking quality and prediction accuracy.
    
    Args:
        pred: Model predictions (probabilities), shape (num_samples, num_tasks)
        y: True binary labels, shape (num_samples, num_tasks)
        verbose: If 1, print metrics for each task
        
    Returns:
        list: Performance metrics [AUROC, AUPR, F1] per task
    """
    num_tasks = y.shape[1]  # Support multi-task classification
    results = []
    
    for i in range(num_tasks):
        # Area Under ROC Curve: measures ranking quality across all thresholds
        auroc = metrics.roc_auc_score(y[:,i], pred[:,i])
        
        # Area Under Precision-Recall Curve: better for imbalanced datasets
        aupr = metrics.average_precision_score(y[:,i], pred[:,i])
        
        # F1 Score: harmonic mean of precision and recall (needs thresholding)
        f1_score = metrics.f1_score(y[:,i], pred[:,i])
        
        results.append([auroc, aupr, f1_score])
        
        if verbose:
            print('Task %d  AUROC = %.4f'%(i, auroc))
            print('Task %d  AUPR  = %.4f'%(i, aupr))
            print('Task %d  F1    = %.4f'%(i, f1_score))
            
    return results


######################################################################
# Training functions
######################################################################


def standard_train_fun(model, x_train, y_train, validation_data, save_prefix, 
                       max_epochs=100, batch_size=100, es_patience=10, 
                       lr_decay=0.1, lr_patience=5, **kwargs):
    """
    Standard neural network training with best practices.
    
    Implements robust training with early stopping and adaptive learning rate
    scheduling to prevent overfitting and ensure convergence.
    
    Args:
        model: Neural network model to train
        x_train: Training input features
        y_train: Training target labels
        validation_data: Validation dataset (x_val, y_val) for monitoring
        save_prefix: Prefix for saving model weights (not used in this implementation)
        max_epochs: Maximum number of training epochs
        batch_size: Mini-batch size for gradient updates
        es_patience: Early stopping patience (epochs without improvement)
        lr_decay: Factor to reduce learning rate when validation plateaus
        lr_patience: Epochs to wait before reducing learning rate
        **kwargs: Additional training parameters
        
    Returns:
        Keras History object: Training history with loss and metric values
    """
    
    # Early stopping: prevent overfitting by stopping when validation loss plateaus
    es_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',           # Monitor validation loss
        patience=es_patience,         # Wait this many epochs for improvement
        verbose=1,                    # Print when stopping
        mode='min',                   # Stop when loss stops decreasing
        restore_best_weights=True     # Revert to best weights found
    )
    
    # Learning rate reduction: adapt learning rate during training
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',           # Monitor validation loss
        factor=lr_decay,              # Multiply LR by this factor when reducing
        patience=lr_patience,         # Wait this many epochs before reducing
        min_lr=1e-7,                  # Don't reduce below this value
        mode='min',                   # Reduce when loss stops decreasing
        verbose=1                     # Print when reducing
    )
    
    # Train the model with callbacks
    # Note: Model should already be compiled before calling this function
    history = model.fit(
        x_train, y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,                 # Shuffle training data each epoch
        validation_data=validation_data,
        callbacks=[es_callback, reduce_lr]
    )
    
    return history.history 



