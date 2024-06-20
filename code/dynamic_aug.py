"""
Tensorflow Model class that dynamically augments training data on each batch
Target labels for augmented seqs are generated using an ensemble provided to the model 
Dynamic augmentations can replace or be added to original training data 
"""

import tensorflow as tf
import tensorflow.keras as keras
import evoaug_tf
from evoaug_tf import evoaug, augment 

# # define augmentation list to use if evoaug augmentations are selected
# augment_list = [augment.RandomDeletion(delete_min=0, delete_max=20),
#                 augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
#                 augment.RandomNoise(noise_mean=0, noise_std=0.2),
#                 augment.RandomMutation(mutate_frac=0.05)]

class DynamicAugModel(keras.Model):
    '''
    Tensorflow keras.Model class with dynamic sequence augmentations 

    To each batch of data, we add EvoAug augmented seqs
    Labels for each augmented seq are generated w/ ensemble avg. prediction

    Arguments
    -----------
    model_func : keras.Model
        function defining a keras Model
    ensemble : list
        list of keras Models comprising the ensemble 
    input_shape : tuple, list
        shape of model input (excluding batch dim)
    augment_list : list
        list of evoaug augmentations, each a callable class from evoaug_tf.augment; default=[]
    max_augs_per_seq : int
        Max number of augmentations to apply to each seq. Value is superceded by # of augs in augment_list. Default=2
    inference_aug : bool
        Flag to turn on augmentations during inference, default=False
    hard_aug : bool 
        Flag to set a hard number of augmentations, otherwise the # of augmentations is set randomly up to max_augs_per_seq, default=True
    append: bool
        Flag to determine whether augmentations replace or are added to original training seqs, default=False
    aug: str
        Type of augmentation to apply: one of random, mutagenesis, evoaug, None (default)
    '''
    def __init__(self, model_func, ensemble=None, input_shape=None, aug=None, augment_list=[],
                max_augs_per_seq=2, inference_aug=False, hard_aug=False, append=False, **kwargs):
        super(DynamicAugModel, self).__init__()
        self.model = model_func
        self.append = append
        self.ensemble = ensemble
        self.aug = aug
        self.inference_aug = inference_aug
        # make sure augment_list is consistent with aug argument
        if aug=='evoaug' and len(augment_list)==0:
            # no augment_list provided for evoaug; set default
            augment_list = [augment.RandomDeletion(delete_min=0, delete_max=20),
                            augment.RandomTranslocationBatch(shift_min=0, shift_max=20),
                            augment.RandomNoise(noise_mean=0, noise_std=0.2),
                            augment.RandomMutation(mutate_frac=0.05)]
        elif aug=='mutagenesis':
            augment_list = [augment.RandomMutation(mutate_frac=0.25)]
        elif len(augment_list)!=0:
            # random aug does not require augment list
            augment_list = []
        
        # params for evoaug/mutagenesis 
        self.augment_list = augment_list
        self.max_num_aug = len(augment_list)
        self.insert_max = augment_max_len(augment_list)
        self.max_augs_per_seq = tf.math.minimum(max_augs_per_seq, len(augment_list))
        self.hard_aug = hard_aug
        ###
        self.kwargs = kwargs
        
        if input_shape is not None:
            self.build_model(input_shape)

    def build_model(self, input_shape):
        # Add batch dimension to input shape2
        augmented_input_shape = [None] + list(input_shape)
        # Extend sequence lengths based on augment_list
        augmented_input_shape[1] += augment_max_len(self.augment_list)

        self.model = self.model(augmented_input_shape[1:], **self.kwargs)
    
    @tf.function
    def call(self, inputs, training=False):
        y_hat = self.model(inputs, training=training)
        return y_hat
    
    @tf.function
    def train_step(self, data):
        # data should be Dataset object, i.e. data from a batch
        X, y = data
        X, y = tf.cast(X, tf.float32), tf.cast(y, tf.float32)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        if self.aug:
            # generate augmentations + labels 
            X_aug, y_aug = self._aug_seqs(seqs=X, labels=y)
        else:
            # training w/o augs
            X_aug, y_aug = X, y
            if self.insert_max!=0:
                # pad seqs if insertions was used 
                X_aug = self._pad_end(X_aug)

        with tf.GradientTape() as tape:
            y_pred = self(X_aug, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compute_loss(y=y_aug, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y_aug, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, batch):
        '''
        predict and evaluate performance 
        '''
        x, y = batch
        if self.inference_aug:
            if self.aug == 'random':
                x = tf.random.shuffle(x)
            else:
                x = self._apply_augment(x)
        elif self.insert_max!=0:
            x = self._pad_end(x)

        y_pred = self(x, training=False)  
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, batch):
        '''
        make prediction with model 
        '''
        x = batch
        if self.inference_aug:
            if self._aug == 'random':
                x = tf.random.shuffle(x)
            else:
                x = self._apply_augment(x)
        elif self.insert_max != 0:
            x = self._pad_end(x)
    
        return self(x)

    def _aug_seqs(self, seqs, labels):
        '''
        generate augmentations for sequences in batch and append to batch
        '''
        assert(self.aug is not None)
        if self.append:
            # append augmented seqs to original training seqs 
            print('appending augmented seqs')
            # make copy of seqs in batch - these will be augmented
            aug_seqs = tf.identity(seqs)
            if self.aug == 'random':
                aug_seqs = tf.random.shuffle(aug_seqs)
            else:
                # evoaug/mutagenesis
                aug_seqs = self._apply_augment(aug_seqs)
            # pad original seqs in batch to match dims of aug_seqs
            if self.insert_max != 0:
                seqs = self._pad_end(seqs)
            assert(type(seqs)==type(aug_seqs))
            # concatenate original batch data w/ complementary augs
            batch_seqs = tf.concat([seqs, aug_seqs], axis=0)
        else:
            print('replacing original seqs w/ augmented seqs')
            if self.aug=='random':
                aug_seqs = tf.random.shuffle(seqs)
            else:
                aug_seqs = self._apply_augment(seqs)
            batch_seqs = aug_seqs 

        # use ensemble to generate labels for batch 
        batch_labels = self._ensemble_predict(batch_seqs)
        batch_labels = tf.cast(tf.convert_to_tensor(batch_labels), tf.float32)

        assert(batch_seqs.shape[0]==batch_labels.shape[0])
        # # use ensemble to generate labels for aug_seqs
        # aug_labels = self._ensemble_predict(aug_seqs)
        # aug_labels = tf.cast(tf.convert_to_tensor(aug_labels), tf.float32)

        # if self.append:
        #     # add labels for augmented seqs to labels for original data in batch
        #     assert labels.shape[-1]==aug_labels.shape[-1], "Error appending augmented data: shape of target labels for augmented seqs differs from shape of original target labels"
        #     # concatenate labels for original and aug seqs
        #     batch_labels = tf.concat([labels, aug_labels], axis=0)
        # else:
        #     batch_labels = aug_labels 
        return batch_seqs, batch_labels

    @tf.function
    def _pad_end(self, x):
        """Add random DNA padding of length insert_max to the end of each sequence in batch."""

        N = tf.shape(x)[0]
        L = tf.shape(x)[1]
        A = tf.cast(tf.shape(x)[2], dtype = tf.float32)
        p = tf.ones((A,)) / A
        padding = tf.transpose(tf.gather(tf.eye(A), tf.random.categorical(tf.math.log([p] * self.insert_max), N)), perm=[1,0,2])

        half = int(self.insert_max/2)
        x_padded = tf.concat([padding[:,:half,:], x, padding[:,half:,:]], axis=1)
        return x_padded
    
    @tf.function
    def _apply_augment(self, x):
        """Apply evoaug augmentations to each sequence in batch, x."""
        # number of augmentations per sequence
        if self.hard_aug:
            batch_num_aug = tf.constant(self.max_augs_per_seq, dtype=tf.int32)
        else:
            batch_num_aug = tf.random.uniform(shape=[], minval=1, maxval=self.max_augs_per_seq+1, dtype=tf.int32)

        # randomly choose which subset of augmentations from augment_list
        aug_indices = tf.sort(tf.random.shuffle(tf.range(self.max_num_aug))[:batch_num_aug])
        print('aug_indices')
        print(aug_indices)
        # apply augmentation combination to sequences
        insert_status = True
        ind = 0
        for augmentation in self.augment_list:
        # for augmentation in augment_list:
            augment_condition = tf.reduce_any(tf.equal(tf.constant(ind), aug_indices))
            if augment_condition:
                print('applying augmentation:')
                print(augmentation)
            x = tf.cond(augment_condition, lambda: augmentation(x), lambda: x)
            if augment_condition and hasattr(augmentation, 'insert_max'):
                insert_status = False
            ind += 1
        if insert_status:
            if self.insert_max != 0:
                x = self._pad_end(x)
        return x

    def _ensemble_predict(self, seqs):
        '''
        generate target labels for augmented seqs using ensemble
        '''
        all_preds = []
        model_ix = 0
        for model in self.ensemble:
            model_preds = model(seqs, training=False)
            all_preds.append(model_preds)
            model_ix += 1
        all_preds = tf.stack(all_preds)
        if self.kwargs['predict_std']:
            return tf.concat([tf.math.reduce_mean(all_preds, axis=0), tf.math.reduce_std(all_preds, axis=0)], axis=1)
        else: 
            return tf.math.reduce_mean(all_preds,axis=0)
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)


#------------------------------------------------------------------------
# Helper function
#------------------------------------------------------------------------


def augment_max_len(augment_list):
    """
    Determine whether insertions are applied to determine the insert_max,
    which will be applied to pad other sequences with random DNA.
    Parameters
    ----------
    augment_list : list
        List of augmentations.
    Returns
    -------
    int
        Value for insert max.
    """
    insert_max = 0
    for augment in augment_list:
        if hasattr(augment, 'insert_max'):
            insert_max = augment.insert_max
    return insert_max