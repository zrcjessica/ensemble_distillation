import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from degu import DEGU, uncertainty_logvar, standard_train_fun, eval_regression


def DeepSTARR(input_shape, output_shape):
    "DeepSTARR model from deAlmeida et al. Nat Genetics (2022)"
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(256, kernel_size=7, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(60, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(60, kernel_size=5, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(120, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(output_shape, activation='linear')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# load dataset
filepath = 'deepstarr_data.h5'
dataset = h5py.File(filepath, 'r')
x_train = np.array(dataset['x_train']).astype(np.float32)
y_train = np.array(dataset['y_train']).astype(np.float32).transpose()
x_valid = np.array(dataset['x_valid']).astype(np.float32)
y_valid = np.array(dataset['y_valid']).astype(np.float32).transpose()
x_test = np.array(dataset['x_test']).astype(np.float32)
y_test = np.array(dataset['y_test']).astype(np.float32).transpose()

# get shapes
N, L, A = x_train.shape
num_targets = y_train.shape[1]

# instantiate DEGU
deepstarr_standard = DeepSTARR(input_shape=(L,A), output_shape=num_targets)
degu = DEGU(deepstarr_standard, num_ensemble=10, uncertainty_fun=uncertainty_logvar)

# train ensemble of DeepSTARR models
optimizer = keras.optimizers.Adam(learning_rate=0.002)
loss='mse'
history = degu.train_ensemble(x_train, y_train,
                              train_fun=standard_train_fun,
                              save_prefix='deepstarr_standard',
                              optimizer=optimizer,
                              loss=loss,
                              validation_data=(x_valid, y_valid))

# evaluate ensemble models
results = degu.eval_ensemble(x_test, y_test, eval_fun=eval_regression)
ensemble_results, standard_results, train_ensemble_mean, train_ensemble_uncertainty = results

# knowledge distillation of ensemble to a single student model
student_model = DeepSTARR(input_shape=(L,A), output_shape=num_targets*2)
student_model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse')
history = degu.distill_student(student_model, x_train, y_train,
                               train_fun=standard_train_fun,
                               save_prefix='deepstarr_distilled',
                               validation_data=(x_valid, y_valid),
                               batch_size=128)

# evaluate student model
student_results, student_pred, y_test_ensemble = degu.eval_student(student_model, x_test, y_test, eval_fun=eval_regression)


