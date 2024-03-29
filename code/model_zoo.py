import keras.layers as kl
from keras.models import Model

def DeepSTARR(input_shape, config, predict_std=False):
    '''
    DeepSTARR model, using same params as in published model
    '''
    # input node to model
    inputs = kl.Input(shape=input_shape)

    x = 0
    # add convolutional blocks
    for i in range(1, (config['n_conv']+1)):
        if (i == 1):
            x = kl.Conv1D(round(config['n_kernels'+str(i)]*config['k']), kernel_size=config['kernel_size'+str(i)], padding='same')(inputs)
        else:
            x = kl.Conv1D(round(config['n_kernels'+str(i)]*config['k']), kernel_size=config['kernel_size'+str(i)], padding='same')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation(config['first_activation'])(x)
        x = kl.MaxPool1D(2)(x)

    # flatten
    x = kl.Flatten()(x)

    # add fully connected layers
    for i in range(1, (config['n_dense']+1)):
        x = kl.Dense(round(config['dense_kernels'+str(i)]*config['k']))(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation(config['activation'])(x)
        x = kl.Dropout(0.4)(x)

    if predict_std:
        # outputs: [Dev-mean, Hk-mean, Dev-std, Hk-std]
        outputs = kl.Dense(4, activation='linear')(x)
    else:
        # outputs: [Dev, Hk]
        outputs = kl.Dense(2, activation='linear')(x) # why specify if default is linear? 

    model = Model(inputs=inputs, outputs=outputs)
    return model 