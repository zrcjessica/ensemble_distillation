import keras.layers as kl
from keras.models import Model

# def DeepSTARR(input_shape):
#     '''
#     DeepSTARR model, using same params as in published model
#     '''
#     # input node to model
#     inputs = kl.Input(shape=input_shape)

#     # first convolutional block 
#     x = kl.Conv1D(256, kernel_size=7, padding='same')(inputs)
#     x = kl.BatchNormalization()(x)
#     x = kl.Activation('relu')(x)
#     x = kl.MaxPool1D(2)(x)

#     # second convolutional block
#     x = kl.Conv1D(60, kernel_size=3, padding='same')(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Activation('relu')(x)
#     x = kl.MaxPool1D(2)(x)

#     # third convolutional block 
#     x = kl.Conv1D(60, kernel_size=5, padding='same')(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Activation('relu')(x)
#     x = kl.MaxPool1D(2)(x)

#     # fourth convolutional block
#     x = kl.Conv1D(120, kernel_size=3, padding='same')(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Activation('relu')(x)
#     x = kl.MaxPool1D(2)(x)

#     # flatten
#     x = kl.Flatten()(x)

#     # fully connected layers - 1
#     x = kl.Dense(256)(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Activation('relu')(x)
#     x = kl.Dropout(0.4)(x)

#     # fully connected layers - 2
#     x = kl.Dense(256)(x)
#     x = kl.BatchNormalization()(x)
#     x = kl.Activation('relu')(x)
#     x = kl.Dropout(0.4)(x)

#     # outputs: [Dev, Hk]
#     outputs = kl.Dense(2, activation='linear')(x) # why specify if default is linear? 

#     model = Model(inputs=inputs, outputs=outputs)
#     return model 

def DeepSTARR(input_shape, config):
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
        x = kl.Activation(config['first_layer_activation'])(x)
        x = kl.MaxPool1D(2)(x)

    # flatten
    x = kl.Flatten()(x)

    # add fully connected layers
    for i in range(1, (config['n_dense']+1)):
        x = kl.Dense(round(config['dense_kernels'+str(i)]*config['k']))(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation(config['activation'])(x)
        x = kl.Dropout(0.4)(x)

    # outputs: [Dev, Hk]
    outputs = kl.Dense(2, activation='linear')(x) # why specify if default is linear? 

    model = Model(inputs=inputs, outputs=outputs)
    return model 