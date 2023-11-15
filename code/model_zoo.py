import keras.layers as kl
from keras.models import Model

def DeepSTARR(input_shape):
    '''
    DeepSTARR model, using same params as in published model
    '''
    # input node to model
    inputs = kl.Input(shape=input_shape)

    # first convolutional block 
    x = kl.Conv1D(256, kernel_size=7, padding='same')(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPool1D(2)(x)

    # second convolutional block
    x = kl.Conv1D(60, kernel_size=3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPool1D(2)(x)

    # third convolutional block 
    x = kl.Conv1D(60, kernel_size=5, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPool1D(2)(x)

    # fourth convolutional block
    x = kl.Conv1D(120, kernel_size=3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPool1D(2)(x)

    # flatten
    x = kl.Flatten()(x)

    # fully connected layers - 1
    x = kl.Dense(256)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(0.4)(x)

    # fully connected layers - 2
    x = kl.Dense(256)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(0.4)(x)

    # outputs: [Dev, Hk]
    outputs = kl.Dense(2, activation='linear')(x) # why specify if default is linear? 

    model = Model(inputs=inputs, outputs=outputs)
    return model 