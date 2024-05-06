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

def lentiMPRA(input_shape, config, predict_std=False):
    '''
    CNN for predicting lentiMPRA data
    '''

    def residual_block(input_layer, filter_size, activation='relu', dilated=5):
        '''
        define residual block for CNN
        '''
        factor = []
        base = 2
        for i in range(dilated):
            factor.append(base**i)
        num_filters = input_layer.shape.as_list()[-1]

        nn = kl.Conv1D(filters=num_filters,
                                        kernel_size=filter_size,
                                        activation=None,
                                        use_bias=False,
                                        padding='same',
                                        # dilation_rate=1, #commenting out bc default
                                        )(input_layer)
        nn = kl.BatchNormalization()(nn)
        for f in factor:
            nn = kl.Activation('relu')(nn)
            nn = kl.Dropout(0.1)(nn)
            nn = kl.Conv1D(filters=num_filters,
                                            kernel_size=filter_size,
                                            # strides=1, # commenting out bc default
                                            activation=None,
                                            use_bias=False,
                                            padding='same',
                                            dilation_rate=f,
                                            )(nn)
            nn = kl.BatchNormalization()(nn)
        nn = kl.add([input_layer, nn])
        return kl.Activation(activation)(nn)

    inputs = kl.Input(shape=input_shape)
    x = kl.Conv1D(196, kernel_size=19, padding='same')(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.2)(x)
    x = residual_block(x, 3, activation=config['dilation_activation'], dilated=config['n_dilations'])
    x = kl.Dropout(0.2)(x)
    x = kl.MaxPooling1D(5)(x) # 55

    x = kl.Conv1D(256, kernel_size=7, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.2)(x)
    x = kl.MaxPooling1D(5)(x) # 10

    #   x = kl.Conv1D(256, kernel_size=5, padding='same')(x)
    #   x = kl.BatchNormalization()(x)
    #   x = kl.Activation('relu')(x)
    #   x = kl.Dropout(0.2)(x)
    #   x = kl.MaxPooling1D(3)(x) # 10

    x = kl.Dense(256)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.5)(x)

    x = kl.GlobalAveragePooling1D()(x)
    x = kl.Flatten()(x)

    x = kl.Dense(256)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('silu')(x)
    x = kl.Dropout(0.5)(x)

    outputs = kl.Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model 