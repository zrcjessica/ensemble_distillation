import keras.layers as kl
from keras.models import Model

def DeepSTARR(input_shape, config, epistemic=False):
    '''
    Build the DeepSTARR model architecture.

    Parameters:
        input_shape (tuple): Shape of the input data.
        config (dict): Configuration dictionary with model hyperparameters, including 
                       the number of convolutional and dense layers, kernel sizes, 
                       and activation functions.
        epistemic (bool): If True, enables epistemic uncertainty prediction.

    Returns:
        Model: Compiled Keras model for DeepSTARR.
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

    if epistemic:
        # outputs: [Dev-mean, Hk-mean, Dev-std, Hk-std]
        outputs = kl.Dense(4, activation='linear')(x)
    elif config['loss_fxn']=='evidential':
        import evidential_deep_learning as edl
        # outputs: [Dev, Hk, Dev-uncertainty, Hk-uncertainty]
        outputs = edl.layers.DenseNormal(2)(x)
    else:
        # outputs: [Dev, Hk]
        outputs = kl.Dense(2, activation='linear')(x) 

    model = Model(inputs=inputs, outputs=outputs)
    return model 

def DeepSTARR_heteroscedastic(input_shape, config):
    '''
    Build the DeepSTARR model with heteroscedastic regression.

    Parameters:
        input_shape (tuple): Shape of the input data.
        config (dict): Configuration dictionary with model hyperparameters.

    Returns:
        Model: Compiled Keras model with heteroscedastic regression.
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

    # output 
    mu = kl.Dense(2, activation='linear', name='mu')(x)
    std = kl.Dense(2, activation='softplus', name='std')(x)
    outputs = kl.Concatenate()([mu, std]) # [Dev-mu, Hk-mu, Dev-std, Hk-std]

    model = Model(inputs=inputs, outputs=outputs)
    return model 

def lentiMPRA(input_shape, config, aleatoric=False, epistemic=False):
    '''
    Wrapper for calling ResidualBind, compatible with earlier script conventions.

    Parameters:
        input_shape (tuple): Shape of the input data.
        config (dict): Configuration dictionary.
        aleatoric (bool): If True, predict aleatoric uncertainty.
        epistemic (bool): If True, predict epistemic uncertainty.

    Returns:
        Model: ResidualBind model for lentiMPRA data.
    '''
    return ResidualBind(input_shape, config, aleatoric, epistemic)

def ResidualBind(input_shape, config, aleatoric=False, epistemic=False):
    '''
    Build the ResidualBind model for predicting lentiMPRA data.

    Parameters:
        input_shape (tuple): Shape of the input data.
        config (dict): Configuration dictionary with model hyperparameters.
        aleatoric (bool): If True, predict aleatoric uncertainty.
        epistemic (bool): If True, predict epistemic uncertainty.

    Returns:
        Model: Compiled Keras model.
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
    
    if aleatoric and epistemic:
        # outputs: [mean, aleatoric std, epistemic std]
        outputs = kl.Dense(3, activation='linear')(x)
    elif aleatoric or epistemic:
        # outputs: [mean, std]
        outputs = kl.Dense(2, activation='linear')(x)
    else:
        if config['loss_fxn']=='evidential':
            import evidential_deep_learning as edl
            # outputs: [mean, uncertainty]
            outputs = edl.layers.DenseNormal(1)(x)
        else:
            # outputs: [mean]
            outputs = kl.Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model 

def ResidualBind_heteroscedastic(input_shape, config):
    '''
    Build the ResidualBind model with heteroscedastic regression.

    Parameters:
        input_shape (tuple): Shape of the input data.
        config (dict): Configuration dictionary.

    Returns:
        Model: Compiled Keras model with heteroscedastic regression.
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
    
    # output 
    # mu = kl.Dense(1, activation='linear', name='mu')(x)
    # std = kl.Dense(1, activation='softplus', name='std')(x)
    # outputs = kl.Concatenate()([mu, std]) # [mu, std]
    outputs = kl.Dense(2, activation='linear')(x) # [mu, logvar]

    model = Model(inputs=inputs, outputs=outputs)
    return model 

def lentiMPRA_v2(input_shape, config, aleatoric=False, epistemic=False):
    '''
    ResidualBind model for predicting lentiMPRA data
    if aleatoric=True, predict aleatoric uncertainty
    if epistemic=True, predict epistemic uncertainty 
    uses separate non-linear output heads instead of a simple Dense layer 
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

    activity_output = kl.Dense(32, activation='silu')(x)
    activity_output = kl.Dropout(0.5)(activity_output)
    activity_output = kl.Dense(1, activation='linear')(activity_output)
    if aleatoric and epistemic:
        aleatoric_output = kl.Dense(32, activation='silu')(x)
        aleatoric_output = kl.Dropout(0.5)(aleatoric_output)
        aleatoric_output = kl.Dense(1, activation='linear')(aleatoric_output)
        epistemic_output = kl.Dense(32, activation='silu')(x)
        epistemic_output = kl.Dropout(0.5)(epistemic_output)
        epistemic_output = kl.Dense(1, activation='linear')(epistemic_output)
        outputs = kl.Concatenate(axis=-1)([activity_output, 
                                           aleatoric_output, 
                                           uncertainty_output])
    elif aleatoric or epistemic:
        uncertainty_output = kl.Dense(32, activation='silu')(x)
        uncertainty_output = kl.Dropout(0.5)(uncertainty_output)
        uncertainty_output = kl.Dense(1, activation='linear')(uncertainty_output)
        outputs = kl.Concatenate(axis=-1)([activity_output, uncertainty_output])
    else:
        outputs = activity_output

    model = Model(inputs=inputs, outputs=outputs)
    return model 


# def MPRAnn(input_shape,output_shape=1,**kwargs):
def MPRAnn(input_shape, aleatoric=False, epistemic=False):
    inputs = kl.Input(shape=(input_shape[0], input_shape[1]), name="input")
    layer = kl.Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs)  # 250 7 relu
    layer = kl.BatchNormalization()(layer)
    layer = kl.Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer)  # 250 8 softmax
    layer = kl.BatchNormalization()(layer)
    layer = kl.MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = kl.Dropout(0.1)(layer)
    layer = kl.Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer)  # 250 3 softmax
    layer = kl.BatchNormalization()(layer)
    layer = kl.Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer)  # 100 3 softmax
    layer = kl.BatchNormalization()(layer)
    layer = kl.MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = kl.Dropout(0.1)(layer)
    layer = kl.Flatten()(layer)
    layer = kl.Dense(300, activation='sigmoid')(layer)  # 300
    layer = kl.Dropout(0.3)(layer)
    layer = kl.Dense(200, activation='sigmoid')(layer)  # 300
    if epistemic and aleatoric:
        predictions = kl.Dense(3, activation='linear')(layer)
    elif epistemic or aleatoric:
        predictions = kl.Dense(2, activation='linear')(layer)
    else:
        predictions = kl.Dense(1, activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return model


def MPRAnn_heteroscedastic(input_shape):
    '''
    MPRAnn with heteroscedastic regression
    '''
    inputs = kl.Input(shape=(input_shape[0], input_shape[1]), name="input")
    layer = kl.Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs)  # 250 7 relu
    layer = kl.BatchNormalization()(layer)
    layer = kl.Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer)  # 250 8 softmax
    layer = kl.BatchNormalization()(layer)
    layer = kl.MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = kl.Dropout(0.1)(layer)
    layer = kl.Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer)  # 250 3 softmax
    layer = kl.BatchNormalization()(layer)
    layer = kl.Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer)  # 100 3 softmax
    layer = kl.BatchNormalization()(layer)
    layer = kl.MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = kl.Dropout(0.1)(layer)
    layer = kl.Flatten()(layer)
    layer = kl.Dense(300, activation='sigmoid')(layer)  # 300
    layer = kl.Dropout(0.3)(layer)
    layer = kl.Dense(200, activation='sigmoid')(layer)  # 300

    # output 
    mu = kl.Dense(1, activation='linear', name='mu')(layer)
    std = kl.Dense(1, activation='softplus', name='std')(layer)
    # var = kl.Dense(1, activation='softplus', name='var')(layer)
    outputs = kl.Concatenate()([mu, std]) # [mu, std]
    # outputs = kl.Concatenate()([mu, var]) # [mu, var]
    model = Model(inputs=inputs, outputs=outputs)
    return model