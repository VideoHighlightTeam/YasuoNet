from tensorflow.keras.layers import Dense, Dropout, Conv3D, Conv2D, Input, MaxPool3D, MaxPool2D, Flatten, concatenate, Reshape
from tensorflow.keras.layers import TimeDistributed, LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def build_basic_model(input_shape_dict):
    video_input_shape = input_shape_dict['video']
    audio_input_shape = input_shape_dict['audio']
    weight_decay = 0.005

    # Video 3D Conv layers
    video_input = Input(video_input_shape)
    x = video_input
    x = Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    video_output = Flatten()(x)

    # Audio 2D Conv layers
    audio_input = Input(audio_input_shape)
    x = Reshape(audio_input_shape + [1])    # add channel dim
    x = Conv2D(4, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D((2, 2), strides=(2, 2), padding='same')(x)
    audio_output = Flatten()(x)

    # Fully-connected layers
    x = concatenate([video_output, audio_output])
    x = Dense(16, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    #     x = Dropout(0.2)(x)
    fc_output = Dense(1, activation='sigmoid', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=[video_input, audio_input], outputs=fc_output)

    return model


def build_sequence_model(input_shape_dict):
    video_input_shape = [None] + input_shape_dict['video']
    audio_input_shape = [None] + input_shape_dict['audio']
    weight_decay = 0.005

    # Video 3D Conv layers
    video_input = Input(video_input_shape)
    x = video_input
    x = TimeDistributed(Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))(x)
    x = TimeDistributed(MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same'))(x)
    video_output = TimeDistributed(Flatten())(x)
    print(x.shape, video_output.shape)

    # Audio 2D Conv layers
    audio_input = Input(audio_input_shape)
    x = Reshape(audio_input_shape + [1])    # add channel dim
    x = TimeDistributed(Conv2D(4, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))(x)
    x = TimeDistributed(MaxPool2D((2, 2), strides=(2, 2), padding='same'))(x)
    audio_output = TimeDistributed(Flatten())(x)

    # LSTM layers
    x = concatenate([video_output, audio_output])
    print(x.shape)
    x = LSTM(16, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    print(x.shape)

    # Fully-connected layers
    x = Dense(16, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    #     x = Dropout(0.2)(x)
    fc_output = Dense(1, activation='sigmoid', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=[video_input, audio_input], outputs=fc_output)

    return model