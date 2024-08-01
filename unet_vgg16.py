import tensorflow as tf

from keras import Model
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D, Add, Activation, concatenate

def build_model(input_shape=(256, 256, 1), classes=1):
    inputs = Input(input_shape)
    # Encoder blocks

    # Encoder Block 1
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    # Encoder Block 2
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    # Encoder Block 3
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    # Encoder Block 4
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    # Encoder Block 5
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
    # Bottleneck
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(p5)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    # Decoder Block 5
    d5 = UpSampling2D((2, 2))(c6)
    d5 = concatenate([c5, d5])
    d5 = Conv2D(256, (3, 3), activation='relu', padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Activation('relu')(d5)
    d5 = Conv2D(256, (3, 3), activation='relu', padding='same')(d5)
    d5 = BatchNormalization()(d5)
    d5 = Activation('relu')(d5)
    # Decoder Block 4
    d4 = UpSampling2D((2, 2))(d5)
    d4 = concatenate([c4, d4])
    d4 = Conv2D(128, (3, 3), activation='relu', padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    d4 = Conv2D(128, (3, 3), activation='relu', padding='same')(d4)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)
    # Decoder Block 3
    d3 = UpSampling2D((2, 2))(d4)
    d3 = concatenate([c3, d3])
    d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(d3)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)
    # Decoder Block 2
    d2 = UpSampling2D((2, 2))(d3)
    d2 = concatenate([c2, d2])
    d2 = Conv2D(32, (3, 3), activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    d2 = Conv2D(32, (3, 3), activation='relu', padding='same')(d2)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)
    # Decoder Block 1
    d1 = UpSampling2D((2, 2))(d2)
    d1 = Conv2D(16, (3, 3), activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    d1 = Conv2D(16, (3, 3), activation='relu', padding='same')(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    # Final Layer
    outputs = Conv2D(classes, (3, 3), activation='relu', padding='same')(d1)
    outputs = Activation('relu')(outputs)
    # Model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
