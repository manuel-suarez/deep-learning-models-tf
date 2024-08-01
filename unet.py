import tensorflow as tf

from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, concatenate

def build_model(input_shape=(256, 256, 1), classes=1):
    inputs = Input(input_shape)
    # Encoder blocks - Contraction path
    # Encoder Block 1
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    # Encoder Block 2
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    # Encoder Block 3
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    # Encoder Block 4
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    # Bottleneck
    bt = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    bt = Dropout(0.3)(bt)
    bt = Conv2D(256, (3, 3), activation='relu', padding='same')(bt)
    # Decoder blocks - Expansive path
    # Decoder Block 1
    u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bt)
    u4 = concatenate([u4, c4])
    u4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    u4 = Dropout(0.2)(u4)
    u4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    # Decoder Block 2
    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u4)
    u3 = concatenate([u3, c3])
    u3 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    u3 = Dropout(0.2)(u3)
    u3 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    # Decoder Block 3
    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u3)
    u2 = concatenate([u2, c2])
    u2 = Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    u2 = Dropout(0.1)(u2)
    u2 = Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    # Decoder Block 4
    u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u2)
    u1 = concatenate([u1, c1])
    u1 = Conv2D(16, (3, 3), activation='relu', padding='same')(u1)
    u1 = Dropout(0.1)(u1)
    u1 = Conv2D(16, (3, 3), activation='relu', padding='same')(u1)
    # Final layer
    if classes == 1:
        activation = 'sigmoid' 
    else:
        activation = 'softmax'
    outputs = Conv2D(classes, (1, 1), activation=activation)(u1)
    # Model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
    
