import tensorflow as tf

from keras import Model
from keras.layers import (
    Input,
    Conv2D,
    ZeroPadding2D,
    BatchNormalization,
    UpSampling2D,
    MaxPooling2D,
    Add,
    Activation,
    add,
    concatenate,
)


def build_model(input_shape=(256, 256, 1), classes=1):
    inputs = Input(input_shape)                                         # 256, 256, 1 
    # Encoder blocks

    # Encoder Block 1
    c1 = BatchNormalization()(inputs)                                   # 256, 256, 1   -> 256, 256, 1
    c1 = ZeroPadding2D((3, 3))(c1)                                      # 256, 256, 1   -> 262, 262, 1
    c1 = Conv2D(64, (7, 7), strides=(2, 2), activation="linear")(c1)    # 262, 262, 1   -> 128, 128, 64
    c1 = BatchNormalization()(c1)                                       # 128, 128, 64  -> 128, 128, 64
    c1 = Activation("relu")(c1)                                         # 128, 128, 64  -> 128, 128, 64, SKIP CONNECTION
    p1 = ZeroPadding2D()(c1)                                            # 128, 128, 64  -> 130, 130, 64
    p1 = MaxPooling2D((3, 3), strides=(2, 2))(p1)                       # 130, 130, 64  -> 64, 64, 64
    # Encoder Block 2
    c2 = BatchNormalization()(p1)                                       # 64, 64, 64    -> 64, 64, 64
    c2res = Activation('relu')(c2)                                      # 64, 64, 64    -> 64, 64, 64, res
    c2 = ZeroPadding2D()(c2res)                                         # 64, 64, 64    -> 66, 66, 64
    c2 = Conv2D(64, (3, 3), activation="linear")(c2)                    # 66, 66, 64    -> 64, 64, 64
    c2res = Conv2D(64, (3, 3), activation='linear', 
                   padding='same')(c2res)                               # 64, 64, 64    -> 64, 64, 64
    c2 = BatchNormalization()(c2)                                       # 64, 64, 64    -> 64, 64, 64
    c2 = Activation('relu')(c2)                                         # 64, 64, 64    -> 64, 64, 64
    c2 = ZeroPadding2D()(c2)                                            # 64, 64, 64    -> 66, 66, 64
    c2 = Conv2D(64, (3, 3), activation='linear')(c2)                    # 66, 66, 64    -> 64, 64, 64
    # Residual Connection 1
    c2res = add([c2, c2res])                                            # [64, 64, 64], [64, 64, 64] -> 64, 64, 64
    c2 = BatchNormalization()(c2res)                                    # 64, 64, 64    -> 64, 64, 64
    c2 = Activation('relu')(c2)                                         # 64, 64, 64    -> 64, 64, 64
    c2 = ZeroPadding2D()(c2)                                            # 64, 64, 64    -> 66, 66, 64
    c2 = Conv2D(64, (3, 3), activation='linear')(c2)                    # 66, 66, 64    -> 64, 64, 64
    c2 = BatchNormalization()(c2)                                       # 64, 64, 64    -> 64, 64, 64
    c2 = Activation('relu')(c2)                                         # 64, 64, 64    -> 64, 64, 64
    c2 = ZeroPadding2D()(c2)                                            # 64, 64, 64    -> 66, 66, 64
    c2 = Conv2D(64, (3, 3), activation='linear')(c2)                    # 66, 66, 64    -> 64, 64, 64
    # Residual Connection 2
    c2res = add([c2, c2res])                                            # [64, 64, 64], [64, 64, 64] -> 64, 64, 64
    c2 = BatchNormalization()(c2res)                                    # 64, 64, 64    -> 64, 64, 64
    c2 = Activation('relu')(c2)                                         # 64, 64, 64    -> 64, 64, 64
    c2 = ZeroPadding2D()(c2)                                              # 64, 64, 64    -> 66, 66, 64
    c2 = Conv2D(64, (3, 3), activation='linear')(c2)                    # 66, 66, 64    -> 64, 64, 64
    c2 = BatchNormalization()(c2)                                       # 64, 64, 64    -> 64, 64, 64
    c2 = Activation('relu')(c2)                                         # 64, 64, 64    -> 64, 64, 64
    c2 = ZeroPadding2D()(c2)                                            # 64, 64, 64    -> 66, 66, 64
    c2 = Conv2D(64, (3, 3), activation='linear')(c2)                    # 66, 66, 64    -> 64, 64, 64
    # Residual Connection 3
    c2 = add([c2, c2res])                                               # [64, 64, 64], [64, 64, 64] -> [64, 64, 64]
    c2 = BatchNormalization()(c2)                                       # 64, 64, 64    -> 64, 64, 64
    c2 = Activation('relu')(c2)                                         # 64, 64, 64    -> 64, 64, 64, SKIP CONNECTION
    # Encoder Block 3
    p2 = ZeroPadding2D()(c2)                                            # 64, 64, 64    -> 66, 66, 64
    p2 = Conv2D(128, (3, 3), strides=(2, 2), activation='linear')(p2)   # 66, 66, 64    -> 32, 32, 128
    c3res = Conv2D(128, (3, 3), strides=(2, 2), activation='linear', 
                 padding='same')(c2)                                    # 64, 64, 64    -> 32, 32, 128
    c3 = BatchNormalization()(p2)                                       # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128
    c3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    c3 = Conv2D(128, (3, 3), activation='linear')(c3)                   # 34, 34, 128   -> 32, 32, 128
    # Residual Connection 1
    c3res = add([c3, c3res])                                            # [32, 32, 128], [32, 32, 128] -> 32, 32, 128
    c3 = BatchNormalization()(c3res)                                    # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128
    c3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    c3 = Conv2D(128, (3, 3), activation='linear')(c3)                   # 34, 34, 128   -> 32, 32, 128
    c3 = BatchNormalization()(c3)                                       # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128
    c3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    c3 = Conv2D(128, (3, 3), activation='linear')(c3)                   # 34, 34, 128   -> 32, 32, 128
    # Residual Connection 2
    c3res = add([c3, c3res])                                            # [32, 32, 128], [32, 32, 128] -> 32, 32, 128
    c3 = BatchNormalization()(c3res)                                    # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128, SKIP CONNECTION
    c3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    c3 = Conv2D(128, (3, 3), activation='linear')(c3)                   # 34, 34, 128   -> 32, 32, 128
    c3 = BatchNormalization()(c3)                                       # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128
    c3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    c3 = Conv2D(128, (3, 3), activation='linear')(c3)                   # 34, 34, 128   -> 32, 32, 128
    # Residual Connection 3
    c3res = add([c3, c3res])                                            # [32, 32, 128], [32, 32, 128] -> 32, 32, 128
    c3 = BatchNormalization()(c3res)                                    # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128
    c3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    c3 = Conv2D(128, (3, 3), activation='linear')(c3)                   # 34, 34, 128   -> 32, 32, 128
    c3 = BatchNormalization()(c3)                                       # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128
    c3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    c3 = Conv2D(128, (3, 3), activation='linear')(c3)                   # 34, 34, 128   -> 32, 32, 128
    # Residual Connection 4
    c3res = add([c3, c3res])                                            # [32, 32, 128], [32, 32, 128] -> 32, 32, 128
    c3 = BatchNormalization()(c3res)                                    # 32, 32, 128   -> 32, 32, 128
    c3 = Activation('relu')(c3)                                         # 32, 32, 128   -> 32, 32, 128
    # Encoder Block 4
    p3 = ZeroPadding2D()(c3)                                            # 32, 32, 128   -> 34, 34, 128
    p3 = Conv2D(256, (3, 3), strides=(2, 2), activation='linear')(p3)   # 34, 34, 128   -> 16, 16, 256
    c4res = Conv2D(256, (3, 3), strides=(2, 2), activation='linear',      
                 padding='same')(c3)                                    # 32, 32, 128   -> 16, 16, 256, res
    c4 = BatchNormalization()(p3)                                       # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    # Residual Connection 1
    c4res = add([c4, c4res])                                            # [16, 16, 256], [16, 16, 256] -> 16, 16, 256
    c4 = BatchNormalization()(c4res)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    c4 = BatchNormalization()(c4)                                       # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    # Residual Connection 2
    c4res = add([c4, c4res])                                            # [16, 16, 256], [16, 16, 256] -> 16, 16, 256
    c4 = BatchNormalization()(c4res)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                              # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    c4 = BatchNormalization()(c4)                                       # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                              # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    # Residual Connection 3 
    c4res = add([c4, c4res])                                            # [16, 16, 256], [16, 16, 256] -> 16, 16, 256
    c4 = BatchNormalization()(c4res)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    c4 = BatchNormalization()(c4)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    # Residual Connection 4
    c4res = add([c4, c4res])                                            # [16, 16, 256], [16, 16, 256] -> 16, 16, 256
    c4 = BatchNormalization()(c4res)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    c4 = BatchNormalization()(c4)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    # Residual Connection 5
    c4res = add([c4, c4res])                                            # [16, 16, 256], [16, 16, 256] -> 16, 16, 256
    c4 = BatchNormalization()(c4res)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    c4 = BatchNormalization()(c4)                                       # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256
    c4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    c4 = Conv2D(256, (3, 3), activation='linear')(c4)                   # 18, 18, 256   -> 16, 16, 256
    # Residual Connection 6
    c4res = add([c4, c4res])                                            # [16, 16, 256], [16, 16, 256] -> 16, 16, 256
    c4 = BatchNormalization()(c4res)                                    # 16, 16, 256   -> 16, 16, 256
    c4 = Activation('relu')(c4)                                         # 16, 16, 256   -> 16, 16, 256, SKIP CONNECTION
    # Encoder Block 5 (Bottleneck)
    p4 = ZeroPadding2D()(c4)                                            # 16, 16, 256   -> 18, 18, 256
    p4 = Conv2D(512, (3, 3), strides=(2, 2), activation='linear')(p4)   # 18, 18, 256   -> 8, 8, 512
    c5res = Conv2D(512, (3, 3), strides=(2, 2), activation='linear', 
                 padding='same')(c4)                                    # 16, 16, 256   -> 8, 8, 512, res
    c5 = BatchNormalization()(p4)                                       # 8, 8, 512     -> 8, 8, 512
    c5 = Activation('relu')(c5)                                         # 8, 8, 512     -> 8, 8, 512
    c5 = ZeroPadding2D()(c5)                                            # 8, 8, 512     -> 10, 10, 512
    c5 = Conv2D(512, (3, 3), activation='linear')(c5)                   # 10, 10, 512   -> 8, 8, 512
    # Residual Connection 1
    c5res = add([c5, c5res])                                            # [8, 8, 512], [8, 8, 512] -> 8, 8, 512
    c5 = BatchNormalization()(c5res)                                    # 8, 8, 512     -> 8, 8, 512
    c5 = Activation('relu')(c5)                                         # 8, 8, 512     -> 8, 8, 512
    c5 = ZeroPadding2D()(c5)                                            # 8, 8, 512     -> 10, 10, 512
    c5 = Conv2D(512, (3, 3), activation='linear')(c5)                   # 10, 10, 512   -> 8, 8, 512
    c5 = BatchNormalization()(c5)                                       # 8, 8, 512     -> 8, 8, 512
    c5 = Activation('relu')(c5)                                         # 8, 8, 512     -> 8, 8, 512
    c5 = ZeroPadding2D()(c5)                                            # 8, 8, 512     -> 10, 10, 512
    c5 = Conv2D(512, (3, 3), activation='linear')(c5)                   # 10, 10, 512   -> 8, 8, 512
    # Residual Connection 2
    c5res = add([c5, c5res])                                            # [8, 8, 512], [8, 8, 512] -> 8, 8, 512
    c5 = BatchNormalization()(c5res)                                    # 8, 8, 512     -> 8, 8, 512
    c5 = Activation('relu')(c5)                                        # 8, 8, 512     -> 8, 8, 512
    c5 = ZeroPadding2D()(c5)                                            # 8, 8, 512     -> 10, 10, 512
    c5 = Conv2D(512, (3, 3), activation='linear')(c5)                   # 10, 10, 512   -> 8, 8, 512
    c5 = BatchNormalization()(c5)                                       # 8, 8, 512     -> 8, 8, 512
    c5 = Activation('relu')(c5)                                         # 8, 8, 512     -> 8, 8, 512
    c5 = ZeroPadding2D()(c5)                                            # 8, 8, 512     -> 10, 10, 512
    c5 = Conv2D(512, (3, 3), activation='linear')(c5)                   # 10, 10, 512   -> 8, 8, 512
    # Residual Connection 3
    c5res = add([c5, c5res])                                            # [8, 8, 512], [8, 8, 512] -> 8, 8, 512
    c5 = BatchNormalization()(c5res)                                    # 8, 8, 512     -> 8, 8, 512
    c5 = Activation('relu')(c5)                                         # 8, 8, 512     -> 8, 8, 512
    # Decoder Block 4
    d4 = UpSampling2D()(c5)                                             # 8, 8, 512     -> 16, 16, 512
    d4 = concatenate([c4, d4])                                          # [16, 16, 512], [16, 16, 512] -> 16, 16, 768
    d4 = Conv2D(256, (3, 3), activation='linear', padding='same')(d4)   # 16, 16, 768   -> 16, 16, 256
    d4 = BatchNormalization()(d4)                                       # 16, 16, 256   -> 16, 16, 256
    d4 = Activation('relu')(d4)                                         # 16, 16, 256   -> 16, 16, 256
    d4 = Conv2D(256, (3, 3), activation='linear', padding='same')(d4)   # 16, 16, 256   -> 16, 16, 256
    d4 = BatchNormalization()(d4)                                       # 16, 16, 256   -> 16, 16, 256
    d4 = Activation('relu')(d4)                                         # 16, 16, 256   -> 16, 16, 256
    # Decoder Block 3
    d3 = UpSampling2D()(d4)                                             # 16, 16, 256   -> 32, 32, 256
    d3 = concatenate([d3, c3])                                          # [32, 32, 256], [32, 32, 128] -> 32, 32, 384
    d3 = Conv2D(128, (3, 3), activation='linear', padding='same')(d3)   # 32, 32, 384   -> 32, 32, 128
    d3 = BatchNormalization()(d3)                                       # 32, 32, 128   -> 32, 32, 128
    d3 = Activation('relu')(d3)                                         # 32, 32, 128   -> 32, 32, 128
    d3 = Conv2D(128, (3, 3), activation='linear', padding='same')(d3)   # 32, 32, 128   -> 32, 32, 128
    d3 = BatchNormalization()(d3)                                       # 32, 32, 128   -> 32, 32, 128
    d3 = Activation('relu')(d3)                                         # 32, 32, 128   -> 32, 32, 128
    # Decoder Block 2
    d2 = UpSampling2D()(d3)                                             # 32, 32, 128   -> 64, 64, 128
    d2 = concatenate([d2, c2])                                          # [64, 64, 128], [64, 64, 64] -> 64, 64, 192
    d2 = Conv2D(64, (3, 3), activation='linear', padding='same')(d2)    # 64, 64, 192   -> 64, 64, 64
    d2 = BatchNormalization()(d2)                                       # 64, 64, 64    -> 64, 64, 64
    d2 = Activation('relu')(d2)                                         # 64, 64, 64    -> 64, 64, 64
    d2 = Conv2D(64, (3, 3), activation='linear', padding='same')(d2)    # 64, 64, 64    -> 64, 64, 64
    d2 = BatchNormalization()(d2)                                       # 64, 64, 64    -> 64, 64, 64
    d2 = Activation('relu')(d2)                                         # 64, 64, 64    -> 64, 64, 64
    # Decoder Block 1
    d1 = UpSampling2D()(d2)                                             # 64, 64, 64    -> 128, 128, 64
    d1 = concatenate([d1, c1])                                          # [128, 128, 64], [128, 128, 64] -> 128, 128, 128
    d1 = Conv2D(32, (3, 3), activation='linear', padding='same')(d1)    # 128, 128, 128 -> 128, 128, 32
    d1 = BatchNormalization()(d1)                                       # 128, 128, 32  -> 128, 128, 32
    d1 = Activation('relu')(d1)                                         # 128, 128, 32  -> 128, 128, 32
    d1 = Conv2D(32, (3, 3), activation='linear', padding='same')(d1)    # 128, 128, 32  -> 128, 128, 32
    d1 = BatchNormalization()(d1)                                       # 128, 128, 32  -> 128, 128, 32
    d1 = Activation('relu')(d1)                                         # 128, 128, 32  -> 128, 128, 32
    # Final Block
    d0 = UpSampling2D()(d1)                                             # 128, 128, 32  -> 256, 256, 32
    d0 = Conv2D(16, (3, 3), activation='linear', padding='same')(d0)    # 256, 256, 32  -> 256, 256, 16
    d0 = BatchNormalization()(d0)                                       # 256, 256, 16  -> 256, 256, 16
    d0 = Activation('relu')(d0)                                         # 256, 256, 16  -> 256, 256, 16
    d0 = Conv2D(16, (3, 3), activation='linear', padding='same')(d0)    # 256, 256, 16  -> 256, 256, 16
    d0 = BatchNormalization()(d0)                                       # 256, 256, 16  -> 256, 256, 16
    d0 = Activation('relu')(d0)                                         # 256, 256, 16  -> 256, 256, 16
    # Final Layer
    outputs = Conv2D(classes, (3, 3), activation="linear", 
                     padding="same")(d0)                                # 256, 256, 16  -> 256, 256, classes
    if classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = Activation(activation)(outputs)                           # 256, 256, classes -> 256, 256, classes
    # Model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
