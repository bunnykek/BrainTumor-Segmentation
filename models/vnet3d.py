from keras.models import Model
from keras.layers import Input, Conv3D, BatchNormalization, Activation, MaxPooling3D, UpSampling3D, concatenate

def conv_block(inputs, filters, kernel_size=(3, 3, 3), activation='relu', padding='same'):
    conv = Conv3D(filters, kernel_size, padding=padding)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    return conv

def vnet3d(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = conv_block(inputs, 16)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv_block(pool1, 32)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv_block(pool2, 64)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv_block(pool3, 128)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Bottleneck
    conv5 = conv_block(pool4, 256)

    # Decoder
    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    concat6 = concatenate([up6, conv4], axis=-1)
    conv6 = conv_block(concat6, 128)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    concat7 = concatenate([up7, conv3], axis=-1)
    conv7 = conv_block(concat7, 64)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    concat8 = concatenate([up8, conv2], axis=-1)
    conv8 = conv_block(concat8, 32)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    concat9 = concatenate([up9, conv1], axis=-1)
    conv9 = conv_block(concat9, 16)

    # Output
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
input_shape = (128, 128, 128, 3)  # Input shape of 3D volumes (64x64x64) with one channel (grayscale)
num_classes = 4  # Number of segmentation classes (background + 4 classes)
model = vnet3d(input_shape, num_classes)
model.summary()
