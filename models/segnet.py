from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate

def conv_block(x, filters, kernel_size, padding='same'):
    x = Conv3D(filters, kernel_size, padding=padding, kernel_initializer='he_uniform', activation='relu')(x)
    return x

def SegNet_3D(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Contracting Path
    conv1 = conv_block(inputs, 16, (3, 3, 3))
    conv2 = conv_block(conv1, 16, (3, 3, 3))
    pool1 = MaxPooling3D((2, 2, 2))(conv2)

    conv3 = conv_block(pool1, 32, (3, 3, 3))
    conv4 = conv_block(conv3, 32, (3, 3, 3))
    pool2 = MaxPooling3D((2, 2, 2))(conv4)

    conv5 = conv_block(pool2, 64, (3, 3, 3))
    conv6 = conv_block(conv5, 64, (3, 3, 3))
    conv7 = conv_block(conv6, 64, (3, 3, 3))
    pool3 = MaxPooling3D((2, 2, 2))(conv7)

    # Bottleneck
    conv8 = conv_block(pool3, 128, (3, 3, 3))
    conv9 = conv_block(conv8, 128, (3, 3, 3))
    conv10 = conv_block(conv9, 128, (3, 3, 3))
    pool4 = MaxPooling3D((2, 2, 2))(conv10)

    # Expansive Path
    up1 = UpSampling3D((2, 2, 2))(pool4)
    up1 = conv_block(up1, 128, (3, 3, 3))
    up1 = conv_block(up1, 128, (3, 3, 3))
    up1 = conv_block(up1, 128, (3, 3, 3))

    up2 = UpSampling3D((2, 2, 2))(up1)
    up2 = conv_block(up2, 64, (3, 3, 3))
    up2 = conv_block(up2, 64, (3, 3, 3))

    up3 = UpSampling3D((2, 2, 2))(up2)
    up3 = conv_block(up3, 32, (3, 3, 3))
    up3 = conv_block(up3, 32, (3, 3, 3))

    up4 = UpSampling3D((2, 2, 2))(up3)
    up4 = conv_block(up4, 16, (3, 3, 3))
    up4 = conv_block(up4, 16, (3, 3, 3))

    # Output Layer
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(up4)

    model = Model(inputs=inputs, outputs=outputs, name='SegNet_3D')

    return model

# Example usage:
input_shape = (32, 32, 32, 1)  # Replace with your input image size and channel count
num_classes = 2  # Replace with the number of segmentation classes you want
model = SegNet_3D(input_shape, num_classes)
model.summary()
