from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate, Activation, BatchNormalization, Multiply, Add

def attention_block(input_tensor, gating_signal, inter_channels):
    theta_x = Conv3D(inter_channels, (1, 1, 1), strides=(1, 1, 1), padding='same')(input_tensor)
    phi_g = Conv3D(inter_channels, (1, 1, 1), strides=(1, 1, 1), padding='same')(gating_signal)
    
    f = Activation('relu')(BatchNormalization()(Add()([theta_x, phi_g])))
    
    psi_f = Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same')(f)
    
    rate = Activation('sigmoid')(psi_f)
    
    att_x = Multiply()([input_tensor, rate])
    
    return att_x

def attention_unet_3d(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    # Bridge
    conv4 = Conv3D(256, (3,3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3,3,3), activation='relu', padding='same')(conv4)
    
    # Attention gate
    gating_signal = Conv3D(256, (1, 1, 1), strides=(1, 1, 1), padding='same')(pool3)
    attention4 = attention_block(conv4, gating_signal, inter_channels=128)
    
    # Decoder
    up5 = UpSampling3D(size=(2, 2, 2))(attention4)  # Use attention4 here
    up5 = Concatenate(axis=-1)([up5, conv3])
    conv5 = Conv3D(128, (3,3,3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(128, (3,3,3), activation='relu', padding='same')(conv5)
    
    # Attention gate
    gating_signal = Conv3D(128, (1, 1, 1), strides=(1, 1, 1), padding='same')(conv5)
    attention5 = attention_block(conv5, gating_signal, inter_channels=64)
    
    up6 = UpSampling3D(size=(2, 2, 2))(attention5)  # Use attention5 here
    up6 = Concatenate(axis=-1)([up6, conv2])
    conv6 = Conv3D(64, (3,3,3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(64, (3,3,3), activation='relu', padding='same')(conv6)
    
    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    up7 = Concatenate(axis=-1)([up7, conv1])
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3,3,3), activation='relu', padding='same')(conv7)
    
    # Output
    output = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(conv7)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

if __name__=='__main__':
    # Instantiate the model
    input_shape = (128, 128, 128, 3)
    num_classes = 4
    model = attention_unet_3d(input_shape, num_classes)

    # Display model summary
    model.summary()
