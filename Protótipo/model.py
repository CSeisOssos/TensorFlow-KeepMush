from keras._tf_keras.keras.models import Sequential
import tensorflow as tf
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras._tf_keras.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from keras._tf_keras.keras.regularizers import l2
from config import IMG_HEIGHT, IMG_WIDTH

def criar_modelo(num_classes):
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # Bloco 1
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 3
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Classificação
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model