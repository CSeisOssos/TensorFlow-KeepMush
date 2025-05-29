from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.applications import EfficientNetB0
from config import IMG_HEIGHT, IMG_WIDTH

def criar_modelo(num_classes):
    """Cria e compila o modelo CNN"""
    print("[MODEL] Criando modelo...")
    
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False  # Congela a base do modelo

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model