from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
from config import IMG_HEIGHT, IMG_WIDTH

def criar_modelo(num_classes):
    """Cria e compila o modelo CNN"""
    print("[MODEL] Criando modelo...")
    
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),  
        
        # Bloco 1
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Bloco 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Bloco 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        # Classificação
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    print("[MODEL] Modelo criado e compilado com sucesso")
    return model