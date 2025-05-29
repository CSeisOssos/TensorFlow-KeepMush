from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
from config import IMG_HEIGHT, IMG_WIDTH

def criar_modelo(num_classes):
    """Cria e compila o modelo CNN"""
    print("[MODEL] Criando modelo...")
    
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),  # Corrigido para usar as dimensões configuradas
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[MODEL] Modelo criado e compilado com sucesso")
    return model