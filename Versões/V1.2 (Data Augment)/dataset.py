from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def preparar_datasets(df_images):
    """Prepara os geradores com augmentation para treino"""
    print("[DATASET] Preparando datasets com augmentation...")
    
    # Divisão treino/validação
    train_df, val_df = train_test_split(
        df_images, 
        test_size=0.2, 
        stratify=df_images['label'], 
        random_state=42
    )
    
    print(f"[DATASET] Amostras de treino: {len(train_df)}")
    print(f"[DATASET] Amostras de validação: {len(val_df)}")

    # Gerador de TREINO com augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,       # Rotação aleatória de ±20 graus
        width_shift_range=0.1,   # Deslocamento horizontal de ±10%
        height_shift_range=0.1,  # Deslocamento vertical de ±10%
        zoom_range=0.2,          # Zoom aleatório de 80% a 120%
        horizontal_flip=True,    # Espelhamento horizontal
        fill_mode='nearest'      # Preenche pixels novos com os mais próximos
    )
    
    # Gerador de VALIDAÇÃO sem augmentation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    print(f"[DATASET] Classes mapeadas: {train_gen.class_indices}")
    print("[DATASET] Geradores criados com sucesso")
    
    return train_gen, val_gen  # Retorna apenas os geradores