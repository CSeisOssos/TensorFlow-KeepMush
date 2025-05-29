from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
def preparar_datasets(df_images):
    """Prepara os geradores de treino e validação"""
    print("[DATASET] Preparando datasets...")
    
    # Divisão treino/validação
    train_df, val_df = train_test_split(
        df_images, 
        test_size=0.2, 
        stratify=df_images['label'], 
        random_state=42
 
    )
    
    print(f"[DATASET] Amostras de treino: {len(train_df)}")
    print(f"[DATASET] Amostras de validação: {len(val_df)}")

    # Geradores de imagem
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    print(f"[DATASET] Classes mapeadas: {train_gen.class_indices}")
    print("[DATASET] Geradores criados com sucesso")
    
    return train_gen, val_gen