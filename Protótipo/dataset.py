from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def preparar_datasets(df_images):
    # Verifica se o dataframe tem as colunas necessárias
    if 'filepath' not in df_images.columns or 'label' not in df_images.columns:
        raise ValueError("DataFrame deve conter colunas 'filepath' e 'label'")

    train_df, val_df = train_test_split(
        df_images,
        test_size=0.2,
        stratify=df_images['label'],
        random_state=42
    )

    # Gerador para treino (com augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        #vertical_flip=True,  # Descomente se fizer sentido para cogumelos
        fill_mode='nearest'
    )

    # Gerador para validação (sem augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Cálculo dos pesos das classes
    counter = Counter(train_df['label'])
    num_classes = len(train_gen.class_indices)  # Número total de classes
    total_samples = len(train_df)  # Número total de amostras de treino

    # Gerar pesos de classe para balanceamento
    counter = Counter(train_df['label'])
    max_count = max(counter.values())
    class_weight = {
    class_idx: (1 / count) * (total_samples / num_classes) 
    for class_name, count in counter.items() 
    for class_idx in [train_gen.class_indices.get(class_name)] 
    if class_idx is not None
}

    return train_gen, val_gen, class_weight