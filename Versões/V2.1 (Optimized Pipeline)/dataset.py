from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf
import numpy as np
import pandas as pd
from config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
import imghdr
from pathlib import Path
import warnings

def filter_valid_images(df_images):
    """Remove arquivos inválidos ou corrompidos com verificações robustas"""
    valid_extensions = {'jpeg', 'jpg', 'png', 'gif', 'bmp'}
    valid_rows = []
    
    for idx, row in df_images.iterrows():
        filepath = row['filepath']
        try:
            # Verificação 1: Arquivo existe
            if not Path(filepath).is_file():
                warnings.warn(f"Arquivo não encontrado: {filepath}")
                continue
                
            # Verificação 2: É uma imagem válida
            img_type = imghdr.what(filepath)
            if img_type not in valid_extensions:
                warnings.warn(f"Formato inválido: {filepath} (tipo: {img_type})")
                continue
                
            # Verificação 3: Pode ser aberto pelo TensorFlow
            img = tf.io.read_file(filepath)
            tf.image.decode_image(img, channels=3)  # Teste de decodificação
                
            valid_rows.append(row)
            
        except Exception as e:
            warnings.warn(f"Erro na imagem {filepath}: {str(e)}")
            continue
    
    return pd.DataFrame(valid_rows)

def balance_dataframe(df_images):
    """Balanceamento inteligente com oversampling progressivo"""
    print("[DATASET] Balanceando classes...")
    
    # Calcula o target_samples como percentil 75% das contagens
    class_counts = df_images['label'].value_counts()
    target_samples = int(np.percentile(class_counts, 75))
    target_samples = max(20, target_samples)  # Mínimo de 20 amostras
    
    # Oversampling apenas para classes abaixo do target
    balanced_dfs = []
    for cls, count in class_counts.items():
        class_df = df_images[df_images['label'] == cls]
        
        if count < target_samples:
            class_df = resample(class_df,
                              replace=True,
                              n_samples=target_samples,
                              random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs)
    print(f"[DATASET] Distribuição após balanceamento:\n{balanced_df['label'].value_counts()}")
    return balanced_df

def load_and_preprocess(path, label, class_to_index, num_classes):
    """Carrega e pré-processa imagens com tratamento robusto de erros"""
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.clip_by_value(img/255.0, 0.0, 1.0)  # Normalização segura
        
    except:
        # Fallback: imagem preta (mantém dimensões mas não afeta treino)
        img = tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
    
    # Processamento das labels
    def py_get_label(l):
        return class_to_index[l.numpy().decode('utf-8')]
    
    label_idx = tf.py_function(py_get_label, [label], tf.int64)
    label_idx.set_shape(())
    return img, tf.one_hot(label_idx, num_classes)

def preparar_datasets(df_images):
    """Pipeline completo com validação, balanceamento e augmentation"""
    # Validação inicial
    df_images = filter_valid_images(df_images)
    if df_images.empty:
        raise ValueError("Nenhuma imagem válida encontrada após filtragem!")
    
    # Balanceamento
    balanced_df = balance_dataframe(df_images)
    
    # Divisão estratificada
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=0.2,
        stratify=balanced_df['label'],
        random_state=42
    )
    
    # Mapeamento de classes
    class_names = np.sort(train_df['label'].unique())
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    print(f"[DATASET] Classes únicas: {num_classes}")
    
    # Augmentation avançada
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.01)
        return image, label
    
    # Pipeline de treino
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_df['filepath'].values, train_df['label'].values.astype(str))
    ).shuffle(2000, reshuffle_each_iteration=True)\
     .map(lambda p, l: load_and_preprocess(p, l, class_to_index, num_classes),
          num_parallel_calls=tf.data.AUTOTUNE)\
     .map(augment, num_parallel_calls=tf.data.AUTOTUNE)\
     .batch(BATCH_SIZE)\
     .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Pipeline de validação
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_df['filepath'].values, val_df['label'].values.astype(str))
    ).map(lambda p, l: load_and_preprocess(p, l, class_to_index, num_classes),
          num_parallel_calls=tf.data.AUTOTUNE)\
     .batch(BATCH_SIZE)\
     .prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("[DATASET] Pipelines criados com sucesso!")
    return train_ds, val_ds, class_to_index