from database import carregar_dados_do_banco
from dataset import preparar_datasets
from model import build_model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, TensorBoard
import tensorflow as tf

print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
print("TensorFlow está usando GPU?", tf.test.is_gpu_available())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} GPUs físicas, {len(logical_gpus)} GPUs lógicas")
    except RuntimeError as e:
        print(e)

def main():
    try:
        # Etapa 1: Carregar dados
        print("\n=== ETAPA 1: Carregando dados ===")
        df_images = carregar_dados_do_banco()
        
        if df_images.empty:
            raise ValueError("Nenhuma imagem foi carregada do banco de dados")
        
        # Etapa 2: Preparar datasets (agora com augmentation)
        print("\n=== ETAPA 2: Preparando datasets ===")
        train_gen, val_gen, class_to_index = preparar_datasets(df_images)
        print(f"\n[DEBUG] Número de classes: {len(class_to_index)}")
        print(f"[DEBUG] Exemplo de mapeamento: {list(class_to_index.items())[:5] if isinstance(class_to_index, dict) else 'Mapeamento não disponível'}")
        model = build_model(num_classes=len(class_to_index))
        model.summary()
        
        # Debug: Verificar um batch
        print("\n[DEBUG] Verificando um batch de treino...")
        batch = next(iter(train_gen))
        print(f"Tipo do batch: {type(batch)}")
        print(f"Quantidade de elementos: {len(batch)}")
        images, labels = batch  # Desempacota diretamente
        print(f"Tipo das imagens: {type(images)}, Shape: {images.shape}")
        print(f"Tipo dos labels: {type(labels)}, Shape: {labels.shape}")
                
        # Etapa 4: Treinamento com callbacks
        print("\n=== ETAPA 4: Iniciando treinamento ===")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,  # Número maior pois temos EarlyStopping
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
                TensorBoard(log_dir='./logs', histogram_freq=1),
            ],
            verbose=1
        )
        
        print("\n[SUCESSO] Treinamento concluído!")
        
    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro: {str(e)}")
        raise

if __name__ == "__main__":
    print("Iniciando script de treinamento...")
    main()