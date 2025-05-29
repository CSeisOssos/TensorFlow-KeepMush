from database import carregar_dados_do_banco
from dataset import preparar_datasets
from model import criar_modelo
import tensorflow as tf
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("GPU disponível:", tf.config.list_physical_devices('GPU'))

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Etapa 1: Carregar dados do banco
    df_images = carregar_dados_do_banco()
    
    # Verificação rápida dos dados
    print(f"Total de imagens carregadas: {len(df_images)}")
    print(f"Classes encontradas: {df_images['label'].unique()}")

    # Etapa 2: Preparar datasets
    train_gen, val_gen, class_weight = preparar_datasets(df_images)
    
    # Debug: Verificar um batch
    batch = next(iter(train_gen))
    print(f"\nDebug - Batch shape: {batch[0].shape}, Labels shape: {batch[1].shape}")
    print(f"Classes mapeadas: {train_gen.class_indices}")

    # Etapa 3: Criar modelo
    model = criar_modelo(num_classes=len(train_gen.class_indices))
    model.summary()

    # Etapa 4: Treinamento
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        class_weight=class_weight,
        callbacks=[
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',  # Note a extensão .keras
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger('training_log.csv')
]
    )
    
    # Plotar resultados
    plot_training_history(history)

if __name__ == "__main__":
    main()
