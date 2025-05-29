from database import carregar_dados_do_banco
from dataset import preparar_datasets
from model import criar_modelo
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, TensorBoard
import tensorflow as tf

def main():
    try:
        # Etapa 1: Carregar dados
        print("\n=== ETAPA 1: Carregando dados ===")
        df_images = carregar_dados_do_banco()
        
        if df_images.empty:
            raise ValueError("Nenhuma imagem foi carregada do banco de dados")
        
        # Etapa 2: Preparar datasets (agora com augmentation)
        print("\n=== ETAPA 2: Preparando datasets ===")
        train_gen, val_gen = preparar_datasets(df_images)
        
        # Debug: Verificar um batch
        print("\n[DEBUG] Verificando um batch de treino...")
        batch = next(iter(train_gen))
        print(f"Tipo do batch: {type(batch)}")
        print(f"Quantidade de elementos: {len(batch)}")
        print(f"Shape das imagens: {batch[0].shape}")
        print(f"Shape dos labels: {batch[1].shape}")
        
        # Etapa 3: Criar modelo
        print("\n=== ETAPA 3: Criando modelo ===")
        model = criar_modelo(num_classes=len(train_gen.class_indices))
        model.summary()
        
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