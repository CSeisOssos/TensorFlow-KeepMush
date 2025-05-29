import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
import pandas as pd
from config import IMG_HEIGHT, IMG_WIDTH
import os

def carregar_classes_do_csv(caminho_csv):
    """Carrega as classes científicas em ordem alfabética do CSV"""
    df = pd.read_csv(caminho_csv)
    classes_cientificas = sorted(df['scientific_name'].unique().tolist())
    return classes_cientificas

def main():
    # 1. Carregar todas as classes do modelo
    caminho_csv = r'C:\SeekMush\Dataset\Fotos + Nomes\translated_names.csv'
    todas_classes = carregar_classes_do_csv(caminho_csv)
    
    print(f"✅ Total de classes carregadas: {len(todas_classes)}")
    print(f"Primeiras 5 classes: {todas_classes[:5]}")
    print(f"Últimas 5 classes: {todas_classes[-5:]}")

    # 2. Carregar modelo
    modelo_path = r'C:\SeekMush\Tensorflow\Treinamento\Versões\V2.1 (Optimized Pipeline)\best_model.keras'
    modelo = tf.keras.models.load_model(modelo_path)
    
    # Verificar compatibilidade
    num_classes_modelo = modelo.output_shape[-1]
    if num_classes_modelo != len(todas_classes):
        raise ValueError(f"Modelo espera {num_classes_modelo} classes, mas CSV tem {len(todas_classes)}")

    # 3. Configurar teste para 3 classes específicas
    classes_teste = ["Amanita muscaria", "Boletus edulis", "Lactarius deliciosus"]  # Nomes científicos
    imagens_teste = [
        r"C:\Users\groch\Downloads\amanita.jpeg",
        r"C:\Users\groch\Downloads\boletus.jpg",
        r"C:\Users\groch\Downloads\lactarius.jpg"
    ]
    
    # 4. Verificar índices das classes de teste
    indices_teste = [todas_classes.index(cls) for cls in classes_teste]
    print(f"\n🔍 Índices das classes de teste: {dict(zip(classes_teste, indices_teste))}")

    # 5. Processar imagens e mostrar resultados detalhados
    resultados = []
    
    for caminho, classe_verdadeira in zip(imagens_teste, classes_teste):
        try:
            # Pré-processamento
            img = tf.keras.preprocessing.image.load_img(caminho, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predição
            pred = modelo.predict(img_array, verbose=0)[0]
            classe_predita_idx = np.argmax(pred)
            classe_predita = todas_classes[classe_predita_idx]
            confianca = pred[classe_predita_idx]
            
            # Top 5 previsões
            top5_idx = np.argsort(pred)[::-1][:5]
            top5 = [(todas_classes[i], pred[i]) for i in top5_idx]
            
            # Armazenar resultados
            resultados.append({
                'imagem': os.path.basename(caminho),
                'verdadeiro': classe_verdadeira,
                'predito': classe_predita,
                'confianca': confianca,
                'top5': top5,
                'acerto': classe_verdadeira == classe_predita
            })
            
            # Exibir resultados
            print(f"\n📌 Imagem: {os.path.basename(caminho)}")
            print(f"🏷️ Verdadeiro: {classe_verdadeira}")
            print(f"🤖 Predito: {classe_predita} (Confiança: {confianca:.2%})")
            print(f"🎯 {'ACERTO' if classe_verdadeira == classe_predita else 'ERRO'}")
            print("\n🔝 Top 5 previsões:")
            for i, (cls, prob) in enumerate(top5, 1):
                print(f"{i}. {cls}: {prob:.2%} {'✅' if cls == classe_verdadeira else ''}")
                
        except Exception as e:
            print(f"\n❌ Erro ao processar {caminho}: {str(e)}")
            resultados.append({
                'imagem': os.path.basename(caminho),
                'erro': str(e)
            })

    # 6. Matriz de confusão para as classes de teste
    try:
        y_true = [classes_teste.index(r['verdadeiro']) for r in resultados if 'verdadeiro' in r]
        y_pred = [classes_teste.index(r['predito']) if r['predito'] in classes_teste else len(classes_teste) 
                 for r in resultados if 'predito' in r]
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes_teste)))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes_teste,
                    yticklabels=classes_teste)
        
        plt.title('Matriz de Confusão (Classes de Teste)', pad=20)
        plt.xlabel('Predito', labelpad=15)
        plt.ylabel('Verdadeiro', labelpad=15)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Erro ao gerar matriz de confusão: {str(e)}")

    # 7. Relatório de classificação
    try:
        print("\n📊 Relatório de Classificação:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=classes_teste,
            zero_division=0
        ))
    except Exception as e:
        print(f"\n❌ Erro ao gerar relatório: {str(e)}")

if __name__ == "__main__":
    print("🔬 Iniciando teste do modelo...")
    main()
    print("\n🧪 Teste concluído!")