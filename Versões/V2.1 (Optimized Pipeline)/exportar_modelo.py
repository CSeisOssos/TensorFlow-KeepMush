import tensorflow as tf
from datetime import datetime
import os

def exportar_modelo(modelo, pasta_saida, formato='h5'):
    """
    Exporta um modelo Keras para o formato especificado.
    
    Args:
        modelo: Modelo Keras treinado
        pasta_saida: Caminho absoluto ou relativo da pasta de destino
        formato: 'h5' ou 'savedmodel' (padrão: 'h5')
    """
    # Cria a pasta se não existir
    os.makedirs(pasta_saida, exist_ok=True)
    
    # Gera nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    nome_modelo = f"cogumelos_{timestamp}"
    
    try:
        if formato.lower() == 'h5':
            caminho = os.path.join(pasta_saida, f"{nome_modelo}.h5")
            modelo.save(caminho)
            print(f"✅ Modelo exportado como .h5 em: {caminho}")
            
        elif formato.lower() == 'savedmodel':
            caminho = os.path.join(pasta_saida, nome_modelo)
            modelo.save(caminho)
            print(f"✅ Modelo exportado como SavedModel em: {caminho}/")
            
        else:
            raise ValueError("Formato inválido. Use 'h5' ou 'savedmodel'")
            
        return caminho
        
    except Exception as e:
        print(f"❌ Falha ao exportar: {str(e)}")
        return None

# Exemplo de uso:
if __name__ == "__main__":
    # 1. Carregue seu modelo treinado (substitua pelo seu modelo real)
    # modelo = tf.keras.models.load_model('caminho/do/modelo_treinado') 
    
    # 2. Especifique ONDE quer salvar (exemplo abaixo)
    pasta_destino = input("C:\SeekMush\Tensorflow\Treinamento\Versões\V2.1 (Optimized Pipeline)").strip()
    
    # 3. Escolha o formato (com validação)
    formato = ''
    while formato.lower() not in ['h5', 'savedmodel']:
        formato = input("Formato desejado (h5/savedmodel): ").strip()
    
    # 4. Exporta (descomente a linha abaixo quando tiver o modelo carregado)
    # exportar_modelo(modelo, pasta_destino, formato)
    print("\n⚠️ Descomente as linhas do modelo real no script antes de usar!")