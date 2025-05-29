import os
import pandas as pd
from config import engine

def carregar_dados_do_banco():
    query = "SELECT scientific_name, photo_path FROM mushrooms"
    df = pd.read_sql(query, engine)
    
    data = []
    for _, row in df.iterrows():
        pasta = row['photo_path']
        if not os.path.isdir(pasta):
            print(f"Diretório não encontrado: {pasta}")  # Debug
            continue
        for arquivo in os.listdir(pasta):
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho_img = os.path.join(pasta, arquivo)
                if not os.path.exists(caminho_img):  # Verificação adicional
                    print(f"Imagem não encontrada: {caminho_img}")
                    continue
                data.append({'filepath': caminho_img, 'label': row['scientific_name']})
    
    print(f"Total de imagens válidas encontradas: {len(data)}")  # Debug
    return pd.DataFrame(data)
