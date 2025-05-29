import os
import pandas as pd
from config import engine

def carregar_dados_do_banco():
    """Carrega os caminhos das imagens e labels do banco de dados"""
    print("[DATABASE] Carregando dados do banco...")
    
    query = "SELECT scientific_name, photo_path FROM mushrooms"
    df = pd.read_sql(query, engine)
    
    data = []
    for _, row in df.iterrows():
        pasta = row['photo_path']
        if not os.path.isdir(pasta):
            print(f"[DATABASE] Atenção: Diretório não encontrado - {pasta}")
            continue
            
        for arquivo in os.listdir(pasta):
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho_img = os.path.join(pasta, arquivo)
                data.append({
                    'filepath': caminho_img, 
                    'label': row['scientific_name']
                })
    
    df_images = pd.DataFrame(data)
    print(f"[DATABASE] Total de imagens carregadas: {len(df_images)}")
    print(f"[DATABASE] Classes encontradas: {df_images['label'].unique()}")
    
    return df_images