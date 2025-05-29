import os
from sqlalchemy import create_engine

# Configurações globais
BASE_IMAGE_PATH = r"C:\SeekMush\Dataset\Fotos + Nomes\Dataset Científico"
IMG_HEIGHT = 512  # Altura das imagens
IMG_WIDTH = 512   # Largura das imagens
BATCH_SIZE = 32   # Tamanho do batch

# Conexão com o banco de dados
DB_URI = "postgresql+psycopg2://rocha:rocha123@localhost:5432/SeekMushBD"
engine = create_engine(DB_URI)

print("[CONFIG] Configurações carregadas com sucesso")