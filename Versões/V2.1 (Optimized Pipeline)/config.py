import os
from sqlalchemy import create_engine
import torch

# Configurações globais
BASE_IMAGE_PATH = r"C:\SeekMush\Dataset\Fotos + Nomes\Dataset Científico"
IMG_HEIGHT = 256  # Altura das imagens
IMG_WIDTH = 256   # Largura das imagens
BATCH_SIZE = 16   # Tamanho do batch

# Conexão com o banco de dados
DB_URI = "postgresql+psycopg2://rocha:rocha123@localhost:5432/SeekMushBD"
engine = create_engine(DB_URI)

print("[CONFIG] Configurações carregadas com sucesso")