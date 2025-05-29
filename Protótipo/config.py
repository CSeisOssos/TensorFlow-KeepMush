import os
from sqlalchemy import create_engine

# Caminho base onde estão as pastas com imagens
BASE_IMAGE_PATH = r"C:\SeekMush\Dataset\Fotos + Nomes\Dataset Científico"

# Parâmetros da imagem
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Conexão com o banco de dados PostgreSQL
DB_URI = "postgresql+psycopg2://rocha:rocha123@localhost:5432/SeekMushBD"
engine = create_engine(DB_URI)
