from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.applications import EfficientNetB3
from config import IMG_HEIGHT, IMG_WIDTH
import tensorflow as tf

def build_model(num_classes):
    # Base Model
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    # Fine-tuning parcial (descongela camadas superiores)
    for layer in base_model.layers[:int(len(base_model.layers)*0.7)]:
        layer.trainable = False
    for layer in base_model.layers[int(len(base_model.layers)*0.7):]:
        layer.trainable = True

    # Head customizada
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)  # Novo
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)  # Aumentado para 512
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Compilação
    model = Model(inputs=base_model.input, outputs=outputs)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9
    )
    
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),  # Novo
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("[MODEL] EfficientNetB0 com cabeça customizada construída.")
    return model