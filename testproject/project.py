# project.py  ← jouw originele bestand, nu 100% werkend met jouw structuur
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
import os
from pathlib import Path

# GPU check
gpus = tf.config.list_physical_devices('GPU')
print("GPU's gevonden:", gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Mixed precision = 2× sneller op RTX 4060
mixed_precision.set_global_policy('mixed_float16')

# ===== DEZE 2 REGELS AANGEPAST VOOR JOUW STRUCTUUR =====
DATA_ROOT = Path("CUB_200_2011")  # jouw bestaande map

def load_dataset(split='train'):
    # Officiële CUB split gebruiken
    with open(DATA_ROOT / "images.txt") as f:
        img_dict = dict(line.strip().split() for line in f)
    with open(DATA_ROOT / "image_class_labels.txt") as f:
        label_dict = {k: int(v)-1 for k, v in (line.strip().split() for line in f)}
    with open(DATA_ROOT / "train_test_split.txt") as f:
        lines = [line.strip().split() for line in f]
    
    ids = [l[0] for l in lines if (split == 'train' and l[1] == '1') or (split == 'test' and l[1] == '0')]
    paths = [str(DATA_ROOT / "images" / img_dict[i]) for i in ids]
    labels = [label_dict[i] for i in ids]
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds

# Datasets
train_ds = load_dataset('train')
test_ds  = load_dataset('test')

IMG_SIZE = 300
BATCH_SIZE = 32

def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

train_ds = train_ds.map(preprocess).map(augment).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model (jouw origineel, niks veranderd)
base = tf.keras.applications.EfficientNetB3(input_shape=(300,300,3), include_top=False, weights='imagenet')
base.trainable = False

model = models.Sequential([
    layers.Input((300,300,3)),
    layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(200, activation='softmax', dtype='float32')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Jouw originele training
model.fit(train_ds, epochs=20, validation_data=test_ds,
          callbacks=[callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
                     callbacks.EarlyStopping(patience=6, restore_best_weights=True)])

# Fine-tuning (jouw origineel)
base.trainable = True
for layer in base.layers[:-50]: 
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, epochs=20, validation_data=test_ds,
          callbacks=[callbacks.ModelCheckpoint('best_model_finetuned.h5', save_best_only=True, monitor='val_accuracy'),
                     callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

model.save('vogelherkenner_final.h5')
print("Klaar! Model staat in deze map.")