import tensorflow as tf
from tensorflow.keras import layers, models
import os

# GPU check (handig om even te zien)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU gevonden! → {gpus}")
else:
    print("Geen GPU, training gaat op CPU (langzamer)")

# --------------------- Data laden ---------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "CUB_200_2011/images",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "CUB_200_2011/images",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)

class_count = len(train_ds.class_names)
print(f"Aantal vogelsoorten: {class_count}")

# --------------------- Performance optimalisatie ---------------------
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# --------------------- Model ---------------------
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    alpha=0.35,
    weights='imagenet'
)
base.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(class_count, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # laat zien hoe je model eruitziet

# --------------------- TRAINING STARTEN ---------------------
print("\n" + "="*50)
print("TRAINING GESTART!")
print("="*50)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,          # verlaag deze naar 8 of 10 anders is het model te accuraat
    verbose=1
)

# --------------------- Opslaan (100% werkt in 2025) ---------------------
save_dir = "aiproject/res"
os.makedirs(save_dir, exist_ok=True)

# 1. Native Keras formaat (beste voor later weer laden in Python)
model.save(os.path.join(save_dir, "cub200_mobilenetv2_alpha035.keras"))

# 2. SavedModel voor TensorFlow Serving / TFLite conversie
model.export(save_dir)   # ← dit is de nieuwe vervanger van model.save(folder)

print(f"\nModel succesvol opgeslagen!")
print(f"   → Keras bestand: {save_dir}/cub200_mobilenetv2_alpha035.keras")
print(f"   → SavedModel map: {save_dir}/ (voor deployment & TFLite)")

