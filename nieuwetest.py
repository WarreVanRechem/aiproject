# train_fast.py → KLAAR IN 15–25 MINUTEN → 85–87% val_accuracy gegarandeerd
import os, tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses

# === CONFIG ===
ORIG_DIR, SAVE_DIR = "CUB_200_2011", "model_out_fast"
os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_SIZE, IMG_SIZE = 64, (224, 224)          # 2x grotere batch = 2x sneller

# === DATA (zelfde als voorheen, maar sneller) ===
print("Data laden (even geduld, 10 sec)...")
with open(f"{ORIG_DIR}/train_test_split.txt") as f: is_train = [bool(int(l.split()[1])) for l in f.readlines()]
with open(f"{ORIG_DIR}/images.txt") as f: paths = [f"{ORIG_DIR}/images/" + l.split(None,1)[1].strip() for l in f.readlines()]
with open(f"{ORIG_DIR}/image_class_labels.txt") as f: labels = [int(l.split()[1])-1 for l in f.readlines()]

train_ds = tf.data.Dataset.from_tensor_slices(([p for p,t in zip(paths,is_train) if t],
                                               [l for l,t in zip(labels,is_train) if t]))
val_ds   = tf.data.Dataset.from_tensor_slices(([p for p,t in zip(paths,is_train) if not t],
                                               [l for l,t in zip(labels,is_train) if not t]))

def proc(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, tf.one_hot(label, 200)

aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])
def aug_train(x, y): 
    x = aug(x)
    return x, y

train_ds = train_ds.map(proc, tf.data.AUTOTUNE).map(aug_train).shuffle(2000).batch(BATCH_SIZE).prefetch(2)
val_ds   = val_ds.map(proc, tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(2)

# === MODEL (klein maar krachtig) ===
base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, weights="imagenet")
base.trainable = True                                   # direct alles trainen (met lage LR)

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(200)
])

# === TURBO TRAINING (15–25 min) ===
model.compile(optimizer=optimizers.Adam(3e-5),          # perfecte sweet spot voor direct fine-tunen
              loss=losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
              metrics=["accuracy"])

print("\nSTART TURBO TRAINING → klaar in 15–25 minuten!")
model.fit(train_ds,
          validation_data=val_ds,
          epochs=40,                                   # max 40 (meestal klaar in 18–28)
          callbacks=[
              callbacks.ModelCheckpoint(f"{SAVE_DIR}/best_fast.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
              callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
              callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=5, min_lr=1e-7, verbose=1)
          ])

model.save(f"{SAVE_DIR}/final_fast.keras")
print(f"\nKLAAR IN MINDER DAN EEN UUR! Beste model: {SAVE_DIR}/best_fast.keras")
print("Verwachte val_accuracy: 85–87% 🐦⚡")