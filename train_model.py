import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # CPU mode
import tensorflow as tf

# Instellingen
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6 # Iets langer trainen omdat we het moeilijker maken
DATA_DIR = 'coral_dataset'

# Data laden (alleen jouw 8 mappen!)
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'val'),
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# STRENGE Data Augmentation
# Dit zorgt dat het model niet lui wordt.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2), # Flink draaien
    tf.keras.layers.RandomZoom(0.3),     # Flink in/uitzoomen
    tf.keras.layers.RandomContrast(0.2), # Contrast aanpassen
    tf.keras.layers.RandomBrightness(0.2), # Helderheid aanpassen
])

# Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs) # Eerst vervormen
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x) # 40% vergeten (dwingt het model om beter te leren)
outputs = tf.keras.layers.Dense(len(train_ds.class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Start 'Strenge' training op 8 vogels...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Converteren
def representative_data_gen():
    for input_value, _ in train_ds.take(100):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open('nabirds_strict_quant.tflite', 'wb') as f:
    f.write(tflite_model)

print("Klaar. Gebruik 'nabirds_strict_quant.tflite' en zet je Threshold op 0.90!")
