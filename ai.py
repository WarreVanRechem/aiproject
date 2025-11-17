"""
CUB-200 -> Coral (Edge TPU) complete pipeline

Dit script bevat:
 - downloaden en uitpakken van CUB_200_2011
 - dataset voorbereiding (train/val splits interactief via image_dataset_from_directory)
 - transfer learning met MobileNetV2 (alpha=0.35, input 224x224)
 - opslaan van het Keras SavedModel
 - conversie naar TFLite met int8 quantization (met representative dataset)
 - (optioneel) aanroepen van edgetpu_compiler als deze beschikbaar is
 - inferentie script voor op de Coral Dev Board (pycoral)

HOW TO USE (voorbeeld):
 1) Trainen op je machine (met GPU indien beschikbaar):
    python cub200_to_coral.py train --data-dir ./CUB_200_2011/images --epochs 15 --batch 32

 2) Converteren & quantizen naar int8 tflite:
    python cub200_to_coral.py convert --saved-model ./cub200_mobilenet_saved --tflite-out ./cub200_int8.tflite

 3) (Op je host) compileer met Edge TPU Compiler:
    edgetpu_compiler cub200_int8.tflite

 4) Kopieer het gecompileerde model naar Coral Dev Board en run inferentie (of run lokaal met USB-accelerator):
    scp cub200_int8_edgetpu.tflite mendel@<board_ip>:/home/mendel/
    python cub200_to_coral.py infer_coral --model ./cub200_int8_edgetpu.tflite --image test_bird.jpg

Requirements:
 pip install tensorflow pillow numpy
 Voor inferentie op Coral (op board of host met USB accelerator): pip install pycoral
 Voor edgetpu compile: installeer edgetpu-compiler (apt package of download van Coral site)

LET OP: training op CUB-200 is relatief groot; je hebt genoeg schijfruimte en bij voorkeur GPU.
"""

import os
import argparse
import tarfile
import urllib.request
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import sys
import subprocess


CUB_URL = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'


def download_and_extract(dest_dir: str):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tgz_path = dest / 'CUB_200_2011.tgz'
    if not tgz_path.exists():
        print('Downloading CUB_200_2011 (~140MB)...')
        urllib.request.urlretrieve(CUB_URL, tgz_path)
    else:
        print('Archive al aanwezig, overslaan download.')

    extracted_flag = dest / 'CUB_200_2011' / 'images'
    if not extracted_flag.exists():
        print('Uitpakken...')
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=dest)
    else:
        print('Dataset al uitgepakt.')

    return str(dest / 'CUB_200_2011' / 'images')


def prepare_datasets(image_dir, img_size=(224,224), batch_size=32, val_split=0.2):
    print('Voorbereiden datasets...')
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=val_split,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        validation_split=val_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f'Klassen: {len(class_names)}')

    # Prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes, input_shape=(224,224,3), alpha=0.35):
    print('Bouwen model (MobileNetV2 alpha=0.35) ...')
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        alpha=alpha,
        weights='imagenet'
    )
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train(args):
    if args.download:
        images_path = download_and_extract(args.data_dir)
    else:
        images_path = args.data_dir
    train_ds, val_ds, class_names = prepare_datasets(images_path, img_size=(224,224), batch_size=args.batch_size, val_split=0.2)

    model = build_model(num_classes=len(class_names), input_shape=(224,224,3), alpha=0.35)

    print('Start training...')
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    save_path = args.saved_model_dir
    print(f'Saving model naar {save_path} ...')
    model.save(save_path)
    print('Klaar met trainen en opslaan.')


def representative_data_gen_from_dataset(train_ds, num_steps=100):
    # produceer batches voor de representative dataset (float32)
    print('Voorbereiden representative dataset...')
    i = 0
    for batch_images, _ in train_ds:
        # Model expects preprocessed float images (MobileNet preprocess_input expects pixels in [-1,1])
        batch = tf.cast(batch_images, tf.float32)
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        for img in batch:
            if i >= num_steps:
                return
            # shape: (224,224,3)
            yield [tf.expand_dims(img, 0).numpy()]
            i += 1


def convert_to_tflite(args):
    print('Laden saved model...')
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Bouw een klein train_ds for representative
    image_dir = args.data_dir
    if image_dir is None:
        raise ValueError('data_dir is required voor representative dataset')

    train_ds, _, _ = prepare_datasets(image_dir, img_size=(224,224), batch_size=args.batch_size, val_split=0.2)

    converter.representative_dataset = lambda: representative_data_gen_from_dataset(train_ds, num_steps=200)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print('Start conversie en quantization... (kan even duren)')
    tflite_model = converter.convert()

    out_path = args.tflite_out
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Geschreven: {out_path}')

    if args.compile:
        print('Proberen te compileren met edgetpu_compiler...')
        try:
            subprocess.run(['edgetpu_compiler', out_path], check=True)
            print('edgetpu_compiler voltooid.')
        except Exception as e:
            print('Kon edgetpu_compiler niet uitvoeren:', e)
            print('Zorg dat edgetpu_compiler geïnstalleerd is op je machine.')


# -------------------- Inferentie op Coral --------------------
# Dit stuk werkt op Coral Dev Board of op host met USB Edge TPU en pycoral geïnstalleerd.
try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters.common import input_size
    from pycoral.adapters.classify import get_classes
    pycoral_available = True
except Exception:
    pycoral_available = False


def infer_on_coral(model_path, image_path, top_k=5):
    if not pycoral_available:
        raise RuntimeError('pycoral niet gevonden. Installeer pycoral (pip install pycoral) op je Coral board of host met USB accelerator.')

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    h, w = input_size(interpreter)
    img = Image.open(image_path).convert('RGB').resize((w, h))

    # Edge TPU model expects uint8 input when int8 quantized
    input_tensor = np.asarray(img)
    input_tensor = np.expand_dims(input_tensor, 0).astype('uint8')

    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    classes = get_classes(interpreter, top_k=top_k)
    print('Top resultaten:')
    for c in classes:
        # c.id is index van klasse volgens trainingsset (0..199)
        print(f'klasse={c.id}, score={c.score:.3f}')


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description='CUB-200 -> Coral pipeline')
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('--data-dir', type=str, default='./CUB_200_2011/images', help='map met images (of root waarin het tar.gz wordt gedownload)')
    p_train.add_argument('--download', action='store_true', help='download en pak CUB dataset uit')
    p_train.add_argument('--saved-model-dir', type=str, default='./cub200_mobilenet_saved', help='output saved model dir')
    p_train.add_argument('--epochs', type=int, default=10)
    p_train.add_argument('--batch-size', type=int, default=32)

    p_conv = sub.add_parser('convert')
    p_conv.add_argument('--saved-model-dir', type=str, default='./cub200_mobilenet_saved')
    p_conv.add_argument('--data-dir', type=str, required=True, help='dataset images dir (voor representative dataset)')
    p_conv.add_argument('--tflite-out', type=str, default='./cub200_int8.tflite')
    p_conv.add_argument('--batch-size', type=int, default=32)
    p_conv.add_argument('--compile', action='store_true', help='probeer edgetpu_compiler aan te roepen als beschikbaar')

    p_inf = sub.add_parser('infer_coral')
    p_inf.add_argument('--model', type=str, required=True, help='path naar *edgetpu gecompileerde* tflite, bijv cub200_int8_edgetpu.tflite')
    p_inf.add_argument('--image', type=str, required=True)
    p_inf.add_argument('--topk', type=int, default=5)

    args = parser.parse_args()

    if args.cmd == 'train':
        if args.download:
            images_dir = download_and_extract(args.data_dir)
            args.data_dir = images_dir
        train(args)
    elif args.cmd == 'convert':
        convert_to_tflite(args)
    elif args.cmd == 'infer_coral':
        infer_on_coral(args.model, args.image, top_k=args.topk)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
