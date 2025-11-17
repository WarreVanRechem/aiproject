import tensorflow as tf
print("TF versie:", tf.__version__)
print("GPU's gedetecteerd:", len(tf.config.list_physical_devices('GPU')))