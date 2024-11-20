import tensorflow as tf

print("GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))
print("Dispositivos físicos:", tf.config.list_physical_devices())
