import tensorflow as tf
try:
    assert len(tf.config.list_physical_devices('GPU')) == 1
except:
    print('GPU problems')