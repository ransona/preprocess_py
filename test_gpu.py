import tensorflow as tf

ngpus = len(tf.config.list_physical_devices('GPU'))

try:
    assert len(ngpus == 2)
except:
    print(f'GPU problems: expecting 2 GPUs, found {ngpus}')
