import tensorflow as tf

ngpus = len(tf.config.list_physical_devices('GPU'))

try:
    assert ngpus > 0
except:
    print(f'GPU problems: expecting at least 1 GPU, found {ngpus}')

    
