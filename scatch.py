import harp
import os
import numpy as np
import matplotlib.pyplot as plt 
data_dir = '/home/adamranson/temp/harp_bin/'
os.chdir(data_dir)
reader = harp.create_reader(data_dir)

# data_read = reader.OperationControl.read('Behavior_10.bin')
data_read = reader.AnalogData.read()
data_read = harp.io.read('Behavior_44.bin')
data_read_np = np.array(data_read)
plt.figure()

# plt.plot(data_read_np[:,0], data_read_np[:,1])
plt.plot(data_read_np[:,0])
plt.show()
x = 0
