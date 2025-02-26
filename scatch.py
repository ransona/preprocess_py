import numpy as np
wheel_diameter = 20 # cm
wheel_circumference = wheel_diameter * np.pi

import harp
import numpy as np
import matplotlib.pyplot as plt 
data_read = harp.io.read('/data/Remote_Repository/ESMT206/2025-02-25_02_ESMT206/2025-02-25_02_ESMT206_Behavior_Event44.bin')
data_read_np = np.array(data_read)
plt.figure()
plt.plot(data_read_np[:,0])
# block with plot
plt.show(block=True)
