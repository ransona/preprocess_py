import numpy as np
import matplotlib.pyplot as plt
rand_traces = np.random.rand(10,100)

FMins = np.min(rand_traces, axis=1)
plt.plot(FMins)
plt.show()

x=0