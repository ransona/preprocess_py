import numpy as np
import matplotlib.pyplot as plt

bv_flip_intervals = [0,0,0,1,0]
tl_flip_intervals = [0,0,1,0,0]
# corr
min_length = min(len(bv_flip_intervals), len(tl_flip_intervals))
correlation = np.correlate(bv_flip_intervals[0:min_length], tl_flip_intervals[0:min_length], mode='full')
lags = np.arange(-len(bv_flip_intervals[0:min_length]) + 1, len(bv_flip_intervals[0:min_length]))
# Find the lag corresponding to the maximum correlation
max_correlation_index = np.argmax(correlation)
lag_in_samples = lags[max_correlation_index]
print('lag = ' + str(lag_in_samples))
plt.plot(bv_flip_intervals)
plt.show()