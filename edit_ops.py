import numpy as np
filename= '/home/yannickbollmann/data/Repository/ESYB007/2025-04-13_03_ESYB007/P2/R001/suite2p/plane0/ops.npy'
# Load the .npy file
ops = np.load(filename, allow_pickle=True).item()
print(ops)
# Modify batch size
ops['batch_size'] = 500
ops['block_size'] = [256, 256]
# Save it back
np.save(filename, ops)