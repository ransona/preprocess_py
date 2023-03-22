import os #, requests
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import suite2p
import timeit
import organise_paths

# expID
expID = '2023-03-01_01_ESMT107'
# user ID to use to place processed data
userID = 'adamranson'
skip_ca = False
animalID, remote_repository_root, \
    processed_root, exp_dir_processed, \
        exp_dir_raw = organise_paths.find_paths(userID, expID)

# Figure Style settings for notebook.

mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': True,
})
jet = mpl.cm.get_cmap('jet')
jet.set_bad(color='k')

ops = suite2p.default_ops()
ops['batch_size'] = 40 # we will decrease the batch_size in case low RAM on computer
ops['threshold_scaling'] = 1.0 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
ops['fs'] = 30 # sampling rate of recording, determines binning for cell detection
ops['tau'] = 1.25 # timescale of gcamp to use for deconvolution
ops['move_bin'] = True
ops['save_mat'] = True

db = {
  'data_path': [exp_dir_raw],
  'save_path0': exp_dir_processed,
  'save_disk': exp_dir_processed,
  #'fast_disk': '/data/fast', # <-- this is the VM's disk space
}

print(db)
print(ops)

output_ops = suite2p.run_s2p(ops=ops, db=db)