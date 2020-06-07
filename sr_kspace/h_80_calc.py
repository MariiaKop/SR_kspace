import numpy as np
import os
from tqdm import tqdm


PATH_TO_LR_TRAIN = 'data/ax_t2_re_im_80_train'
PATH_TO_LR_MEAN_KSPACE = 'data/lr_80_mean_kspace.npy'

lr_samples = [s for s in os.listdir(PATH_TO_LR_TRAIN) if s.endswith('npy')]

mean_k_space = np.zeros((80, 80), dtype=np.float32)

for s in tqdm(lr_samples):
    slice = np.load(os.path.join(PATH_TO_LR_TRAIN, s))
    k_space_abs = np.sqrt(slice[0]**2 + slice[1]**2)
    mean_k_space += k_space_abs

mean_k_space /= len(lr_samples)
with open(PATH_TO_LR_MEAN_KSPACE, 'wb') as f:
    pass
    np.save(f, mean_k_space)