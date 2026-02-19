import os
import sys

import dolfin as dl
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.environ.get("HIPPYFLOW_PATH"))
import hippyflow as hf

# Since the problems are

data_dir = f"/storage/joshua/hyperelasticity/amortizedProblem/samples/"

hf.compress_dataset(
    data_dir,
    derivatives=(0, 0, 0),
    clean_up=False,
    has_z_data=False,
    has_y_data=True,
    derivatives_only=False,
    save=True,
    n_problems=15,
)

hf.compress_dataset(
    data_dir,
    derivatives=(1, 0, 0),
    clean_up=False,
    has_z_data=False,
    has_y_data=True,
    derivatives_only=True,
    save=True,
    n_problems=15,
)

hf.compress_dataset(
    data_dir,
    derivatives=(0, 0, 1),
    clean_up=True,
    has_z_data=False,
    has_y_data=True,
    derivatives_only=True,
    save=True,
    n_problems=15,
)

mqy_data = np.load(data_dir + "mqy_data.npz")
np.save(data_dir + "X_data.npy", mqy_data["m_data"])
np.save(data_dir + "fX_data.npy", mqy_data["q_data"])
np.save(
    data_dir + "dfXdX_data.npy",
    np.load(data_dir + "JstarPhi_data.npz")["JstarPhi_data"],
)
np.save(data_dir + "Y_data.npy", mqy_data["y_data"])  # in case we need them for the future
scores = np.load(data_dir + "score_data.npz")
np.save(data_dir + "score_data.npy", scores["score_data"])  # in case we need them for the future
