"""Renders one real LIBERO frame with the backend in MUJOCO_GL. Exit code 0 means rendering works."""
import os
import sys

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

task = benchmark.get_benchmark_dict()["libero_10"]().get_task(0)
env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                         camera_heights=64, camera_widths=64)
image = env.reset()["agentview_image"]
env.close()
if not np.std(image) > 0:
    sys.exit("rendered image is blank")
print("render ok:", os.environ.get("MUJOCO_GL"), "image std", round(float(np.std(image)), 2))
