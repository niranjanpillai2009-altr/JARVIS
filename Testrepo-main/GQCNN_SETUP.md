GQ-CNN Setup and Model Download
================================

This project includes a lightweight `gqcnn_wrapper.py` scaffold that calls
the Berkeley GQ-CNN tooling. To get real grasp candidates you must install
and configure the official `gqcnn` package and download the pretrained
models (e.g. `GQCNN-4.0-PJ`). Follow one of the approaches below.

1) Quick (pip) install — recommended for offline evaluation on saved images
-----------------------------------------------------------------------
- Clone the repository and install into a virtual environment:

```bash
git clone https://github.com/BerkeleyAutomation/gqcnn.git
cd gqcnn
python -m venv .venv
source .venv/bin/activate   # on Windows use `.venv\Scripts\activate`
pip install -U pip setuptools
pip install .
```

- Download pre-trained models and sample data (from repo root):

```bash
./scripts/downloads/download_example_data.sh
./scripts/downloads/models/download_models.sh
```

Notes:
- The gqcnn repo historically targets Linux; on Windows prefer WSL2 (Ubuntu) or use Docker.
- The official docs note Python 3.5–3.7 compatibility; newer installs *may* work but use a virtualenv.

2) From-source + Docker (recommended if you want GPU acceleration)
---------------------------------------------------------------
- Build/run the provided Docker image (recommended on Windows):

```bash
git clone https://github.com/BerkeleyAutomation/gqcnn.git
cd gqcnn
./scripts/docker/build-docker.sh
# CPU image
docker run --rm -it -v $(pwd):/workspace gqcnn/cpu /bin/bash
# or GPU (requires nvidia-docker / NVIDIA runtime)
docker run --rm -it --gpus all -v $(pwd):/workspace gqcnn/gpu /bin/bash
```

Inside the container you can run the example policy scripts and download models as in the tutorial.

3) ROS installation (for physical robots)
-----------------------------------------
If you plan to use the ROS grasp planning service (recommended for integration with a real robot), follow the ROS installation steps in the gqcnn docs and launch the service:

```bash
# clone into a catkin workspace
cd <catkin_ws>/src
git clone https://github.com/BerkeleyAutomation/gqcnn.git
cd <catkin_ws>
catkin_make
source devel/setup.bash
# launch the grasp planning service with the parallel-jaw pretrained model
roslaunch gqcnn grasp_planning_service.launch model_name:=GQCNN-4.0-PJ
```

How to test with the example Python policy
-----------------------------------------
From the top-level `gqcnn` repository run:

```bash
python examples/policy.py GQCNN-4.0-PJ --depth_image data/examples/clutter/phoxi/dex-net_4.0/depth_0.npy \
  --segmask data/examples/clutter/phoxi/dex-net_4.0/segmask_0.png --camera_intr data/calib/phoxi/phoxi.intr
```

If this outputs candidate grasps, the model and policy are working.

Integrating with this simulator (`jarvis4`)
------------------------------------------
- The simulator's `jarvis4` command writes the EE depth image to a temporary
  `.npy` file and calls the wrapper `gqcnn_wrapper.detect_gqcnn_grasps(...)`.
- Ensure your Python environment used to run the simulator has access to the
  same `gqcnn` installation (activate the same virtualenv or run inside the
  same Docker/WSL session).

Troubleshooting
---------------
- If `detect_gqcnn_grasps` raises an ImportError, confirm the package is
  installed in the Python interpreter used by the simulator (`python -c "import gqcnn; print(gqcnn.__file__)"`).
- On Windows, the model downloader scripts are shell scripts — run them from
  WSL, Git Bash, or inside the Docker container.
- If the pretrained model was downloaded to a non-default location, set the
  `GQCNN_MODEL_DIR` environment variable or launch the ROS grasp planning
  service with `model_dir:=/path/to/models`.

References
----------
- Tutorial: https://berkeleyautomation.github.io/gqcnn/tutorials/tutorial.html#grasp-planning
- Install:  https://berkeleyautomation.github.io/gqcnn/install/install.html
- Repo:     https://github.com/BerkeleyAutomation/gqcnn
