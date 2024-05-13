# sudo apt-get install -y \
#     libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran \
#     libgfortran5 libatlas3-base libatlas-base-dev \
#     libopenblas-dev libopenblas-base libblas-dev \
#     liblapack-dev cython3 libatlas-base-dev openmpi-bin \
#     libopenmpi-dev python3-dev python-is-python3
# pip3 install pip --upgrade

conda env remove -n squat

conda create -n squat python=3.9


conda activate squat

pip install tensorflow==2.10.0

pip install jupyterlab
pip install opencv-python
pip3 install matplotlib
pip3 install imageio

python
>>> import tensorflow as tf
>>> print(tf.__version__)
