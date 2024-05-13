sudo apt-get install -y \
    libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran \
    libgfortran5 libatlas3-base libatlas-base-dev \
    libopenblas-dev libopenblas-base libblas-dev \
    liblapack-dev cython3 libatlas-base-dev openmpi-bin \
    libopenmpi-dev python3-dev python-is-python3
pip3 install pip --upgrade
pip3 install keras_applications==1.0.8 --no-deps
pip3 install keras_preprocessing==1.1.2 --no-deps
pip3 install numpy==1.24.2
pip3 install h5py==3.6.0
pip3 install pybind11==2.9.2
pip3 install -U --user six wheel mock
pip3 uninstall tensorflow

TFVER=2.10.0

PYVER=39

ARCH=`python -c 'import platform; print(platform.machine())'`
echo CPU ARCH: ${ARCH}

pip3 install \
--no-cache-dir \
https://github.com/PINTO0309/Tensorflow-bin/releases/download/v${TFVER}/tensorflow-${TFVER}-cp${PYVER}-none-linux_${ARCH}.whl


pip3 install tensorflow-hub
pip3 install git+https://github.com/tensorflow/docs

pip3 install imageio
pip3 install matplotlib