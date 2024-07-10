FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
WORKDIR /workspace

# Install system packages
RUN apt update && \
    apt install -y python3 python3-pip git build-essential prusa-slicer libfuse2 \
    dbus-x11 \
    libgtk2.0-dev \
    libwx-perl \
    libxmu-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    python3-dev \
    ninja-build \
    mesa-utils \
    xdg-utils \
    locales \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Install pip packages
RUN pip install --upgrade setuptools

# Download and setup the project
RUN git clone https://github.com/Eldoprano/Project-3D-Gen-4-Print
WORKDIR /workspace/Project-3D-Gen-4-Print
RUN pip install -r ./requirements.txt
RUN pip install git+https://github.com/tatsy/torchmcubes.git

# Make ports available to the world outside this container
EXPOSE 8888 7860

# For some reason, torchcubes doesn't build with GPU support
# so, for faster (2X speed) TripoSR, you should attach a terminal to the docker image, and run:
#  pip uninstall torchmcubes -y
#  pip install git+https://github.com/tatsy/torchmcubes.git