FROM ros:humble-perception-jammy
ARG USERNAME=ros2user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG CUDA_COMPUTE_CAPABILITY="8.6"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

ENV DEBIAN_FRONTEND=noninteractive

# Create the user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd -rm -d /home/$USERNAME --uid $USER_UID --gid $USER_GID -s /bin/bash -g root -G sudo -u 1001 -p "$(openssl passwd -1 user)" $USERNAME
# -ms /bin/bash

# Add sudo support
# RUN apt-get -y update
# RUN apt-get install -y sudo
# RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
# RUN chmod 0440 /etc/sudoers.d/$USERNAME
# RUN apt-get update && apt-get upgrade -y

RUN passwd -d $USERNAME && \
    adduser $USERNAME sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" | tee -a /etc/sudoers

# Install a few important dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ament-cmake \
    ccls \
    # python3-colcon-common-extensions \
    python3-pip \
    python3-dev \
    build-essential \
    nano \
    vim \
    clang \
    clang-format \
    clang-tidy \
    libusb-1.0-0-dev \
    # ros-humble-turtlesim \
    # ros-humble-rqt* \
    terminator

RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc
ENV SHELL=/bin/bash

# WORKDIR /app
WORKDIR /home/${USERNAME}

# COPY install_nvidia_drivers.sh /home/${USERNAME}/
# RUN chmod +x /home/${USERNAME}/install_nvidia_drivers.sh
# COPY install_cuda_toolkit.sh /
# RUN chmod +x /install_cuda_toolkit.sh
COPY install_cuda_toolkit.sh ./
RUN chmod +x ./install_cuda_toolkit.sh

RUN apt-get install wget

# RUN cd /home/${USERNAME}/ && \
#     ./install_nvidia_drivers.sh
RUN ./install_cuda_toolkit.sh
# RUN cd /home/${USERNAME}/ && \
#     ./install_cuda_toolkit.sh

RUN echo 'export PATH="/usr/local/cuda-13/bin:$PATH"' >> /home/$USERNAME/.bashrc
RUN echo 'export LD_LIBRARY_PATH="/usr/local/cuda-13/lib64:$LD_LIBRARY_PATH"' >> /home/$USERNAME/.bashrc

RUN python3 -m pip install --upgrade wheel
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools==79.0.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

RUN mkdir ./submodules
COPY ./submodules ./submodules

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# EXPOSE 8080

COPY ./src ./src
COPY ./pyproject.toml ./pyproject.toml

RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$CUDA_COMPUTE_CAPABILITY pip install --no-build-isolation --no-cache-dir -e ./submodules/pytorch3d && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$CUDA_COMPUTE_CAPABILITY pip install --no-build-isolation --no-cache-dir ./submodules/Pointcept/libs/pointops && \
    pip install --no-build-isolation --no-cache-dir ./submodules/urdf_parser_py && \
    pip install --no-build-isolation --no-cache-dir .

# RUN chown -R ${USERNAME} /home/${USERNAME}/DexGrasperUni

COPY ./scripts ./scripts

COPY ./params ./params
COPY ./datasets ./datasets
COPY ./assets ./assets

RUN chown -R ${USERNAME} datasets

# Set the default user
USER ${USERNAME}
# WORKDIR /home/${USERNAME}
# ENV SHELL /bin/terminator

CMD ["/usr/bin/terminator"]
