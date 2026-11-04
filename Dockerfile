FROM ros:humble-perception-jammy
# ARG USERNAME=ros2user
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID
ARG CUDA_COMPUTE_CAPABILITY="8.6"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

ENV DEBIAN_FRONTEND=noninteractive

# Create the user
# RUN groupadd --gid $USER_GID $USERNAME
# RUN useradd -rm -d /home/$USERNAME --uid $USER_UID --gid $USER_GID -ms /bin/bash -g root -G sudo -u 1001 -p "$(openssl passwd -1 user)" $USERNAME

# Add sudo support
# RUN apt-get -y update
# RUN apt-get install -y sudo
# RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
# RUN chmod 0440 /etc/sudoers.d/$USERNAME
# RUN apt-get update && apt-get upgrade -y

# passwd -d $USER && \
    # adduser $USER sudo && \
    # echo "$USER ALL=(ALL) NOPASSWD:ALL" | tee -a /etc/sudoers

# Install a few important dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ament-cmake \
    ccls \
    python3-colcon-common-extensions \
    python3-pip \
    python3-dev \
    build-essential \
    nano \
    vim \
    clang \
    clang-format \
    clang-tidy \
    # ros-humble-turtlesim \
    # ros-humble-rqt* \
    terminator

# RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc
# ENV SHELL /bin/bash

WORKDIR /app

# COPY install_nvidia_drivers.sh /home/${USERNAME}/
# RUN chmod +x /home/${USERNAME}/install_nvidia_drivers.sh
COPY install_cuda_toolkit.sh /
RUN chmod +x /install_cuda_toolkit.sh
# COPY install_cuda_toolkit.sh /home/${USERNAME}/
# RUN chmod +x /home/${USERNAME}/install_cuda_toolkit.sh

RUN apt-get install wget

# RUN cd /home/${USERNAME}/ && \
#     ./install_nvidia_drivers.sh
RUN /install_cuda_toolkit.sh
# RUN cd /home/${USERNAME}/ && \
#     ./install_cuda_toolkit.sh

RUN echo "export PATH="/usr/local/cuda-13/bin:$PATH"" >> /home/$USERNAME/.bashrc
RUN echo "export LD_LIBRARY_PATH="/usr/local/cuda-13/lib64:$LD_LIBRARY_PATH"" >> /home/$USERNAME/.bashrc

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

RUN mkdir ./submodules
COPY ./submodules ./submodules

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080
# RUN pip install \
#     numpy==2.2.6 \
#     matplotlib==3.10.9 \
#     opencv-python==4.6.0.66 \
#     # torch==2.14.0
#     # torchvision==0.29.0
#     # pytorch-cuda==13.0
#     pyyaml==6.0.3 \
#     scipy==1.15.3 \
#     open3d==0.20.0

# RUN pip install torch torchvision --ignore-installed --index-url https://download.pytorch.org/whl/cu130

# RUN pip install \
#     pytorch_kinematics==0.10.0

COPY ./src ./src
COPY ./pyproject.toml ./pyproject.toml

RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$CUDA_COMPUTE_CAPABILITY pip install --no-cache-dir -e ./submodules/pytorch3d && \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$CUDA_COMPUTE_CAPABILITY pip install --no-cache-dir ./submodules/Pointcept/libs/pointops && \
    pip install --no-cache-dir ./submodules/urdf_parser_py && \
    pip install --no-cache-dir .

# RUN cd /home/$USERNAME/ && \
#     git clone https://github.com/Krasnikovs/DexGrasperUni.git --recurse-submodules

# RUN cd /home/$USERNAME/DexGrasperUni && \
#     FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -e ./submodules/pytorch3d && \
#     FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install --no-build-isolation -e ./submodules/Pointcept/libs/pointops && \
#     pip install --no-build-isolation -e ./submodules/urdf_parser_py && \
#     pip install -e ./

# RUN chown -R ${USERNAME} /home/${USERNAME}/DexGrasperUni

# RUN cd /home/${USERNAME}/ && \ 
#     ./build_packages.sh

COPY ./scripts ./scripts

# Set the default user
# USER ${USERNAME}
# WORKDIR /home/${USERNAME}
# ENV SHELL /bin/terminator

CMD ["/usr/bin/terminator"]
