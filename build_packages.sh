
echo "Building venv and upgrading pip"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

echo "Installing needed libraries"
.venv/bin/pip install -r requirements.txt

echo "Installing torch and torchvision"
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo "Building and installing packages from submodules"

CUDA_COMPUTE_CAPABILITY="8.6"

FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$CUDA_COMPUTE_CAPABILITY .venv/bin/pip install --no-build-isolation -e ./submodules/pytorch3d
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$CUDA_COMPUTE_CAPABILITY .venv/bin/pip install --no-build-isolation -e ./submodules/Pointcept/libs/pointops
.venv/bin/pip install --no-build-isolation -e ./submodules/urdf_parser_py

echo "Building src"
.venv/bin/pip install --no-build-isolation -e ./

echo "Completed building and installing"