
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

.venv/bin/pip install -r requirements.txt

.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" .venv/bin/pip install --no-build-isolation -e ./submodules/pytorch3d
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" .venv/bin/pip install --no-build-isolation -e ./submodules/Pointcept/libs/pointops
.venv/bin/pip install --no-build-isolation -e ./submodules/urdf_parser_py

.venv/bin/pip install -e ./
