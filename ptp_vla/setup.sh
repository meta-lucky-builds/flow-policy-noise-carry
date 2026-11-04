#!/usr/bin/env bash
# Installs everything on a fresh Ubuntu machine. Run once, from inside this folder, as root or with sudo.
set -euo pipefail
LEROBOT_COMMIT=ba83e4cf43737e2665be0aafcb5f513b83c7d0f3

echo "--- system packages"
if command -v apt-get >/dev/null; then
  SUDO=""
  [ "$(id -u)" -eq 0 ] || SUDO=sudo
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y
  $SUDO apt-get install -y curl unzip ffmpeg build-essential pciutils kmod \
    libegl1 libgl1 libglvnd0 libglx0 libopengl0 libgles2 libglu1-mesa \
    libosmesa6 libosmesa6-dev libegl-dev libglvnd-dev libxext6 libx11-6
fi

echo "--- Python 3.12 environment"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install "lerobot[smolvla,libero] @ https://github.com/huggingface/lerobot/archive/${LEROBOT_COMMIT}.zip" pytest

echo "--- GPU"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; x = torch.ones(8, 8).cuda() @ torch.ones(8, 8).cuda(); print(torch.__version__, torch.cuda.device_count(), 'GPUs,', torch.cuda.get_device_name(0), ', compute ok', x.sum().item())"

echo "--- LIBERO first-time configuration (answers N to its custom-path question)"
printf 'N\n' | python -c "from libero.libero import benchmark; print('suites', sorted(benchmark.get_benchmark_dict())[:3])"

bash fix_render.sh

echo "--- code tests against this LeRobot commit"
python -m pytest test_vla.py test_parallel.py -q -p no:cacheprovider
echo "SETUP OK"
